#include "fixed_block_read_plan.h"
#include "kvstore.h"
#include "read_bandwidth_limit.h"
#include <torch/torch.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <unordered_set>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <linux/io_uring.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <pybind11/stl.h>
#include <omp.h>
#include <thread>
#include <future>
#include <condition_variable>
#include <chrono>

namespace {

constexpr unsigned int kDefaultIoUringQueueDepth = 32;
constexpr unsigned int kDefaultReorderIoUringQueueDepth = 8;
constexpr size_t kDirectIoAlignment = 4096;

struct ForegroundReadGuard {
    std::atomic<int>& readers;
    explicit ForegroundReadGuard(std::atomic<int>& value) : readers(value) { ++readers; }
    ~ForegroundReadGuard() { --readers; }
};

struct FreeDeleter {
    void operator()(void* pointer) const {
        std::free(pointer);
    }
};

using AlignedBuffer = std::unique_ptr<void, FreeDeleter>;

AlignedBuffer allocate_aligned_buffer(size_t size) {
    void* pointer = nullptr;
    int result = posix_memalign(&pointer, kDirectIoAlignment, size);
    if (result != 0) {
        throw std::runtime_error(
            "Failed to allocate aligned I/O buffer: " +
            std::string(std::strerror(result)));
    }
    return AlignedBuffer(pointer);
}

bool align_direct_read(
    uint64_t offset,
    uint64_t length,
    uint64_t& aligned_offset,
    size_t& data_offset,
    size_t& aligned_length
) {
    aligned_offset = offset - offset % kDirectIoAlignment;
    uint64_t prefix = offset - aligned_offset;
    if (length == 0 || length > std::numeric_limits<uint64_t>::max() - prefix) {
        return false;
    }

    uint64_t required = prefix + length;
    if (required > std::numeric_limits<uint64_t>::max() -
                       (kDirectIoAlignment - 1)) {
        return false;
    }
    uint64_t rounded =
        (required + kDirectIoAlignment - 1) / kDirectIoAlignment *
        kDirectIoAlignment;
    if (prefix > std::numeric_limits<size_t>::max() ||
        rounded > std::numeric_limits<size_t>::max()) {
        return false;
    }
    data_offset = static_cast<size_t>(prefix);
    aligned_length = static_cast<size_t>(rounded);
    return true;
}

unsigned int get_iouring_queue_depth() {
    static const unsigned int queue_depth = []() {
        const char* value = std::getenv("KVSTORE_IOURING_QD");
        if (value == nullptr || *value == '\0') {
            return kDefaultIoUringQueueDepth;
        }

        char* end = nullptr;
        errno = 0;
        unsigned long parsed = std::strtoul(value, &end, 10);
        if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
            std::cerr << "Invalid KVSTORE_IOURING_QD='" << value
                      << "'; using " << kDefaultIoUringQueueDepth << std::endl;
            return kDefaultIoUringQueueDepth;
        }
        return static_cast<unsigned int>(std::min<unsigned long>(parsed, 4096));
    }();
    return queue_depth;
}

unsigned int get_reorder_iouring_queue_depth() {
    static const unsigned int queue_depth = []() {
        const char* value = std::getenv("KVSTORE_REORDER_IOURING_QD");
        if (value == nullptr || *value == '\0') {
            return kDefaultReorderIoUringQueueDepth;
        }

        char* end = nullptr;
        errno = 0;
        unsigned long parsed = std::strtoul(value, &end, 10);
        if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
            std::cerr << "Invalid KVSTORE_REORDER_IOURING_QD='" << value
                      << "'; using " << kDefaultReorderIoUringQueueDepth
                      << std::endl;
            return kDefaultReorderIoUringQueueDepth;
        }
        return static_cast<unsigned int>(
            std::min<unsigned long>(parsed, 4096));
    }();
    return queue_depth;
}

class RawIoUring {
public:
    explicit RawIoUring(unsigned int entries) {
        io_uring_params params{};
        ring_fd_ = static_cast<int>(syscall(__NR_io_uring_setup, entries, &params));
        if (ring_fd_ < 0) {
            error_ = "io_uring_setup failed: " + std::string(std::strerror(errno));
            return;
        }

        sq_ring_size_ = params.sq_off.array + params.sq_entries * sizeof(unsigned int);
        cq_ring_size_ = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);

        if (params.features & IORING_FEAT_SINGLE_MMAP) {
            single_mmap_ = true;
            sq_ring_size_ = std::max(sq_ring_size_, cq_ring_size_);
            cq_ring_size_ = sq_ring_size_;
        }

        sq_ring_ = mmap(nullptr, sq_ring_size_, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_POPULATE, ring_fd_, IORING_OFF_SQ_RING);
        if (sq_ring_ == MAP_FAILED) {
            sq_ring_ = nullptr;
            error_ = "mmap submission ring failed: " + std::string(std::strerror(errno));
            cleanup();
            return;
        }

        if (single_mmap_) {
            cq_ring_ = sq_ring_;
        } else {
            cq_ring_ = mmap(nullptr, cq_ring_size_, PROT_READ | PROT_WRITE,
                            MAP_SHARED | MAP_POPULATE, ring_fd_, IORING_OFF_CQ_RING);
            if (cq_ring_ == MAP_FAILED) {
                cq_ring_ = nullptr;
                error_ = "mmap completion ring failed: " + std::string(std::strerror(errno));
                cleanup();
                return;
            }
        }

        sqes_size_ = params.sq_entries * sizeof(io_uring_sqe);
        sqes_ = static_cast<io_uring_sqe*>(
            mmap(nullptr, sqes_size_, PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_POPULATE, ring_fd_, IORING_OFF_SQES));
        if (sqes_ == MAP_FAILED) {
            sqes_ = nullptr;
            error_ = "mmap SQEs failed: " + std::string(std::strerror(errno));
            cleanup();
            return;
        }

        sq_head_ = ptr<unsigned int>(sq_ring_, params.sq_off.head);
        sq_tail_ = ptr<unsigned int>(sq_ring_, params.sq_off.tail);
        sq_mask_ = ptr<unsigned int>(sq_ring_, params.sq_off.ring_mask);
        sq_entries_ = ptr<unsigned int>(sq_ring_, params.sq_off.ring_entries);
        sq_array_ = ptr<unsigned int>(sq_ring_, params.sq_off.array);

        cq_head_ = ptr<unsigned int>(cq_ring_, params.cq_off.head);
        cq_tail_ = ptr<unsigned int>(cq_ring_, params.cq_off.tail);
        cq_mask_ = ptr<unsigned int>(cq_ring_, params.cq_off.ring_mask);
        cqes_ = ptr<io_uring_cqe>(cq_ring_, params.cq_off.cqes);
        valid_ = true;
    }

    ~RawIoUring() {
        cleanup();
    }

    RawIoUring(const RawIoUring&) = delete;
    RawIoUring& operator=(const RawIoUring&) = delete;

    bool valid() const { return valid_; }
    const std::string& error() const { return error_; }
    unsigned int capacity() const { return valid_ ? *sq_entries_ : 0; }

    bool prepare_read(int fd, void* buffer, unsigned int length,
                      uint64_t offset, uint64_t user_data) {
        unsigned int head = __atomic_load_n(sq_head_, __ATOMIC_ACQUIRE);
        unsigned int tail = __atomic_load_n(sq_tail_, __ATOMIC_RELAXED);
        if (tail - head >= *sq_entries_) {
            return false;
        }

        unsigned int index = tail & *sq_mask_;
        io_uring_sqe& sqe = sqes_[index];
        std::memset(&sqe, 0, sizeof(sqe));
        pending_read_bytes_ += length;
        sqe.opcode = IORING_OP_READ;
        // O_DIRECT callers reach native asynchronous block I/O, while
        // buffered callers can complete from page cache or use io-wq.
        sqe.fd = fd;
        sqe.off = offset;
        sqe.addr = reinterpret_cast<uint64_t>(buffer);
        sqe.len = length;
        sqe.user_data = user_data;
        sq_array_[index] = index;
        __atomic_store_n(sq_tail_, tail + 1, __ATOMIC_RELEASE);
        return true;
    }

    bool prepare_fsync(int fd, uint64_t user_data) {
        unsigned int head = __atomic_load_n(sq_head_, __ATOMIC_ACQUIRE);
        unsigned int tail = __atomic_load_n(sq_tail_, __ATOMIC_RELAXED);
        if (tail - head >= *sq_entries_) {
            return false;
        }

        unsigned int index = tail & *sq_mask_;
        io_uring_sqe& sqe = sqes_[index];
        std::memset(&sqe, 0, sizeof(sqe));
        sqe.opcode = IORING_OP_FSYNC;
        sqe.fd = fd;
        sqe.user_data = user_data;
        sq_array_[index] = index;
        __atomic_store_n(sq_tail_, tail + 1, __ATOMIC_RELEASE);
        return true;
    }

    int submit(unsigned int count) {
        kv_read_bandwidth_limit().acquire(pending_read_bytes_);
        pending_read_bytes_ = 0;
        unsigned int submitted = 0;
        while (submitted < count) {
            int ret = static_cast<int>(syscall(__NR_io_uring_enter, ring_fd_,
                count - submitted, 0, 0, nullptr, 0));
            if (ret < 0) {
                if (errno == EINTR) continue;
                return -errno;
            }
            if (ret == 0) return -EIO;
            submitted += static_cast<unsigned int>(ret);
        }
        return static_cast<int>(submitted);
    }

    int wait_for_completion() {
        while (__atomic_load_n(cq_head_, __ATOMIC_ACQUIRE) ==
               __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE)) {
            int ret = static_cast<int>(syscall(__NR_io_uring_enter, ring_fd_,
                0, 1, IORING_ENTER_GETEVENTS, nullptr, 0));
            if (ret < 0 && errno != EINTR) return -errno;
        }
        return 0;
    }

    bool pop_completion(uint64_t& user_data, int& result) {
        unsigned int head = __atomic_load_n(cq_head_, __ATOMIC_RELAXED);
        unsigned int tail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
        if (head == tail) return false;

        const io_uring_cqe& cqe = cqes_[head & *cq_mask_];
        user_data = cqe.user_data;
        result = cqe.res;
        __atomic_store_n(cq_head_, head + 1, __ATOMIC_RELEASE);
        return true;
    }

private:
    template <typename T>
    static T* ptr(void* base, uint64_t offset) {
        return reinterpret_cast<T*>(static_cast<char*>(base) + offset);
    }

    void cleanup() {
        valid_ = false;
        if (sqes_ != nullptr) {
            munmap(sqes_, sqes_size_);
            sqes_ = nullptr;
        }
        if (!single_mmap_ && cq_ring_ != nullptr) {
            munmap(cq_ring_, cq_ring_size_);
            cq_ring_ = nullptr;
        }
        if (sq_ring_ != nullptr) {
            munmap(sq_ring_, sq_ring_size_);
            sq_ring_ = nullptr;
            cq_ring_ = nullptr;
        }
        if (ring_fd_ >= 0) {
            close(ring_fd_);
            ring_fd_ = -1;
        }
    }

    uint64_t pending_read_bytes_ = 0;
    int ring_fd_ = -1;
    bool valid_ = false;
    bool single_mmap_ = false;
    std::string error_;

    void* sq_ring_ = nullptr;
    void* cq_ring_ = nullptr;
    io_uring_sqe* sqes_ = nullptr;
    size_t sq_ring_size_ = 0;
    size_t cq_ring_size_ = 0;
    size_t sqes_size_ = 0;

    unsigned int* sq_head_ = nullptr;
    unsigned int* sq_tail_ = nullptr;
    unsigned int* sq_mask_ = nullptr;
    unsigned int* sq_entries_ = nullptr;
    unsigned int* sq_array_ = nullptr;
    unsigned int* cq_head_ = nullptr;
    unsigned int* cq_tail_ = nullptr;
    unsigned int* cq_mask_ = nullptr;
    io_uring_cqe* cqes_ = nullptr;
};


struct PageCopy {
    size_t destination_offset;
    size_t page_offset;
    size_t length;
};

struct PageRead {
    uint64_t offset;
    size_t required_bytes;
    std::vector<PageCopy> copies;
};

void scatter_page(
    const PageRead& page,
    const char* page_buffer,
    std::vector<char>& destination
) {
    for (const PageCopy& copy : page.copies) {
        std::memcpy(
            destination.data() + copy.destination_offset,
            page_buffer + copy.page_offset,
            copy.length);
    }
}

void read_reorder_pages_iouring(
    int fd,
    const std::vector<PageRead>& pages,
    std::vector<char>& destination
) {
    if (pages.empty()) return;

    thread_local std::unique_ptr<RawIoUring> ring;
    thread_local bool fallback_warning_printed = false;
    if (!ring) {
        unsigned int queue_depth = get_reorder_iouring_queue_depth();
        ring = std::make_unique<RawIoUring>(queue_depth);
        std::cout << "reorder io_uring queue depth: " << queue_depth
                  << std::endl;
    }

    // Retain a functional buffered-I/O path on kernels where io_uring is
    // disabled. Keep the same 4-KiB page granularity as the io_uring path.
    if (!ring->valid()) {
        if (!fallback_warning_printed) {
            std::cerr << "reorder io_uring unavailable (" << ring->error()
                      << "); using synchronous buffered page reads"
                      << std::endl;
            fallback_warning_printed = true;
        }
        AlignedBuffer page_buffer =
            allocate_aligned_buffer(kDirectIoAlignment);
        for (const PageRead& page : pages) {
            ssize_t result;
            do {
                kv_read_bandwidth_limit().acquire(kDirectIoAlignment);
                result = pread(
                    fd, page_buffer.get(), kDirectIoAlignment,
                    static_cast<off_t>(page.offset));
            } while (result < 0 && errno == EINTR);
            if (result < 0) {
                throw std::runtime_error(
                    "reorder buffered read failed at offset " +
                    std::to_string(page.offset) + ": " +
                    std::string(std::strerror(errno)));
            }
            if (static_cast<size_t>(result) < page.required_bytes) {
                throw std::runtime_error(
                    "reorder buffered read returned " +
                    std::to_string(result) + " bytes at offset " +
                    std::to_string(page.offset) + "; at least " +
                    std::to_string(page.required_bytes) + " required");
            }
            scatter_page(
                page, static_cast<const char*>(page_buffer.get()),
                destination);
        }
        return;
    }

    const size_t queue_depth = ring->capacity();
    size_t next = 0;
    while (next < pages.size()) {
        size_t window_size = std::min(queue_depth, pages.size() - next);
        if (window_size >
            std::numeric_limits<size_t>::max() / kDirectIoAlignment) {
            throw std::runtime_error("reorder io_uring window is too large");
        }
        AlignedBuffer window = allocate_aligned_buffer(
            window_size * kDirectIoAlignment);
        char* window_data = static_cast<char*>(window.get());

        for (size_t i = 0; i < window_size; ++i) {
            if (!ring->prepare_read(
                    fd, window_data + i * kDirectIoAlignment,
                    static_cast<unsigned int>(kDirectIoAlignment),
                    pages[next + i].offset, static_cast<uint64_t>(i + 1))) {
                throw std::runtime_error(
                    "reorder io_uring submission queue unexpectedly full");
            }
        }

        int submitted = ring->submit(static_cast<unsigned int>(window_size));
        if (submitted < 0) {
            throw std::runtime_error(
                "reorder io_uring submit failed: " +
                std::string(std::strerror(-submitted)));
        }

        size_t completed = 0;
        while (completed < window_size) {
            int wait_result = ring->wait_for_completion();
            if (wait_result < 0) {
                throw std::runtime_error(
                    "reorder io_uring wait failed: " +
                    std::string(std::strerror(-wait_result)));
            }

            uint64_t user_data = 0;
            int result = 0;
            while (ring->pop_completion(user_data, result)) {
                if (user_data == 0 || user_data > window_size) {
                    throw std::runtime_error(
                        "reorder io_uring returned invalid user_data");
                }
                size_t window_index = static_cast<size_t>(user_data - 1);
                const PageRead& page = pages[next + window_index];
                if (result < 0) {
                    throw std::runtime_error(
                        "reorder buffered io_uring read failed at offset " +
                        std::to_string(page.offset) + ": " +
                        std::string(std::strerror(-result)));
                }
                if (static_cast<size_t>(result) < page.required_bytes) {
                    throw std::runtime_error(
                        "reorder buffered io_uring read returned " +
                        std::to_string(result) + " bytes at offset " +
                        std::to_string(page.offset) + "; at least " +
                        std::to_string(page.required_bytes) + " required");
                }
                scatter_page(
                    page,
                    window_data + window_index * kDirectIoAlignment,
                    destination);
                ++completed;
            }
        }
        next += window_size;
    }
}

void write_all_or_throw(
    int fd,
    const char* data,
    size_t size,
    const std::string& file_name
) {
    size_t written = 0;
    while (written < size) {
        ssize_t result = write(fd, data + written, size - written);
        if (result < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                "Failed to write KV data to " + file_name +
                ": " + std::string(std::strerror(errno)));
        }
        if (result == 0) {
            throw std::runtime_error(
                "Zero-byte write while writing KV data to " +
                file_name);
        }
        written += static_cast<size_t>(result);
    }
}

void fsync_fd_or_throw(int fd, const std::string& file_name) {
    int result;
    do {
        result = fsync(fd);
    } while (result < 0 && errno == EINTR);
    if (result < 0) {
        throw std::runtime_error(
            "Failed to fsync " + file_name + ": " +
            std::string(std::strerror(errno)));
    }
}

void fsync_file_or_throw(const std::string& file_name) {
    int fd = open(file_name.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(
            "Failed to open " + file_name + " for fsync: " +
            std::string(std::strerror(errno)));
    }

    try {
        fsync_fd_or_throw(fd, file_name);
    } catch (...) {
        close(fd);
        throw;
    }

    if (close(fd) < 0) {
        throw std::runtime_error(
            "Failed to close " + file_name + " after fsync: " +
            std::string(std::strerror(errno)));
    }
}

void fsync_files_iouring_or_throw(
    const std::vector<std::string>& file_names
) {
    if (file_names.empty()) return;

    std::vector<int> file_descriptors;
    file_descriptors.reserve(file_names.size());
    auto close_all = [&]() {
        for (int fd : file_descriptors) {
            if (fd >= 0) close(fd);
        }
        file_descriptors.clear();
    };

    for (const std::string& file_name : file_names) {
        int fd = open(file_name.c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) {
            int saved_errno = errno;
            close_all();
            throw std::runtime_error(
                "Failed to open " + file_name +
                " for persist barrier: " +
                std::string(std::strerror(saved_errno)));
        }
        file_descriptors.push_back(fd);
    }

    RawIoUring ring(get_iouring_queue_depth());
    if (!ring.valid()) {
        // Some restricted kernels disable io_uring. The isolation guarantee is
        // more important than concurrency, so retain a synchronous fallback.
        try {
            for (size_t i = 0; i < file_descriptors.size(); ++i) {
                fsync_fd_or_throw(file_descriptors[i], file_names[i]);
            }
        } catch (...) {
            close_all();
            throw;
        }
        close_all();
        return;
    }

    try {
        size_t next = 0;
        while (next < file_descriptors.size()) {
            size_t window_size = std::min(
                static_cast<size_t>(ring.capacity()),
                file_descriptors.size() - next);
            for (size_t i = 0; i < window_size; ++i) {
                if (!ring.prepare_fsync(
                        file_descriptors[next + i],
                        static_cast<uint64_t>(i + 1))) {
                    throw std::runtime_error(
                        "Persist barrier io_uring queue unexpectedly full");
                }
            }

            int submitted = ring.submit(
                static_cast<unsigned int>(window_size));
            if (submitted < 0) {
                throw std::runtime_error(
                    "Persist barrier io_uring submit failed: " +
                    std::string(std::strerror(-submitted)));
            }

            size_t completed = 0;
            while (completed < window_size) {
                int wait_result = ring.wait_for_completion();
                if (wait_result < 0) {
                    throw std::runtime_error(
                        "Persist barrier io_uring wait failed: " +
                        std::string(std::strerror(-wait_result)));
                }

                uint64_t user_data = 0;
                int result = 0;
                while (ring.pop_completion(user_data, result)) {
                    if (user_data == 0 || user_data > window_size) {
                        throw std::runtime_error(
                            "Persist barrier returned invalid user_data");
                    }
                    if (result < 0) {
                        size_t file_index =
                            next + static_cast<size_t>(user_data - 1);
                        throw std::runtime_error(
                            "Failed to fsync " + file_names[file_index] +
                            " in persist barrier: " +
                            std::string(std::strerror(-result)));
                    }
                    ++completed;
                }
            }
            next += window_size;
        }
    } catch (...) {
        close_all();
        throw;
    }
    close_all();
}

void fsync_directory_or_throw(const std::string& directory) {
    int fd = open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(
            "Failed to open persist directory " + directory + ": " +
            std::string(std::strerror(errno)));
    }
    try {
        fsync_fd_or_throw(fd, directory);
    } catch (...) {
        close(fd);
        throw;
    }
    if (close(fd) < 0) {
        throw std::runtime_error(
            "Failed to close persist directory " + directory + ": " +
            std::string(std::strerror(errno)));
    }
}

}  // namespace



struct KVReadResources {
    // One persistent hot reader per store. Its TLS io_uring survives layers.
    struct HotReader {
        std::mutex mutex;
        std::condition_variable ready;
        std::packaged_task<void()> task;
        bool stopping = false;
        std::thread thread;
        std::future<void> submit(std::function<void()> fn) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!thread.joinable()) thread = std::thread([this] {
                for (;;) {
                    std::packaged_task<void()> work;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        ready.wait(lock, [this] { return stopping || task.valid(); });
                        if (stopping && !task.valid()) return;
                        work = std::move(task);
                    }
                    work();
                }
            });
            task = std::packaged_task<void()>(std::move(fn));
            auto result = task.get_future();
            ready.notify_one();
            return result;
        }
        ~HotReader() {
            { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
            ready.notify_one();
            if (thread.joinable()) thread.join();
        }
    };
    struct Buffer {
        AlignedBuffer data;
        size_t capacity = 0;
        std::vector<std::tuple<uint64_t, uint64_t, int>> entries;
        std::vector<int> token_order;
        std::vector<MergeRead> reads;
    };
    struct RangeWorkspace {
        AlignedBuffer data;
        size_t capacity = 0;
    } range;
    Buffer hot_buffer;
    std::mutex mutex;
    std::map<std::string, int> files;
    std::map<std::thread::id, Buffer> buffers;
    std::map<std::string, double> profile;
    std::map<uint64_t, std::map<std::string, uint64_t>> merge_stats;
    HotReader hot_reader;

    ~KVReadResources() { close_files(); }
    void close_files() {
        std::lock_guard<std::mutex> lock(mutex);
        for (const auto& item : files) ::close(item.second);
        files.clear();
    }
    int open_file(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = files.find(path);
        if (found != files.end()) { profile["fd_cache_hits"]++; return found->second; }
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd >= 0) { files.emplace(path, fd); profile["fd_opens"]++; }
        return fd;
    }
    Buffer& worker_buffer() {
        std::lock_guard<std::mutex> lock(mutex);
        return buffers[std::this_thread::get_id()];
    }
    void add(const std::string& name, double value) {
        std::lock_guard<std::mutex> lock(mutex);
        profile[name] += value;
    }
};

using ReadClock = std::chrono::steady_clock;
static double read_seconds(ReadClock::time_point start) {
    return std::chrono::duration<double>(ReadClock::now() - start).count();
}



static int kv_read_threads(int tasks) {
    int requested = omp_get_max_threads();
    const char* value = std::getenv("OMP_NUM_THREADS");
    if (value && *value) {
        char* end = nullptr;
        long parsed = std::strtol(value, &end, 10);
        if (parsed > 0 && parsed <= INT_MAX && (*end == '\0' || *end == ','))
            requested = static_cast<int>(parsed);
    }
    return std::max(1, std::min(tasks, requested));
}



std::map<uint64_t, std::map<std::string, uint64_t>> KVStore::get_merge_read_stats(bool reset) {
    std::lock_guard<std::mutex> lock(read_resources->mutex);
    auto result = read_resources->merge_stats;
    if (reset) read_resources->merge_stats.clear();
    return result;
}

std::map<std::string, double> KVStore::get_read_profile(bool reset) {
    std::lock_guard<std::mutex> lock(read_resources->mutex);
    auto result = read_resources->profile;
    for (const char* name : {"plan_thread_seconds", "setup_thread_seconds",
            "submit_thread_seconds", "wait_thread_seconds", "scatter_thread_seconds",
            "fallback_thread_seconds", "fd_opens", "fd_cache_hits", "buffer_growths",
            "iouring_head_calls", "fallback_head_calls",
            "cache_lookup_wall_seconds", "cache_admission_wall_seconds",
            "cache_candidate_prepare_wall_seconds", "cache_decision_wall_seconds",
            "cache_hot_decision_wall_seconds", "cache_cold_decision_wall_seconds",
            "cache_group_update_wall_seconds", "cache_group_admission_wall_seconds",
            "cache_nonoverlapped_prepare_wall_seconds", "cache_post_read_admission_wall_seconds",
            "cache_read_admission_pipeline_wall_seconds",
            "promotion_snapshot_wall_seconds"})
        result.emplace(name, 0.0);
    result["fixed_block_reads_enabled"] = hot_growth_fixed_reads;
    result["fixed_block_bytes"] = hot_growth_fixed_reads ? hot_growth_fixed_block_bytes : 0;
    result["cache_read_ranges_enabled"] = cache_read_ranges;
    result["cache_range_buffer_capacity_bytes"] = read_resources->range.capacity;
    result["cache_diagnostics_enabled"] = cache_diagnostics;
    result["buffer_capacity_bytes"] = 0;
    for (const auto& item : read_resources->buffers)
        result["buffer_capacity_bytes"] += item.second.capacity;
    result["cached_fds"] = read_resources->files.size();
    result["read_threads_requested"] = kv_read_threads(num_key_value_heads);
    if (reset) read_resources->profile.clear();
    return result;
}

std::map<std::string, uint64_t> KVStore::get_reorder_memory_stats(bool reset) {
    std::unique_lock<std::shared_mutex> lock(kv_io_mutex);
    auto result = reorder_memory_stats;
    for (const char* name : {"promotion_cache_tokens", "promotion_snapshot_tokens",
            "promotion_disk_tokens", "promotion_source_pages", "promotion_source_page_bytes",
            "demotion_cache_tokens", "demotion_snapshot_tokens", "demotion_disk_tokens"})
        result.emplace(name, 0);
    result["snapshot_payload_bytes"] = 0;
    result["snapshot_buffer_capacity_bytes"] = 0;
    for (const auto& item : promotion_snapshots) {
        result["snapshot_payload_bytes"] += item.second.payload_size;
        result["snapshot_buffer_capacity_bytes"] += item.second.payload_capacity;
    }
    if (reset) reorder_memory_stats.clear();
    return result;
}

void KVStore::capture_promotion_snapshot(int prefix, int layer, const int* ids, const int* counts) {
    if (!hot_growth_layout_updates) return;
    const auto started = ReadClock::now();
    const size_t half = head_dim * sizeof(DTYPE), entry = 2 * half;
    auto& snapshot = promotion_snapshots[{prefix, layer}];
    snapshot.reset();
    snapshot.sources.resize(num_key_value_heads);
    std::exception_ptr failure;
    std::mutex failure_mutex;
    const auto scan_started = ReadClock::now();
    const char* preview_setting = std::getenv("KVSTORE_PROMOTION_SNAPSHOT_PREVIEW");
    // With v2's small incoming payloads, sorting just to discard some copies can
    // cost more than packing them. Keep exact compaction preview opt-in for v2;
    // cheap append/write-cap checks still use the actual selection request.
    const bool preview_compaction = preview_setting ? std::string(preview_setting) != "0"
                                                    : !false;
    int preview_heads = 0, append_heads = 0, preview_skipped_heads = 0;
    #pragma omp parallel for num_threads(kv_read_threads(num_key_value_heads)) reduction(+:preview_heads,append_heads,preview_skipped_heads)
    for (int head = 0; head < num_key_value_heads; ++head) {
        try {
            auto& sources = snapshot.sources[head];
            sources.clear();
            // A head with no uncached cold payload needs no admission preview.
            for (int j = 0; j < counts[head]; ++j) {
                const int pos = head * max_length + j;
                const int token = ids[pos];
                if (kv_meta->at(get_meta_id(token, layer, head)).file_index != 1) continue;
                        }
            if (sources.empty()) continue;
            bool filtered = hot_growth_enabled && promotion_snapshot_request >= 0 &&
                hot_growth_heads.count(layer) &&
                promotion_snapshot_request > hot_growth_heads.at(layer)[head].request;
            if (filtered) {
                const auto& state = hot_growth_heads.at(layer)[head];
                // Even counting unique incoming IDs is unnecessary if all
                // selections would fit an append's write cap and full preview
                // is disabled. Compaction heads keep the same owned snapshot.
                if (!preview_compaction && (!false ||
                        size_t(counts[head]) * entry <= hot_growth_max_write_bytes)) {
                    ++preview_skipped_heads;
                    continue;
                }
                std::vector<int> selected(ids + head * max_length,
                                          ids + head * max_length + counts[head]);
                if (!std::is_sorted(selected.begin(), selected.end()))
                    std::sort(selected.begin(), selected.end());
                selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
                ;
                if (!preview_compaction) { ++preview_skipped_heads; continue; }
                ++preview_heads;
                std::vector<int> eligible;
                const auto plan = state.preview(hot_growth_config, selected, promotion_snapshot_request);
                const auto [first, end] = plan.changed_span(state.order, state.hot);
                if (size_t(end - first) * entry <= hot_growth_max_write_bytes)
                    for (int i = first; i < end; ++i)
                        if (state.position[plan.warm[i]] < 0) eligible.push_back(plan.warm[i]);
                std::sort(eligible.begin(), eligible.end());
                sources.erase(std::remove_if(sources.begin(), sources.end(), [&](int pos) {
                    return !std::binary_search(eligible.begin(), eligible.end(), ids[pos]);
                }), sources.end());
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) failure = std::current_exception();
        }
    }
    if (failure) std::rethrow_exception(failure);
    read_resources->add("promotion_snapshot_scan_wall_seconds", read_seconds(scan_started));
    read_resources->add("promotion_snapshot_preview_heads", preview_heads);
    read_resources->add("promotion_snapshot_preview_skipped_heads", preview_skipped_heads);
    read_resources->add("promotion_snapshot_append_heads", append_heads);
    const auto pack_started = ReadClock::now();
    size_t upper = 0;
    for (const auto& sources : snapshot.sources) upper += sources.size();
    if (upper && snapshot.positions.empty())
        snapshot.positions.assign(size_t(num_key_value_heads) * max_length, -1);
    snapshot.touched.reserve(upper);
    // Deduplicate independently within each head. Positions from different
    // heads never alias; a prefix sum then assigns disjoint packing ranges.
    #pragma omp parallel for num_threads(kv_read_threads(num_key_value_heads))
    for (int head = 0; head < num_key_value_heads; ++head) {
        auto& sources = snapshot.sources[head];
        size_t unique = 0;
        for (int pos : sources) {
            const int flat = head * max_length + ids[pos];
            if (snapshot.positions[flat] >= 0) continue;
            snapshot.positions[flat] = 0;
            sources[unique++] = pos;
        }
        sources.resize(unique);
    }
    std::vector<size_t> offsets(num_key_value_heads + 1, 0);
    for (int head = 0; head < num_key_value_heads; ++head)
        offsets[head + 1] = offsets[head] + snapshot.sources[head].size();
    snapshot.touched.resize(offsets.back());
    const size_t bytes = offsets.back() * entry;
    read_resources->add("promotion_snapshot_buffer_growths", bytes > snapshot.payload_capacity);
    try {
        snapshot.reserve_payload(bytes);
    } catch (...) {
        for (int head = 0; head < num_key_value_heads; ++head)
            for (int pos : snapshot.sources[head])
                snapshot.positions[head * max_length + ids[pos]] = -1;
        snapshot.touched.clear();
        throw;
    }
    snapshot.payload_size = bytes;
    read_resources->add("promotion_snapshot_copied_tokens", snapshot.touched.size());
    #pragma omp parallel for num_threads(kv_read_threads(num_key_value_heads))
    for (int head = 0; head < num_key_value_heads; ++head) {
        size_t slot = offsets[head];
        for (int pos : snapshot.sources[head]) {
            const int flat = head * max_length + ids[pos];
            snapshot.positions[flat] = slot;
            snapshot.touched[slot] = flat;
            char* dest = snapshot.payload.get() + slot++ * entry;
            std::memcpy(dest, queried_key + query_output_offset(size_t(pos)), half);
            std::memcpy(dest + half, queried_value + query_output_offset(size_t(pos)), half);
        }
    }
    read_resources->add("promotion_snapshot_pack_wall_seconds", read_seconds(pack_started));
    read_resources->add("promotion_snapshot_wall_seconds", read_seconds(started));
}

int KVStore::copy_reorder_kv(int prefix, int layer, uint64_t meta_id, char* destination) {
    size_t flat = meta_id - get_meta_id(0, layer, 0);
    size_t half = head_dim * sizeof(DTYPE);
    ;
    auto found = promotion_snapshots.find({prefix, layer});
    if (found == promotion_snapshots.end() || flat >= found->second.positions.size()) return 0;
    int slot = found->second.positions[flat];
    if (slot < 0) return 0;
    std::memcpy(destination, found->second.payload.get() + static_cast<size_t>(slot) * 2 * half, 2 * half);
    return 2;
}

void KVStore::update_reorder_location(int prefix, int layer, uint64_t meta_id,
                                    uint64_t file, uint64_t offset) {
    // Invalidate old physical ownership, never immutable token-ID payloads.
    const size_t entry = 2 * head_dim * sizeof(DTYPE);
    auto& old = kv_meta->at(meta_id);
    auto update_index = [&](uint64_t region, uint64_t position, int64_t owner) {
        if (region == 0) {
            auto it = hot_region_indices.find({prefix, layer});
            if (it == hot_region_indices.end()) return;
            size_t slot = position / entry;
            if (it->second.size() <= slot) it->second.resize(slot + 1, -1);
            if (owner >= 0 || it->second[slot] ==
                    static_cast<int64_t>(meta_id - get_meta_id(0, layer, 0)))
                it->second[slot] = owner;
            return;
        }
        if (region != 1) return;
        auto it = cold_region_indices.find({prefix, layer});
        if (it == cold_region_indices.end()) return;
        size_t slot = position / entry;
        if (it->second.size() <= slot) it->second.resize(slot + 1, -1);
        // A reused cold slot may already belong to the demoted victim.
        if (owner >= 0 || it->second[slot] == static_cast<int64_t>(meta_id - get_meta_id(0, layer, 0)))
            it->second[slot] = owner;
    };
    if (old.file_index <= 1) update_index(old.file_index, old.offset, -1);
    if (file <= 1) update_index(file, offset, meta_id - get_meta_id(0, layer, 0));
    old = FileOffsetInfo(file, offset);
}

void KVStore::clear_read_cache() {
    // Call only at a quiescent boundary, never while reads use cached fds.
    std::unique_lock<std::shared_mutex> lock(kv_io_mutex);
    read_resources->close_files();
    cold_region_indices.clear();
    hot_region_indices.clear();
    promotion_snapshots.clear();
    read_resources->buffers.clear();
}

KVStore::KVStore(){
    const char* diagnostics = std::getenv("KVSTORE_CACHE_DIAGNOSTICS");
    const char* overlap = std::getenv("KVSTORE_CACHE_IO_OVERLAP");
    const char* cold_reuse = std::getenv("KVSTORE_COLD_REUSE_DIAGNOSTICS");
    read_resources = std::make_unique<KVReadResources>();
    this->allocated = false;
    this->kv_meta = new std::unordered_map<uint64_t, FileOffsetInfo>();
    this->persisted = false;
    this->num_io = 0;
    this->loaded_bytes = 0;
    this->reorder_io_bytes = 0;
    this->compaction_threshold = 1;
    this->collect_use_iouring = true;

    const char* collect_backend = std::getenv("KVSTORE_COLLECT_IO_BACKEND");
    if (collect_backend != nullptr && *collect_backend != '\0') {
        try {
            this->set_collect_io_backend(collect_backend);
        } catch (const std::invalid_argument& error) {
            std::cerr << error.what() << "; using fread" << std::endl;
        }
    }
}

KVStore::~KVStore(){
    if (this->allocated) {
        for (int i = 0; i < this->num_layers; ++i)
        {
            delete [] this->key_cache[i];
            delete [] this->value_cache[i];
        }
        delete [] this->queried_key;
        delete [] this->queried_value;
        this->allocated = false;
    }
    delete kv_meta;
}

void KVStore::alloc(
    int num_layers,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    int max_length
) {
    {
        std::unique_lock<std::shared_mutex> lock(kv_io_mutex);
        promotion_snapshots.clear();
        hot_growth_heads.clear();
        hot_growth_io_failed.store(false);
        cold_region_indices.clear();
        hot_region_indices.clear();
        hybrid_cache.clear();
    }
    this->num_layers = num_layers;
    this->num_attention_heads = num_attention_heads;
    this->num_key_value_heads = num_key_value_heads;
    this->head_dim = head_dim;
    this->max_length = max_length;

    for (int i = 0; i < this->num_layers; i++){
        DTYPE * k = new DTYPE[this->num_key_value_heads * this->max_length * this->head_dim];
        DTYPE * v = new DTYPE[this->num_key_value_heads * this->max_length * this->head_dim];

        memset(k, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
        memset(v, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));

        this->key_cache.push_back(k);
        this->value_cache.push_back(v);
    }

    this->queried_key = new DTYPE[this->num_key_value_heads * this->max_length * this->head_dim];
    this->queried_value = new DTYPE[this->num_key_value_heads * this->max_length * this->head_dim];
    memset(this->queried_key, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
    memset(this->queried_value, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));

    this->num_io = 0;
    this->loaded_bytes = 0;
    this->reorder_io_bytes = 0;
    // this->compaction_threshold = 1;
    //this->layer_stats.assign(num_layers, LayerStats());

}

void KVStore::clear() {
    hot_growth_heads.clear();
    offload_len = 0;
    clear_read_cache();
    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for(int i = 0; i < this->num_layers; ++i){
        memset(this->key_cache[i], 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
        memset(this->value_cache[i], 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
    }

    memset(this->queried_key, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));
    memset(this->queried_value, 0, this->num_key_value_heads * this->max_length * this->head_dim * sizeof(DTYPE));

    this->num_io = 0;
    this->loaded_bytes = 0;
    this->reorder_io_bytes = 0;
}

void KVStore::fill(
    int layer_id,
    torch::Tensor k,
    torch::Tensor v)
{
    clear_cache();
    DTYPE * key = this->key_cache[layer_id];
    DTYPE * value = this->value_cache[layer_id];

    int seq_len = k.size(1);
    this->offload_len = seq_len;

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->num_key_value_heads; ++i){
        memcpy(key + i * this->max_length * this->head_dim, static_cast<DTYPE *>(k.data_ptr()) + i * seq_len * this->head_dim, seq_len * this->head_dim * sizeof(DTYPE));
        memcpy(value + i * this->max_length * this->head_dim, static_cast<DTYPE *>(v.data_ptr()) + i * seq_len * this->head_dim, seq_len * this->head_dim * sizeof(DTYPE));

    }
}

uint64_t KVStore::get_meta_id(int token_id, int layer_id, int head_id){
    return layer_id * this->num_key_value_heads * this->max_length + head_id * this->max_length + token_id;
}

void KVStore::persist_meta(int prefix_id) {
    std::string meta_file = this->store_path + "/" + std::to_string(prefix_id) + ".meta";
    //std::ofstream file(meta_file, std::ios::app);
    std::ofstream file(meta_file, std::ios::binary | std::ios::trunc);
    std::stringstream ss;
    for (const auto &[id, meta] : *kv_meta) {
        ss << id << " " << meta.file_index << " " << meta.offset << "\n";
    }
    file.write(ss.str().data(), ss.str().size());
    file.flush();
    file.close();

    std::cout<<"Successfully persist kvstore meta"<<std::endl;
}

void KVStore::recover_meta(std::string path, int prefix_id) {
    clear_read_cache();
    auto start_time = std::chrono::high_resolution_clock::now();

    this->store_path = path;
    std::string meta_file = this->store_path + "/" + std::to_string(prefix_id) + ".meta";
    std::ifstream file(meta_file);
    assert(this->file_meta.size() == 0);

    file.seekg(0, std::ios::beg);
    int meta_id;
    uint64_t file_index, offset;
    int lines = 0;
    while(file >> meta_id >> file_index >> offset) {
        kv_meta->insert({meta_id, FileOffsetInfo(file_index, offset)});
    }
    file.close();
    this->persisted = true;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "[TIMER] recover_meta for prefix " << prefix_id << " took " << duration.count() << " seconds." << std::endl;
}

void KVStore::reorder_persist(
    std::string path,
    int prefix_id,
    int layer_id,
    const torch::Tensor& input_token_info
) {
    const auto& reorder_token_info = input_token_info;

    TORCH_CHECK(layer_id >= 0 && layer_id < this->num_layers,
                "Invalid layer_id: ", layer_id);
    TORCH_CHECK(this->offload_len > 0 &&
                    this->offload_len <= this->max_length,
                "Invalid offload length: ", this->offload_len);
    TORCH_CHECK(!reorder_token_info.is_cuda(),
                "reorder_token_info must be a CPU tensor");
    TORCH_CHECK(reorder_token_info.scalar_type() == torch::kInt64,
                "reorder_token_info must have dtype torch.int64");
    TORCH_CHECK(reorder_token_info.dim() == 2 &&
                    reorder_token_info.size(1) == 2,
                "reorder_token_info must have shape [N, 2]");
    TORCH_CHECK(reorder_token_info.is_contiguous(),
                "reorder_token_info must be contiguous");

    if (hot_growth_enabled) {

    }
    const size_t head_bytes =
        static_cast<size_t>(this->head_dim) * sizeof(DTYPE);
    const size_t entry_size = 2 * head_bytes;
    const size_t entry_count =
        static_cast<size_t>(this->num_key_value_heads) * this->offload_len;
    std::vector<unsigned char> is_important(entry_count, 0);
    std::vector<std::pair<int, int>> important_tokens;
    important_tokens.reserve(
        static_cast<size_t>(reorder_token_info.size(0)));

    auto token_accessor = reorder_token_info.accessor<int64_t, 2>();
    for (int64_t row = 0; row < token_accessor.size(0); ++row) {
        int64_t head_id = token_accessor[row][0];
        int64_t token_id = token_accessor[row][1];
        TORCH_CHECK(head_id >= 0 && head_id < this->num_key_value_heads,
                    "Invalid reorder head_id: ", head_id);
        TORCH_CHECK(token_id >= 0 && token_id < this->offload_len,
                    "Invalid reorder token_id: ", token_id);
        size_t flat_index =
            static_cast<size_t>(head_id) * this->offload_len + token_id;
        if (!is_important[flat_index]) {
            is_important[flat_index] = 1;
            important_tokens.emplace_back(
                static_cast<int>(head_id), static_cast<int>(token_id));
        }
    }

    if (hot_growth_enabled) {
        // The native transaction addresses fixed per-head spans. Preserve score
        // order inside each head even if a caller interleaves the input rows.
        std::stable_sort(important_tokens.begin(), important_tokens.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<std::vector<int>> initial(num_key_value_heads);
        for (auto pair : important_tokens) initial[pair.first].push_back(pair.second);
        std::vector<hot_growth::Head> heads;
        for (const auto& tokens : initial) heads.emplace_back(hot_growth_config, offload_len, tokens);
        for (auto& head : heads) {
                head.cache_include_hot = hybrid_cache_policy == "access_control";
                head.publish_cache_scores();
            }
        hot_growth_heads[layer_id] = std::move(heads);
    }

    TORCH_CHECK(entry_size == 0 ||
                    entry_count <=
                        std::numeric_limits<size_t>::max() / entry_size,
                "Initial reorder KV buffer is too large");
    const size_t resident_slots = hot_growth_enabled
        ? num_key_value_heads * hot_growth_heads.at(layer_id)[0].storage_capacity()
        : important_tokens.size();
    std::vector<char> important_buffer(resident_slots * entry_size);
    std::vector<char> unimportant_buffer(
        (hot_growth_enabled ? entry_count : entry_count - important_tokens.size()) * entry_size);
    std::vector<std::pair<uint64_t, FileOffsetInfo>> new_metadata;
    new_metadata.reserve(entry_count);

    DTYPE* key = this->key_cache[layer_id];
    DTYPE* value = this->value_cache[layer_id];
    auto copy_entry = [&](int head_id, int token_id, char* destination) {
        size_t cache_offset =
            (static_cast<size_t>(head_id) * this->max_length + token_id) *
            this->head_dim;
        std::memcpy(destination, key + cache_offset, head_bytes);
        std::memcpy(
            destination + head_bytes, value + cache_offset, head_bytes);
    };

    std::vector<size_t> head_positions(num_key_value_heads, 0);
    for (size_t index = 0; index < important_tokens.size(); ++index) {
        auto [head_id, token_id] = important_tokens[index];
        const size_t slot = hot_growth_enabled
            ? head_id * hot_growth_heads.at(layer_id)[head_id].storage_capacity() + head_positions[head_id]++
            : index;
        copy_entry(
            head_id, token_id,
            important_buffer.data() + slot * entry_size);
        new_metadata.emplace_back(
            get_meta_id(token_id, layer_id, head_id),
            FileOffsetInfo(0, slot * entry_size));
    }
    size_t unimportant_index = 0;
    for (int head_id = 0; head_id < this->num_key_value_heads; ++head_id) {
        std::vector<int> order;
            for (int token = 0; token < offload_len; ++token) order.push_back(token);

        for (int token_id : order) {
            size_t flat_index =
                static_cast<size_t>(head_id) * this->offload_len + token_id;
            if (is_important[flat_index] && !hot_growth_enabled) continue;
            copy_entry(
                head_id, token_id,
                unimportant_buffer.data() +
                    unimportant_index * entry_size);
            if (!is_important[flat_index]) new_metadata.emplace_back(
                get_meta_id(token_id, layer_id, head_id),
                FileOffsetInfo(1, unimportant_index * entry_size));
            ++unimportant_index;
        }
    }

    const std::string base_name = path + "/" + std::to_string(prefix_id) +
        "_layer" + std::to_string(layer_id);
    const std::string file_name0 = base_name + "_part0.bin";
    const std::string file_name1 = base_name + "_part1.bin";
    const std::string temporary_file0 = file_name0 + ".reorder.tmp";
    const std::string temporary_file1 = file_name1 + ".reorder.tmp";

    auto write_temporary = [](const std::string& file_name,
                              const std::vector<char>& data) {
        int fd = open(
            file_name.c_str(),
            O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
            0644);
        if (fd < 0) {
            throw std::runtime_error(
                "Failed to open initial reorder file " + file_name + ": " +
                std::string(std::strerror(errno)));
        }
        try {
            write_all_or_throw(fd, data.data(), data.size(), file_name);
        } catch (...) {
            close(fd);
            unlink(file_name.c_str());
            throw;
        }
        if (close(fd) < 0) {
            int saved_errno = errno;
            unlink(file_name.c_str());
            throw std::runtime_error(
                "Failed to close initial reorder file " + file_name + ": " +
                std::string(std::strerror(saved_errno)));
        }
    };

    std::unique_lock<std::shared_mutex> io_lock(this->kv_io_mutex);
    read_resources->close_files();
    cold_region_indices.erase({prefix_id, layer_id});
    hot_region_indices.erase({prefix_id, layer_id});
    promotion_snapshots.clear();
    try {
        write_temporary(temporary_file0, important_buffer);
        write_temporary(temporary_file1, unimportant_buffer);
    } catch (...) {
        unlink(temporary_file0.c_str());
        unlink(temporary_file1.c_str());
        throw;
    }
    // Publish both completed temporary files before exposing their locations
    // through metadata. A failure leaves persisted=false, so no reader can
    // consume a partial initial layout.
    if (rename(temporary_file1.c_str(), file_name1.c_str()) < 0) {
        int saved_errno = errno;
        unlink(temporary_file0.c_str());
        unlink(temporary_file1.c_str());
        throw std::runtime_error(
            "Failed to publish initial reorder file " + file_name1 + ": " +
            std::string(std::strerror(saved_errno)));
    }
    if (rename(temporary_file0.c_str(), file_name0.c_str()) < 0) {
        int saved_errno = errno;
        unlink(temporary_file0.c_str());
        throw std::runtime_error(
            "Failed to publish initial reorder file " + file_name0 + ": " +
            std::string(std::strerror(saved_errno)));
    }
    std::vector<int32_t>* hot_owners = nullptr;
    std::vector<int64_t>* cold_owners = nullptr;
    if (cache_read_ranges) {
        hot_owners = &hot_region_indices[{prefix_id, layer_id}];
        cold_owners = &cold_region_indices[{prefix_id, layer_id}];
        hot_owners->assign(important_buffer.size() / entry_size, -1);
        cold_owners->assign(unimportant_buffer.size() / entry_size, -1);
    }
    for (const auto& [meta_id, location] : new_metadata) {
        (*this->kv_meta)[meta_id] = location;
        if (hot_owners) {
            const int64_t owner = meta_id - get_meta_id(0, layer_id, 0);
            if (location.file_index == 0) (*hot_owners)[location.offset / entry_size] = owner;
            else (*cold_owners)[location.offset / entry_size] = owner;
        }
    }
    this->store_path = std::move(path);
    this->pending_persist_files.push_back(file_name0);
    this->pending_persist_files.push_back(file_name1);
    this->reorder_io_bytes +=
        static_cast<long long>(important_buffer.size() + unimportant_buffer.size());

    this->persisted = true;
}

void KVStore::finish_persist() {
    if (this->pending_persist_files.empty()) return;

    // All layers have already been written and closed. Submit their durability
    // barriers together so SSD writeback completes before prefix reuse starts,
    // without serially fsyncing each layer in the write loop.
    fsync_files_iouring_or_throw(this->pending_persist_files);
    fsync_directory_or_throw(this->store_path);
    this->pending_persist_files.clear();
    this->persisted = true;
}

void KVStore::set_collect_io_backend(const std::string& backend) {
    if (backend == "iouring" || backend == "io_uring") {
        this->collect_use_iouring = true;
    } else if (backend == "fread" || backend == "ifstream") {
        this->collect_use_iouring = false;
    } else {
        throw std::invalid_argument(
            "Invalid KVSTORE collect I/O backend '" + backend +
            "' (expected fread or iouring)");
    }
}

std::string KVStore::get_collect_io_backend() const {
    return this->collect_use_iouring ? "iouring" : "fread";
}

void KVStore::prepare_direct_output(const std::vector<int>& counts) {
    if (!direct_collect) return;
    const auto started = ReadClock::now();
    direct_stride = *std::max_element(counts.begin(), counts.end()) + direct_suffix_len;
    // Writes only padding/suffix rows. Valid rows are filled by the gather or
    // disk scatter below; no second pass over the valid K/V payload is needed.
    #pragma omp parallel for num_threads(kv_read_threads(num_key_value_heads))
    for (int head = 0; head < num_key_value_heads; ++head) {
        const size_t offset = (size_t(head) * direct_stride + counts[head]) * head_dim;
        const size_t bytes = size_t(direct_stride - counts[head]) * head_dim * sizeof(DTYPE);
        std::memset(queried_key + offset, 0, bytes);
        std::memset(queried_value + offset, 0, bytes);
    }
    read_resources->add("direct_staging_tail_zero_wall_seconds", read_seconds(started));
    read_resources->add("direct_staging_bytes", size_t(num_key_value_heads) * direct_stride * head_dim * sizeof(DTYPE) * 2);
    read_resources->add("direct_staging_collections", 1);
}

std::tuple<torch::Tensor, int> KVStore::merge_collect_into(
    int prefix_id, int layer_id, torch::Tensor ids, torch::Tensor counts,
    torch::Tensor out_key, torch::Tensor out_value, int suffix_len) {
    TORCH_CHECK(!direct_collect, "Nested direct collection is not supported");
    TORCH_CHECK(!collect_hot_extras, "AdaKVs uses selected-only output");
    TORCH_CHECK(num_key_value_heads > 0 && max_length > 0 && suffix_len >= 0 && suffix_len <= max_length,
                "Invalid direct staging geometry");
    TORCH_CHECK(layer_id >= 0 && layer_id < num_layers, "Invalid direct staging layer");
    TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type() == torch::kInt32 && ids.is_contiguous() &&
                ids.dim() == 2 && ids.size(0) == num_key_value_heads && ids.size(1) == max_length,
                "Direct staging IDs must be contiguous CPU int32 [heads, max_length]");
    TORCH_CHECK(counts.device().is_cpu() && counts.scalar_type() == torch::kInt32 && counts.is_contiguous() &&
                counts.dim() == 1 && counts.numel() == num_key_value_heads,
                "Direct staging counts must be contiguous CPU int32 [heads]");
    const int* lengths = counts.data_ptr<int>();
    const int* indices = ids.data_ptr<int>();
    for (int h = 0; h < num_key_value_heads; ++h) {
        TORCH_CHECK(lengths[h] >= 0 && lengths[h] <= max_length, "Invalid direct staging count");
        for (int j = 0; j < lengths[h]; ++j)
            TORCH_CHECK(indices[h * max_length + j] >= 0 && indices[h * max_length + j] < offload_len,
                        "Invalid direct staging token ID");
    }
    const int64_t required = int64_t(num_key_value_heads) * (max_length + suffix_len) * head_dim;
    for (const auto& tensor : {out_key, out_value})
        TORCH_CHECK(tensor.device().is_cpu() && tensor.scalar_type() == torch::kFloat16 &&
                    tensor.is_contiguous() && tensor.dim() == 1 && tensor.numel() >= required,
                    "Direct staging outputs must be sufficiently large contiguous CPU float16 slabs");
    const auto k_address = reinterpret_cast<uintptr_t>(out_key.data_ptr());
    const auto v_address = reinterpret_cast<uintptr_t>(out_value.data_ptr());
    TORCH_CHECK(k_address + required * sizeof(DTYPE) <= v_address ||
                v_address + required * sizeof(DTYPE) <= k_address, "Direct staging outputs overlap");
    auto* original_key = queried_key;
    auto* original_value = queried_value;
    queried_key = reinterpret_cast<DTYPE*>(out_key.data_ptr());
    queried_value = reinterpret_cast<DTYPE*>(out_value.data_ptr());
    direct_collect = true;
    direct_suffix_len = suffix_len;
    try {
        collect_planned_key_value(prefix_id, layer_id, ids, counts, false);
        const int width = direct_stride - suffix_len;
        // Core IDs must outlive the reusable selector, including full-width
        // selections where contiguous() alone could alias the input tensor.
        auto selected = collect_hot_extras
            ? collected_hot_indices.narrow(1, 0, width).contiguous()
            : ids.narrow(1, 0, width).clone();
        if (!collect_hot_extras && width) {
            auto output = selected.data_ptr<int>();
            for (int head = 0; head < num_key_value_heads; ++head)
                std::fill(output + head * width + lengths[head],
                          output + (head + 1) * width, -1);
        }
        queried_key = original_key;
        queried_value = original_value;
        direct_collect = false;
        direct_stride = direct_suffix_len = 0;
        return {selected, width};
    } catch (...) {
        // Ownership always stays with the caller; reset/destruction must never
        // free external slabs, including after a failed/short disk read.
        queried_key = original_key;
        queried_value = original_value;
        direct_collect = false;
        direct_stride = direct_suffix_len = 0;
        throw;
    }
}

void KVStore::merge_collect_queried_key_value(
    int prefix_id, int layer_id, torch::Tensor ind_pt, torch::Tensor nnz_pt,
    std::optional<torch::Tensor> importance_scores
) {
    collect_planned_key_value(prefix_id, layer_id, ind_pt, nnz_pt, false, importance_scores);
}

void KVStore::collect_planned_key_value(
    int prefix_id,
    int layer_id,
    torch::Tensor ind_pt,
    torch::Tensor nnz_pt,
    bool adjacent_pages_only,
    std::optional<torch::Tensor> importance_scores,
    bool group_only
) {
    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());

    int attn_group = this->num_attention_heads / this->num_key_value_heads;


    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE);
    std::vector<int> collected_counts;
    if (collect_hot_extras || direct_collect)
        collected_counts.assign(nnz, nnz + num_key_value_heads);
    ;

    //handle persisted=false case; collect kv from memory
    if (this->persisted == false) {
      prepare_direct_output(collected_counts);
      int stride = this->max_length * this->head_dim; // 一个头完整数据的长度
      DTYPE* key = this->key_cache[layer_id];
      DTYPE* value = this->value_cache[layer_id];
      for (int i = 0; i < this->num_key_value_heads; i++) {
        auto head_ind = ind + i * this->max_length;
        int num_indices = nnz[i];
        auto queried_key_ptr = this->queried_key + query_output_offset(size_t(i) * max_length); // 当前头要查询的数据位置
        auto queried_value_ptr = this->queried_value + query_output_offset(size_t(i) * max_length);
        auto key_ptr = key + i * stride; // 当前头完整数据位置
        auto value_ptr = value + i * stride;
        for (int j = 0; j < num_indices; j++) {
          auto cur_ind = head_ind[j];
          memcpy(queried_key_ptr + j * this->head_dim, key_ptr + cur_ind * this->head_dim, this->head_dim * sizeof(DTYPE));
          memcpy(queried_value_ptr + j * this->head_dim, value_ptr + cur_ind * this->head_dim, this->head_dim * sizeof(DTYPE));
        }
      }
      return;
    }

    ForegroundReadGuard foreground_guard(foreground_readers);
    std::shared_lock<std::shared_mutex> io_lock(this->kv_io_mutex);
    TORCH_CHECK(!hot_growth_io_failed.load(), "Hot growth I/O failure: reload prefix");

    TORCH_CHECK(hot_growth_enabled && hot_growth_hybrid_reads,
                "AdaKVs requires guarded hot growth and the hybrid reader");
    prepare_direct_output(collected_counts);
    collect_hybrid_cached(prefix_id, layer_id, ind, nnz);
}


void KVStore::merge_load_key_value_from_file(
    std::vector<MergeRead>& content,
    std::vector<int> &token_order,
    int prefix_id,
    int layer_id,
    int head_id,
    const CacheReadCopies* captures,
    size_t capture_begin
) {
    // if(head_id == 0 || head_id == 5){
    //     merge_analyze_content_segments(content, layer_id, head_id, this->layer_stats);
    // }

    #pragma omp atomic update
    this->num_io += static_cast<int>(content.size());
    size_t entry_size = 2 * this->head_dim * sizeof(DTYPE); // key + value
    int count = 0;

    uint64_t cur_file_index = std::numeric_limits<uint64_t>::max();
    std::ifstream file;

    long long bytes_loaded = 0;
    long long region_bytes[2] = {0, 0};
    size_t capture_index = capture_begin;
    for(const auto& [file_index, start_offset, length, positions] : content){
        // change file
        if(file_index != cur_file_index){
            if (file.is_open()) file.close();
            //std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_part" + std::to_string(file_index) + ".bin";
            std::string file_name = this->store_path + "/" + std::to_string(prefix_id) + "_layer" + std::to_string(layer_id)  + "_part" + std::to_string(file_index) + ".bin";
            //std::string file_name = this->store_path + "/" + std::to_string(prefix_id)  + "_layer" + std::to_string(layer_id) + ".bin";
            file.open(file_name, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open KV file: " + file_name);
            }
            cur_file_index = file_index;
        }

        int num_entries = length / entry_size;
        const bool retained = captures && !captures->read_buffers.empty();
        std::vector<char> buffer(retained ? 0 : length);
        char* data = retained ? captures->read_buffers.at(capture_index) + start_offset % 4096 : buffer.data();
        file.clear();
        file.seekg(start_offset, std::ios::beg);
        kv_read_bandwidth_limit().acquire(length);
        file.read(data, length);
        bytes_loaded += static_cast<long long>(file.gcount());
        if (file_index <= 1) region_bytes[file_index] += file.gcount();
        // All selected KV must exist, including tokens straddling disk pages.
        uint64_t required = (retained || positions.empty()) ? length : 0;
        for (uint64_t pos : positions)
            required = std::max(required, (pos + 1) * entry_size);
        TORCH_CHECK(static_cast<uint64_t>(file.gcount()) >= required,
                    "Short KV read: selected payload is incomplete");

        if (captures) {
            for (const auto& copy : captures->at(capture_index)) {
                TORCH_CHECK(copy.offset + entry_size <= static_cast<uint64_t>(file.gcount()),
                            "Short KV read: prefetched payload is incomplete");
                std::memcpy(copy.key, data + copy.offset, entry_size / 2);
                std::memcpy(copy.value, data + copy.offset + entry_size / 2, entry_size / 2);
            }
        }
        ++capture_index;
        if (captures && captures->read_only) continue;

        if(positions.empty()){
            for (int i = 0; i < num_entries; i++) {
                DTYPE* entry = reinterpret_cast<DTYPE*>(data + i * entry_size);
                DTYPE* cur_key = entry;
                DTYPE* cur_value = entry + this->head_dim;

                // 写入 queried_key 和 queried_value 中对应 head_id 和 position
                // int key_offset = head_id * this->max_length * this->head_dim + count * this->head_dim;
                // int value_offset = head_id * this->max_length * this->head_dim + count * this->head_dim;
                int key_index = token_order[count];
                size_t key_offset = query_output_offset(size_t(head_id) * max_length + key_index);
                size_t value_offset = key_offset;

                memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
                memcpy(this->queried_value + value_offset, cur_value, this->head_dim * sizeof(DTYPE));

                count++;
            }
        }else{
            for (uint64_t pos : positions) {

                DTYPE* entry = reinterpret_cast<DTYPE*>(data + pos * entry_size);
                DTYPE* cur_key = entry;
                DTYPE* cur_value = entry + this->head_dim;

                int key_index = token_order[count];
                size_t key_offset = query_output_offset(size_t(head_id) * max_length + key_index);
                size_t value_offset = key_offset;

                memcpy(this->queried_key + key_offset, cur_key, this->head_dim * sizeof(DTYPE));
                memcpy(this->queried_value + value_offset, cur_value, this->head_dim * sizeof(DTYPE));

                count++;
            }
        }
    }
    #pragma omp atomic update
    this->loaded_bytes += bytes_loaded;
    read_resources->add("hot_loaded_bytes", region_bytes[0]);
    read_resources->add("cold_loaded_bytes", region_bytes[1]);
    if (file.is_open()) file.close();
}

void KVStore::load_key_value_from_file_iouring(
    std::vector<std::tuple<uint64_t, uint64_t, int>>& content,
    std::vector<int>& token_order,
    int prefix_id,
    int layer_id,
    int head_id
) {
    // collect_queried_key_value only creates exact contiguous ranges, so every
    // entry in each range is selected. Adapt those ranges to the selection-aware
    // representation without changing collection or scatter ordering.
    std::vector<MergeRead>
        mapped_content;
    mapped_content.reserve(content.size());
    for (const auto& [file_index, offset, length] : content) {
        mapped_content.emplace_back(
            file_index, offset, static_cast<uint64_t>(length), std::vector<uint64_t>{});
    }
    this->load_key_value_from_file_iouring(
        mapped_content, token_order, prefix_id, layer_id, head_id);
}

void KVStore::load_key_value_from_file_iouring(
    std::vector<MergeRead>& content,
    std::vector<int>& token_order,
    int prefix_id,
    int layer_id,
    int head_id,
    const CacheReadCopies* captures,
    size_t capture_begin
) {
    if (content.empty()) return;
    double setup_seconds = 0, submit_seconds = 0, wait_seconds = 0, scatter_seconds = 0;
    auto setup_start = ReadClock::now();

    // Each OpenMP worker owns a persistent ring. This avoids paying setup and
    // mmap costs once per head/layer while keeping the existing per-head merge
    // and scatter semantics unchanged.
    thread_local std::unique_ptr<RawIoUring> ring;
    thread_local bool fallback_warning_printed = false;
    if (!ring) {
        ring = std::make_unique<RawIoUring>(get_iouring_queue_depth());
    }
    if (!ring->valid()) {
        if (!fallback_warning_printed) {
            std::cerr << "io_uring unavailable (" << ring->error()
                      << "); falling back to synchronous buffered reads"
                      << std::endl;
            fallback_warning_printed = true;
        }
        auto fallback_start = ReadClock::now();
        merge_load_key_value_from_file(
            content, token_order, prefix_id, layer_id, head_id, captures, capture_begin);
        read_resources->add("fallback_thread_seconds", read_seconds(fallback_start));
        read_resources->add("fallback_head_calls", 1);
        return;
    }

    const size_t entry_size = 2 * this->head_dim * sizeof(DTYPE);

    struct PendingRead {
        size_t content_index;
        size_t token_order_start;
        char* io_buffer;
        size_t io_size;
        size_t data_offset;
        size_t logical_size;
    };

    // token_order is in physical-offset order. Record the beginning of each
    // merged segment so completions may be scattered out of submission order.
    std::vector<size_t> token_starts(content.size());
    size_t token_cursor = 0;
    for (size_t i = 0; i < content.size(); ++i) {
        const auto& [file_index, offset, length, positions] = content[i];
        (void)file_index;
        (void)offset;
        token_starts[i] = token_cursor;
        if (positions.empty()) {
            token_cursor += length / entry_size;
        } else {
            token_cursor += positions.size();
        }
    }
    if (!(captures && captures->read_only) && token_cursor != token_order.size()) {
        std::cerr << "io_uring KV read: token-order size mismatch for layer "
                  << layer_id << ", head " << head_id << ": expected "
                  << token_cursor << ", got " << token_order.size()
                  << std::endl;
        auto fallback_start = ReadClock::now();
        merge_load_key_value_from_file(
            content, token_order, prefix_id, layer_id, head_id, captures, capture_begin);
        read_resources->add("fallback_thread_seconds", read_seconds(fallback_start));
        read_resources->add("fallback_head_calls", 1);
        return;
    }

    std::unordered_map<uint64_t, int> fd_table;
    for (const auto& [file_index, offset, length, positions] : content) {
        (void)offset;
        (void)length;
        (void)positions;
        if (fd_table.find(file_index) != fd_table.end()) continue;

        std::string file_name = this->store_path + "/" +
            std::to_string(prefix_id) + "_layer" + std::to_string(layer_id) +
            "_part" + std::to_string(file_index) + ".bin";
        int fd = read_resources->open_file(file_name);
        if (fd < 0) {
            std::cerr << "io_uring O_DIRECT KV read: open failed for "
                      << file_name
                      << ": " << std::strerror(errno)
                      << "; falling back to synchronous reads" << std::endl;

            auto fallback_start = ReadClock::now();
            merge_load_key_value_from_file(
                content, token_order, prefix_id, layer_id, head_id, captures, capture_begin);
            read_resources->add("fallback_thread_seconds", read_seconds(fallback_start));
            read_resources->add("fallback_head_calls", 1);
            return;
        }
        fd_table.emplace(file_index, fd);
    }

    auto scatter = [&](const PendingRead& pending) -> bool {
        if (captures && captures->read_only) return true;
        auto scatter_start = ReadClock::now();
        const auto& [file_index, offset, length, positions] =
            content[pending.content_index];
        (void)file_index;
        (void)offset;

        size_t order_pos = pending.token_order_start;
        auto copy_entry = [&](size_t entry_pos) -> bool {
            if (order_pos >= token_order.size() ||
                (entry_pos + 1) * entry_size > pending.logical_size) {
                return false;
            }

            const DTYPE* entry = reinterpret_cast<const DTYPE*>(
                pending.io_buffer + pending.data_offset +
                entry_pos * entry_size);
            int token_index = token_order[order_pos++];
            size_t output_offset =
                query_output_offset(size_t(head_id) * max_length + token_index);
            memcpy(this->queried_key + output_offset,
                   entry, this->head_dim * sizeof(DTYPE));
            memcpy(this->queried_value + output_offset,
                   entry + this->head_dim, this->head_dim * sizeof(DTYPE));
            return true;
        };

        if (positions.empty()) {
            size_t num_entries = length / entry_size;
            for (size_t i = 0; i < num_entries; ++i) {
                if (!copy_entry(i)) return false;
            }
        } else {
            for (uint64_t position : positions) {
                if (!copy_entry(position)) return false;
            }
        }
        if (captures) {
            for (const auto& copy : captures->at(capture_begin + pending.content_index)) {
                if (copy.offset + entry_size > pending.logical_size) return false;
                const char* source = pending.io_buffer + pending.data_offset + copy.offset;
                std::memcpy(copy.key, source, entry_size / 2);
                std::memcpy(copy.value, source + entry_size / 2, entry_size / 2);
            }
        }
        scatter_seconds += read_seconds(scatter_start);
        return true;
    };

    auto& worker_buffer = read_resources->worker_buffer();
    setup_seconds += read_seconds(setup_start);
    bool success = true;
    long long bytes_loaded = 0;
    long long region_bytes[2] = {0, 0};
    size_t next = 0;
    const size_t queue_depth = ring->capacity();

    // Submit in queue-depth-sized windows. All buffers in a window stay alive
    // until every CQE has been consumed.
    while (next < content.size() && success) {
        setup_start = ReadClock::now();
        const size_t window_size =
            std::min(queue_depth, content.size() - next);

        size_t window_bytes = 0;
        std::vector<uint64_t> aligned_offsets(window_size);
        std::vector<size_t> data_offsets(window_size);
        std::vector<size_t> aligned_lengths(window_size);
        for (size_t i = 0; i < window_size; ++i) {
            uint64_t offset = std::get<1>(content[next + i]);
            uint64_t length = std::get<2>(content[next + i]);
            if (!align_direct_read(
                    offset, length, aligned_offsets[i], data_offsets[i],
                    aligned_lengths[i]) ||
                aligned_lengths[i] >
                    std::numeric_limits<unsigned int>::max() ||
                aligned_lengths[i] >
                    std::numeric_limits<size_t>::max() - window_bytes) {
                std::cerr << "io_uring O_DIRECT KV read: invalid range offset="
                          << offset << ", length=" << length
                          << std::endl;
                success = false;
                break;
            }
            window_bytes += aligned_lengths[i];
        }
        if (!success) break;

        // Reuse this worker's largest window across heads, layers and requests.
        // Its base and every SQE slice remain 4-KiB aligned.
        try {
            if ((!captures || captures->read_buffers.empty()) && worker_buffer.capacity < window_bytes) {
                worker_buffer.data = allocate_aligned_buffer(window_bytes);
                worker_buffer.capacity = window_bytes;
                read_resources->add("buffer_growths", 1);
            }
        } catch (const std::exception& error) {
            std::cerr << "io_uring O_DIRECT KV read: " << error.what()
                      << std::endl;
            success = false;
            break;
        }
        char* window_data = static_cast<char*>(worker_buffer.data.get());
        std::vector<PendingRead> pending;
        pending.reserve(window_size);
        size_t buffer_offset = 0;

        for (size_t i = 0; i < window_size; ++i) {
            size_t content_index = next + i;
            const auto& [file_index, offset, length, positions] =
                content[content_index];
            (void)offset;
            (void)positions;
            pending.push_back(PendingRead{
                content_index, token_starts[content_index],
                captures && !captures->read_buffers.empty()
                    ? captures->read_buffers.at(capture_begin + content_index)
                    : window_data + buffer_offset, aligned_lengths[i],
                data_offsets[i], static_cast<size_t>(length)});
            PendingRead& request = pending.back();
            if (!ring->prepare_read(
                    fd_table.at(file_index), request.io_buffer,
                    static_cast<unsigned int>(request.io_size),
                    aligned_offsets[i],
                    static_cast<uint64_t>(pending.size()))) {
                std::cerr << "io_uring O_DIRECT KV read: submission queue "
                             "unexpectedly full"
                          << std::endl;
                success = false;
                break;
            }
            buffer_offset += request.io_size;
        }

        if (!success) break;

        setup_seconds += read_seconds(setup_start);
        auto submit_start = ReadClock::now();
        int submitted = ring->submit(static_cast<unsigned int>(pending.size()));
        submit_seconds += read_seconds(submit_start);
        if (submitted < 0) {
            std::cerr << "io_uring KV read: submit failed: "
                      << std::strerror(-submitted) << std::endl;
            success = false;
            break;
        }

        size_t completed = 0;
        while (completed < pending.size()) {
            auto wait_start = ReadClock::now();
            int wait_result = ring->wait_for_completion();
            wait_seconds += read_seconds(wait_start);
            if (wait_result < 0) {
                std::cerr << "io_uring KV read: wait failed: "
                          << std::strerror(-wait_result) << std::endl;
                success = false;
                break;
            }

            uint64_t user_data = 0;
            int result = 0;
            while (ring->pop_completion(user_data, result)) {
                if (user_data == 0 || user_data > pending.size()) {
                    std::cerr << "io_uring KV read: invalid completion user_data"
                              << std::endl;
                    success = false;
                } else {
                    PendingRead& request = pending[user_data - 1];
                    size_t required =
                        request.data_offset + request.logical_size;
                    if (result >= 0 && static_cast<size_t>(result) < required &&
                        !(captures && (!captures->read_buffers.empty() || captures->read_only))) {
                        const auto& positions = std::get<3>(content[request.content_index]);
                        if (!positions.empty()) {
                            // Fixed blocks can extend beyond EOF. Match the
                            // buffered reader: selected/captured KV must be
                            // complete; unused block padding need not exist.
                            size_t payload_end = 0;
                            for (uint64_t pos : positions)
                                payload_end = std::max(payload_end, size_t(pos + 1) * entry_size);
                            if (captures) for (const auto& copy : captures->at(capture_begin + request.content_index))
                                payload_end = std::max(payload_end, size_t(copy.offset) + entry_size);
                            required = request.data_offset + payload_end;
                            // Scatter must validate against bytes actually
                            // returned, rather than the requested block size.
                            request.logical_size = static_cast<size_t>(result) > request.data_offset
                                ? static_cast<size_t>(result) - request.data_offset : 0;
                        }
                    }
                    if (result < 0) {
                        std::cerr << "io_uring O_DIRECT KV read failed: "
                                  << std::strerror(-result) << std::endl;
                        success = false;
                    } else if (static_cast<size_t>(result) < required) {
                        // Files are currently not padded to 4 KiB. A direct
                        // read covering the final logical entry may therefore
                        // complete short at EOF; it is valid if that entry is
                        // fully present in the returned prefix.
                        std::cerr << "io_uring O_DIRECT KV read returned "
                                  << result << " bytes; at least " << required
                                  << " bytes are required" << std::endl;
                        success = false;
                    } else if (!scatter(request)) {
                        std::cerr << "io_uring KV read: invalid scatter metadata"
                                  << std::endl;
                        success = false;
                    } else {
                        bytes_loaded += static_cast<long long>(result);
                        const auto region = std::get<0>(content[request.content_index]);
                        if (region <= 1) region_bytes[region] += result;
                    }
                }
                ++completed;
            }
            if (!success) break;
        }
        next += window_size;
    }

    read_resources->add("setup_thread_seconds", setup_seconds);
    read_resources->add("submit_thread_seconds", submit_seconds);
    read_resources->add("wait_thread_seconds", wait_seconds);
    read_resources->add("scatter_thread_seconds", scatter_seconds);
    read_resources->add("iouring_head_calls", 1);

    if (!success) {
        // Do not silently return partially populated KV buffers. Fatal runtime
        // I/O failures indicate that this request cannot be computed safely.
        std::cerr << "io_uring KV loading failed for layer " << layer_id
                  << ", head " << head_id << std::endl;
        std::abort();
    }

    #pragma omp atomic update
    this->num_io += static_cast<int>(content.size());
    #pragma omp atomic update
    this->loaded_bytes += bytes_loaded;
    read_resources->add("hot_loaded_bytes", region_bytes[0]);
    read_resources->add("cold_loaded_bytes", region_bytes[1]);
}

torch::Tensor KVStore::get_queried_key_cache()
{
    auto options = torch::TensorOptions().dtype(torch::kFloat16);
    // auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::from_blob(this->queried_key, {this->num_key_value_heads, this->max_length, this->head_dim}, options);
    return tensor;
}

torch::Tensor KVStore::get_initial_key_view(int layer) {
    TORCH_CHECK(layer >= 0 && static_cast<size_t>(layer) < key_cache.size() && offload_len > 0,
                "Initial key view requires a populated valid layer");
    // Startup-only borrowed view; the binding retains this KVStore's lifetime.
    return torch::from_blob(key_cache[layer],
        {num_key_value_heads, max_length, head_dim},
        torch::TensorOptions().dtype(torch::kFloat16)).narrow(1, 0, offload_len);
}

torch::Tensor KVStore::get_queried_value_cache()
{
    auto options = torch::TensorOptions().dtype(torch::kFloat16);
    // auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::from_blob(this->queried_value, {this->num_key_value_heads, this->max_length, this->head_dim}, options);
    return tensor;
}

int KVStore::get_num_io_and_reset(){
    int n = this->num_io;
    this->num_io = 0;
    return n;
}

long long KVStore::get_loaded_bytes_and_reset(){
    long long n = this->loaded_bytes;
    this->loaded_bytes = 0;
    return n;
}

long long KVStore::get_reorder_io_bytes_and_reset(){
    long long n = this->reorder_io_bytes;
    this->reorder_io_bytes = 0;
    return n;
}

void KVStore::set_compaction_threshold(double threshold){
    this->compaction_threshold = threshold;
}
void KVStore::configure_cache(int64_t capacity_bytes, const std::string& policy) {
    std::unique_lock<std::shared_mutex> lock(kv_io_mutex);
    TORCH_CHECK(capacity_bytes >= 0 && policy == "access_control", "AdaKVs requires access_control cache");
    hybrid_cache_policy = policy;
    if (const char* p = std::getenv("HYBRID_CACHE_READ_US")) hybrid_cache_read_us = std::stod(p);
    if (const char* p = std::getenv("HYBRID_CACHE_READ_BPS")) hybrid_cache_read_bps = std::stod(p);
    TORCH_CHECK(std::isfinite(hybrid_cache_read_us) && hybrid_cache_read_us >= 0 &&
                std::isfinite(hybrid_cache_read_bps) && hybrid_cache_read_bps > 0, "Invalid hybrid planner cost");
    const char* miss_mode=std::getenv("HYBRID_CACHE_MISS_READ_MODE");
    hybrid_cache_miss_read_mode=miss_mode?miss_mode:"auto";
    const char* hot_overlap = std::getenv("HYBRID_CACHE_HOT_OVERLAP");
    TORCH_CHECK(!hot_overlap || std::string(hot_overlap)=="0" || std::string(hot_overlap)=="1",
                "HYBRID_CACHE_HOT_OVERLAP must be 0 or 1");
    hybrid_cache_hot_overlap = false;
    TORCH_CHECK(!hot_overlap || std::string(hot_overlap)=="0", "AdaKVs uses non-overlapped hot reads");
    TORCH_CHECK(hybrid_cache_miss_read_mode=="auto" || hybrid_cache_miss_read_mode=="paged" ||
                hybrid_cache_miss_read_mode=="coalesced", "Invalid hybrid miss read mode");
    double access_alpha=1., access_l0=hybrid_cache_read_us*1e-6*hybrid_cache_read_bps;
    bool access_bypass=false;
    {
        if(const char* p=std::getenv("ACCESS_CONTROL_ALPHA")) access_alpha=std::stod(p);
        if(const char* p=std::getenv("ACCESS_CONTROL_L0_BYTES")) access_l0=std::stod(p);
        if(const char* p=std::getenv("ACCESS_CONTROL_HOT_BYPASS")) {
            TORCH_CHECK(std::string(p)=="0" || std::string(p)=="1", "Invalid access-control hot bypass");
            access_bypass=std::string(p)=="1";
        }
    }
    hybrid_cache.configure(policy == "hybrid_layout" || policy == "access_control" ? capacity_bytes : 0,
                           policy=="access_control",access_alpha,access_l0,access_bypass);

    TORCH_CHECK(!access_bypass, "AdaKVs caches all three regions");
}
void KVStore::clear_cache() {
    std::unique_lock<std::shared_mutex> lock(kv_io_mutex);
    cold_region_indices.clear(); hot_region_indices.clear();
    promotion_snapshots.clear(); hybrid_cache.clear();
}
std::map<std::string, uint64_t> KVStore::get_cache_stats(bool reset) {
    std::unique_lock<std::shared_mutex> lock(kv_io_mutex);
    return hybrid_cache.stats(reset);
}
std::vector<std::map<std::string, uint64_t>> KVStore::get_cache_partition_stats() {
    std::unique_lock<std::shared_mutex> lock(kv_io_mutex);
    return hybrid_cache.partition_stats();
}

#include "hot_growth_impl.h"
#include "hybrid_cache_impl.h"

PYBIND11_MODULE(kvstore, m) {
    pybind11::class_<KVStore>(m, "KVStore")
        .def(pybind11::init<>())
        .def("alloc", &KVStore::alloc)
        .def("clear", &KVStore::clear)
        .def("fill", &KVStore::fill)
        .def("persist_meta", &KVStore::persist_meta)
        .def("recover_meta", &KVStore::recover_meta)
        .def("reorder_persist", &KVStore::reorder_persist)
        .def("finish_persist", &KVStore::finish_persist)
        .def("merge_collect_into", &KVStore::merge_collect_into)
        .def("merge_collect_queried_key_value", &KVStore::merge_collect_queried_key_value,
             pybind11::arg("prefix_id"), pybind11::arg("layer_id"),
             pybind11::arg("indices"), pybind11::arg("counts"),
             pybind11::arg("importance_scores") = std::nullopt)
        .def("get_queried_key_cache", &KVStore::get_queried_key_cache)
        .def("get_queried_value_cache", &KVStore::get_queried_value_cache)
        .def("get_initial_key_view", &KVStore::get_initial_key_view)
        .def("set_compaction_threshold", &KVStore::set_compaction_threshold)
        .def("configure_cache", &KVStore::configure_cache)
        .def("clear_cache", &KVStore::clear_cache)
        .def("get_cache_partition_stats", &KVStore::get_cache_partition_stats)
        .def("clear_read_cache", &KVStore::clear_read_cache)
        .def("configure_hot_growth", &KVStore::configure_hot_growth)
        .def("get_hot_growth_layout", &KVStore::get_hot_growth_layout)
        .def("hot_growth_update", &KVStore::hot_growth_update)
        .def("set_hot_growth_foreground", &KVStore::set_hot_growth_foreground)
        .def("set_promotion_snapshot_request", &KVStore::set_promotion_snapshot_request)
        .def("set_hot_growth_read_mode", &KVStore::set_hot_growth_read_mode)
        .def("set_merge_threshold_bytes", &KVStore::set_merge_threshold_bytes)
        .def("set_merge_extra_read_budget", &KVStore::set_merge_extra_read_budget)
        .def("set_region_merge_extra_read_budgets", &KVStore::set_region_merge_extra_read_budgets)
        .def("set_collect_io_backend", &KVStore::set_collect_io_backend)
        .def("get_collect_io_backend", &KVStore::get_collect_io_backend)
        .def("get_num_io", &KVStore::get_num_io_and_reset)
        .def("get_loaded_bytes", &KVStore::get_loaded_bytes_and_reset)
        .def("get_reorder_io_bytes", &KVStore::get_reorder_io_bytes_and_reset)
        .def("get_cache_stats", &KVStore::get_cache_stats, pybind11::arg("reset") = false)
        .def("get_read_profile", &KVStore::get_read_profile, pybind11::arg("reset") = true)
        .def("get_merge_read_stats", &KVStore::get_merge_read_stats, pybind11::arg("reset") = true)
        .def("get_reorder_memory_stats", &KVStore::get_reorder_memory_stats, pybind11::arg("reset") = true)
        ;
}
