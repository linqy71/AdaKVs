#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

// Process-wide budget shared by every KVStore and ImpressChunkReader instance.
// Reserve one submission window at a time; sleep outside the mutex so OpenMP
// workers retain their own rings and asynchronous in-flight reads.
class KVReadBandwidthLimit {
public:
    struct Stats {
        uint64_t bytes_per_second, burst_bytes, charged_bytes, reservations;
        double wait_thread_seconds;
    };
    KVReadBandwidthLimit()
        : rate_(environment("KVSTORE_READ_BPS", 7000000000ULL)),
          burst_(environment("KVSTORE_READ_BURST_BYTES", 4 * 1024 * 1024)),
          tokens_(burst_), updated_(Clock::now()) {}

    void acquire(uint64_t bytes) {
        if (bytes == 0) return;
        Clock::time_point deadline;
        bool wait = false;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            bytes_ += bytes;
            ++reservations_;
            if (!rate_) return;
            const auto now = Clock::now();
            tokens_ = std::min<long double>(burst_, tokens_ +
                std::chrono::duration<long double>(now - updated_).count() * rate_);
            updated_ = now;
            // Negative credits represent reservations by already-waiting
            // threads. Do not reset debt at layer/request boundaries.
            tokens_ -= bytes;
            if (tokens_ < 0) {
                const long double nanos = std::ceil(-tokens_ * 1e9L / rate_);
                if (nanos > std::numeric_limits<int64_t>::max() / 2)
                    throw std::runtime_error("KV read pacing deadline overflow");
                deadline = now + std::chrono::nanoseconds(static_cast<int64_t>(nanos));
                wait = true;
            }
        }
        if (wait) {
            const auto start = Clock::now();
            std::this_thread::sleep_until(deadline);
            const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
            std::lock_guard<std::mutex> guard(mutex_);
            wait_seconds_ += elapsed;
        }
    }
    Stats stats() {
        std::lock_guard<std::mutex> guard(mutex_);
        return {rate_, burst_, bytes_, reservations_, wait_seconds_};
    }
private:
    using Clock = std::chrono::steady_clock;
    static uint64_t environment(const char* name, uint64_t fallback) {
        const char* value = std::getenv(name);
        if (!value) return fallback;
        if (!*value) throw std::invalid_argument(std::string(name) + " must be an integer (0 disables)");
        uint64_t result = 0;
        for (const char* p = value; *p; ++p) {
            if (*p < '0' || *p > '9' || result > (1000000000000000ULL - (*p - '0')) / 10)
                throw std::invalid_argument(std::string(name) + " must be in [0, 1000000000000000]");
            result = result * 10 + (*p - '0');
        }
        return result;
    }
    const uint64_t rate_, burst_;
    std::mutex mutex_;
    long double tokens_;
    Clock::time_point updated_;
    uint64_t bytes_ = 0, reservations_ = 0;
    double wait_seconds_ = 0;
};

inline KVReadBandwidthLimit& kv_read_bandwidth_limit() {
    static KVReadBandwidthLimit limiter;
    return limiter;
}
