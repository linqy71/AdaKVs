#pragma once

#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

// Empty positions means every entry is selected (strict contiguous readers).
using MergeRead = std::tuple<uint64_t, uint64_t, uint64_t, std::vector<uint64_t>>;

// Optional cache-only scatter destinations inside the same physical read.
// offset is relative to the logical MergeRead start, in bytes.
struct CacheReadCopy { uint64_t offset; void* key; void* value; };
struct CacheReadCopies : std::vector<std::vector<CacheReadCopy>> {
    // Optional 4-KiB-aligned destinations retained until cache commit.
    std::vector<char*> read_buffers;
    // Read complete extents into retained buffers without selected-token scatter.
    bool read_only = false;
};

// Input is sorted by (file, physical offset, destination index).
inline void plan_merge_reads_into(
    const std::vector<std::tuple<uint64_t, uint64_t, int>>& entries,
    uint64_t entry_size, uint64_t threshold, std::vector<MergeRead>& reads) {
    size_t used = 0;
    size_t start = 0;
    while (start < entries.size()) {
        const auto [file, offset, destination] = entries[start];
        size_t end = start + 1;
        while (end < entries.size() && std::get<0>(entries[end]) == file &&
               std::get<1>(entries[end]) == std::get<1>(entries[end - 1]) + entry_size) {
            ++end;
        }
        uint64_t length = (end - start) * entry_size;
        bool merge = false;
        if (used != 0) {
            const auto& [last_file, last_offset, last_length, positions] = reads[used - 1];
            merge = last_file == file && offset >= last_offset + last_length &&
                    offset - last_offset <= threshold &&
                    length <= threshold - (offset - last_offset);
        }
        if (!merge) {
            if (used == reads.size())
                reads.emplace_back(file, offset, length, std::vector<uint64_t>{});
            auto& [f, o, n, positions] = reads[used++];
            f = file; o = offset; n = length;
            positions.clear();  // Empty means the whole contiguous run is selected.
        } else {
            auto& [f, read_offset, read_length, positions] = reads[used - 1];
            if (positions.empty()) {
                positions.reserve(read_length / entry_size + end - start);
                for (uint64_t i = 0; i < read_length / entry_size; ++i)
                    positions.push_back(i);
            }
            for (size_t i = start; i < end; ++i)
                positions.push_back((std::get<1>(entries[i]) - read_offset) / entry_size);
            read_length = offset - read_offset + length;
        }
        start = end;
    }
    reads.resize(used);
}

inline std::vector<MergeRead> plan_merge_reads(
    const std::vector<std::tuple<uint64_t, uint64_t, int>>& entries,
    uint64_t entry_size, uint64_t threshold) {
    std::vector<MergeRead> reads;
    plan_merge_reads_into(entries, entry_size, threshold, reads);
    return reads;
}

// Collection-wide runs retain a slice of the sorted entries, including duplicate
// destinations. Offsets in a KV file are multiples of the fixed entry size.
struct PhysicalMergeRun {
    uint64_t file, offset, length;
    size_t begin, end;
};

inline uint64_t merge_aligned_end(uint64_t end) {
    return (end + 4095) / 4096 * 4096;
}

inline std::vector<PhysicalMergeRun> collect_merge_runs(
    const std::vector<std::tuple<uint64_t, uint64_t, int>>& entries,
    uint64_t entry_size) {
    std::vector<PhysicalMergeRun> runs;
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto [file, offset, destination] = entries[i];
        if (!runs.empty() && runs.back().file == file &&
            offset <= runs.back().offset + runs.back().length) {
            auto& run = runs.back();
            run.length = std::max(run.length, offset + entry_size - run.offset);
            run.end = i + 1;
        } else {
            runs.push_back({file, offset, entry_size, i, i + 1});
        }
    }
    return runs;
}

// Overlapping aligned ranges must share a read even when their logical span
// exceeds threshold. Adjacent aligned ranges are NOT merged on that basis.
// This guarantees each (file, 4-KiB page) is submitted at most once per call.
// With adjacent_pages=true and threshold=0, also join touching aligned ranges
// without introducing any additional physical pages (Placement-only reads).
inline void plan_unique_page_reads_into(
    const std::vector<PhysicalMergeRun>& runs, uint64_t threshold,
    std::vector<PhysicalMergeRun>& reads, bool adjacent_pages = false) {
    reads.clear();
    for (const auto& run : runs) {
        if (!reads.empty() && reads.back().file == run.file) {
            auto& last = reads.back();
            uint64_t end = run.offset + run.length;
            bool shared_page = run.offset / 4096 * 4096 <
                merge_aligned_end(last.offset + last.length);
            bool adjacent_page = adjacent_pages && run.offset / 4096 * 4096 ==
                merge_aligned_end(last.offset + last.length);
            if (shared_page || adjacent_page || end - last.offset <= threshold) {
                last.length = end - last.offset;
                last.end = run.end;
                continue;
            }
        }
        reads.push_back(run);
    }
}

inline uint64_t merge_read_bytes(const std::vector<PhysicalMergeRun>& reads) {
    uint64_t bytes = 0;
    for (const auto& read : reads)
        bytes += merge_aligned_end(read.offset + read.length) -
            read.offset / 4096 * 4096;
    return bytes;
}

struct MergeCandidate {
    uint64_t threshold, io_count, bytes;
};

// Evaluate a candidate without allocating or writing a full read plan. Keep
// only the current merged span; count its aligned bytes when it is closed.
inline MergeCandidate summarize_unique_page_reads(
    const std::vector<PhysicalMergeRun>& runs, uint64_t threshold,
    bool adjacent_pages = false) {
    MergeCandidate result{threshold, 0, 0};
    uint64_t file = 0, offset = 0, end = 0;
    for (const auto& run : runs) {
        const uint64_t next_end = run.offset + run.length;
        const uint64_t page = run.offset / 4096 * 4096;
        if (result.io_count && file == run.file &&
            (page < merge_aligned_end(end) ||
             (adjacent_pages && page == merge_aligned_end(end)) ||
             next_end - offset <= threshold)) {
            end = next_end;
            continue;
        }
        if (result.io_count)
            result.bytes += merge_aligned_end(end) - offset / 4096 * 4096;
        ++result.io_count;
        file = run.file;
        offset = run.offset;
        end = next_end;
    }
    if (result.io_count)
        result.bytes += merge_aligned_end(end) - offset / 4096 * 4096;
    return result;
}

// Candidates are nonempty. Tie-break by bytes, then by threshold; evaluate the
// budget against the least expensive physical plan, not against useful bytes.
inline uint64_t select_budgeted_merge_threshold(
    const std::vector<MergeCandidate>& candidates, double extra_read_budget) {
    uint64_t minimum_bytes = candidates.front().bytes;
    for (const auto& candidate : candidates)
        minimum_bytes = std::min(minimum_bytes, candidate.bytes);
    const MergeCandidate* best = nullptr;
    for (const auto& candidate : candidates) {
        // Compare fractions at the same precision as the supplied budget so
        // an exact decimal boundary (e.g. 3%) is not rejected by promotion.
        if (candidate.bytes > minimum_bytes &&
            (minimum_bytes == 0 ||
             static_cast<double>(candidate.bytes - minimum_bytes) / minimum_bytes >
                 extra_read_budget)) continue;
        if (!best || std::tie(candidate.io_count, candidate.bytes, candidate.threshold) <
                     std::tie(best->io_count, best->bytes, best->threshold))
            best = &candidate;
    }
    return best->threshold;
}
