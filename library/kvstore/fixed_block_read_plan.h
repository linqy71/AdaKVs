#pragma once
#include "merge_read_plan.h"
#include <stdexcept>

// Input sorted by (file, offset, destination). A block is shared across heads,
// but never across files. Adjacent blocks deliberately remain separate reads.
inline std::vector<PhysicalMergeRun> plan_fixed_block_reads(
        const std::vector<std::tuple<uint64_t, uint64_t, int>>& entries,
        uint64_t entry_bytes, uint64_t block_bytes = 16384) {
    if (!entry_bytes || block_bytes < 4096 || block_bytes % 4096 || block_bytes % entry_bytes)
        throw std::invalid_argument("Fixed block must be page aligned and contain whole KV entries");
    std::vector<PhysicalMergeRun> reads;
    for (size_t begin = 0; begin < entries.size();) {
        const auto file = std::get<0>(entries[begin]);
        const auto offset = std::get<1>(entries[begin]) / block_bytes * block_bytes;
        size_t end = begin + 1;
        while (end < entries.size() && std::get<0>(entries[end]) == file &&
               std::get<1>(entries[end]) / block_bytes * block_bytes == offset) ++end;
        reads.push_back({file, offset, block_bytes, begin, end});
        begin = end;
    }
    return reads;
}
