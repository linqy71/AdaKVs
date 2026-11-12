#pragma once
#include "merge_read_plan.h"

struct WarmCacheReadPlan {
    std::vector<PhysicalMergeRun> reads;
    uint64_t dropped_ranges = 0, coalesced_ranges = 0, paged_ranges = 0;
    uint64_t page_reads = 0, trimmed_bytes = 0;
};

// Reuse the sorted full plan. Scan each selected entry once to find miss pages;
// a second linear scan assigns scatter slices (including covered cache hits).
// Requires whole entries within 4-KiB pages and nonoverlapping full-plan pages.
template<class IsHit>
WarmCacheReadPlan plan_warm_cache_ranges(
        const std::vector<std::tuple<uint64_t,uint64_t,int>>& entries,
        const std::vector<PhysicalMergeRun>& full, IsHit is_hit,
        bool allow_pages, bool force_pages, double request_cost, double bandwidth) {
    WarmCacheReadPlan result;
    result.reads.reserve(full.size());
    std::vector<uint64_t> pages;
    for (const auto& run : full) {
        pages.clear();
        for (size_t i=run.begin; i<run.end; ++i) {
            if (is_hit(std::get<2>(entries[i]))) continue;
            uint64_t page=std::get<1>(entries[i])/4096*4096;
            if (pages.empty() || pages.back()!=page) pages.push_back(page);
        }
        if (pages.empty()) { ++result.dropped_ranges; continue; }
        result.page_reads+=pages.size();
        uint64_t first=pages.front(), end=pages.back()+4096;
        result.trimmed_bytes+=end-first;
        bool paged=allow_pages && (force_pages ||
            pages.size()*request_cost+pages.size()*4096.0/bandwidth <
            request_cost+(end-first)/bandwidth);
        size_t cursor=run.begin;
        auto append=[&](uint64_t start, uint64_t stop) {
            while (cursor<run.end && std::get<1>(entries[cursor])<start) ++cursor;
            size_t begin=cursor;
            while (cursor<run.end && std::get<1>(entries[cursor])<stop) ++cursor;
            result.reads.push_back({run.file,start,stop-start,begin,cursor});
        };
        if (paged) {
            ++result.paged_ranges;
            for (uint64_t page : pages) append(page,page+4096);
        } else {
            ++result.coalesced_ranges;
            append(first,end);
        }
    }
    return result;
}
