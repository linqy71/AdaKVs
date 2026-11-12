#pragma once
#include "merge_read_plan.h"

// One hot extent, arbitrary selected order. Gather missing pages during lookup;
// planning and scatter routing are linear in pages/tokens, with no token sort.
// The page table doubles as a bitmap before finish() and a read index after it.
class HotCacheReadPlan {
    uint64_t base=0, first=0, last=0;
    size_t missing=0;
    std::vector<int> page_reads;
public:
    std::vector<PhysicalMergeRun> reads;
    bool paged=false;
    void reset(uint64_t start,uint64_t end) {
        base=start/4096*4096;
        page_reads.assign(end>start ? (merge_aligned_end(end)-base)/4096 : 0,-1);
        reads.clear();missing=0;first=last=0;paged=false;
    }
    void add_miss(uint64_t offset) {
        const uint64_t page=(offset-base)/4096;
        if(page_reads[page]!=-1) return;
        page_reads[page]=-2;
        if(!missing++) first=last=page;
        else {first=std::min(first,page);last=std::max(last,page);}
    }
    size_t miss_pages() const { return missing; }
    uint64_t trimmed_bytes() const { return missing ? (last-first+1)*4096 : 0; }
    void finish(bool allow_pages,bool force_pages,double request_cost,double bandwidth) {
        if(!missing) return;
        paged=allow_pages && (force_pages ||
            missing*request_cost+missing*4096.0/bandwidth <
            request_cost+trimmed_bytes()/bandwidth);
        if(paged) {
            reads.reserve(missing);
            for(uint64_t i=first;i<=last;++i) if(page_reads[i]==-2) {
                page_reads[i]=reads.size();
                reads.push_back({0,base+i*4096,4096,0,0});
            }
        } else reads.push_back({0,base+first*4096,trimmed_bytes(),0,0});
    }
    // -1 means the selected token must be served by cache. All misses have a
    // read; hits on a submitted page/span are scattered from that read once.
    int read_index(uint64_t offset) const {
        if(!missing) return -1;
        const uint64_t page=(offset-base)/4096;
        if(paged) return page_reads[page];
        return page>=first && page<=last ? 0 : -1;
    }
};
