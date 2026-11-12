#pragma once

// Included by hybrid_cache.h. Rebuild groups from CURRENT physical positions;
// logical payloads survive reorder. Input pointers must outlive this call and
// must not alias cache slots (the collector supplies completed output tensors).
// rejected_hot_misses is the exact count of unique nonresident hot tokens
// omitted from missing after hot_group_oversized() proves rejection inevitable.
template<class Locate>
HybridCache::GroupResult HybridCache::put_groups(
        int l,int h,const std::vector<Admission>& missing,
        const std::vector<GroupRange>& warm_ranges,
        uint64_t hot_start,uint64_t hot_end,uint64_t cold_block,Locate locate,
        size_t rejected_hot_misses,const std::vector<Location>* missing_locations,
        const FileRanges* coalesced_ranges) {
    if(!access_control || cold_block<4096 || cold_block%4096)
        throw std::logic_error("Invalid access-control group configuration");
    if(missing_locations && missing_locations->size()!=missing.size())
        throw std::logic_error("Admission locations must match missing tokens");
    if(coalesced_ranges && (rejected_hot_misses || bypass_hot))
        throw std::logic_error("Coalesced groups include hot and have no whole-hot prefilter");
    using Clock=std::chrono::steady_clock;
    const auto started=Clock::now();
    auto& p=partitions.at(l*heads+h);
    if(!p.scores) throw std::logic_error("Missing group EMA publication");
    ++p.batches;
    // Collection is serialized by the I/O lock: share scratch across all
    // partitions instead of retaining a selected-token workspace per head.
    auto& work=group_workspace;
    work.members.clear();work.groups.clear();work.group_order.clear();work.hot_victims.clear();
    work.members.reserve(p.heap.size()+missing.size());
    for(int slot:p.heap) {
        int token=p.tokens[slot];
        if(rejected_hot_misses && p.hot_tokens[token]) {
            work.hot_victims.push_back(slot);continue;
        }
        work.members.push_back({{token,nullptr,nullptr},locate(token)});
    }
    const size_t residents=work.members.size();
    // Reuse the dense token index for batch deduplication. Hits already occur
    // in the resident portion; they are not copied or charged a second time.
    size_t locations_reused=0;
    for(size_t i=0;i<missing.size();++i) {
        const auto& item=missing[i];
        if(item.token<0 || size_t(item.token)>=p.index.size()) {++p.rejected;continue;}
        if(p.index[item.token]!=-1) continue;
        auto location=missing_locations ? (*missing_locations)[i] : locate(item.token);
        locations_reused+=missing_locations!=nullptr;
        if(bypass_hot && location.region==0) continue;
        work.members.push_back({item,location});p.index[item.token]=-2;
    }
    for(size_t i=residents;i<work.members.size();++i) p.index[work.members[i].item.token]=-1;
    auto create=[&](int region,uint64_t offset,uint64_t length) {
        work.groups.push_back({-1,0,0,region,offset,length,0.});
        return int(work.groups.size()-1);
    };
    if(coalesced_ranges) {
        // Coalesced reads can span hot/warm or head boundaries. Quotas remain
        // per head, but hot and warm members in the SAME read are atomic.
        // Residents outside the current full selection plan fall back to pages.
        uint64_t first[2]={UINT64_MAX,UINT64_MAX},last[2]={};
        for(const auto& member:work.members) {
            const int file=member.location.region==2;
            const uint64_t page=member.location.offset/4096;
            first[file]=std::min(first[file],page);last[file]=std::max(last[file],page);
        }
        for(int file=0;file<2;++file) {
            auto& table=file ? work.cold_blocks : work.warm_pages;
            table.assign(first[file]==UINT64_MAX?0:last[file]-first[file]+1,-1);
            if(table.empty()) continue;
            const auto& ranges=(*coalesced_ranges)[file];
            auto start=std::lower_bound(ranges.begin(),ranges.end(),first[file]*4096,
                [](const GroupRange& range,uint64_t offset) {return range.offset+range.length<=offset;});
            for(auto it=start;it!=ranges.end() && it->offset/4096<=last[file];++it) {
                uint64_t begin=std::max(first[file],it->offset/4096);
                uint64_t end=std::min(last[file]+1,(it->offset+it->length)/4096);
                if(begin>=end) continue;
                int group=create(file?2:1,it->offset,it->length);
                std::fill(table.begin()+begin-first[file],table.begin()+end-first[file],group);
            }
        }
        for(size_t i=0;i<work.members.size();++i) {
            auto& member=work.members[i];const auto loc=member.location;
            const int file=loc.region==2;
            auto& table=file ? work.cold_blocks : work.warm_pages;
            int& mapped=table[loc.offset/4096-first[file]];
            if(mapped<0) mapped=create(loc.region,loc.offset/4096*4096,4096);
            auto& g=work.groups[mapped];
            if(!g.count) g.region=loc.region;
            else if(g.region!=loc.region) g.region=1; // Mixed hot/warm group.
            member.next=g.first;member.group=mapped;g.first=i;++g.count;
            g.residents+=p.index[member.item.token]>=0;
        }
    } else {
    // Dense physical page/block indexes avoid sorting or hashing tokens. Each
    // table spans only this head's observed positions, never preceding heads.
    uint64_t first[3]={UINT64_MAX,UINT64_MAX,UINT64_MAX}, last[3]={};
    for(const auto& member:work.members) {
        auto loc=member.location;
        if(!loc.region) continue;
        uint64_t unit=loc.offset/(loc.region==1?4096:cold_block);
        first[loc.region]=std::min(first[loc.region],unit);
        last[loc.region]=std::max(last[loc.region],unit);
    }
    work.warm_pages.assign(first[1]==UINT64_MAX?0:last[1]-first[1]+1,-1);
    work.cold_blocks.assign(first[2]==UINT64_MAX?0:last[2]-first[2]+1,-1);
    // Full warm ranges are sorted, nonoverlapping and page-aligned. Seed the
    // page lookup once from the existing plan, without a second merge search.
    if(!work.warm_pages.empty()) for(const auto& range:warm_ranges) {
        uint64_t begin=std::max(first[1],range.offset/4096);
        uint64_t end=std::min(last[1]+1,(range.offset+range.length)/4096);
        if(begin>=end) continue;
        int group=create(1,range.offset,range.length);
        std::fill(work.warm_pages.begin()+(begin-first[1]),
                  work.warm_pages.begin()+(end-first[1]),group);
    }
    int hot_group=-1;
    for(size_t i=0;i<work.members.size();++i) {
        auto& member=work.members[i];const auto loc=member.location;
        int group;
        if(loc.region==0) {
            if(hot_group<0) {
                uint64_t offset=hot_start/4096*4096;
                hot_group=create(0,offset,std::max<uint64_t>(4096,(hot_end+4095)/4096*4096-offset));
            }
            group=hot_group;
        } else {
            const uint64_t unit=loc.region==1?4096:cold_block;
            auto& table=loc.region==1?work.warm_pages:work.cold_blocks;
            int& mapped=table[loc.offset/unit-first[loc.region]];
            // Unselected warm residents outside current full ranges use their
            // current page; cold residents always use current fixed blocks.
            if(mapped<0) mapped=create(loc.region,loc.offset/unit*unit,unit);
            group=mapped;
        }
        auto& g=work.groups[group];member.next=g.first;member.group=group;g.first=i;++g.count;
        g.residents+=p.index[member.item.token]>=0;
    }
    }
    const auto grouped=Clock::now();
    // Reject impossible groups before EMA reads, exponentiation or sorting.
    auto reject_oversized=[&](int region,size_t count,size_t resident_count) {
        ++p.group_candidates;p.group_max_tokens=std::max<uint64_t>(p.group_max_tokens,count);
        ++p.group_rejected;p.rejected+=count-resident_count;
        ++p.group_oversized;p.group_evicted+=resident_count!=0;
        ++p.region_rejected[region];++p.region_oversized[region];
        ++p.fast_groups;p.fast_members+=count;
    };
    if(rejected_hot_misses) {
        const size_t count=rejected_hot_misses+work.hot_victims.size();
        if(count<=p.slots) throw std::logic_error("Hot early rejection must exceed the partition quota");
        reject_oversized(0,count,work.hot_victims.size());
        p.group_members+=count;++p.hot_early_groups;p.hot_skipped_misses+=rejected_hot_misses;
    }
    size_t eligible_members=0;
    for(size_t i=0;i<work.groups.size();++i) {
        auto& g=work.groups[i];
        if(!g.count) continue;
        if(g.count>p.slots) {reject_oversized(g.region,g.count,g.residents);continue;}
        eligible_members+=g.count;
        p.group_max_tokens=std::max<uint64_t>(p.group_max_tokens,g.count);
        work.group_order.push_back(i);
    }
    double score_sort_seconds=0;
    if(eligible_members<=p.slots) {
        // Every remaining group wins, so heat and ordering cannot affect the
        // result. Keep group/payload iteration order identical to ranked mode.
        ++p.all_fit_heads;p.all_fit_groups+=work.group_order.size();
        p.all_fit_members+=eligible_members;
    } else {
        ++p.ranked_heads;p.scored_groups+=work.group_order.size();
        const auto scoring=Clock::now();
        // Preserve accumulation order and the original tie breaks exactly.
        for(const auto& member:work.members) {
            auto& g=work.groups[member.group];
            if(g.count<=p.slots) g.score+=(*p.scores)[member.item.token];
        }
        for(size_t i:work.group_order) {
            auto& g=work.groups[i];g.score=group_score(g.score/g.count,g.length);
        }
        std::sort(work.group_order.begin(),work.group_order.end(),[&](size_t a,size_t b) {
            const auto& x=work.groups[a];const auto& y=work.groups[b];
            if(x.score!=y.score) return x.score>y.score;
            bool xr=x.residents==x.count, yr=y.residents==y.count;
            if(xr!=yr) return xr;
            if(x.residents!=y.residents) return x.residents>y.residents;
            return x.region!=y.region?x.region<y.region:x.offset<y.offset;
        });
        score_sort_seconds=std::chrono::duration<double>(Clock::now()-scoring).count();
    }
    size_t available=p.slots, inserted=0;
    p.resident_groups=0;
    for(int r=0;r<3;++r) p.region_used[r]=0;
    p.group_candidates+=work.group_order.size();p.group_members+=work.members.size();
    p.materialized_members+=work.members.size();
    for(size_t i:work.group_order) {
        auto& g=work.groups[i];
        g.keep=g.count<=available;
        if(g.keep) {
            available-=g.count;++p.group_selected;++p.resident_groups;
            for(int member=g.first;member>=0;member=work.members[member].next)
                p.region_used[work.members[member].location.region]+=p.entry;
            if(g.count==g.residents) ++p.group_retained;
            else {++p.group_admitted;inserted+=g.count-g.residents;}
        } else {
            ++p.group_rejected;p.rejected+=g.count-g.residents;
            p.group_oversized+=g.count>p.slots;
            p.group_evicted+=g.residents!=0;
            ++p.region_rejected[g.region];
            p.region_oversized[g.region]+=g.count>p.slots;
        }
    }
    const auto planned=Clock::now();
    // Decide everything before touching payloads. Losers are removed as whole
    // current groups; winner residents retain their slots and raw KV pointers.
    if(inserted && !p.payload) {p.payload.reset(new char[p.slots*p.entry]);++p.allocations;}
    // Existing hot hits stay valid throughout read/scatter; only now remove
    // all resident members of the early-rejected group, including unselected KV.
    for(int slot:work.hot_victims) {p.remove_unordered(slot);++p.evictions;}
    for(const auto& g:work.groups) if(!g.keep) {
        for(int i=g.first;i>=0;i=work.members[i].next) {
            int slot=p.index[work.members[i].item.token];
            if(slot>=0) {p.remove_unordered(slot);++p.evictions;}
        }
    }
    for(const auto& g:work.groups) if(g.keep) {
        for(int i=g.first;i>=0;i=work.members[i].next) {
            const auto& item=work.members[i].item;
            if(p.index[item.token]>=0) continue;
            int slot=p.free.back();p.free.pop_back();
            p.positions[slot]=p.heap.size();p.heap.push_back(slot);
            p.tokens[slot]=item.token;p.index[item.token]=slot;
            char* target=p.payload.get()+size_t(slot)*p.entry;
            std::memcpy(target,item.key,p.entry/2);
            std::memcpy(target+p.entry/2,item.value,p.entry/2);
            ++p.insertions;
        }
    }
    const double decision=std::chrono::duration<double>(planned-started).count();
    const double membership=std::chrono::duration<double>(grouped-started).count();
    return {inserted,decision,std::chrono::duration<double>(Clock::now()-planned).count(),
            membership,score_sort_seconds,std::max(0.,decision-membership-score_sort_seconds),
            locations_reused};
}
