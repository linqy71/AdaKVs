#include "warm_cache_read_plan.h"
#include "hot_cache_read_plan.h"

// Included by kvstore.cc; caller holds kv_io_mutex against layout publication.
void KVStore::collect_hybrid_cached(int prefix, int layer, const int* ids, const int* counts) {
    const auto total_start = ReadClock::now();
    const bool access = true;
    const bool layout = true;
    const bool cache_hot = true;
    const size_t entry = 2 * head_dim * sizeof(DTYPE), half = entry/2;
    const uint64_t block = hot_growth_cold_merge_bytes;
    TORCH_CHECK(block >= 4096 && block % 4096 == 0 && block % entry == 0,
                "Hybrid fixed cold block must be a multiple of 4096 and KV entry bytes");
    const auto& states = hot_growth_heads.at(layer);
    {
        auto published = ReadClock::now();
        hybrid_cache.begin(prefix, num_layers, num_key_value_heads, layer, block);
        for (int h=0; h<num_key_value_heads; ++h)
            if (!hybrid_cache.published(layer,h))
                hybrid_cache.publish(layer,h,states[h].order,states[h].hot,entry,states[h].cache_scores);
        read_resources->add("hybrid_cache_rank_publish_wall_seconds", read_seconds(published));
    }
    using E = std::tuple<uint64_t,uint64_t,int>;
    struct Read { MergeRead io; std::vector<int> destinations; int region; };
    std::vector<Read> reads;
    std::vector<std::pair<int,int>> warm_misses, cold_misses;
    std::vector<E> cold_selected;
    struct Hit { int destination; const char* payload; };
    std::vector<Hit> pending_hits;
    std::vector<std::vector<HybridCache::Admission>> admissions(num_key_value_heads);
    std::vector<std::vector<HybridCache::GroupRange>> access_ranges(access ? num_key_value_heads : 0);
    std::vector<std::vector<HybridCache::Location>> admission_locations(access ? num_key_value_heads : 0);
    struct LocatedMiss { int head, destination; HybridCache::Location location; };
    std::vector<LocatedMiss> access_misses, access_cold_misses;
    std::vector<size_t> rejected_hot_misses(cache_hot ? num_key_value_heads : 0,0);
    std::vector<uint8_t> hot_missing_seen(cache_hot ? offload_len : 0,0);
    std::map<std::string,double> metrics;
    HotCacheReadPlan hot_plan;
    uint64_t position_lookups=0, position_checks=0;
    // The caller holds kv_io_mutex: position/hot/order reflect the same commit
    // as kv_meta. Cold backing is immutable and indexed by logical token ID.
    auto locate_access = [&](int h,int token) {
        const auto& state=states[h];
        const int position=state.position.at(token);
        ++position_lookups;
        HybridCache::Location location{
            position<0 ? 2 : position<state.hot ? 0 : 1,
            position<0 ? (uint64_t(h)*offload_len+token)*entry :
                         (uint64_t(h)*state.storage_capacity()+position)*entry};
        if(cache_diagnostics) {
            const auto& original=kv_meta->at(get_meta_id(token,layer,h));
            TORCH_CHECK(original.file_index==uint64_t(location.region==2) &&
                        original.offset==location.offset,"Committed layout position mismatch");
            ++position_checks;
        }
        return location;
    };
    const char* names[] = {"hot", "warm", "cold"};
    const double request_cost = hybrid_cache_read_us * 1e-6, bandwidth = hybrid_cache_read_bps;
    uint64_t selected_count[3]{}, hit_count[3]{}, miss_count[3]{};
    auto copy = [&](int destination, const char* k, const char* v) {
        auto pos = query_output_offset(destination);
        std::memcpy(queried_key+pos,k,half); std::memcpy(queried_value+pos,v,half);
    };
    auto plan_warm = [&](std::vector<E> selected) {
        std::vector<PhysicalMergeRun> output;
        if (selected.empty()) return std::make_pair(selected,output);
        std::sort(selected.begin(), selected.end());
        auto runs = collect_merge_runs(selected,entry);
        uint64_t threshold = merge_threshold_bytes;
        auto budget = region_merge_budget(0);
        if (budget) {
            std::vector<MergeCandidate> options;
            for (uint64_t t : {0,4096,8192,16384,32768,65536})
                options.push_back(summarize_unique_page_reads(runs,t));
            threshold = select_budgeted_merge_threshold(options,*budget);
        }
        plan_unique_page_reads_into(runs,threshold,output);
        return std::make_pair(selected,output);
    };
    auto bytes = [](const std::vector<PhysicalMergeRun>& r) { return merge_read_bytes(r); };
    auto append = [&](const std::vector<E>& selected, const std::vector<PhysicalMergeRun>& runs, int region) {
        for (const auto& r:runs) {
            Read read{{r.file,r.offset,r.length,{}},{},region};
            for (size_t i=r.begin;i<r.end;++i) {
                std::get<3>(read.io).push_back((std::get<1>(selected[i])-r.offset)/entry);
                read.destinations.push_back(std::get<2>(selected[i]));
            }
            reads.push_back(std::move(read));
        }
    };
    const bool overlap = layout && !cache_hot && hybrid_cache_hot_overlap && kv_read_threads(num_key_value_heads)>1;
    const int reader_threads=kv_read_threads(num_key_value_heads);
    std::vector<MergeRead> hot_reads;
    std::vector<size_t> hot_offsets(num_key_value_heads);
    std::vector<std::vector<E>> hot_scatter(num_key_value_heads);
    auto& hot_buffer=read_resources->hot_buffer;
    auto hot_started = ReadClock::now(), hot_finished = hot_started;
    double hot_seconds = 0;
    std::future<void> hot_future;
    struct JoinHot {
        std::future<void>& future;
        ~JoinHot() { if (future.valid()) future.wait(); }
    } join_hot{hot_future};
    const auto early_start = ReadClock::now();
    size_t hot_bytes=0;
    ;
    metrics["hot_prepare_wall_seconds"]=read_seconds(early_start);
    const bool hot_active=!hot_reads.empty();
    metrics["hot_overlap_collections"]=hot_active;
    if(hot_active) {
        hot_started=ReadClock::now();
        hot_future=read_resources->hot_reader.submit([&] {
            const auto begin=ReadClock::now();
            std::exception_ptr failure;
            std::mutex failure_mutex;
            #pragma omp parallel num_threads(reader_threads)
            {
                try {
                    const int worker=omp_get_thread_num(), workers=omp_get_num_threads();
                    const size_t first=hot_reads.size()*worker/workers;
                    const size_t last=hot_reads.size()*(worker+1)/workers;
                    std::vector<MergeRead> local(hot_reads.begin()+first,hot_reads.begin()+last);
                    std::vector<int> no_scatter;
                    CacheReadCopies captures;
                    captures.resize(local.size()); captures.read_only=true;
                    size_t offset=0;
                    for(size_t i=0;i<last;++i) {
                        auto start=std::get<1>(hot_reads[i]), length=std::get<2>(hot_reads[i]);
                        if(i>=first) captures.read_buffers.push_back(
                            static_cast<char*>(hot_buffer.data.get())+offset);
                        offset+=merge_aligned_end(start+length)-start/4096*4096;
                    }
                    if(!local.empty()) load_key_value_from_file_iouring(local,no_scatter,prefix,layer,0,&captures);
                } catch(...) {
                    std::lock_guard<std::mutex> lock(failure_mutex);
                    if(!failure) failure=std::current_exception();
                }
            }
            if(failure) std::rethrow_exception(failure);
            hot_finished=ReadClock::now();
            hot_seconds=read_seconds(begin);
        });
        metrics["hot_dispatch_wall_seconds"]=read_seconds(hot_started);
    }
    const auto lookup_start = ReadClock::now();
    // Lookup and planning are head-local; no permanent physical warm groups.
    double plan_seconds = 0;
    for (int h=0; h<num_key_value_heads; ++h) {
        const uint64_t hot_start = uint64_t(h)*states[h].storage_capacity()*entry;
        const uint64_t hot_end = hot_start+states[h].hot*entry;
        std::vector<E> hot, warm, missing_warm;
        std::vector<Hit> warm_hits;
        std::vector<const char*> warm_payload(layout ? counts[h] : 0, nullptr);
        size_t unique_hot_misses=0, hot_hits=0;
        const auto token_lookup_start=ReadClock::now();
        {
            TORCH_CHECK(4096%entry==0,"access_control requires KV entries dividing 4 KiB");
            std::fill(hot_missing_seen.begin(),hot_missing_seen.end(),0);
            hot_plan.reset(hot_start,hot_end);
            hot.reserve(counts[h]);
        }
        for (int j=0;j<counts[h];++j) {
            int destination=h*max_length+j, token=ids[destination];
            uint64_t file,offset;
            int region;
            {
                const auto loc=locate_access(h,token);
                region=loc.region;offset=loc.offset;file=region==2;
            }
            ++selected_count[region];
            E e{file,offset,destination};
            ;
            bool hit=false;
            {
                const char* p=hybrid_cache.get(layer,h,HybridCache::warm(token));
                if(p) {
                    hit=true;
                    if(region==0) {
                        ++hot_hits;
                        warm_payload[j]=p;
                    } else if(region==1) {
                        warm_hits.push_back({destination,p});
                        warm_payload[j]=p;
                    }
                    else pending_hits.push_back({destination,p});
                }
            }
            if(hit) ++hit_count[region]; else ++miss_count[region];
            if(region==0) {
                hot.push_back(e);
                if(!hit && !hot_missing_seen[token]) {
                    hot_missing_seen[token]=1;++unique_hot_misses;
                    hot_plan.add_miss(offset);
                }
            } else if(region==1) {
                warm.push_back(e);
                if(!hit) {
                    {
                        access_misses.push_back({h,destination,{region,offset}});
                        if(4096%entry!=0) missing_warm.push_back(e);
                    }
                }
            } else {
                if(!hit) {
                    cold_selected.push_back(e);
                    access_cold_misses.push_back({h,destination,{region,offset}});
                }
            }
        }
        metrics["access_token_lookup_wall_seconds"]+=read_seconds(token_lookup_start);
        if(!hot.empty() && cache_hot) {
            const auto hot_plan_start=ReadClock::now();
            const auto prefilter_start=ReadClock::now();
            if(hybrid_cache.hot_group_oversized(layer,h,unique_hot_misses)) {
                rejected_hot_misses[h]=unique_hot_misses;
            } else {
                // Keep hits/misses unchanged for read planning. Only eligible
                // hot misses need per-token admission pointers after I/O.
                for(const auto& e:hot) {
                    int destination=std::get<2>(e);
                    if(!warm_payload[destination-h*max_length])
                        access_misses.push_back({h,destination,{0,std::get<1>(e)}});
                }
            }
            metrics["access_group_prefilter_wall_seconds"]+=read_seconds(prefilter_start);
            const auto length=merge_aligned_end(hot_end)-hot_start/4096*4096;
            ++metrics["hot_without_cache_reads"];
            metrics["hot_without_cache_bytes"]+=length;
            const auto page_plan_start=ReadClock::now();
            hot_plan.finish(hybrid_cache_miss_read_mode!="coalesced",
                            hybrid_cache_miss_read_mode=="paged",request_cost,bandwidth);
            metrics["access_hot_page_plan_wall_seconds"]+=read_seconds(page_plan_start);
            ++metrics["access_hot_linear_heads"];
            metrics["access_hot_no_hit_heads"]+=hot_hits==0;
            metrics["access_hot_paged_heads"]+=hot_plan.paged;
            metrics["access_hot_miss_pages"]+=hot_plan.miss_pages();
            const auto submitted=bytes(hot_plan.reads);
            metrics["hot_range_saved_bytes"]+=length-submitted;
            if(!hot_hits && !hot_plan.paged && submitted==length) {
                ++metrics["access_hot_full_read_heads"];
                metrics["access_hot_full_read_tokens"]+=hot.size();
            }
            metrics["access_hot_refined_heads"]+=hot_hits!=0 || hybrid_cache_miss_read_mode=="paged";
            metrics["access_hot_trimmed_no_hit_heads"]+=!hot_hits && submitted<length;
            const auto scatter_plan_start=ReadClock::now();
            const size_t read_base=reads.size();
            for(const auto& run:hot_plan.reads) {
                reads.push_back({{run.file,run.offset,run.length,{}},{},0});
                if(hot_plan.reads.size()==1) {
                    std::get<3>(reads.back().io).reserve(hot.size());
                    reads.back().destinations.reserve(hot.size());
                }
            }
            // Offset buckets replace sorted slices. Each selected destination,
            // including duplicates, is written exactly once by read or cache.
            uint64_t reread_hits=0;
            for(const auto& selected:hot) {
                const auto offset=std::get<1>(selected);
                const int destination=std::get<2>(selected);
                const auto payload=warm_payload[destination-h*max_length];
                const int index=hot_plan.read_index(offset);
                if(index>=0) {
                    auto& read=reads[read_base+index];
                    std::get<3>(read.io).push_back((offset-std::get<1>(read.io))/entry);
                    read.destinations.push_back(destination);
                    reread_hits+=payload!=nullptr;
                } else if(payload) pending_hits.push_back({destination,payload});
            }
            metrics["hot_reread_hit_tokens"]+=reread_hits;
            metrics["access_hot_scatter_plan_wall_seconds"]+=read_seconds(scatter_plan_start);
            const auto hot_plan_seconds=read_seconds(hot_plan_start);
            metrics["access_hot_plan_wall_seconds"]+=hot_plan_seconds;
            plan_seconds+=hot_plan_seconds;
        } else if(!hot.empty()) append(hot,{{0,hot_start,hot_end-hot_start,0,hot.size()}},0);
        auto start=ReadClock::now();
        auto full=plan_warm(warm);
        {
            const auto score_start=ReadClock::now();
            access_ranges[h].reserve(full.second.size());
            for(const auto& run:full.second) {
                const auto offset=run.offset/4096*4096;
                access_ranges[h].push_back({offset,merge_aligned_end(run.offset+run.length)-offset});
            }
            metrics["access_group_range_wall_seconds"]+=read_seconds(score_start);
        }
        metrics["warm_without_cache_reads"]+=full.second.size();
        metrics["warm_without_cache_bytes"]+=bytes(full.second);
        if(layout && 4096%entry==0) {
            const auto range_start=ReadClock::now();
            auto is_hit=[&](int destination) {
                return warm_payload[destination-h*max_length]!=nullptr;
            };
            auto plan=plan_warm_cache_ranges(full.first,full.second,is_hit,
                hybrid_cache_miss_read_mode!="coalesced",
                hybrid_cache_miss_read_mode=="paged",request_cost,bandwidth);
            metrics["warm_range_plan_wall_seconds"]+=read_seconds(range_start);
            metrics["warm_full_ranges"]+=full.second.size();
            metrics["warm_dropped_hit_ranges"]+=plan.dropped_ranges;
            metrics["warm_coalesced_ranges"]+=plan.coalesced_ranges;
            metrics["warm_paged_ranges"]+=plan.paged_ranges;
            metrics["warm_paged_candidate_reads"]+=plan.page_reads;
            metrics["warm_paged_candidate_bytes"]+=plan.page_reads*4096;
            metrics["warm_trimmed_candidate_bytes"]+=plan.trimmed_bytes;
            metrics["warm_range_saved_bytes"]+=bytes(full.second)-bytes(plan.reads);
            metrics["warm_paged_heads"]+=plan.paged_ranges!=0;
            metrics["warm_coalesced_heads"]+=plan.coalesced_ranges!=0;
            uint64_t reread_hits=0;
            // Slices reference the full sorted selection. Null only hits that
            // the actual submitted ranges will scatter, not the entire head.
            for(const auto& run:plan.reads) {
                for(size_t i=run.begin;i<run.end;++i) {
                    auto& payload=warm_payload[std::get<2>(full.first[i])-h*max_length];
                    if(payload) { ++reread_hits; payload=nullptr; }
                }
            }
            append(full.first,plan.reads,1);
            for(const auto& hit:warm_hits)
                if(warm_payload[hit.destination-h*max_length]) pending_hits.push_back(hit);
            metrics["warm_reread_heads"]+=reread_hits!=0;
            metrics["warm_reread_hit_tokens"]+=reread_hits;
            metrics["warm_hit_copy_avoided_tokens"]+=reread_hits;
        } else {
            decltype(full) miss;
            const auto* missing=&full;
            if(missing_warm.size()!=warm.size()) {miss=plan_warm(missing_warm);missing=&miss;}
            auto cost=[&](const std::vector<PhysicalMergeRun>& r) {
                return r.size()*request_cost+bytes(r)/bandwidth;
            };
            bool reread=!missing->second.empty() && cost(full.second)<cost(missing->second);
            const auto* chosen=reread?&full:missing;
            std::vector<PhysicalMergeRun> pages;
            bool paged=false;
            if(layout && hybrid_cache_miss_read_mode!="coalesced" && !missing->first.empty() && 4096%entry==0) {
                pages=plan_fixed_block_reads(missing->first,entry,4096);
                metrics["warm_paged_candidate_reads"]+=pages.size();
                metrics["warm_paged_candidate_bytes"]+=bytes(pages);
                paged=hybrid_cache_miss_read_mode=="paged" || cost(pages)<cost(chosen->second);
            }
            if(paged) {reread=false;append(missing->first,pages,1);++metrics["warm_paged_heads"];}
            else {append(chosen->first,chosen->second,1);++metrics[reread?"warm_reread_heads":"warm_coalesced_heads"];}
            if(reread) {
                metrics["warm_reread_hit_tokens"]+=warm.size()-missing_warm.size();
                metrics["warm_hit_copy_avoided_tokens"]+=warm_hits.size();
            } else pending_hits.insert(pending_hits.end(),warm_hits.begin(),warm_hits.end());
        }
        plan_seconds+=read_seconds(start);
    }
    metrics["lookup_wall_seconds"] = std::max(0.0,read_seconds(lookup_start)-plan_seconds);
    auto planning=ReadClock::now();
    std::sort(cold_selected.begin(),cold_selected.end());
    auto cold_fixed=plan_fixed_block_reads(cold_selected,entry,block);
    std::vector<PhysicalMergeRun> cold_pages;
    bool cold_paged=false;
    if(layout && hybrid_cache_miss_read_mode!="coalesced" && !cold_selected.empty() && 4096%entry==0) {
        cold_pages=plan_fixed_block_reads(cold_selected,entry,4096);
        metrics["cold_paged_candidate_reads"]+=cold_pages.size();
        metrics["cold_paged_candidate_bytes"]+=bytes(cold_pages);
        cold_paged=hybrid_cache_miss_read_mode=="paged" ||
            cold_pages.size()*request_cost+bytes(cold_pages)/bandwidth <
            cold_fixed.size()*request_cost+bytes(cold_fixed)/bandwidth;
    }
    metrics["cold_fixed_candidate_reads"]+=cold_fixed.size();
    metrics["cold_fixed_candidate_bytes"]+=bytes(cold_fixed);
    append(cold_selected,cold_paged?cold_pages:cold_fixed,2);
    ++metrics[cold_paged?"cold_paged_collections":"cold_fixed_collections"];
    metrics["plan_wall_seconds"]=plan_seconds+read_seconds(planning);
    auto hit_copy=ReadClock::now();
    for(const auto& hit:pending_hits) copy(hit.destination,hit.payload,hit.payload+half);
    metrics["hit_copy_wall_seconds"]=read_seconds(hit_copy);
    metrics["hit_copy_tokens"]=pending_hits.size();
    metrics["hit_copy_bytes"]=pending_hits.size()*entry;
    // Optional counterfactual cold plan: avoid its set construction in latency runs.
    auto diagnostics_start=ReadClock::now();
    if(cache_diagnostics) {
        std::set<uint64_t> all;
        for(int h=0;h<num_key_value_heads;++h) for(int j=0;j<counts[h];++j) {
            auto& loc=kv_meta->at(get_meta_id(ids[h*max_length+j],layer,h));
            if(loc.file_index==1) all.insert(loc.offset/block);
        }
        metrics["cold_without_cache_reads"]=all.size();
        metrics["cold_without_cache_bytes"]=all.size()*block;
    }
    metrics["diagnostics_wall_seconds"]=read_seconds(diagnostics_start);
    const auto buffer_start=ReadClock::now();
    auto& workspace=read_resources->range;
    size_t required=0;
    std::vector<size_t> offsets;
    for(auto& r:reads) {
        auto& [file,offset,length,positions]=r.io;
        const uint64_t end = uint64_t(num_key_value_heads)*(file ? offload_len : states[0].storage_capacity())*entry;
        length=std::min(length,end-offset);
        offsets.push_back(required);
        required+=merge_aligned_end(offset+length)-offset/4096*4096;
        ++metrics[std::string(names[r.region])+"_submitted_reads"];
        metrics[std::string(names[r.region])+"_submitted_bytes"]+=merge_aligned_end(offset+length)-offset/4096*4096;
    }
    if(workspace.capacity<required) { workspace.data=allocate_aligned_buffer(required); workspace.capacity=required; }
    metrics["buffer_wall_seconds"]=read_seconds(buffer_start);
    // Finish hot I/O before starting the full-width miss team. Lookup/planning
    // overlap the hot team; miss I/O and selected hot scatter then share the
    // original worker budget, without two competing reader teams.
    const auto pre_io_finished=ReadClock::now();
    const auto join_start=ReadClock::now();
    if(hot_future.valid()) hot_future.get();
    metrics["hot_join_wait_wall_seconds"]=read_seconds(join_start);
    metrics["hot_read_wall_seconds"]=hot_seconds;
    if(hot_active) {
        metrics["hot_completion_wall_seconds"]=std::chrono::duration<double>(hot_finished-hot_started).count();
        metrics["hot_remaining_at_miss_submit_seconds"]=std::max(0.0,
            std::chrono::duration<double>(hot_finished-pre_io_finished).count());
        metrics["hot_hidden_before_miss_submit_seconds"]=std::max(0.0,
            std::chrono::duration<double>(std::min(hot_finished,pre_io_finished)-hot_started).count());
    }
    const auto scatter_start=ReadClock::now();
    if(hot_active) {
        #pragma omp parallel for num_threads(reader_threads) schedule(static)
        for(int h=0;h<num_key_value_heads;++h) {
            if(hot_scatter[h].empty()) continue;
            const uint64_t start=uint64_t(h)*states[h].storage_capacity()*entry;
            const char* data=static_cast<char*>(hot_buffer.data.get())+hot_offsets[h]+start%4096;
            for(const auto& selected:hot_scatter[h]) {
                const char* source=data+std::get<1>(selected)-start;
                copy(std::get<2>(selected),source,source+half);
            }
        }
    }
    metrics["hot_scatter_wall_seconds"]=read_seconds(scatter_start);
    const auto io_start=ReadClock::now();
    std::exception_ptr error; std::mutex error_mutex;
    const int threads=reader_threads;
    #pragma omp parallel num_threads(threads)
    {
        int worker=omp_get_thread_num(), count=omp_get_num_threads();
        size_t begin=reads.size()*worker/count,end=reads.size()*(worker+1)/count;
        try {
            auto& io=read_resources->worker_buffer(); io.reads.clear(); io.token_order.clear();
            CacheReadCopies captures; captures.resize(end-begin);
            for(size_t i=begin;i<end;++i) {
                io.reads.push_back(reads[i].io);
                io.token_order.insert(io.token_order.end(),reads[i].destinations.begin(),reads[i].destinations.end());
                captures.read_buffers.push_back(static_cast<char*>(workspace.data.get())+offsets[i]);
            }
            if(begin!=end) load_key_value_from_file_iouring(io.reads,io.token_order,prefix,layer,0,&captures);
        } catch(...) { std::lock_guard<std::mutex> lock(error_mutex); if(!error) error=std::current_exception(); }
    }
    metrics["read_scatter_wall_seconds"]=read_seconds(io_start);
    if(error) std::rethrow_exception(error); // Publish cache only after all reads succeed.
    auto admission=ReadClock::now();
    for(const auto* batch:{&access_misses,&access_cold_misses}) for(const auto& miss:*batch) {
        const auto pos=query_output_offset(miss.destination);
        admissions[miss.head].push_back({ids[miss.destination],queried_key+pos,queried_value+pos});
        admission_locations[miss.head].push_back(miss.location);
    }
    for(auto [h,destination]:warm_misses) {
        auto pos=query_output_offset(destination); int token=ids[destination];
        admissions[h].push_back({token,queried_key+pos,queried_value+pos});
    }
    for(auto [h,destination]:cold_misses) {
        auto pos=query_output_offset(destination);
        admissions[h].push_back({ids[destination],queried_key+pos,queried_value+pos});
    }
    for(int h=0;h<num_key_value_heads;++h) {
        {
            const uint64_t start=uint64_t(h)*states[h].storage_capacity()*entry;
            const uint64_t end=start+states[h].hot*entry;
            auto result=hybrid_cache.put_groups(layer,h,admissions[h],access_ranges[h],start,end,block,
                [&](int token) {
                    return locate_access(h,token);
                },cache_hot ? rejected_hot_misses[h] : 0,&admission_locations[h]);
            metrics["access_group_decision_wall_seconds"]+=result.decision_seconds;
            metrics["access_group_apply_wall_seconds"]+=result.apply_seconds;
            metrics["access_group_membership_wall_seconds"]+=result.membership_seconds;
            metrics["access_group_score_sort_wall_seconds"]+=result.score_sort_seconds;
            metrics["access_group_selection_wall_seconds"]+=result.selection_seconds;
            metrics["access_admission_locations_reused"]+=result.locations_reused;
        }
    }
    metrics["admission_wall_seconds"]=read_seconds(admission);
    metrics["collections"]=1;
    {
        metrics["access_control_collections"]=1;
        metrics["access_control_hot_bypass"]=hybrid_cache.hot_bypass();
        metrics["access_position_lookups"]=position_lookups;
        metrics["access_position_checks"]=position_checks;
        // Snapshot after admission; logical quotas include all eligible regions.
        uint64_t hot_entries=0;
        const auto occupancy_start=ReadClock::now();
        hot_entries=hybrid_cache.hot_entries(layer);
        metrics["access_hot_used_bytes"]=hot_entries*entry;
        metrics["access_stats_wall_seconds"]=read_seconds(occupancy_start);
    }
    metrics["collect_wall_seconds"]=read_seconds(total_start);
    for(int r=0;r<3;++r) {
        metrics[std::string(names[r])+"_selected_tokens"]=selected_count[r];
        metrics[std::string(names[r])+"_token_hits"]=hit_count[r];
        metrics[std::string(names[r])+"_token_misses"]=miss_count[r];
    }
    for(auto& m:metrics) read_resources->add("hybrid_cache_"+m.first,m.second);
}
