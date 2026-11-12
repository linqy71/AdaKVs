#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

// Foreground and publication are serialized by KVStore's existing I/O lock.
// Scores use a common EMA scale, so unobserved tokens need no decay updates.
class HybridCache {
public:
    using Scores = std::shared_ptr<const std::vector<double>>;
    struct Admission { int token; const void* key; const void* value; };
    struct Location { int region; uint64_t offset; };
    struct GroupRange { uint64_t offset, length; };
    using FileRanges = std::array<std::vector<GroupRange>, 2>;
    struct GroupResult {
        size_t inserted;
        double decision_seconds, apply_seconds;
        double membership_seconds, score_sort_seconds, selection_seconds;
        size_t locations_reused;
    };
private:
    struct GroupMember { Admission item; Location location; int next=-1, group=-1; };
    struct Group {
        int first;
        size_t count, residents;
        int region;
        uint64_t offset, length;
        double score;
        bool keep=false;
    };
    struct GroupWorkspace {
        std::vector<GroupMember> members;
        std::vector<Group> groups;
        std::vector<size_t> group_order;
        std::vector<int> warm_pages, cold_blocks;
        std::vector<int> hot_victims;
    } group_workspace;
    struct Partition {
        std::vector<int> index, tokens, positions, heap, free;
        std::vector<double> values;
        std::vector<uint8_t> hot_tokens;
        uint64_t group_candidates=0, group_members=0, group_selected=0;
        uint64_t group_admitted=0, group_rejected=0, group_oversized=0;
        uint64_t group_evicted=0, group_retained=0, resident_groups=0, group_max_tokens=0;
        uint64_t region_oversized[3]{}, region_rejected[3]{}, region_used[3]{};
        uint64_t fast_groups=0, fast_members=0, materialized_members=0;
        uint64_t hot_early_groups=0, hot_skipped_misses=0;
        uint64_t all_fit_heads=0, all_fit_groups=0, all_fit_members=0;
        uint64_t ranked_heads=0, scored_groups=0;
        std::vector<Admission> candidates;
        std::unique_ptr<char[]> payload;
        Scores scores;
        size_t entry=0, slots=0;
        int hot=0;
        uint64_t hits=0, misses=0, evictions=0, insertions=0, rejected=0;
        uint64_t hot_removed=0, publications=0, allocations=0, rebuilds=0, updates=0, batches=0;
        bool less(int a, int b) const {
            return values[a] != values[b] ? values[a] < values[b] : tokens[a] < tokens[b];
        }
        void exchange(int a, int b) {
            std::swap(heap[a],heap[b]); positions[heap[a]]=a; positions[heap[b]]=b;
        }
        void up(int i) { while(i && less(heap[i],heap[(i-1)/2])) { int p=(i-1)/2; exchange(i,p); i=p; } }
        void down(int i) {
            for (;;) {
                int child=2*i+1;
                if(child>=int(heap.size())) break;
                if(child+1<int(heap.size()) && less(heap[child+1],heap[child])) ++child;
                if(!less(heap[child],heap[i])) break;
                exchange(i,child); i=child;
            }
        }
        void fix(int slot) { int i=positions[slot]; up(i); down(positions[slot]); }
        void rebuild() {
            for(size_t i=0;i<heap.size();++i) positions[heap[i]]=i;
            for(int i=int(heap.size())/2-1;i>=0;--i) down(i);
            ++rebuilds;
        }
        void remove(int slot) {
            int i=positions[slot], last=heap.back(); heap.pop_back();
            if(i<int(heap.size())) { heap[i]=last; positions[last]=i; fix(last); }
            index[tokens[slot]]=-1; tokens[slot]=-1; positions[slot]=-1; free.push_back(slot);
        }
        // Access-control uses this vector as a dense resident list, not a
        // token priority heap. Group decisions need no per-token heap repairs.
        void remove_unordered(int slot) {
            int i=positions[slot], last=heap.back();heap.pop_back();
            if(i<int(heap.size())) {heap[i]=last;positions[last]=i;}
            index[tokens[slot]]=-1;tokens[slot]=-1;positions[slot]=-1;free.push_back(slot);
        }
    };
    std::vector<Partition> partitions;
    uint64_t capacity=0;
    int heads=0, prefix=-1;
    bool access_control=false, bypass_hot=true;
    double access_alpha=1., access_l0=56000.;
public:
    static uint64_t warm(int token) { return uint64_t(token); }
    void configure(uint64_t bytes, bool access=false, double alpha=1.,
                   double l0=56000., bool hot_bypass=true) {
        if(!std::isfinite(alpha) || alpha<=0 || alpha>2 || !std::isfinite(l0) || l0<0)
            throw std::invalid_argument("Invalid access-control score parameters");
        capacity=bytes;access_control=access;access_alpha=alpha;
        access_l0=l0;bypass_hot=hot_bypass;clear();
    }
    bool hot_bypass() const { return !access_control || bypass_hot; }
    void clear() { partitions.clear();group_workspace=GroupWorkspace{};heads=0;prefix=-1; }
    void begin(int p,int layers,int h,int=-1,uint64_t=8192) {
        if(prefix!=p || heads!=h || partitions.size()!=size_t(layers*h)) {
            clear();prefix=p;heads=h;partitions.resize(layers*h);
        }
    }
    uint64_t quota() const { return partitions.empty()?0:capacity/partitions.size(); }
    bool published(int l,int h) const { return bool(partitions.at(l*heads+h).scores); }
    void publish(int l,int h,const std::vector<int>& order,int hot,size_t entry,
                 Scores scores,const std::vector<int>* changed=nullptr,bool rebase=false) {
        if(partitions.empty()) return;
        if(!scores || !entry) throw std::logic_error("Missing EMA scores/entry geometry");
        auto& p=partitions.at(l*heads+h);
        if(!p.scores) {
            p.entry=entry;p.slots=std::min<uint64_t>(quota()/entry,scores->size());
            p.index.assign(scores->size(),-1);p.tokens.assign(p.slots,-1);
            p.positions.assign(p.slots,-1);p.values.resize(p.slots);
            p.heap.reserve(p.slots);p.free.reserve(p.slots);
            for(int i=int(p.slots)-1;i>=0;--i) p.free.push_back(i);
        }
        p.scores=std::move(scores);
        if(hot_bypass())
            for(int i=p.hot;i<hot;++i) {
                int slot=p.index.at(order[i]);
                if(slot>=0) {
                    if(access_control) p.remove_unordered(slot);else p.remove(slot);
                    ++p.hot_removed;
                }
            }
        if(access_control) {
            // Keep logical payloads. Current physical groups and mean EMA
            // priorities are rebuilt once at admission, using committed metadata.
            if(p.hot_tokens.empty()) p.hot_tokens.assign(p.index.size(),0);
            for(int i=p.hot;i<hot;++i) p.hot_tokens[order[i]]=1;
            p.hot=hot;
            ++p.publications;
            return;
        }
        p.hot=hot;
        if(!changed || rebase) {
            for(int slot:p.heap) p.values[slot]=p.scores->at(p.tokens[slot]);
            if(!p.heap.empty()) p.rebuild();
        } else {
            // Sparse updates repair only affected resident positions. For dense
            // changes heapify once, rather than paying log(N) for every resident.
            size_t count=0;
            for(int t:*changed) count+=p.index[t]>=0;
            bool dense=count>p.heap.size()/2;
            for(int t:*changed) {
                int slot=p.index[t];if(slot<0) continue;
                p.values[slot]=p.scores->at(t);++p.updates;
                if(!dense) p.fix(slot);
            }
            if(dense) p.rebuild();
        }
        ++p.publications;
    }
    const char* get(int l,int h,uint64_t token) {
        auto& p=partitions.at(l*heads+h);
        int slot=token<p.index.size()?p.index[token]:-1;
        if(slot<0) {++p.misses;return nullptr;}
        ++p.hits;return p.payload.get()+size_t(slot)*p.entry;
    }
    template<class Locate>
    GroupResult put_groups(int l,int h,const std::vector<Admission>& selected,
                          const std::vector<GroupRange>& warm_ranges,
                          uint64_t hot_start,uint64_t hot_end,uint64_t cold_block,
                          Locate locate,size_t rejected_hot_misses=0,
                          const std::vector<Location>* missing_locations=nullptr,
                          const FileRanges* coalesced_ranges=nullptr);
    // The caller counts UNIQUE nonresident hot tokens during lookup. All
    // resident hot members participate, including ones not selected this round.
    bool hot_group_oversized(int l,int h,size_t unique_misses) const {
        const auto& p=partitions.at(l*heads+h);
        if(unique_misses>p.slots) return true;
        size_t count=unique_misses;
        for(int slot:p.heap) count+=p.hot_tokens[p.tokens[slot]];
        return count>p.slots;
    }
    uint64_t hot_entries(int layer) const {
        uint64_t count=0;
        if(access_control) for(int h=0;h<heads;++h) {
            const auto& p=partitions.at(layer*heads+h);
            for(int slot:p.heap) count+=p.hot_tokens[p.tokens[slot]];
        }
        return count;
    }
    size_t put_batch(int l,int h,const std::vector<Admission>& input) {
        if(access_control) throw std::logic_error("access_control requires group admission");
        auto& p=partitions.at(l*heads+h);++p.batches;p.candidates.clear();
        for(const auto& item:input) {
            if(!p.scores || item.token<0 || size_t(item.token)>=p.index.size() || !p.slots) {++p.rejected;continue;}
            if(p.index[item.token]!=-1) continue;
            p.index[item.token]=-2; // Deduplicate this batch without another map.
            p.candidates.push_back(item);
        }
        for(const auto& item:p.candidates) p.index[item.token]=-1;
        auto better=[&](const Admission& a,const Admission& b) {
            double x=priority(p,a.token),y=priority(p,b.token);
            return x!=y?x>y:a.token<b.token;
        };
        if(p.candidates.size()>p.slots) {
            std::nth_element(p.candidates.begin(),p.candidates.begin()+p.slots,p.candidates.end(),better);
            p.rejected+=p.candidates.size()-p.slots;p.candidates.resize(p.slots);
        }
        std::sort(p.candidates.begin(),p.candidates.end(),better);
        if(!p.candidates.empty() && !p.payload) {
            p.payload.reset(new char[p.slots*p.entry]);++p.allocations;
        }
        bool dirty=false;size_t admitted=0;
        for(size_t i=0;i<p.candidates.size();++i) {
            const auto& item=p.candidates[i];double score=priority(p,item.token);int slot;
            if(!p.free.empty()) {
                slot=p.free.back();p.free.pop_back();p.heap.push_back(slot);dirty=true;
            } else {
                if(dirty) {p.rebuild();dirty=false;}
                slot=p.heap.front();
                if(score<=p.values[slot]) {p.rejected+=p.candidates.size()-i;break;}
                p.index[p.tokens[slot]]=-1;++p.evictions;
            }
            p.tokens[slot]=item.token;p.index[item.token]=slot;p.values[slot]=score;
            char* destination=p.payload.get()+size_t(slot)*p.entry;
            std::memcpy(destination,item.key,p.entry/2);
            std::memcpy(destination+p.entry/2,item.value,p.entry/2);
            if(!dirty) p.down(0);
            ++p.insertions;++admitted;
        }
        if(dirty) p.rebuild();
        return admitted;
    }
    bool put(int l,int h,uint64_t token,const void* a,size_t n,const void* b=nullptr,size_t m=0) {
        auto& p=partitions.at(l*heads+h);
        if(n+m!=p.entry || !n) {++p.rejected;return false;}
        const char* value=b?static_cast<const char*>(b):static_cast<const char*>(a)+n/2;
        return put_batch(l,h,{{int(token),a,value}})!=0;
    }
    std::vector<std::map<std::string,uint64_t>> partition_stats() const {
        std::vector<std::map<std::string,uint64_t>> out;
        for(size_t i=0;i<partitions.size();++i) {
            const auto& p=partitions[i];
            out.push_back({{"layer_id",i/heads},{"head_id",i%heads},{"capacity_bytes",quota()},
                {"used_bytes",p.heap.size()*p.entry},{"entries",p.heap.size()},{"hits",p.hits},{"misses",p.misses},
                {"evictions",p.evictions},{"insertions",p.insertions},{"ema_rejections",p.rejected},
                {"hot_promotion_removals",p.hot_removed},{"rank_publications",p.publications},
                {"ema_heap_entries",access_control?0:p.heap.size()},{"ema_resident_updates",p.updates},{"ema_heap_rebuilds",p.rebuilds},
                {"payload_allocations",p.allocations},{"payload_capacity_bytes",p.payload?p.slots*p.entry:0},
                {"admission_batches",p.batches}});
            if(access_control) {
                auto& row=out.back();
                row["access_group_candidates"]=p.group_candidates;
                row["access_group_members"]=p.group_members;
                row["access_group_selected"]=p.group_selected;
                row["admitted_groups"]=p.group_admitted;
                row["rejected_groups"]=p.group_rejected;
                row["oversized_rejected_groups"]=p.group_oversized;
                row["access_group_evictions"]=p.group_evicted;
                row["access_group_retained"]=p.group_retained;
                row["access_resident_groups"]=p.resident_groups;
                row["access_group_max_tokens"]=p.group_max_tokens;
                row["access_oversized_fast_groups"]=p.fast_groups;
                row["access_oversized_fast_members"]=p.fast_members;
                row["access_group_materialized_members"]=p.materialized_members;
                row["access_hot_early_rejected_groups"]=p.hot_early_groups;
                row["access_hot_admission_skipped_tokens"]=p.hot_skipped_misses;
                row["access_group_all_fit_heads"]=p.all_fit_heads;
                row["access_group_all_fit_groups"]=p.all_fit_groups;
                row["access_group_all_fit_members"]=p.all_fit_members;
                row["access_group_ranked_heads"]=p.ranked_heads;
                row["access_group_scored_groups"]=p.scored_groups;
                const char* regions[]={"hot","warm","cold"};
                for(int r=0;r<3;++r) {
                    const std::string name=std::string("access_")+regions[r];
                    row[name+"_oversized_groups"]=p.region_oversized[r];
                    row[name+"_rejected_groups"]=p.region_rejected[r];
                    row[name+"_used_bytes"]=p.region_used[r];
                }
            }
        }
        return out;
    }
    std::map<std::string,uint64_t> stats(bool reset=false) {
        std::map<std::string,uint64_t> out{{"capacity_bytes",capacity},{"hybrid_layout",1},{"hybrid_layout_version",4},
            {"partition_capacity_bytes",quota()},{"equal_layer_head_quotas",1},{"logical_token_cache",1}};
        out["access_control"]=access_control;
        out["access_control_version"]=access_control?4:0;
        out["access_group_decisions"]=access_control;
        out["hot_bypass"]=hot_bypass();
        for(const auto& p:partition_stats()) for(const auto& f:p)
            if(f.first=="access_group_max_tokens") out[f.first]=std::max(out[f.first],f.second);
            else if(f.first!="layer_id" && f.first!="head_id" && f.first!="capacity_bytes") out[f.first]+=f.second;
        if(reset) for(auto& p:partitions) {
            p.hits=p.misses=p.evictions=p.insertions=p.rejected=p.hot_removed=p.publications=0;
            p.allocations=p.rebuilds=p.updates=p.batches=0;
            p.group_candidates=p.group_members=p.group_selected=p.group_admitted=p.group_rejected=0;
            p.group_oversized=p.group_evicted=p.group_retained=p.group_max_tokens=0;
            p.fast_groups=p.fast_members=p.materialized_members=p.hot_early_groups=p.hot_skipped_misses=0;
            p.all_fit_heads=p.all_fit_groups=p.all_fit_members=p.ranked_heads=p.scored_groups=0;
            for(int r=0;r<3;++r) p.region_oversized[r]=p.region_rejected[r]=0;
        }
        return out;
    }
private:
    double group_score(double heat,uint64_t length) const {
        return (access_alpha==1. ? heat : std::pow(heat,access_alpha)) *
            (1.+access_l0/length);
    }
    double priority(const Partition& p,int token) const {
        return (*p.scores)[token];
    }
};

#include "access_control_cache_impl.h"
