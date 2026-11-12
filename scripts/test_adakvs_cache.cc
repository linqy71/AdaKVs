// g++ -std=c++17 -O2 -I library/kvstore scripts/test_adakvs_cache.cc -o /tmp/test_adakvs_cache
#include "hybrid_cache.h"
#include "hot_growth_policy.h"
#include <array>
#include <cassert>
#include <numeric>
#include <random>
#include <set>

struct Fixture {
    HybridCache cache;
    std::shared_ptr<std::vector<double>> heat=std::make_shared<std::vector<double>>(64,1.);
    std::array<std::array<char,512>,64> data{};
    std::vector<HybridCache::Location> locations;
    std::vector<HybridCache::GroupRange> ranges;
    uint64_t hot_start=0,hot_end=4096;
    Fixture(int slots,double l0=0.,double alpha=1.) {
        cache.configure(slots*512*2,true,alpha,l0,false);cache.begin(1,1,2);
        for(int i=0;i<64;++i) {
            locations.push_back({1,uint64_t(i+1)*4096});
            data[i].fill(char(i+1));
        }
        cache.publish(0,0,{},0,512,heat);
    }
    HybridCache::GroupResult put(const std::vector<int>& tokens) {
        std::vector<HybridCache::Admission> input;
        for(int t:tokens) input.push_back({t,data[t].data(),data[t].data()+256});
        return cache.put_groups(0,0,input,ranges,hot_start,hot_end,8192,
                                [&](int t){return locations[t];});
    }
    HybridCache::GroupResult put_prefiltered(const std::vector<int>& tokens) {
        std::set<int> hot_misses;
        for(int t:tokens) if(locations[t].region==0 && !cache.get(0,0,t)) hot_misses.insert(t);
        bool reject=cache.hot_group_oversized(0,0,hot_misses.size());
        std::vector<HybridCache::Admission> input;
        for(int t:tokens) if(!reject || locations[t].region!=0)
            input.push_back({t,data[t].data(),data[t].data()+256});
        return cache.put_groups(0,0,input,ranges,hot_start,hot_end,8192,
            [&](int t){return locations[t];},reject?hot_misses.size():0);
    }
    bool hit(int t) {
        auto ptr=cache.get(0,0,t);
        if(ptr) assert(std::memcmp(ptr,data[t].data(),512)==0);
        return ptr;
    }
};

int main() {
    { // Mean heat, not sum or hottest member; an unfittable group is never cut.
        Fixture f(2);
        (*f.heat)[0]=9.;(*f.heat)[1]=1.;(*f.heat)[2]=6.;
        f.locations[1]={1,f.locations[0].offset+512};
        f.put({0,1,2});
        assert(!f.hit(0) && !f.hit(1) && f.hit(2));
        assert(f.cache.stats()["used_bytes"]==512);
        assert(f.cache.stats()["rejected_groups"]==1);
    }
    { // Whole-group eviction can require several victims and leave slack.
        Fixture f(4);
        f.locations[1]={1,f.locations[0].offset+512};
        f.locations[3]={1,f.locations[2].offset+512};
        (*f.heat)[2]=(*f.heat)[3]=2.;
        f.put({0,1,2,3});
        for(int t:{4,5,6}) {f.locations[t]={1,uint64_t(8*4096+(t-4)*512)};(*f.heat)[t]=3.;}
        f.put({4,5,6});
        for(int t=0;t<4;++t) assert(!f.hit(t));
        for(int t=4;t<7;++t) assert(f.hit(t));
        auto stats=f.cache.stats();
        assert(stats["access_group_evictions"]==2 && stats["evictions"]==4);
        assert(stats["payload_allocations"]==1 && stats["ema_heap_rebuilds"]==0);
        assert(stats["used_bytes"]==3*512 && stats["access_resident_groups"]==1);
    }
    { // The H-L length term beats hotter sequential access when it should.
        Fixture f(2,56000.);
        f.locations[0]={0,0};(*f.heat)[0]=1.;f.hot_end=1024*1024;
        f.locations[2]={1,f.locations[1].offset+512};
        (*f.heat)[1]=(*f.heat)[2]=.2;
        f.put({0});f.put({1,2});
        assert(!f.hit(0) && f.hit(1) && f.hit(2));
        auto ptr=f.cache.get(0,0,1);
        f.put({1,2,2});
        assert(f.cache.get(0,0,1)==ptr && f.cache.stats()["insertions"]==3);
    }
    { // Long warm coalesced ranges are one decision, even across many pages.
        Fixture f(2,56000.);
        f.ranges={{4096,64*4096}};
        f.put({0,1,2});
        assert(f.cache.stats()["used_bytes"]==0);
        assert(f.cache.stats()["oversized_rejected_groups"]==1);
    }
    { // A mixed hit/miss group must either retain every member or lose all.
        Fixture f(2);
        for(int t:{0,1,2}) {f.locations[t]={0,uint64_t(t*512)};(*f.heat)[t]=100.;}
        f.cache.publish(0,0,{0,1,2},3,512,f.heat);
        f.put({0,1});assert(f.cache.hot_entries(0)==2);
        f.locations[4]={1,f.locations[3].offset+512};
        f.put({0,1,2,2,3,4});
        for(int t=0;t<3;++t) assert(!f.hit(t));
        assert(f.hit(3) && f.hit(4) && f.cache.hot_entries(0)==0);
        auto stats=f.cache.stats();
        assert(stats["oversized_rejected_groups"]==1 && stats["access_group_evictions"]==1);
        assert(stats["access_group_max_tokens"]==3 && stats["used_bytes"]==2*512);
        assert(f.cache.partition_stats()[1]["used_bytes"]==0); // No cross-head borrowing.
        assert(stats["hot_bypass"]==0 && stats["access_control_version"]==4);
    }
    { // Equal scores retain complete residents, including nonselected ones.
        Fixture f(2);f.put({0,1});f.put({2,3});
        assert(f.hit(0) && f.hit(1) && !f.hit(2) && !f.hit(3));
        auto before=f.cache.stats(true);
        assert(f.cache.stats()["access_group_candidates"]==0);
        assert(f.cache.stats()["used_bytes"]==before["used_bytes"]);
        f.cache.begin(2,1,2);assert(f.cache.stats()["used_bytes"]==0);
    }
    { // Reorder/region conversion must regroup CURRENT physical positions.
        Fixture f(3);
        f.locations[1]={1,f.locations[0].offset+512};
        (*f.heat)[0]=1.;(*f.heat)[1]=3.;(*f.heat)[2]=3.;
        f.put({0,1,2});
        auto ptr=f.cache.get(0,0,1);
        // The old warm group splits; 1 joins 2 in cold, 0 is promoted hot.
        f.locations[0]={0,0};f.locations[1]={2,8192};f.locations[2]={2,8704};
        f.cache.publish(0,0,{0},1,512,f.heat);
        f.ranges.clear();f.put({});
        assert(f.cache.get(0,0,1)==ptr && f.hit(0) && f.hit(2));
        (*f.heat)[3]=4.;f.locations[3]={1,32768};f.put({3});
        assert(!f.hit(0) && f.hit(1) && f.hit(2) && f.hit(3));
        assert(f.cache.get(0,0,1)==ptr);
        // A newly planned coalesced range merges old 1 with 3; together they
        // exceed quota after new selected members join, so both are evicted.
        f.locations[1]={1,32768+512};f.locations[4]={1,32768+1024};
        f.locations[5]={1,32768+1536};f.ranges={{32768,4096}};
        f.put({4,5});
        assert(!f.hit(1) && !f.hit(3) && !f.hit(4) && !f.hit(5) && f.hit(2));
    }
    { // Empty quotas reject groups without allocation or underflow.
        Fixture f(0);f.put({0,0,1});
        assert(f.cache.stats()["used_bytes"]==0);
        assert(f.cache.stats()["oversized_rejected_groups"]==2);
        assert(f.cache.stats()["payload_allocations"]==0);
    }
    { // All fitting groups bypass heat/sort, including after oversized losers
      // are excluded. Equal-score competition still retains old residents.
        Fixture f(2,56000.,1.5);
        f.locations[1]={1,f.locations[0].offset+512};
        f.locations[2]={1,f.locations[0].offset+1024};
        auto result=f.put({0,1,2,3,4,4});
        assert(!f.hit(0) && !f.hit(1) && !f.hit(2) && f.hit(3) && f.hit(4));
        auto stats=f.cache.stats();
        assert(stats["access_group_all_fit_heads"]==1);
        assert(stats["access_group_all_fit_groups"]==2 && stats["access_group_all_fit_members"]==2);
        assert(stats["access_group_ranked_heads"]==0 && stats["access_group_scored_groups"]==0);
        assert(result.score_sort_seconds==0 && result.membership_seconds<=result.decision_seconds);
        result=f.put({5});
        assert(f.hit(3) && f.hit(4) && !f.hit(5));
        stats=f.cache.stats(true);
        assert(stats["access_group_ranked_heads"]==1 && stats["access_group_scored_groups"]==3);
        assert(result.score_sort_seconds>0);
        assert(f.cache.stats()["access_group_all_fit_heads"]==0);
        assert(f.cache.stats()["access_group_scored_groups"]==0);
    }
    { // Newly missing locations can be reused; residents must still resolve
      // their current positions after reorder. Duplicate inputs resolve once.
        Fixture f(4);f.put({0});
        f.locations[0]={2,8192};
        std::vector<HybridCache::Admission> input;
        std::vector<HybridCache::Location> locations;
        for(int t:{0,1,1,2}) {
            input.push_back({t,f.data[t].data(),f.data[t].data()+256});
            locations.push_back(f.locations[t]);
        }
        int resolved=0;
        auto result=f.cache.put_groups(0,0,input,{},0,4096,8192,[&](int t) {
            assert(t==0);++resolved;return f.locations[t];
        },0,&locations);
        assert(resolved==1 && result.locations_reused==2);
        assert(f.hit(0) && f.hit(1) && f.hit(2));
        locations.pop_back();bool rejected=false;
        try {f.cache.put_groups(0,0,input,{},0,4096,8192,
            [&](int t){return f.locations[t];},0,&locations);}
        catch(const std::logic_error&) {rejected=true;}
        assert(rejected && f.cache.stats()["entries"]==3);
    }
    { // Early rejection includes unselected residents and defers eviction
      // until admission. Its decisions/counters match the fully expanded path.
        Fixture full(3), fast(3);
        for(auto f:{&full,&fast}) {
            f->put({0,1,8});
            f->cache.publish(0,0,{0,1,2,3},4,512,f->heat);
            for(int t=0;t<4;++t) f->locations[t]={0,uint64_t(t*512)};
        }
        assert(fast.cache.hot_group_oversized(0,0,2)); // Two cached + two new.
        assert(fast.hit(0) && fast.hit(1)); // Prefilter does not mutate residency.
        full.put({2,3,3,10});fast.put_prefiltered({2,3,3,10});
        for(int t=0;t<64;++t) assert(full.hit(t)==fast.hit(t));
        auto a=full.cache.stats(),b=fast.cache.stats();
        for(const auto& item:a) if(item.first!="hits" && item.first!="misses" &&
            item.first!="access_group_materialized_members" &&
            item.first!="access_hot_early_rejected_groups" &&
            item.first!="access_hot_admission_skipped_tokens") assert(item.second==b[item.first]);
        assert(b["access_hot_admission_skipped_tokens"]==2);
        assert(b["access_oversized_fast_members"]==4 && b["access_group_evictions"]==1);
        assert(b["access_group_materialized_members"]+4==a["access_group_materialized_members"]);
        fast.cache.stats(true);
        assert(fast.cache.stats()["access_oversized_fast_groups"]==0);
    }
    { // Zero-quota hot rejection works without constructing a single member.
        Fixture f(0);f.locations[0]={0,0};f.cache.publish(0,0,{0},1,512,f.heat);
        f.put_prefiltered({0,0});auto s=f.cache.stats();
        assert(s["access_group_members"]==1 && s["access_group_materialized_members"]==0);
        assert(s["oversized_rejected_groups"]==1 && s["payload_allocations"]==0);
    }
    { // Randomized independent reference: relocation, duplicates, variable
      // group sizes, nonselected residents, changing scores and exact payloads.
        Fixture f(9,56000.,1.5);std::mt19937 rng(42);std::set<int> resident;
        for(int round=0;round<150;++round) {
            std::vector<int> physical(64);std::iota(physical.begin(),physical.end(),0);
            std::shuffle(physical.begin(),physical.end(),rng);
            for(int t=0;t<64;++t) {
                f.locations[t]={2,uint64_t(physical[t])*512};
                (*f.heat)[t]=1.+rng()%1000;
            }
            std::vector<int> selected;
            for(int j=0;j<12;++j) selected.push_back(rng()%64);
            std::set<int> pool=resident;pool.insert(selected.begin(),selected.end());
            std::map<uint64_t,std::set<int>> blocks;
            for(int t:pool) blocks[f.locations[t].offset/8192].insert(t);
            std::vector<uint64_t> order;
            std::map<uint64_t,double> scores;
            std::map<uint64_t,size_t> cached;
            for(auto& pair:blocks) {
                double sum=0;for(int t:pair.second) {sum+=(*f.heat)[t];cached[pair.first]+=resident.count(t);}
                scores[pair.first]=std::pow(sum/pair.second.size(),1.5)*(1.+56000./8192);
                order.push_back(pair.first);
            }
            std::sort(order.begin(),order.end(),[&](uint64_t a,uint64_t b) {
                if(scores[a]!=scores[b]) return scores[a]>scores[b];
                bool ar=cached[a]==blocks[a].size(),br=cached[b]==blocks[b].size();
                if(ar!=br) return ar;
                return cached[a]!=cached[b]?cached[a]>cached[b]:a<b;
            });
            resident.clear();size_t remaining=9;
            for(auto b:order) if(blocks[b].size()<=remaining) {
                remaining-=blocks[b].size();resident.insert(blocks[b].begin(),blocks[b].end());
            }
            f.put(selected);
            for(int t=0;t<64;++t) assert(f.hit(t)==bool(resident.count(t)));
            assert(f.cache.stats()["used_bytes"]==resident.size()*512);
        }
    }
    { // Compare expanded and prefiltered admission across growth, warm
      // relocation, cold conversion, duplicates, and partially hit hot groups.
        Fixture full(9,56000.,1.5),fast(9,56000.,1.5);std::mt19937 rng(71);
        const char* counters[]={"entries","insertions","evictions","ema_rejections",
            "access_group_members","access_group_candidates","access_group_selected",
            "access_group_retained","admitted_groups","rejected_groups",
            "oversized_rejected_groups","access_group_evictions","access_hot_used_bytes",
            "access_warm_used_bytes","access_cold_used_bytes"};
        for(int round=0;round<160;++round) {
            int hot=round/10;
            std::vector<int> order(hot);std::iota(order.begin(),order.end(),0);
            std::vector<int> physical(64);std::iota(physical.begin(),physical.end(),0);
            std::shuffle(physical.begin(),physical.end(),rng);
            for(int t=0;t<64;++t) {
                (*full.heat)[t]=(*fast.heat)[t]=1.+rng()%1000;
                auto loc=HybridCache::Location{t<hot?0:int(1+rng()%2),
                    t<hot?uint64_t(t)*512:65536+uint64_t(physical[t])*512};
                full.locations[t]=fast.locations[t]=loc;
            }
            for(auto f:{&full,&fast}) {
                f->hot_end=hot*512;
                f->ranges={{65536,16384},{65536+24576,8192}};
                f->cache.publish(0,0,order,hot,512,f->heat);
            }
            std::vector<int> selected;
            for(int j=0;j<18;++j) selected.push_back(rng()%64);
            if(round%2) selected.insert(selected.end(),order.begin(),order.end());
            full.put(selected);fast.put_prefiltered(selected);
            for(int t=0;t<64;++t) assert(full.hit(t)==fast.hit(t));
            auto a=full.cache.stats(),b=fast.cache.stats();
            for(auto key:counters) assert(a[key]==b[key]);
        }
        assert(fast.cache.stats()["access_hot_early_rejected_groups"]>0);
    }
    hot_growth::Config config;config.retention=.2;
    hot_growth::Head state(config,20,{0,1,2,3,4,5});
    state.cache_include_hot=true;state.publish_cache_scores();
    double old=state.cache_scores->at(0);
    state.observe({0},1);state.publish_cache_scores();
    assert(state.cache_scores->at(0)>old);
}
