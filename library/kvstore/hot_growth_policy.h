#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Worker-owned policy. No tensor operations, full-prefix sorting or per-request
// frequency sweep. Timestamps implement exact lazy EMA decay.
namespace hot_growth {
struct Config {
    double retention = .05, budget = 1.5, initial = .5, minimum = .2;
    double threshold = .6, growth = .2, half_life = 8., margin = .02;
    int warmup = 1;
    void validate() const {
        ;
        for (double x : {retention, budget, initial, minimum, threshold, growth, half_life, margin})
            if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite hot growth parameter");
        if (retention <= 0 || retention > 1 || budget <= 0 || retention * budget > 1 ||
            initial < 0 || minimum <= 0 || initial + minimum > budget ||
            threshold < 0 || threshold > 1 || growth <= 0 || half_life <= 0 || margin < 0 || warmup < 1)
            throw std::invalid_argument("Invalid hot growth budgets or gate");
    }
    int slots(int n, double multiplier) const {
        return int(std::floor(n * retention * multiplier + 1e-9));
    }
};
struct Plan {
    std::vector<int> warm;
    int freeze = 0, retained = 0, compared = 0;
    int compactions = 0, evicted = 0, appended = 0;
    std::pair<int, int> changed_span(const std::vector<int>& order, int hot) const {
        int first = 0, end = warm.size(), old_width = int(order.size()) - hot;
        while (first < std::min(end, old_width) && warm[first] == order[hot + first]) ++first;
        if (end == old_width)
            while (end > first && warm[end - 1] == order[hot + end - 1]) --end;
        return {first, end};
    }
};
struct Head {
    std::vector<int> order, rank, position, previous, stamp;
    std::vector<double> frequency;
    int hot = 0, cap = 0, step = 0, request = 0, observations = 1;
    double beta;
    // Physical stride is independent of the currently occupied prefix.
    int storage_slots = 0;
    size_t storage_capacity() const { return storage_slots ? storage_slots : order.size(); }
    std::shared_ptr<std::vector<double>> cache_scores;
    std::vector<uint8_t> cache_dirty_flags;
    std::vector<int> cache_dirty, cache_updates;
    std::vector<double> cache_inverse_decay;
    int cache_score_epoch=0;
    bool cache_include_hot=false;
    // Writer runs under commit exclusion. Collectors see only published values;
    // observation records dirty IDs but never mutates this score table.
    bool publish_cache_scores() {
        if(!cache_scores) {
            cache_scores=std::make_shared<std::vector<double>>(frequency);
            cache_dirty_flags.assign(frequency.size(),0);
            cache_inverse_decay={1.};
            return false;
        }
        cache_updates.swap(cache_dirty);cache_dirty.clear();
        bool rebase=(request-cache_score_epoch)*(-std::log2(beta))>256.;
        if(rebase) {
            for(size_t t=0;t<frequency.size();++t) (*cache_scores)[t]=score(t);
            cache_score_epoch=request;cache_inverse_decay={1.};
        } else {
            while(cache_inverse_decay.size()<=size_t(request-cache_score_epoch))
                cache_inverse_decay.push_back(cache_inverse_decay.back()/beta);
            for(int t:cache_updates)
                (*cache_scores)[t]=frequency[t]*cache_inverse_decay[stamp[t]-cache_score_epoch];
        }
        for(int t:cache_updates) cache_dirty_flags[t]=0;
        return rebase;
    }
    Head(const Config& c, int n, const std::vector<int>& initial) : order(initial),
            rank(n), position(n, -1), stamp(n, 0), frequency(n, 0.),
            hot(c.slots(n, c.initial)), cap(int(initial.size()) - std::max(1, c.slots(n, c.minimum))),
            step(std::max(1, c.slots(n, std::min(c.growth, c.budget)))), beta(std::exp2(-1. / c.half_life)) {
        if (int(order.size()) != c.slots(n, c.budget) || hot > cap || order.empty()) {
            throw std::invalid_argument("Prefix too small or incomplete initial hot growth layout");
        }
        for (int t = 0; t < n; ++t) rank[t] = int(order.size()) + t;
        for (int i = 0; i < int(order.size()); ++i) {
            int t = order[i];
            if (t < 0 || t >= n || position[t] >= 0) throw std::invalid_argument("Invalid initial token");
            position[t] = i; rank[t] = i;
            if (i < std::max(1, c.slots(n, 1.))) frequency[t] = 1. - beta;
        }
        snapshot();
    }
    double score(int t) const {
        const int age = request - stamp[t];
        if (!frequency[t] || !age) return frequency[t];
        return frequency[t] * (age == 1 ? beta : std::pow(beta, age));
    }
    std::vector<int> observe(const std::vector<int>& selected, int sequence) {
        if (sequence <= request) throw std::invalid_argument("Hot growth requests must increase");
        if (sequence != request + 1) previous.clear();
        request = sequence; ++observations;
        auto unique = selected;
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        for (int t : unique) {
            if (t < 0 || t >= int(frequency.size())) throw std::invalid_argument("Invalid selected token");
            frequency[t] = score(t) + 1. - beta; stamp[t] = request;
            if(cache_scores && (cache_include_hot || !(position[t]>=0 && position[t]<hot)) && !cache_dirty_flags[t]) {
                cache_dirty_flags[t]=1;cache_dirty.push_back(t);
            }
        }
        return unique;
    }
    Plan plan(const Config& c, const std::vector<int>& selected) const {
        return plan_scored(c, selected, [&](int t) { return score(t); });
    }
    // Preview admission without advancing EMA, observation count or layout.
    Plan preview(const Config& c, const std::vector<int>& selected, int sequence) const {
        // Plan candidates are residents or newly selected tokens. Mark resident
        // observations by physical slot instead of binary-searching selected IDs
        // for every score, and calculate each common decay age only once.
        std::vector<uint8_t> observed(order.size(), 0);
        for (int t : selected) if (position[t] >= 0) observed[position[t]] = 1;
        std::array<double, 64> decay;
        decay[0] = 1.; decay[1] = beta;
        uint64_t ready = 3;
        return plan_scored(c, selected, [&](int t) {
            const int age = sequence - stamp[t];
            double value = frequency[t];
            if (value && age) {
                if (age < 0 || age >= int(decay.size())) value *= std::pow(beta, age);
                else {
                    const uint64_t bit = uint64_t(1) << age;
                    if (!(ready & bit)) { decay[age] = std::pow(beta, age); ready |= bit; }
                    value *= decay[age];
                }
            }
            if (position[t] < 0 || observed[position[t]]) value += 1. - beta;
            return value;
        });
    }
    template<class Score>
    Plan plan_scored(const Config& c, const std::vector<int>& selected, Score value) const {
        ;
        struct Candidate { int token; double score; bool incoming; };
        std::vector<Candidate> pool;
        pool.reserve(order.size() - hot + selected.size());
        for (int i = hot; i < int(order.size()); ++i) pool.push_back({order[i], value(order[i]), false});
        for (int t : selected) if (position[t] < 0) pool.push_back({t, value(t), true});
        const size_t width = order.size() - hot;
        auto admission = [&](const Candidate& a, const Candidate& b) {
            double x = a.score - c.margin * a.incoming, y = b.score - c.margin * b.incoming;
            if (x != y) return x > y;
            if (a.incoming != b.incoming) return !a.incoming;
            return rank[a.token] < rank[b.token];
        };
        if (pool.size() > width) {
            std::nth_element(pool.begin(), pool.begin() + width, pool.end(), admission);
            pool.resize(width);
        }
        std::sort(pool.begin(), pool.end(), [&](const Candidate& a, const Candidate& b) {
            return a.score != b.score ? a.score > b.score : rank[a.token] < rank[b.token];
        });
        Plan p;
        for (auto candidate : pool) p.warm.push_back(candidate.token);
        if (observations >= c.warmup && hot < cap) {
            p.compared = std::min(step, cap - hot);
            for (int i = 0; i < p.compared; ++i)
                p.retained += std::binary_search(previous.begin(), previous.end(), p.warm[i]);
            if (p.retained + 1e-12 >= c.threshold * p.compared) p.freeze = p.compared;
        }
        return p;
    }
    void snapshot() {
        previous.assign(order.begin() + hot, order.begin() + hot + std::min(step, cap - hot));
        std::sort(previous.begin(), previous.end());
    }
    void commit(const Plan& p) {
        for (int i = hot; i < int(order.size()); ++i) position[order[i]] = -1;
        order.resize(hot + p.warm.size());
        std::copy(p.warm.begin(), p.warm.end(), order.begin() + hot);
        for (int i = hot; i < int(order.size()); ++i) position[order[i]] = i;
        hot += p.freeze;
        if (!storage_slots) snapshot();
    }
};
}  // namespace hot_growth
