#pragma once

#include <torch/extension.h>
#include "merge_read_plan.h"
#include "hybrid_cache.h"
#include "hot_growth_policy.h"
#include <atomic>
#include <math.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <shared_mutex>
#include <memory>
#include <map>
#include <optional>
#include <tuple>
#include <cmath>
#include <mutex>

struct KVReadResources;
// #define ATTENTION_THREADS 64
#define ATTENTION_THREADS 32
#define READ_THREADS 8

using bfloat16 = std::uint16_t;
#define DTYPE bfloat16

struct FileOffsetInfo {
    uint64_t file_index;
    uint64_t offset;
    FileOffsetInfo() : file_index(0), offset(0) {}
    FileOffsetInfo(uint64_t idx, uint64_t off) : file_index(idx), offset(off) {}
};

struct LayerStats{
    uint64_t    segment_cnt;
    double      avg_continuous_length;
    double      tot_time;
    double      collect_time;

    LayerStats(): segment_cnt(0), avg_continuous_length(0.0), tot_time(0.0){}

    void update_len(uint64_t new_segment_cnt, double new_avg_len){
        avg_continuous_length = (avg_continuous_length * segment_cnt + new_avg_len * new_segment_cnt) / (segment_cnt + new_segment_cnt);
        segment_cnt += new_segment_cnt;
    }

    void update_time(double new_time) { tot_time += new_time; }
};

struct IOTask{
    uint64_t file_index;
    uint64_t offset;
    uint64_t length;

    int bitmap;

    std::vector<int> token_indices;

    int head_id;
};

class KVStore {
    public:
        KVStore();
        ~KVStore();
        void alloc(int num_layers, int num_attention_heads, int num_key_value_heads, int head_dim, int max_length);
        void fill(int layer_id, torch::Tensor k, torch::Tensor v);
        void reorder_persist(std::string path, int prefix_id, int layer_id, const torch::Tensor& reorder_token_info);
        void finish_persist();
        void clear();
        uint64_t get_meta_id(int token_id, int layer_id, int head_id);
        void persist_meta(int prefix_id);
        void recover_meta(std::string path, int prefix_id);
        std::map<std::string, double> get_read_profile(bool reset = true);
        std::map<uint64_t, std::map<std::string, uint64_t>> get_merge_read_stats(bool reset = true);
        void clear_read_cache();
        std::map<std::string, uint64_t> get_reorder_memory_stats(bool reset = true);
        void configure_cache(int64_t capacity_bytes, const std::string& policy = "access_control");
        void clear_cache();
        std::map<std::string, uint64_t> get_cache_stats(bool reset = false);
        std::vector<std::map<std::string, uint64_t>> get_cache_partition_stats();
        void set_collect_io_backend(const std::string& backend);
        std::string get_collect_io_backend() const;
        void merge_collect_queried_key_value(int prefix_id, int layer_id, torch::Tensor ind_pt,
            torch::Tensor nnz_pt, std::optional<torch::Tensor> importance_scores = std::nullopt);
        std::tuple<torch::Tensor, int> merge_collect_into(int prefix_id, int layer_id,
            torch::Tensor ids, torch::Tensor counts, torch::Tensor out_key,
            torch::Tensor out_value, int suffix_len);
        void collect_planned_key_value(int prefix_id, int layer_id, torch::Tensor ind_pt,
            torch::Tensor nnz_pt, bool adjacent_pages_only,
            std::optional<torch::Tensor> importance_scores = std::nullopt,
            bool group_only = false);
        void merge_load_key_value_from_file(std::vector<MergeRead>& content, std::vector<int> &token_order, int prefix_id, int layer_id, int head_id,
            const CacheReadCopies* captures = nullptr, size_t capture_begin = 0);
        void load_key_value_from_file_iouring(std::vector<MergeRead>& content, std::vector<int> &token_order, int prefix_id, int layer_id, int head_id,
            const CacheReadCopies* captures = nullptr, size_t capture_begin = 0);
        void load_key_value_from_file_iouring(std::vector<std::tuple<uint64_t, uint64_t, int>>& content, std::vector<int> &token_order, int prefix_id, int layer_id, int head_id);
        void configure_hot_growth(double retention, double budget, double initial,
            double minimum, double threshold, double growth, double half_life,
            double margin, int warmup, int64_t max_write_bytes);
        std::map<std::string, double> hot_growth_update(std::string path, int prefix,
            int layer, const torch::Tensor& selected, int request);
        torch::Tensor get_hot_growth_layout(int layer);
        torch::Tensor get_initial_key_view(int layer);
        void set_hot_growth_foreground(bool active) { hot_growth_foreground.store(active); }
        void set_promotion_snapshot_request(int request) { promotion_snapshot_request = request; }
        void set_hot_growth_read_mode(const std::string& mode, int64_t cold_bytes = 8192) {
            TORCH_CHECK(mode == "hybrid", "AdaKVs uses the hybrid reader");
            TORCH_CHECK(cold_bytes >= 4096 && cold_bytes % 4096 == 0, "Invalid cold block size");
            hot_growth_hybrid_reads = true;
            hot_growth_cold_merge_bytes = cold_bytes;
        }
        void set_merge_threshold_bytes(int64_t threshold) {
            TORCH_CHECK(threshold >= 0, "merge_threshold_bytes must be nonnegative");
            merge_threshold_bytes = static_cast<uint64_t>(threshold);
        }
        void set_merge_extra_read_budget(std::optional<double> budget) {
            TORCH_CHECK(!budget || (std::isfinite(*budget) && *budget >= 0),
                        "merge_extra_read_budget must be finite and nonnegative, or None");
            merge_extra_read_budget = budget;
        }
        void set_region_merge_extra_read_budgets(std::optional<double> hot,
                                                std::optional<double> cold) {
            TORCH_CHECK(!hot || (std::isfinite(*hot) && *hot >= 0), "hot budget must be finite and nonnegative, or None");
            TORCH_CHECK(!cold || (std::isfinite(*cold) && *cold >= 0), "cold budget must be finite and nonnegative, or None");
            hot_merge_extra_read_budget = hot;
            cold_merge_extra_read_budget = cold;
        }
        void set_compaction_threshold(double threshold);
        torch::Tensor get_queried_key_cache();
        torch::Tensor get_queried_value_cache();
        int get_num_io_and_reset();
        long long get_loaded_bytes_and_reset();
        long long get_reorder_io_bytes_and_reset();
    private:
        bool hot_growth_enabled = false;
        bool hot_growth_layout_updates = true;
        int promotion_snapshot_request = -1;
        bool hot_growth_hybrid_reads = false;
        bool hot_growth_fixed_reads = false;
        uint64_t hot_growth_fixed_block_bytes = 16384;
        HybridCache hybrid_cache;
        std::string hybrid_cache_miss_read_mode = "auto";
        bool hybrid_cache_hot_overlap = true;
        double hybrid_cache_read_us = 8.0, hybrid_cache_read_bps = 7e9;
        std::string hybrid_cache_policy;
        void collect_hybrid_cached(int prefix, int layer, const int* ids, const int* counts);
        uint64_t hot_growth_cold_merge_bytes = 8192;
        hot_growth::Config hot_growth_config;
        size_t hot_growth_max_write_bytes = 1024 * 1024;
        std::map<int, std::vector<hot_growth::Head>> hot_growth_heads;
        std::atomic<bool> hot_growth_foreground{false};
        std::atomic<bool> hot_growth_io_failed{false};
        std::atomic<int> foreground_readers{0};
        // Logical token destinations still use max_length. Only this query's
        // external payload is compacted to direct_stride rows per head.
        int direct_stride = 0;
        int direct_suffix_len = 0;
        bool direct_collect = false;
        size_t query_output_offset(size_t logical_position) const {
            const size_t stride = direct_collect ? direct_stride : max_length;
            return ((logical_position / max_length) * stride +
                    logical_position % max_length) * head_dim;
        }
        void prepare_direct_output(const std::vector<int>& counts);
        bool collect_hot_extras = false;
        bool cache_read_ranges = false;
        torch::Tensor collected_hot_indices;
        struct PromotionSnapshot {
            std::vector<int> positions;
            std::vector<int> touched;
            std::vector<std::vector<int>> sources;
            // Every used byte is overwritten before publication. A raw buffer
            // avoids vector::resize zeroing GBs of K/V on each collection.
            std::unique_ptr<char[]> payload;
            size_t payload_size = 0, payload_capacity = 0;
            void reserve_payload(size_t bytes) {
                if (bytes <= payload_capacity) return;
                payload.reset(new char[bytes]);
                payload_capacity = bytes;
            }
            void reset() {
                for (int flat : touched) positions[flat] = -1;
                touched.clear(); payload_size = 0;
            }
        };
        std::map<std::pair<int, int>, PromotionSnapshot> promotion_snapshots;
        std::map<std::string, uint64_t> reorder_memory_stats;
        void capture_promotion_snapshot(int prefix, int layer, const int* ids, const int* counts);
        int copy_reorder_kv(int prefix, int layer, uint64_t meta_id, char* destination);
        void update_reorder_location(int prefix, int layer, uint64_t meta_id,
                                     uint64_t file, uint64_t offset);
        std::map<std::pair<int, int>, std::vector<int64_t>> cold_region_indices;
        std::map<std::pair<int, int>, std::vector<int32_t>> hot_region_indices;
        bool cache_diagnostics = false;
        std::unique_ptr<KVReadResources> read_resources;
        int num_layers;
        int num_attention_heads;
        int num_key_value_heads;
        int head_dim;
        int max_length;
        int offload_len = 0;
        int num_io;
        long long loaded_bytes;
        long long reorder_io_bytes;
        uint64_t merge_threshold_bytes = 4096;
        std::optional<double> merge_extra_read_budget;
        std::optional<double> hot_merge_extra_read_budget, cold_merge_extra_read_budget;
        std::optional<double> region_merge_budget(uint64_t file) const {
            auto override = file == 0 ? hot_merge_extra_read_budget : cold_merge_extra_read_budget;
            return override ? override : merge_extra_read_budget;
        }
        double compaction_threshold;
        bool collect_use_iouring;

        bool allocated;
        std::vector<DTYPE *>key_cache;// 每一层的 一个kv_head*max_length*head_dim的开辟好的空间  lsh_server存下来的用于持久化
        std::vector<DTYPE *>value_cache;
        DTYPE* queried_key; //返回request所需要的kv
        DTYPE* queried_value;

        std::string store_path;
        std::vector<std::string> pending_persist_files;
        bool persisted;

        std::unordered_map<uint64_t, FileOffsetInfo>* kv_meta; // metaid -> pos
        mutable std::shared_mutex kv_io_mutex;

        std::vector<LayerStats> layer_stats;
};
