// Included by kvstore.cc after its I/O helpers. One reorder worker owns plans;
// readers use kv_io_mutex. The cold file is immutable for this strategy.
void KVStore::configure_hot_growth(double retention, double budget, double initial,
        double minimum, double threshold, double growth, double half_life,
        double margin, int warmup, int64_t max_write_bytes) {
    TORCH_CHECK(!persisted, "Configure hot growth before initial persistence");
    hot_growth::Config config{retention, budget, initial, minimum, threshold,
                             growth, half_life, margin, warmup};
    config.validate();
    TORCH_CHECK(max_write_bytes > 0, "Hot growth write cap must be positive");
    hot_growth_config = config;
    hot_growth_max_write_bytes = max_write_bytes;
    hot_growth_enabled = true;
    hot_growth_layout_updates = true;
}



namespace {
struct GrowthFile {
    int fd;
    GrowthFile(const std::string& path, int flags) : fd(open(path.c_str(), flags | O_CLOEXEC)) {
        TORCH_CHECK(fd >= 0, "Cannot open hot growth file: ", path, ": ", std::strerror(errno));
    }
    ~GrowthFile() { close(fd); }
};
void growth_io(int fd, char* data, size_t size, uint64_t offset, bool writing) {
    while (size) {
        ssize_t count = writing ? pwrite(fd, data, size, offset) : pread(fd, data, size, offset);
        if (count < 0 && errno == EINTR) continue;
        TORCH_CHECK(count > 0, "Hot growth ", writing ? "write" : "read", " failed: ", std::strerror(errno));
        size -= count; offset += count; data += count;
    }
}
}

torch::Tensor KVStore::get_hot_growth_layout(int layer) {
    std::shared_lock<std::shared_mutex> lock(kv_io_mutex);
    const auto& heads = hot_growth_heads.at(layer);
    // Unoccupied slots are -1; column zero contains the hot count.
    auto result = torch::full({num_key_value_heads, int(heads[0].storage_capacity()) + 1}, -1, torch::kInt64);
    auto out = result.accessor<int64_t, 2>();
    for (int h = 0; h < num_key_value_heads; ++h) {
        out[h][0] = heads[h].hot;
        std::copy(heads[h].order.begin(), heads[h].order.end(), &out[h][1]);
    }
    return result;
}

std::map<std::string, double> KVStore::hot_growth_update(std::string path, int prefix,
        int layer, const torch::Tensor& selected, int request) {
    TORCH_CHECK(!hot_growth_io_failed.load(), "Hot growth I/O failure: reload prefix");
    TORCH_CHECK(hot_growth_enabled && persisted && hot_growth_heads.count(layer), "Missing hot growth layout");
    TORCH_CHECK(selected.device().is_cpu() && selected.scalar_type() == torch::kInt64 &&
        selected.dim() == 2 && selected.size(1) == 2 && selected.is_contiguous(), "Expected CPU int64 [N,2] selection");
    std::vector<std::vector<int>> selections(num_key_value_heads);
    auto ids = selected.accessor<int64_t, 2>();
    for (int64_t i = 0; i < selected.size(0); ++i) {
        TORCH_CHECK(ids[i][0] >= 0 && ids[i][0] < num_key_value_heads && ids[i][1] >= 0 && ids[i][1] < offload_len,
                    "Invalid hot growth selection");
        selections[ids[i][0]].push_back(ids[i][1]);
    }
    std::map<std::string, double> stats{{"replacement_write_bytes", 0.},
        {"resident_move_write_bytes", 0.}, {"unchanged_span_write_bytes", 0.}};
    auto& heads = hot_growth_heads.at(layer);
    const size_t entry = 2 * head_dim * sizeof(DTYPE);
    ;
    const std::string base = path + "/" + std::to_string(prefix) + "_layer" + std::to_string(layer);
    GrowthFile hot_file(base + "_part0.bin", O_RDWR), cold_file(base + "_part1.bin", O_RDONLY);
    auto busy = [&] { return hot_growth_foreground.load() || foreground_readers.load() > 0; };
    for (int h = 0; h < num_key_value_heads; ++h) {
        auto& state = heads[h];
        auto unique = state.observe(selections[h], request);
        auto skip = [&](const char* reason) { ++stats[reason]; state.previous.clear(); };
        if (busy()) { skip("foreground_skipped_heads"); continue; }
        auto plan_start = ReadClock::now();
        auto plan = state.plan(hot_growth_config, unique);
        stats["plan_seconds"] += read_seconds(plan_start);
        const int width = state.order.size() - state.hot;
        const auto [first, end] = plan.changed_span(state.order, state.hot);
        size_t bytes = size_t(end - first) * entry;
        size_t replacement_slots = 0, moved_slots = 0, unchanged_slots = 0;
        if (bytes > hot_growth_max_write_bytes) { skip("write_cap_skipped_heads"); continue; }
        std::vector<char> old, staged;
        const uint64_t offset = (size_t(h) * state.storage_capacity() + state.hot + first) * entry;
        if (bytes) {
            if (busy()) { skip("foreground_skipped_heads"); continue; }
            const int source_end = width == int(plan.warm.size()) ? end : std::max(width, end);
            old.resize(size_t(source_end - first) * entry); staged.resize(bytes);
            auto stage_start = ReadClock::now();
            // No payload I/O under the metadata lock. Only this worker changes
            // part0; prefix reset drains it before deleting/replacing the store.
            growth_io(hot_file.fd, old.data(), old.size(), offset, false);
            stats["read_bytes"] += old.size();
            std::vector<int> incoming, missing;
            incoming.reserve(end - first); missing.reserve(end - first);
            for (int i = first; i < end; ++i) {
                const int pos = state.position[plan.warm[i]];
                // Count only committed writes, not interrupted staging attempts.
                if (pos < 0) ++replacement_slots;
                else if (pos == state.hot + i) ++unchanged_slots;
                else ++moved_slots;
                if (pos >= state.hot + first && pos < state.hot + source_end)
                    std::memcpy(staged.data() + size_t(i - first) * entry,
                                old.data() + size_t(pos - state.hot - first) * entry, entry);
                else incoming.push_back(i);
            }
            if (!incoming.empty()) {
                std::unique_lock<std::shared_mutex> lock(kv_io_mutex, std::try_to_lock);
                if (!lock.owns_lock() || busy()) { skip("foreground_skipped_heads"); continue; }
                auto prepare_start = ReadClock::now();
                for (int i : incoming) {
                    int source = copy_reorder_kv(prefix, layer, get_meta_id(plan.warm[i], layer, h),
                                                staged.data() + size_t(i - first) * entry);
                    if (source) ++stats["memory_source_tokens"];
                    else missing.push_back(i);
                }
                const double seconds = read_seconds(prepare_start);
                lock.unlock();
                stats["prepare_lock_seconds"] += seconds;
                stats["max_prepare_lock_seconds"] = std::max(stats["max_prepare_lock_seconds"], seconds);
            }
            stats["disk_source_tokens"] += missing.size();
            std::map<uint64_t, PageRead> pages;
            for (int i : missing) {
                size_t remaining = entry, copied = 0;
                uint64_t source_offset = (size_t(h) * offload_len + plan.warm[i]) * entry;
                while (remaining) {
                    uint64_t page_offset = source_offset - source_offset % kDirectIoAlignment;
                    size_t within = source_offset - page_offset;
                    size_t length = std::min(remaining, kDirectIoAlignment - within);
                    auto it = pages.emplace(page_offset, PageRead{page_offset, 0, {}}).first;
                    it->second.required_bytes = std::max(it->second.required_bytes, within + length);
                    it->second.copies.push_back(PageCopy{size_t(i - first) * entry + copied, within, length});
                    remaining -= length; copied += length; source_offset += length;
                }
            }
            // One buffered read per source page. Check foreground demand between
            // pages instead of filling an io_uring queue ahead of the frontend.
            std::vector<char> page(kDirectIoAlignment);
            bool interrupted = false;
            for (const auto& pair : pages) {
                if (busy()) { interrupted = true; break; }
                const auto& p = pair.second;
                growth_io(cold_file.fd, page.data(), p.required_bytes, p.offset, false);
                scatter_page(p, page.data(), staged);
                stats["read_bytes"] += p.required_bytes;
            }
            stats["stage_seconds"] += read_seconds(stage_start);
            if (interrupted) { skip("foreground_skipped_heads"); continue; }
        }
        std::unique_lock<std::shared_mutex> lock(kv_io_mutex, std::try_to_lock);
        if (!lock.owns_lock() || busy()) { skip("foreground_skipped_heads"); continue; }
        const auto commit_start = ReadClock::now();
        if (bytes) {
            try {
                growth_io(hot_file.fd, staged.data(), bytes, offset, true);
            } catch (...) {
                // Readers cannot observe partially overwritten bytes. If even
                // rollback fails, poison the store and fail subsequent reads.
                try { growth_io(hot_file.fd, old.data(), bytes, offset, true); }
                catch (...) { hot_growth_io_failed.store(true); }
                throw;
            }
            // Restore old warm tokens to immutable backing, then publish all
            // winners. Cache payloads are immutable and keyed by logical IDs.
            stats["write_bytes"] += bytes;
            stats["replacement_write_bytes"] += replacement_slots * entry;
            stats["resident_move_write_bytes"] += moved_slots * entry;
            stats["unchanged_span_write_bytes"] += unchanged_slots * entry;
            stats["written_heads"] += 1;
            if (!replacement_slots) {
                stats["reorder_only_heads"] += 1;
                stats["reorder_only_write_bytes"] += bytes;
            }
            if (state.hot == state.cap) {
                stats["at_cap_write_bytes"] += bytes;
                stats["at_cap_replacement_write_bytes"] += replacement_slots * entry;
                stats["at_cap_resident_move_write_bytes"] += moved_slots * entry;
                stats["at_cap_unchanged_span_write_bytes"] += unchanged_slots * entry;
            }
            reorder_io_bytes += bytes;
        }
        // A truncation can evict tokens without writing payload bytes. Restore
        // every old warm location before publishing the occupied new span.
        if (bytes || width != int(plan.warm.size())) {
            const bool resized = width != int(plan.warm.size());
            for (int i = first; i < (resized ? width : end); ++i) {
                int token = state.order[state.hot + i];
                update_reorder_location(prefix, layer, get_meta_id(token, layer, h), 1,
                                        (size_t(h) * offload_len + token) * entry);
            }
            for (int i = first; i < (resized ? int(plan.warm.size()) : end); ++i)
                update_reorder_location(prefix, layer, get_meta_id(plan.warm[i], layer, h), 0,
                                        offset + size_t(i - first) * entry);
        }
        const int old_hot = state.hot;
        state.commit(plan);
        {
            auto publish_start = ReadClock::now();
            bool rebase=state.publish_cache_scores();
            hybrid_cache.publish(layer,h,state.order,state.hot,entry,state.cache_scores,&state.cache_updates,rebase);
            stats["cache_ema_published_tokens"] += state.cache_updates.size();
            stats["cache_ema_rebase_tokens"] += rebase ? state.frequency.size() : 0;
            stats["cache_rank_publish_seconds"] += read_seconds(publish_start);
        }
        ;
        const double commit_seconds = read_seconds(commit_start);
        lock.unlock();
        stats["commit_seconds"] += commit_seconds;
        stats["max_commit_seconds"] = std::max(stats["max_commit_seconds"], commit_seconds);
        ++stats["committed_heads"];
        stats["frozen_tokens"] += plan.freeze;
        stats["freeze_checks"] += plan.compared > 0;
        stats["deferred_checks"] += plan.compared > 0 && !plan.freeze;
        stats["candidate_retained"] += plan.retained;
        stats["candidate_compared"] += plan.compared;
        stats["compactions"] += plan.compactions;
        stats["evicted_tokens"] += plan.evicted;
        stats["appended_tokens"] += plan.appended;
    }
    // Flush once per changed layer, outside kv_io_mutex. A completed synchronous
    // job must not leave dirty warm pages for the next foreground direct read.
    if (stats["write_bytes"] > 0) {
        const auto flush_start = ReadClock::now();
        try {
            fsync_fd_or_throw(hot_file.fd, base + "_part0.bin");
        } catch (...) {
            hot_growth_io_failed.store(true);
            throw;
        }
        stats["flush_seconds"] += read_seconds(flush_start);
        stats["flushed_layers"] += 1;
    }
    for (const auto& state : heads) {
        stats["hot_tokens"] += state.hot;
        stats["warm_tokens"] += state.order.size() - state.hot;
        stats["free_tokens"] += state.storage_capacity() - state.order.size();
    }
    return stats;
}
