import os
from pathlib import Path

import torch

from flexllmgen.retrieval_server import BaseRetrievalServer
from flexllmgen.timer import timers

try:
    from pqcode import PQCodeIndex
except ImportError:
    PQCodeIndex = None


class PQCodeSharedServer(BaseRetrievalServer):
    """KV retrieval server backed exclusively by a shared PQCode index."""

    retrieval_policy = "pqcode_shared"
    selection_block_size = 1

    def __init__(self, config, num_layers, kv_store_path, batch_size=1, max_length=8192, device='cuda:0', dtype=torch.float16, compaction_threshold=0.0, num_subvectors=2, num_bits=6, retention_ratio=0.25, query_chunk_size=32, kmeans_max_iter=20, kmeans_seed=0, merge_threshold_bytes=4096, merge_extra_read_budget=None):
        if PQCodeIndex is None:
            raise ImportError(
                "PQCode is not installed. Run `pip install -e "
                "library/pqcode` from the repository root.")
        self.pq_token_groups = False or False
        self.pq_group_layout = False
        self.pq_paged_reads = False
        self.pq_read_chunk_size = 0
        self.group_by_kv = False
        self.pq_num_subvectors = num_subvectors
        self.pq_num_bits = num_bits
        self.selection_retention_ratio = retention_ratio
        super().__init__(
            config=config,
            num_layers=num_layers,
            kv_store_path=kv_store_path,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            dtype=dtype,
            compaction_threshold=compaction_threshold,
            merge_threshold_bytes=merge_threshold_bytes,
            merge_extra_read_budget=merge_extra_read_budget,


        )
        self.pqcode_index = PQCodeIndex()
        self.pqcode_index.alloc(
            num_subvectors=num_subvectors,
            num_bits=num_bits,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            batch_size=self.batch_size,
            max_length=self.max_length,
            head_dim=self.head_dim,
            retention_ratio=retention_ratio,
            query_chunk_size=query_chunk_size,
            kmeans_max_iter=kmeans_max_iter,
            kmeans_seed=kmeans_seed,
        )
        self.group_selection_stats = []
        self.importance_observer = None
        self.hot_recovery_observer = None
        from flexllmgen.pq_online_correction import PQOnlineCorrection
        self.correction_hot_extras = False
        self.cache_admission = 'selected'
        self.correction_loaded_ids = {}
        self.online_correction = PQOnlineCorrection(
            alpha=float(os.environ.get('PQ_CORRECTION_ALPHA', '0.2')),
            epsilon=float(os.environ.get('PQ_CORRECTION_EPSILON', '1e-12')),
            beta=float(os.environ.get('PQ_CORRECTION_BETA', '0.5')))
        self.pqcode_index.online_correction = self.online_correction
        self._initialize_reorder_worker()

    def load_kv(self, req_id, layer_idx, prefix_id, suffix_len=None):
        if hasattr(self.kv_store, 'set_promotion_snapshot_request'):
            # req_id addresses stored prefix metadata (OPT passes 0). Admission
            # preview instead needs the request that produced this selection.
            identity = getattr(self, '_promotion_selection_identity', None)
            snapshot_request = (identity[0] if identity is not None and
                                identity[1:] == (layer_idx, prefix_id) else -1)
            self.kv_store.set_promotion_snapshot_request(
                snapshot_request if os.environ.get('KVSTORE_REORDER_SYNC', '0') == '1' else -1)
        self.direct_staging_active = False
        correction_enabled = getattr(self, 'online_correction', None) is not None
        use_direct_staging = (suffix_len is not None and (
                not getattr(self, 'kv_cache_capacity_bytes', 0)
                or getattr(self, 'kv_cache_policy', '').startswith('hybrid_')
                or getattr(self, 'kv_cache_policy', '') == 'access_control')
                and hasattr(self.kv_store, 'merge_collect_into'))
        if not self.offloaded or prefix_id == 0:
            return None, None
        if use_direct_staging:
            if getattr(self, '_prefix_workspace', None) is None:
                from flexllmgen.retrieval_workspace import PrefixKVWorkspace
                self._prefix_workspace = PrefixKVWorkspace(
                    self.num_key_value_heads, self.head_dim, self.device,
                    host_copy_backend='numpy')
            # Includes only buffer acquisition/waits, not a valid-payload copy.
            timers('kv postprocess').start()
            slot = self._prefix_workspace.acquire_direct(self.max_length, suffix_len)
            timers('kv postprocess').stop()
            timers('io part test').start()
            try:
                ids, width = self.kv_store.merge_collect_into(
                    prefix_id, layer_idx, self.selection_results, self.nnz,
                    slot['host_k'], slot['host_v'], suffix_len)
            finally:
                timers('io part test').stop()
            self.correction_loaded_ids[layer_idx] = ids
            self.direct_staging_active = True
            return self._prefix_workspace.transfer_direct(slot, width, suffix_len)
        # Collect once, bypassing the base loader's unused core-only packing.
        super().load_kv(req_id, layer_idx, prefix_id, collect_only=True)
        timers("correction ids").start()
        # The native getter already returns an owned clone.
        width = int(self.nnz.max())
        ids = self.selection_results[:, :width].clone()
        invalid = torch.arange(width).unsqueeze(0) >= self.nnz.unsqueeze(1)
        ids.masked_fill_(invalid, -1)
        counts = (ids >= 0).sum(1).tolist()
        width = max(counts)
        ids = ids[:, :width].contiguous()
        self.correction_loaded_ids[layer_idx] = ids
        timers("correction ids").stop()
        native_key = self.kv_store.get_queried_key_cache()
        native_value = self.kv_store.get_queried_value_cache()
        if suffix_len is not None:
            if getattr(self, '_prefix_workspace', None) is None:
                from flexllmgen.retrieval_workspace import PrefixKVWorkspace
                self._prefix_workspace = PrefixKVWorkspace(
                    self.num_key_value_heads, self.head_dim, self.device,
                    host_copy_backend='numpy')
            return self._prefix_workspace.stage(
                native_key, native_value, width, suffix_len, valid_counts=counts)
        timers("correction pack").start()
        key = self.pinned_queried_key[:width]
        value = self.pinned_queried_value[:width]
        # Preserve the legacy token-major interface for quality/debug callers,
        # while using the same bounded valid-row copy and tail initialization.
        from flexllmgen.retrieval_workspace import PrefixKVWorkspace
        PrefixKVWorkspace.copy_host_prefix(
            key.transpose(0, 1), native_key, width, 'numpy', counts)
        PrefixKVWorkspace.copy_host_prefix(
            value.transpose(0, 1), native_value, width, 'numpy', counts)
        timers("correction pack").stop()
        return key, value

    def get_imp_idx(self, layer_idx):
        if (getattr(self, 'online_correction', None) is not None and
                layer_idx in self.correction_loaded_ids):
            ids = self.correction_loaded_ids[layer_idx]
            average = (ids >= 0).sum(1).float().mean()
            self.imp_lens.append(average)
            return ids, average
        return super().get_imp_idx(layer_idx)

    def correction_callback(self, layer):
        correction = getattr(self, 'online_correction', None)
        if correction is None:
            return None
        def observe(ids, q, keys, actual, mask):
            metadata = self.pqcode_index.prepare_device_metadata(layer, 0, q.device)
            correction.defer_observation(layer, metadata,
                                         ids, q, keys, actual, mask)
        return observe


    def reset(self, switch=False):
        self._promotion_selection_identity = None
        super().reset(switch=switch)
        if getattr(self, 'online_correction', None) is not None:
            self.correction_loaded_ids.clear()

    def configure_kv_cache(self, capacity_bytes, policy="access_control"):
        super().configure_kv_cache(capacity_bytes, policy)
        self.selection_scores = (
            None)

    def prepare_offload_key(self, layer_idx, request_id, key_states):
        """Build PQ metadata while the freshly computed Keys are on GPU."""
        sync = None
        if key_states.device.type == "cuda":
            sync = lambda: torch.cuda.synchronize(key_states.device)

        timer = timers("build pq meta")
        timer.start(sync)
        try:
            self.pqcode_index.build(layer_idx, request_id, key_states)
            self.pqcode_index.prepare_device_metadata(
                layer_idx, request_id, key_states.device)
        finally:
            timer.stop(sync)
        return key_states

    def build_metadata(
            self, layer_idx, request_id, prefix_id, key_states, seq_len):
        # Metadata was built before BaseRetrievalServer moved the KV to CPU.
        pass


    def finish_metadata_build(self, layer_idx, request_id, prefix_id, key_states, seq_len):
        pass

    def _build_reorder_inputs(self, query_results):
        selections = super()._build_reorder_inputs(query_results)
        return selections

    def _metadata_path(self, prefix_id, layer_idx):
        return Path(self.kv_store_path) / (
            f"pqcode_prefix_{prefix_id}_layer_{layer_idx}.pt")

    def persist_metadata(
            self, layer_idx, request_id, prefix_id, seq_len):
        metadata_file = self._metadata_path(prefix_id, layer_idx)
        temporary_file = metadata_file.with_name(metadata_file.name + ".tmp")
        state = self.pqcode_index.state_dict(layer_idx, request_id)
        state["group_layout"] = False
        torch.save(state, temporary_file)
        os.replace(temporary_file, metadata_file)


    def retrieve(
        self,
        req_id,
        layer_idx,
        query_states,
        prefix_id,
        max_index,
        save_res,
    ):
        self._promotion_selection_identity = None
        if not self.offloaded or prefix_id == 0:
            return None, None
        fast = getattr(self, "fast_retrieval", False)
        if fast:
            timers("pq score").start_cuda()
        else:
            timers("pq score").start(torch.cuda.synchronize)
        selection_options = {}
        self.pqcode_index.batch_retrieve_multi(
            layer_idx=layer_idx,
            query_states=query_states,
            results=self.selection_results,
            nnz=self.nnz,
            max_index=max_index,
            request_id=0,
            **selection_options,
        )
        if fast:
            # batch_retrieve_multi already makes CPU selection indices ready
            # for KVStore; do not wait for unrelated weight-copy streams.
            timers("pq score").stop_cuda()
        else:
            timers("pq score").stop(torch.cuda.synchronize)
        if save_res:
            self._save_current_selection(req_id, layer_idx)
        self.record_query_results(layer_idx)
        self.prefix_lens.append(max_index)
        self._promotion_selection_identity = (req_id, layer_idx, prefix_id)

    pqcode_shared_retrieve = retrieve

    def selection_filename(self, layer_idx):
        return (
            f"pqcode_m{self.pq_num_subvectors}_b{self.pq_num_bits}_"
            f"layer_{layer_idx}_prefill_imp_token_idx.pt")


    def reset_policy(self):
        self._promotion_selection_identity = None
        self.pqcode_index.clear()
        self.group_selection_stats.clear()
        if getattr(self, 'online_correction', None) is not None:
            self.online_correction.clear()

    def load_pqcode_meta(self, prefix_id):
        timers("load PQ meta").start()
        try:
            states = []
            loaded_prefix_len = None
            for layer_idx in range(self.num_layers):
                metadata_file = self._metadata_path(prefix_id, layer_idx)
                if not metadata_file.exists():
                    raise FileNotFoundError(
                        f"Missing PQCode metadata file: {metadata_file}")
                state = torch.load(str(metadata_file), map_location="cpu")
                if int(state.get("layer_idx", -1)) != layer_idx:
                    raise ValueError(
                        f"PQCode metadata {metadata_file} belongs to layer "
                        f"{state.get('layer_idx')}, expected {layer_idx}")
                layer_prefix_len = int(state["prefix_len"])
                if loaded_prefix_len is None:
                    loaded_prefix_len = layer_prefix_len
                elif loaded_prefix_len != layer_prefix_len:
                    raise ValueError(
                        "PQCode metadata has inconsistent prefix lengths: "
                        f"{loaded_prefix_len} and {layer_prefix_len}")
                states.append(state)

            self.pqcode_index.clear()
            for state in states:
                self.pqcode_index.load_state_dict(state)
                self.pqcode_index.prepare_device_metadata(
                    int(state["layer_idx"]),
                    int(state["request_id"]),
                    self.device,
                )
            self.offload_len = loaded_prefix_len
            self.offloaded = True
            self.current_prefix_id = prefix_id
        finally:
            timers("load PQ meta").stop()

    def prepare_retrieval(self, prefix_id):
        missing_layers = [
            layer_idx for layer_idx in range(self.num_layers)
            if (layer_idx, 0) not in self.pqcode_index.metadata
        ]
        if self.current_prefix_id != prefix_id or missing_layers:
            self.load_pqcode_meta(prefix_id)
