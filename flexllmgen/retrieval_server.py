import os
import threading
import time
import json
from pathlib import Path

def _parse_cpu_list(value, variable_name):
    """Parse Linux CPU-list syntax such as ``0-7,12,14-15``."""
    if value is None or not value.strip():
        return None

    cpus = set()
    try:
        for item in value.split(","):
            item = item.strip()
            if not item:
                raise ValueError
            if "-" in item:
                bounds = item.split("-")
                if len(bounds) != 2:
                    raise ValueError
                first, last = (int(bound.strip()) for bound in bounds)
                if first < 0 or last < first:
                    raise ValueError
                cpus.update(range(first, last + 1))
            else:
                cpu = int(item)
                if cpu < 0:
                    raise ValueError
                cpus.add(cpu)
    except ValueError as error:
        raise ValueError(
            f"Invalid {variable_name} CPU list: {value!r}") from error

    if not cpus:
        raise ValueError(f"{variable_name} must select at least one CPU")
    return cpus


def _format_cpu_list(cpus):
    """Return a compact, deterministic Linux CPU-list representation."""
    ordered = sorted(cpus)
    ranges = []
    first = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(
            str(first) if first == previous else f"{first}-{previous}")
        first = previous = cpu
    ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(ranges)


# Apply the foreground mask before importing PyTorch/KVStore so their CPU
# worker pools inherit it. Avoid forcing libgomp placements: PyTorch also uses
# OpenMP while building PQ metadata, and a process-wide placement list can
# serialize or severely oversubscribe those kernels.
_INITIAL_CPU_AFFINITY = (
    os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None)
_BOOTSTRAP_FOREGROUND_CPUS = _parse_cpu_list(
    os.environ.get("FLEXGEN_FOREGROUND_CPUS"),
    "FLEXGEN_FOREGROUND_CPUS")
if _BOOTSTRAP_FOREGROUND_CPUS is not None:
    if _INITIAL_CPU_AFFINITY is None:
        raise RuntimeError(
            "FLEXGEN CPU affinity requires Linux sched affinity support")
    try:
        # GOMP_CPU_AFFINITY may already have narrowed the OpenMP master to its
        # first place. sched_setaffinity can still widen it within the job's
        # cgroup, so let the kernel validate the requested mask directly.
        os.sched_setaffinity(0, _BOOTSTRAP_FOREGROUND_CPUS)
    except OSError as error:
        raise ValueError(
            "FLEXGEN_FOREGROUND_CPUS cannot be applied: "
            f"{_format_cpu_list(_BOOTSTRAP_FOREGROUND_CPUS)}") from error

import torch
from kvstore import KVStore

from flexllmgen.timer import timers
def validate_merge_extra_read_budget(value):
    import math
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Merge read budget must be finite and nonnegative')
    return value


# torch.cuda.set_device(0)



class BaseRetrievalServer:

    retrieval_policy = None
    selection_retention_ratio = 1.0
    selection_block_size = 1

    def __init__(self, config, num_layers: int, kv_store_path: str, batch_size: int=1, max_length: int=8192, device: str='cuda:0', dtype=torch.float16, compaction_threshold: float=0.0, merge_threshold_bytes=4096, merge_extra_read_budget=None):
        if merge_threshold_bytes < 0:
            raise ValueError("merge_threshold_bytes must be nonnegative")
        self.merge_threshold_bytes = merge_threshold_bytes
        self.merge_extra_read_budget = (
            validate_merge_extra_read_budget(merge_extra_read_budget)
            if merge_extra_read_budget is not None else None)
        self._configure_cpu_affinity()
        self.config = config  ### OptConfig or LlamaConfig
        self.num_layers = num_layers
        self.batch_size = batch_size
        if "opt" in config.name:
              self.num_key_value_heads = config.n_head
        else: ## llama
            self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.n_head
        self.max_length = max_length ### max overall length (prefill+decode)
        self.device = device
        self.dtype = dtype
        self.head_dim = config.hidden_size // config.n_head

        self.offloaded = False
        self.persisted = False
        self.merge = True
        self.collect_attention_mass = False
        self.attention_warmup = False
        self.fall_back = False
        self.attention_mass_keys = (
            None)
        self.attention_mass_results = {}

        self.current_prefix_id = 0 ### 0 means nothing
        self.kv_store = KVStore()
        self.kv_store.alloc(self.num_layers, self.num_attention_heads, self.num_key_value_heads, self.head_dim, max_length)
        self.kv_store.set_merge_threshold_bytes(self.merge_threshold_bytes)
        self.kv_store.set_merge_extra_read_budget(self.merge_extra_read_budget)
        self.kv_store.set_compaction_threshold(compaction_threshold)
        self.compaction_threshold = compaction_threshold
        self.kv_store_path = kv_store_path
        # Per-head selection counts and indices shared by all retrieval APIs.
        selection_heads = (self.num_attention_heads)
        self.nnz = torch.zeros((self.batch_size * selection_heads,), pin_memory=True).to(torch.int32)
        self.selection_results = torch.zeros((self.batch_size * selection_heads, self.max_length), pin_memory=True).to(torch.int32)

        self.query_results = [
            (torch.zeros_like(self.nnz), torch.zeros_like(self.selection_results))
            for _ in range(self.num_layers)]

        self.pinned_queried_key = torch.zeros((self.max_length, self.num_key_value_heads, self.head_dim), dtype=torch.float16).pin_memory()
        self.pinned_queried_value = torch.zeros((self.max_length, self.num_key_value_heads, self.head_dim), dtype=torch.float16).pin_memory()

        ### store token group strategy of each layer
        self.persist_strategy = [None for i in range(self.num_layers)]

        self.copy_stream = torch.cuda.Stream(priority=-1)
        self.build_stream = torch.cuda.Stream()

        self.prefix_lens = []
        self.imp_lens = []

    def _configure_cpu_affinity(self):
        """Partition foreground and reorder work using optional CPU masks."""
        global _INITIAL_CPU_AFFINITY
        foreground = _parse_cpu_list(
            os.environ.get("FLEXGEN_FOREGROUND_CPUS"),
            "FLEXGEN_FOREGROUND_CPUS")
        reorder = _parse_cpu_list(
            os.environ.get("FLEXGEN_REORDER_CPUS"),
            "FLEXGEN_REORDER_CPUS")
        self._foreground_cpus = foreground
        self._reorder_cpus = reorder

        if foreground is None and reorder is None:
            return
        if not hasattr(os, "sched_getaffinity"):
            raise RuntimeError(
                "FLEXGEN CPU affinity requires Linux sched affinity support")

        if _INITIAL_CPU_AFFINITY is None:
            _INITIAL_CPU_AFFINITY = os.sched_getaffinity(0)
        if (foreground is not None and reorder is not None and
                foreground.intersection(reorder)):
            overlap = _format_cpu_list(foreground.intersection(reorder))
            raise ValueError(
                "Foreground and reorder CPU masks must be disjoint; "
                f"overlap: {overlap}")

        # Probe the union so an invalid background-only CPU fails during model
        # initialization rather than asynchronously inside the worker. This
        # also works when libgomp has narrowed the current thread to one CPU.
        original = os.sched_getaffinity(0)
        requested = set()
        if foreground is not None:
            requested.update(foreground)
        if reorder is not None:
            requested.update(reorder)
        try:
            os.sched_setaffinity(0, requested)
        except OSError as error:
            raise ValueError(
                "Requested FLEXGEN CPU affinity cannot be applied: "
                f"{_format_cpu_list(requested)}") from error
        finally:
            os.sched_setaffinity(
                0, foreground if foreground is not None else original)

        if foreground is not None:
            # KVStore and PyTorch worker pools inherit this mask without a
            # process-wide OMP placement policy.
            os.sched_setaffinity(0, foreground)
            print(
                "foreground CPU affinity: "
                f"{_format_cpu_list(foreground)}",
                flush=True)
        if reorder is not None:
            print(
                "reorder CPU affinity: "
                f"{_format_cpu_list(reorder)}",
                flush=True)

    def warmup_retrieval(self, q_len):
        """Warm policy-specific kernels before the measured request."""

    def offload(self,
        layer_idx:int,
        request_id: int,
        seq_len:int,
        prefix_id: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor):

        """Offload prefix KV and build metadata owned by this policy."""
        offload_key = key_states[:seq_len].transpose(0,1).contiguous()
        offload_value = value_states[:seq_len].transpose(0,1).contiguous()
        original_key = None
        offload_key = self.prepare_offload_key(
            layer_idx, request_id, offload_key)
        offload_len = offload_key.shape[1]
        offload_key = offload_key.cpu()
        offload_value = offload_value.cpu()
        self.build_metadata(
            layer_idx, request_id, prefix_id, offload_key, offload_len)

        ### offload to kv store
        self.kv_store.fill(layer_idx, offload_key, offload_value)
        self.finish_metadata_build(
            layer_idx, request_id, prefix_id, offload_key, offload_len)

        self.offloaded = True
        self.offload_len = offload_len
        self.current_prefix_id = prefix_id

        if not self.persisted:
            self.persist_metadata(
                layer_idx, request_id, prefix_id, offload_len)

    def prepare_offload_key(self, layer_idx, request_id, key_states):
        """Return the representation that should be stored in KVStore."""
        return key_states

    def attention_mass_key(self, original_key, stored_key):
        """Return the dense CPU Keys used to evaluate selection quality."""
        return stored_key

    def build_metadata(
            self, layer_idx, request_id, prefix_id, key_states, seq_len):
        raise NotImplementedError

    def finish_metadata_build(
            self, layer_idx, request_id, prefix_id, key_states, seq_len):
        """Finish metadata that must be built after KVStore is populated."""

    def persist_metadata(
            self, layer_idx, request_id, prefix_id, seq_len):
        """Persist policy metadata when the policy supports recovery."""

    def record_query_results(self, layer_idx, selected_width=None):
        if selected_width is not None:
            # KVStore still receives its fixed-stride scratch tensor, but
            # layer-owned snapshots only need the active selection rectangle.
            self.query_results[layer_idx] = (
                self.nnz.clone(), self.selection_results[:, :selected_width].clone())
            return
        nnz, res = self.query_results[layer_idx]
        if res.shape != self.selection_results.shape:
            res = torch.empty_like(self.selection_results)
            self.query_results[layer_idx] = (nnz, res)
        nnz.copy_(self.nnz)
        res.copy_(self.selection_results)

    def record_initial_attention_topk(self, layer_idx, token_indices):
        """Record first-prefill attention top-k for initial persistence."""
        if token_indices.ndim != 2:
            raise ValueError(
                "Initial attention top-k must have shape [heads, tokens]")
        selection_heads = self.query_results[layer_idx][0].shape[0]
        if token_indices.shape[0] != selection_heads:
            raise ValueError(
                "Initial attention top-k head count mismatch: expected "
                f"{selection_heads}, got {token_indices.shape[0]}")
        selected_count = token_indices.shape[1]
        if selected_count <= 0 or selected_count > self.max_length:
            raise ValueError(
                f"Invalid initial attention top-k size: {selected_count}")

        # query_results is layer-owned, so the non-blocking device-to-host
        # copy can overlap the remainder of the layer pipeline. OptLM.sync()
        # completes it before initial persistence consumes these tensors.
        nnz, results = self.query_results[layer_idx]
        if results.shape[1] < selected_count:
            results = results.new_empty((results.shape[0], selected_count))
            self.query_results[layer_idx] = (nnz, results)
        nnz.fill_(selected_count)
        results.zero_()
        if token_indices.is_cuda:
            timer = timers("initial attention topk")
            timer.start_cuda()
            results[:, :selected_count].copy_(
                token_indices.to(dtype=torch.int32), non_blocking=True)
            timer.stop_cuda()
        else:
            results[:, :selected_count].copy_(
                token_indices.to(dtype=torch.int32))
        if getattr(self, 'hot_growth_config', None) is not None:
            if not hasattr(self, '_initial_persist_results'):
                self._initial_persist_results = {}
            # Independent ownership: later core PQ queries overwrite query_results.
            self._initial_persist_results[layer_idx] = (
                nnz.clone(), token_indices.detach().clone())

    def _save_current_selection(self, req_id, layer_idx):
        if self.analysis_dir is None:
            raise ValueError(
                "analysis_dir must be configured when save_res is enabled")
        save_dir = Path(self.analysis_dir) / str(req_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = self.selection_filename(layer_idx)
        torch.save(
            (self.nnz.clone(), self.selection_results.clone()),
            str(save_dir / filename))

    def selection_filename(self, layer_idx):
        raise NotImplementedError

    def get_imp_idx(self, layer_idx):
        nnz, res = self.query_results[layer_idx]
        for head_id, n in enumerate(nnz):
            res[head_id, n:] = -1
        max_len = nnz.max()
        avg_len = nnz.sum() / len(nnz)
        self.imp_lens.append(avg_len)
        return res[:, :max_len], avg_len


    def configure_region_merge_budgets(self, hot=None, cold=None):
        hot = validate_merge_extra_read_budget(hot) if hot is not None else None
        cold = validate_merge_extra_read_budget(cold) if cold is not None else None
        self.kv_store.set_region_merge_extra_read_budgets(hot, cold)
        self.hot_merge_extra_read_budget = hot
        self.cold_merge_extra_read_budget = cold

    def configure_kv_cache(self, capacity_bytes, policy="access_control"):
        """Keep configuration across store replacement, and payloads across requests."""
        if not hasattr(self.kv_store, "configure_cache"):
            raise RuntimeError("Rebuild library/kvstore for KV payload caching")
        self.kv_store.configure_cache(capacity_bytes, policy)
        self.kv_cache_capacity_bytes = capacity_bytes
        self.kv_cache_policy = policy




    def retrieve(
        self,
        req_id: int,
        layer_idx: int,
        query_states: torch.Tensor,
        prefix_id: int,
        max_index: int,
        save_res: bool,
    ):
        raise NotImplementedError

    def use_full_attention(self, layer_idx):
        return False




    def load_kv(self,
        req_id: int,
        layer_idx: int,
        prefix_id: int,
        suffix_len=None,
        collect_only=False):
        if not self.offloaded or prefix_id == 0:
            return None, None
        timers("io part test").start()
        ### collect key value from kv_store
        #timers("io part test").start()
        # self.kv_store.collect_queried_key_value(prefix_id, layer_idx, self.selection_results, self.nnz)
        self.kv_store.merge_collect_queried_key_value(
            prefix_id, layer_idx, self.selection_results, self.nnz)

        #self.kv_store.concurrent_merge_collect_queried_key_value(prefix_id, layer_idx, self.selection_results, self.nnz)
        timers("io part test").stop()
        if collect_only:
            return None, None
        #timers("io part test").stop()
        ### shape : n_head, max_length, head_dim
        res_len = self.nnz.max().data

        if suffix_len is not None:
            # Both PQCode and Quest store uncentered Keys.
            if getattr(self, "_prefix_workspace", None) is None:
                from flexllmgen.retrieval_workspace import PrefixKVWorkspace
                self._prefix_workspace = PrefixKVWorkspace(
                    self.num_key_value_heads, self.head_dim, self.device,
                    host_copy_backend=("torch"))
            return self._prefix_workspace.stage(
                self.kv_store.get_queried_key_cache(),
                self.kv_store.get_queried_value_cache(),
                int(res_len), suffix_len)

        timers("kv postprocess").start()
        queried_key = self.kv_store.get_queried_key_cache()  # collect后保存在kvstore里,这里将数据包装成tensor后拿出来
        queried_value = self.kv_store.get_queried_value_cache()
        self.restore_loaded_keys(
            queried_key[:, :res_len, :], layer_idx, req_id)
        self.pinned_queried_key[:res_len].copy_(queried_key[:, :res_len, :].transpose(0, 1), non_blocking=True)
        self.pinned_queried_value[:res_len].copy_(queried_value[:, :res_len, :].transpose(0, 1), non_blocking=True)
        timers("kv postprocess").stop()

        return self.pinned_queried_key[:res_len], self.pinned_queried_value[:res_len]

    ### for debug...
    ### get full kv from kv_store by generating indices of range(offloaded_len)

    def restore_loaded_keys(self, queried_key, layer_idx, request_id):
        """Restore any policy-specific transform applied before storage."""


    def record_initial_persist_selection(self, layer_idx, query_states, prefix_len):
        """Select a separate initial layout without changing attention's KV set."""
        if getattr(self, 'hot_growth_config', None) is not None:
            if self.persisted:
                return
            if not hasattr(self, '_initial_hot_growth_queries'):
                self._initial_hot_growth_queries = {}
            if layer_idx not in self._initial_hot_growth_queries:
                # Only the small suffix Q is retained. Full scoring is deferred
                # until finish_one_query, outside the measured frontend prefill.
                self._initial_hot_growth_queries[layer_idx] = (
                    query_states.detach().clone(), prefix_len)
            return

    def reorder_persist(self, prefix_id):
        """Directly persist initial important/unimportant token partitions."""
        self.wait_for_reorder()
        config = getattr(self, 'hot_growth_config', None)
        queries = getattr(self, '_initial_hot_growth_queries', {})
        selections = getattr(self, '_initial_persist_results', {})
        if config is not None and queries:
            from flexllmgen.hot_growth import initial_attention_topk
            if set(queries) != set(range(self.num_layers)):
                raise ValueError('Incomplete first-request queries for hot growth initialization')
            if not hasattr(self.kv_store, 'get_initial_key_view'):
                raise RuntimeError('Rebuild library/kvstore for first-request hot growth initialization')
            selections = {}
            budget = config.initial_count(self.offload_len, self.selection_retention_ratio)
            ranking_started = time.perf_counter()
            for layer in range(self.num_layers):
                query, prefix_len = queries[layer]
                if prefix_len != self.offload_len:
                    raise ValueError('Hot growth initialization requires the complete stored prefix')
                selections[layer] = initial_attention_topk(
                    query, self.kv_store.get_initial_key_view(layer), budget)
                del queries[layer]
            self._initial_persist_results = selections
            print(f'initial hot growth attention ranking: layers={self.num_layers}, '
                  f'tokens_per_head={budget}, seconds={time.perf_counter()-ranking_started:.6f}',
                  flush=True)
        if config is not None and not selections:
            raise ValueError('Missing initial attention ranking for hot growth; core PQ selections cannot fill B')
        if selections and len(selections) != self.num_layers:
            raise ValueError('Incomplete initial persistence selection')
        source = ([selections[i] for i in range(self.num_layers)]
                  if selections else self.query_results)
        if config is not None:
            # Full-prefill attention top-k is score ordered. Trim ceil(top-k)
            # to the exact per-head floor budget used by the native policy.
            budget = config.initial_count(self.offload_len, self.selection_retention_ratio)
            for layer, (counts, ids) in enumerate(source):
                if int(counts.min()) < budget or ids.shape[1] < budget:
                    raise ValueError(f'Initial hot growth ranking too short: layer={layer}, '
                                     f'expected={budget}, min_count={int(counts.min())}, width={ids.shape[1]}')
            source = [(counts.clamp(max=budget).cpu(), ids[:, :budget].cpu()) for counts, ids in source]
        reorder_tensors = self._build_reorder_inputs(tuple(source))
        for layer_idx, reorder_token_info in enumerate(reorder_tensors):
            self.kv_store.reorder_persist(
                self.kv_store_path,
                prefix_id,
                layer_idx,
                reorder_token_info,
            )
        self.kv_store.finish_persist()
        reorder_io_bytes = self.kv_store.get_reorder_io_bytes()
        print(
            f"initial reorder persist I/O: "
            f"{reorder_io_bytes / 1024 / 1024:.2f} MB",
            flush=True,
        )
        self.persisted = True


            #print(f"Successfully promote prefix {prefix_id} to storage in {self.kv_store_path}")

    def _build_reorder_input(self, nnz, results):
        """Build one layer's promotions in head-major order."""
        counts = nnz[:self.num_key_value_heads].to(torch.long)
        max_count = int(counts.max().item()) if counts.numel() else 0
        if max_count == 0:
            return torch.empty((0, 2), dtype=torch.long, device="cpu")

        positions = torch.arange(max_count, dtype=torch.long).unsqueeze(0)
        valid = positions < counts.unsqueeze(1)
        head_ids = torch.arange(
            self.num_key_value_heads, dtype=torch.long).unsqueeze(1)
        head_ids = head_ids.expand(-1, max_count)[valid]
        token_ids = results[
            :self.num_key_value_heads, :max_count][valid].to(torch.long)
        return torch.stack((head_ids, token_ids), dim=1)

    def _build_reorder_inputs(self, query_results):
        # gc.get_objects() can retain an in-progress tuple while tensor ops
        # release the GIL. CPython 3.8 then rejects _PyTuple_Resize. Build a
        # list first so tuple construction needs neither resizing nor Python
        # callbacks that let the memory-statistics thread observe it.
        return tuple([
            self._build_reorder_input(nnz, results)
            for nnz, results in query_results])

    def _snapshot_reorder_query_results(self):
        """Freeze only observable selection data before request reset."""
        snapshots = []
        for nnz, results in self.query_results:
            counts = nnz[:self.num_key_value_heads].clone()
            max_count = int(counts.max().item()) if counts.numel() else 0
            selected = results[
                :self.num_key_value_heads, :max_count].clone()
            snapshots.append((counts, selected))
        return tuple(snapshots)


    def _initialize_reorder_worker(self):
        # Keep at most one running job and one mergeable pending batch. This
        # bounds memory and prevents stale per-request work from growing an
        # unbounded FIFO when foreground inference is faster than reordering.
        self._reorder_condition = threading.Condition()
        self._reorder_running = None
        self._reorder_pending = None
        self._reorder_completed = []
        self._reorder_failure = None
        self._reorder_shutdown = False
        self._hot_growth_foreground = False
        self._hot_growth_cancelled = False
        self._next_reorder_job_id = 0
        self._reorder_worker = threading.Thread(
            target=self._reorder_worker_loop,
            name="kv-reorder",
            daemon=True)
        self._reorder_worker.start()

    def _execute_reorder_job(self, batch):
        return self._execute_hot_growth_job(batch)

    def set_hot_growth_foreground(self, active):
        if getattr(self, 'hot_growth_config', None) is None:
            return
        with self._reorder_condition:
            self._hot_growth_foreground = active
            self.kv_store.set_hot_growth_foreground(active)
            self._reorder_condition.notify_all()

    def _execute_hot_growth_job(self, batch):
        started = time.perf_counter()
        if not getattr(self, '_hot_growth_niced', False):
            try:
                os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
            except (AttributeError, OSError):
                pass  # Affinity and foreground priority remain effective.
            self._hot_growth_niced = True
        # There is only one compact snapshot in a pending growth job. A slow
        # worker never replays an unbounded backlog between frontend requests.
        query_results = batch['query_result_batches'][0]
        selected_count = 0
        totals = {}
        request = batch['request_ids'][0]
        for layer, (counts, ids) in enumerate(query_results):
            with self._reorder_condition:
                while self._hot_growth_foreground and not self._hot_growth_cancelled:
                    self._reorder_condition.wait()
                if self._hot_growth_cancelled:
                    totals['cancelled_layers'] = self.num_layers - layer
                    break
            selected = self._build_reorder_input(counts, ids)
            selected_count += selected.size(0)
            stats = self.kv_store.hot_growth_update(
                self.kv_store_path, batch['prefix_id'], layer, selected, request)
            for key, value in stats.items():
                if key.startswith('max_'):
                    totals[key] = max(totals.get(key, 0), value)
                else:
                    totals[key] = totals.get(key, 0) + value
        finished = time.perf_counter()
        total_bytes = self.num_layers * self.num_key_value_heads * self.offload_len * 4 * self.head_dim
        totals.update(prefix_id=batch['prefix_id'], request_id=request,
                      dropped_pending_requests=batch.get('dropped_pending_requests', 0),
                      elapsed_seconds=finished-started)
        print('HOT_GROWTH ' + json.dumps(totals), flush=True)
        # Keep native cumulative accounting consistent with existing strategies.
        io_bytes = self.kv_store.get_reorder_io_bytes()
        return dict(job_id=batch['job_id'], submission_ids=batch['submission_ids'],
                    request_ids=batch['request_ids'], merged_requests=1,
                    selected_promotions=selected_count,
                    queue_wait=started-batch['submitted_at'], duration=finished-started,
                    observer_seconds=0., execution_seconds=finished-started,
                    total_latency=finished-batch['submitted_at'], io_bytes=io_bytes,
                    total_prefix_kv_bytes=total_bytes,
                    io_ratio=io_bytes/total_bytes if total_bytes else 0.)

    def _reorder_worker_loop(self):
        if self._reorder_cpus is not None:
            # RawIoUring is thread-local and is first constructed below this
            # point, after the reorder thread has received its own CPU mask.
            os.sched_setaffinity(0, self._reorder_cpus)
        while True:
            with self._reorder_condition:
                while (self._reorder_pending is None and
                       not self._reorder_shutdown):
                    self._reorder_condition.wait()
                if self._reorder_shutdown:
                    return
                batch = self._reorder_pending
                self._reorder_pending = None
                self._reorder_running = batch

            try:
                result = self._execute_reorder_job(batch)
            except BaseException as error:
                with self._reorder_condition:
                    self._reorder_failure = error
                    self._reorder_running = None
                    self._reorder_pending = None
                    self._reorder_shutdown = True
                    self._reorder_condition.notify_all()
                return

            with self._reorder_condition:
                self._reorder_completed.append(result)
                self._reorder_running = None
                self._reorder_condition.notify_all()


    def _merge_reorder_batches(self, pending, incoming):
        if pending['prefix_id'] != incoming['prefix_id']:
            raise RuntimeError('Cannot merge hot growth work across prefixes')
        incoming['dropped_pending_requests'] = pending.get('dropped_pending_requests', 0) + 1
        return incoming

    @staticmethod
    def _print_reorder_result(result):
        print(
            "async incremental reorder complete: "
            f"job={result['job_id']}, requests={result['request_ids']}, "
            f"merged requests={result['merged_requests']}, "
            f"selected promotions={result['selected_promotions']}, "
            f"queue wait={result['queue_wait']:.6f} s, "
            f"duration={result['duration']:.6f} s, "
            f"total latency={result['total_latency']:.6f} s, "
            f"I/O={result['io_bytes'] / 1024 / 1024:.2f} MB, "
            f"I/O ratio={result['io_ratio']:.4f}",
            flush=True)

    def _reap_finished_reorders(self):
        with self._reorder_condition:
            completed = self._reorder_completed
            self._reorder_completed = []
            failure = self._reorder_failure
        for result in completed:
            self._print_reorder_result(result)
        if failure is not None:
            raise RuntimeError(
                "Background incremental KV reorder failed") from failure


    def incremental_reorder_async(self, prefix_id, request_id=None, hot_growth=False):
        self._reap_finished_reorders()
        query_results = self._snapshot_reorder_query_results()
        job_id = self._next_reorder_job_id
        self._next_reorder_job_id += 1
        submitted_at = time.perf_counter()
        incoming = {
            "hot_growth": hot_growth,
            "hot_stability": False,
            "frequency": False,
            "conservative": False,
            "job_id": job_id,
            "submission_ids": [job_id],
            "request_ids": [request_id],
            "prefix_id": prefix_id,
            "query_result_batches": [query_results],
            "submitted_at": submitted_at,
            "last_merged_at": submitted_at,
        }
        with self._reorder_condition:
            if self._reorder_shutdown:
                raise RuntimeError("Background reorder worker is not running")
            if self._reorder_pending is None:
                self._reorder_pending = incoming
                disposition = "queued"
            else:
                pending_job_id = self._reorder_pending["job_id"]
                self._reorder_pending = self._merge_reorder_batches(
                    self._reorder_pending, incoming)
                disposition = f"merged into pending job {pending_job_id}"
            running = self._reorder_running is not None
            pending_requests = len(self._reorder_pending["request_ids"])
            self._reorder_condition.notify()
        print(
            f"async incremental reorder submitted: submission={job_id}, "
            f"request={request_id}, disposition={disposition}, "
            f"running={int(running)}, pending batches=1, "
            f"pending requests={pending_requests}",
            flush=True)

    def report_reorder_status(self, request_id=None):
        self.set_hot_growth_foreground(True)
        self._reap_finished_reorders()
        with self._reorder_condition:
            running = self._reorder_running is not None
            pending = self._reorder_pending is not None
            pending_requests = (
                len(self._reorder_pending["request_ids"])
                if pending else 0)
        print(
            f"request start reorder status: request={request_id}, "
            f"running={int(running)}, pending batches={int(pending)}, "
            f"pending requests={pending_requests}",
            flush=True)

    def wait_for_reorder(self):
        self.set_hot_growth_foreground(False)
        with self._reorder_condition:
            while ((self._reorder_running is not None or
                    self._reorder_pending is not None) and
                   self._reorder_failure is None):
                self._reorder_condition.wait()
        self._reap_finished_reorders()

    def _cancel_hot_growth(self):
        """Drop obsolete work, wait only for the current native layer to yield."""
        with self._reorder_condition:
            self._hot_growth_cancelled = True
            self._hot_growth_foreground = False
            self.kv_store.set_hot_growth_foreground(True)
            self._reorder_pending = None
            self._reorder_condition.notify_all()
            while self._reorder_running is not None and self._reorder_failure is None:
                self._reorder_condition.wait()
        self._reap_finished_reorders()

    def shutdown_reorder_worker(self):
        if self._reorder_worker is None:
            return
        wait_error = None
        try:
            if getattr(self, 'hot_growth_config', None) is not None:
                self._cancel_hot_growth()
            else:
                self.wait_for_reorder()
        except BaseException as error:
            wait_error = error
        with self._reorder_condition:
            self._reorder_shutdown = True
            self._reorder_condition.notify_all()
        self._reorder_worker.join()
        self._reorder_worker = None
        if wait_error is not None:
            raise wait_error
    def reset(self, switch=False):
        self._initial_persist_results = {}
        self._initial_hot_growth_queries = {}
        if switch:
            # Prefix switches occur after request synchronization. Release
            # pooled buffers so long prefixes do not inflate later residency.
            self._prefix_workspace = None
            # Do not replace the KVStore or its files while background reorder
            # is still using the old instance.
            if getattr(self, 'hot_growth_config', None) is not None:
                self._cancel_hot_growth()
                self._hot_growth_cancelled = False
            else:
                self.wait_for_reorder()
            self.reset_policy()
            del self.kv_store
            self.kv_store = KVStore()
            self.kv_store.alloc(self.num_layers, self.num_attention_heads, self.num_key_value_heads, self.head_dim, self.max_length)
            self.kv_store.set_merge_threshold_bytes(self.merge_threshold_bytes)
            self.kv_store.set_merge_extra_read_budget(self.merge_extra_read_budget)
            if (getattr(self, "hot_merge_extra_read_budget", None) is not None or
                    getattr(self, "cold_merge_extra_read_budget", None) is not None):
                self.configure_region_merge_budgets(
                    self.hot_merge_extra_read_budget, self.cold_merge_extra_read_budget)
            self.kv_store.set_compaction_threshold(self.compaction_threshold)
            config = getattr(self, 'hot_growth_config', None)
            if config is not None:
                config.configure_store(self.kv_store, self.selection_retention_ratio)
            self.configure_kv_cache(self.kv_cache_capacity_bytes, 'access_control')
            self.persisted = False

        if hasattr(self.kv_store, "get_read_profile"):
            self.kv_store.get_read_profile()  # Discard unreported warmup/request counters.
        if hasattr(self.kv_store, "get_merge_read_stats"):
            self.kv_store.get_merge_read_stats()
        print("req done, sum of io count: ", self.kv_store.get_num_io())
        self.nnz.zero_()
        self.selection_results.zero_()
        self.clear_policy_buffers()
        self.query_results = [
            (torch.zeros_like(self.nnz), torch.zeros_like(self.selection_results))
            for _ in range(self.num_layers)]

    def reset_policy(self):
        raise NotImplementedError

    def clear_policy_buffers(self):
        """Clear request-local metadata buffers."""


    ### TODO ----- handle recover
    def persist_kv_store_meta(self, prefix_id):
        self.kv_store.persist_meta(prefix_id)

    def recover_kv_store_meta(self, prefix_id):
        self.offloaded = True
        self.kv_store.recover_meta(self.kv_store_path, prefix_id)


    def persist_queried_result(self, prefix_id):
        for layer_idx in range(self.num_layers):
            nnz, results = self.query_results[layer_idx]
            file_path = os.path.join(self.kv_store_path, f"queried_results_prefix_{prefix_id}_layer_{layer_idx}.pt")
            data_to_save = {
                'nnz': nnz,
                'results': results
            }
            torch.save(data_to_save, file_path)


    def recover_queried_result(self, prefix_id):
        for layer_idx in range(self.num_layers):
            file_path = os.path.join(self.kv_store_path, f"queried_results_prefix_{prefix_id}_layer_{layer_idx}.pt")
            if os.path.exists(file_path):
                data_loaded = torch.load(file_path, map_location="cpu")
                nnz = data_loaded['nnz']
                results = data_loaded['results']
                # print(f"--- Debug: Recovering Layer {layer_idx} ---")
                # print(f"  Loaded nnz (shape: {nnz.shape}):")
                # print(nnz)
                # print(f"  Loaded results (shape: {results.shape}):")
                # print(results)
                self.query_results[layer_idx] = (nnz.to(self.nnz.device), results.to(self.selection_results.device))
            else:
                print(f"File {file_path} does not exist. Cannot recover queried results for layer {layer_idx}.")

    def prepare_retrieval(self, prefix_id):
        """Prepare policy metadata for one request without touching KV data."""
        raise NotImplementedError
