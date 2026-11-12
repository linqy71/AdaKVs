"""AdaKVs OPT runtime, migrated from the FlexLLMGen retrieval engine."""

import argparse
import dataclasses
import faulthandler
from functools import partial
import os
import pickle
import sys
import time
import json
from pathlib import Path
from typing import Union, List, Optional

# Preserve all Python thread stacks on native faults, including SIGBUS.
# Enable before importing any native extension or starting background workers.
faulthandler.enable(all_threads=True)

# Import the affinity-aware base before importing PyTorch. When configured,
# the main thread receives its foreground mask before worker pools are created.
from flexllmgen.hot_growth import from_arguments as hot_growth_from_arguments

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import gc

from flexllmgen.compression import CompressionConfig
from flexllmgen.opt_config import (
    OptConfig, get_opt_config, download_opt_weights, add_opt_context_arguments,
    get_opt_position_weight_path, validate_opt_context_length)
from flexllmgen.pytorch_backend_adakvs import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
from flexllmgen.timer import timers
from flexllmgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)

from flexllmgen.metadata_manage import RadixTree, RadixTreeNode, RadixToken, CachePointer
fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes


def _start_gpu_stage(name, fast, sync):
    if fast:
        timers(name).start_cuda()
    else:
        timers(name).start(sync)


def _stop_gpu_stage(name, fast, sync):
    if fast:
        timers(name).stop_cuda()
    else:
        timers(name).stop(sync)


def create_retrieval_server(config, num_layers, kv_store_path, *, K, L, compaction_threshold, pq_num_subvectors, pq_num_bits, pq_retention_ratio, pq_query_chunk_size, pq_kmeans_max_iter, pq_kmeans_seed, merge_threshold_bytes=4096, merge_extra_read_budget=None, kv_cache_capacity_bytes=0, hot_merge_extra_read_budget=None, cold_merge_extra_read_budget=None):
    from flexllmgen.pqcode_shared_server import PQCodeSharedServer
    server = PQCodeSharedServer(
        config, num_layers, kv_store_path, max_length=config.max_seq_len,
        num_subvectors=pq_num_subvectors, num_bits=pq_num_bits,
        retention_ratio=pq_retention_ratio, query_chunk_size=pq_query_chunk_size,
        kmeans_max_iter=pq_kmeans_max_iter, kmeans_seed=pq_kmeans_seed,
        compaction_threshold=compaction_threshold,
        merge_threshold_bytes=merge_threshold_bytes,
        merge_extra_read_budget=merge_extra_read_budget)
    server.attention_kernel_K, server.attention_kernel_L = K, L
    if hot_merge_extra_read_budget is not None or cold_merge_extra_read_budget is not None:
        server.configure_region_merge_budgets(hot_merge_extra_read_budget, cold_merge_extra_read_budget)
    server.configure_kv_cache(kv_cache_capacity_bytes, 'access_control')
    return server

# torch.cuda.set_device(0)
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from collections import defaultdict
import logging

logging.basicConfig(#filename="test.log", filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%m-%d %H:%M:%S", level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    cpu_cache_compute: bool

    # Sparsity of attention weights
    attn_sparsity: float

    # Compress weights with group-wise quantization
    compress_weight: bool
    comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    compress_cache: bool
    comp_cache_config: CompressionConfig

    # the ratio of important tokens in prefix kv cache
    important_ratio: float = 0.3

    # Config of Chunk Pool
    chunk_size: int = 4
    #gpu_heap_size: int = 0
    #cpu_heap_size: int = 0

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
    return ret


class InputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def set_kv_server(self, kv_server):
        pass

    def init_weight(self, weight_home, path):
        v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.max_seq_len, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
            # w_pos
            ((s + 2, h), dtype, get_opt_position_weight_path(self.config, path)),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token, w_pos = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst), w_pos.smart_copy(dst)))

    def init_cache_one_gpu_batch(self, cache_home, max_prompt_len, max_gen_len):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i, j):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, j):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        h = self.compute.opt_input_embed(h, mask,
            w_token, w_pos, self.config.pad_token_id, donate)
        hidden.val = h


class OutputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def set_kv_server(self, kv_server):
        pass

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "decoder.layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "decoder.layer_norm.bias"),
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, b_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2), b_ln.smart_copy(dst2),
                w_token.smart_copy(dst1)))

    def init_cache_one_gpu_batch(self, cache_home, max_prompt_len, max_gen_len):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i, j):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, j):
        donate = [False] * 4
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (b_ln, donate[2]), (w_token, donate[3]) = weight_read_buf.pop()
        else:
            (w_ln, _), (b_ln, _), (w_token, _) = weight_read_buf.val

        h = self.compute.opt_output_embed(h, w_ln, b_ln, w_token, donate,
            self.task.do_sample, self.task.temperature)
        hidden.val = h


class SelfAttention:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)

        self.task = None
        self.copy_stream = torch.cuda.Stream(priority=-1)
        # self.cpu2gpu_stream = torch.cuda.Stream()
        self.prefill_cache_shape = 0 #prefill阶段的cache第一维长度，可能是n_imp + NR(jaccard超过threshold)  可能是R+NR(没有超过threshold)

    def set_task(self, task):
        self.task = task

    def set_kv_server(self, kv_server):
        self.kv_server = kv_server

    def _sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()


    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + ".q_proj.weight"),
            # b_q
            ((h,), dtype, path + ".q_proj.bias"),
            # w_k
            ((h, h), dtype, path + ".k_proj.weight"),
            # b_k
            ((h,), dtype, path + ".k_proj.bias"),
            # w_v
            ((h, h), dtype, path + ".v_proj.weight"),
            # b_v
            ((h,), dtype, path + ".v_proj.bias"),
            # w_out
            ((h, h), dtype, path + ".out_proj.weight"),
            # b_out
            ((h,), dtype, path + ".out_proj.bias"),
            # w_ln
            ((h,), dtype, path + "_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_out.smart_copy(dst1), b_out.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))
        if not getattr(self.kv_server, "fast_retrieval", False):
            typical_shapes = [(1, 1024, 4096, 128)]
            self.compute.warmup_gpu(typical_shapes)

    def init_cache_one_gpu_batch(self, cache_home, max_prompt_len, max_gen_len):
        if (max_gen_len == 1
                and True
                and os.environ.get("FLEXGEN_RETRIEVAL_PIPELINE", "1") != "0"):
            # Prefix warmup offloads its computed K/V directly to KVStore;
            # measured TTFT requests never read or write a decoding cache.
            return
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy, max_prompt_len, max_gen_len)
        cache_home.store(cache)

    def alloc_prefix_kv(self, cache_len):
        n_head = self.config.n_head
        batch_size = self.policy.gpu_batch_size
        head_dim = self.config.input_dim // n_head

        if cache_len <= 0:
            raise ValueError("prefix KV cache length must be positive")
        dst = self.attention_compute
        shape = (cache_len, batch_size * n_head, head_dim)

        pin_memory = True if dst.device_type == DeviceType.CPU else False

        k_cache = dst.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = dst.allocate(shape, np.float16, pin_memory=pin_memory)

        return k_cache, v_cache

    ### Fetch selected prefix KV and copy before the next server load.
    ### todo: manage the shape of k_cache_data
    ### dst: TorchTensor, src: torch.Tensor
    ### copy src[0:length] to dst[start:start+length]
    def copy_prefix(self, dst, src, start):
        ### copy
        assert(dst.device.device_type == DeviceType.CPU or DeviceType.CUDA)
        length = src.shape[0]
        dst = dst.data[start: start + length]
        dst.copy_(src, non_blocking=True)
        return length


    def get_prefix_kv(self, imp_token_idx, layer):
        n_head = self.config.n_head
        batch_size = self.policy.gpu_batch_size
        head_dim = self.config.input_dim // n_head

        n_important = len(imp_token_idx) # 如果超过threshold长度就是重要token个数，如果没超过就是common_prefix_len
        dst = self.attention_compute
        shape = (n_important, batch_size * n_head, head_dim)

        pin_memory = True if dst.device_type == DeviceType.CPU else False

        k_cache = dst.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = dst.allocate(shape, np.float16, pin_memory=pin_memory)

        for j in range(n_important):
            kv_ptr = self.task.common_prefix_kv_ptr[imp_token_idx[j]][layer]
            chunk_id, offset = kv_ptr.chunk_id, kv_ptr.offset
            logger.info(f"Get prefix kv: chunk_id={chunk_id}, offset={offset}")
            self.chunk_pool.get_full_head_cache(k_cache, v_cache, j, chunk_id, offset)

        return k_cache, v_cache

    def load_cache(self, cache_home, cache_read_buf, i, j):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, self.task.prompt_len + i),
                       slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")

    # Prefix-only construction stores KV through the retrieval server.
    ### not prefix_only: store important_kv+non_prefix_kv to cache_home
    def store_cache(self, cache_home, cache_write_buf, i):
        # shape: (s, b * n_head, head_dim)
        k_new, v_new = cache_write_buf.pop()
        seq_len, _, _ = k_new.shape

        if self.task.prefix_only:
            print(
                f"offloading prefix {self.task.new_prefix_id} to "
                f"{'pqcode_shared'}")
            self.kv_server.offload(
                self.layer_id, 0, seq_len, self.task.new_prefix_id,
                k_new.data, v_new.data)
            return

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        k_home, v_home = cache_home.val
        if i == 0:  # prefix prefill
            # prefill阶段整个cache存入
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            # 在原先cache的基础上拼接
            # pos = self.task.prompt_len + i
            pos = self.prefill_cache_shape + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, j):
        n_head = self.config.n_head

        donate = [False] * 14
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
             (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
             (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop()
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (b_out, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            if not self.task.prefix_only:
                ### should get important kv first

                matched_prefix = self.task.matched_prefix
                k_cache = v_cache = None
                workspace = None
                fast = getattr(self.kv_server, "fast_retrieval", False)

                cur_pos = 0
                _start_gpu_stage("imp calc", fast, self._sync)
                query_states, suffix_hidden = (
                    self.compute.prepare_suffix_query_states(
                        h, mask, w_q, b_q, w_ln, b_ln, n_head,
                        matched_prefix))
                _stop_gpu_stage("imp calc", fast, self._sync)

                for prefix_id, max_common_len in matched_prefix.items():
                    ## j is layer_id
                    # Retrieval implementations synchronize only the work
                    # needed by their own selector. A global disk/CUDA sync at
                    # this boundary also waits for unrelated pipeline work and
                    # substantially inflates the reported retrieval time.
                    timers("retrieval").start()
                    self.kv_server.retrieve(
                        self.task.req_id, self.layer_id, query_states,
                        prefix_id, max_common_len,
                        self.task.save_res)
                    # A separate selector pass changes only the initial disk
                    # partition; retrieval scratch and attention remain intact.
                    self.kv_server.record_initial_persist_selection(
                        self.layer_id, query_states, max_common_len)
                    timers("retrieval").stop()

                    if fast:
                        timers("imp load and compute").start()
                        workspace = self.kv_server.load_kv(
                            0, self.layer_id, prefix_id,
                            suffix_len=self.task.prompt_len - max_common_len)
                        timers("imp load and compute").stop()
                        continue
                    correction = getattr(self.kv_server, 'online_correction', None)
                    timers("imp load and compute").start(self._sync)

                    k_cache_data, v_cache_data = self.kv_server.load_kv(
                        0, self.layer_id, prefix_id)
                    timers("imp load and compute").stop(self._sync)
                    if k_cache is None:
                        cache_len = (k_cache_data.shape[0] if len(matched_prefix) == 1
                                     else sum(matched_prefix.values()))
                        k_cache, v_cache = self.alloc_prefix_kv(cache_len)

                    # k_cache_data, v_cache_data = self.kv_server.get_full_kv(0, self.layer_id, query_states, prefix_id)
                    # print(k_cache_data)
                    timers("copy prefix").start(self._sync)
                    with torch.cuda.stream(self.copy_stream):
                        # k_cache_data, v_cache_data = self.kv_server.get_full_kv(0, j, query_states, prefix_id)
                        length = self.copy_prefix(k_cache, k_cache_data, cur_pos)
                        length = self.copy_prefix(v_cache, v_cache_data, cur_pos)
                    self.copy_stream.synchronize()
                    timers("copy prefix").stop(self._sync)
                    cur_pos += length
                # n_imp = cur_pos
                ### kv_server的layer统一用layer_id管理
                imp_token_idx, _ = self.kv_server.get_imp_idx(self.layer_id)
                # imp_token_idx = self.kv_server.get_full_idx(self.layer_id)
                # print(imp_token_idx[:3])
                _start_gpu_stage("compute", fast, self._sync)
                h, new_k_cache, new_v_cache = self.compute.mha_prefill_with_kv(h, mask, w_q, b_q,
                    w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                    self.policy.compress_cache, self.policy.comp_cache_config, matched_prefix,
                    imp_token_idx, self.kv_server.attention_kernel_K,
                    self.kv_server.attention_kernel_L,
                    prepared_query_states=query_states,
                    prepared_suffix_hidden=suffix_hidden,
                    kv_workspace=workspace,
                    **({'attention_observer': self.kv_server.correction_callback(self.layer_id)}
                       if getattr(self.kv_server, 'online_correction', None) is not None else {}))
                _stop_gpu_stage("compute", fast, self._sync)
                if workspace is not None:
                    self.kv_server._prefix_workspace.release(workspace)
                self.prefill_cache_shape = new_k_cache.shape[0]
                # logger.info(f"SelfAttention Prefix cache shape: {self.prefill_cache_shape}")
            else:
                # The first request has no known reusable prefix yet. Compute
                # the normal full prefill and record attention-based token
                # importance while the attention probabilities are resident.
                topk_ratio = self.kv_server.initial_attention_topk_ratio
                score_observer = getattr(self.kv_server, "initial_score_observer", None)
                result = self.compute.mha(
                    h, mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out,
                    w_ln, b_ln, n_head, donate,
                    self.policy.compress_cache,
                    self.policy.comp_cache_config,
                    self.kv_server.attention_kernel_K,
                    self.kv_server.attention_kernel_L,
                    attention_topk_ratio=topk_ratio,
                    prefill_query_chunk_size=getattr(
                        self.kv_server, "prefill_query_chunk_size", None),
                    **({"score_observer": partial(score_observer, self.layer_id),
                        "score_query_start": getattr(
                            self.kv_server, "initial_score_query_start", 0)}
                       if score_observer is not None else {}))
                if topk_ratio is None:
                    h, new_k_cache, new_v_cache = result
                else:
                    h, new_k_cache, new_v_cache, attention_topk = result
                    self.kv_server.record_initial_attention_topk(
                        self.layer_id, attention_topk)

                self.prefill_cache_shape = self.task.prompt_len
                # logger.info(f"SelfAttention Prefix cache shape: {self.prefill_cache_shape}")
            # 存入的cache shape可能是(s, b * n_head, head_dim) 也可能是 (n_imp + s - common_prefix_len[0], ..., ...)
            cache_write_buf.store((new_k_cache, new_v_cache))

        else:  # decoding
            imp_token_idx, _ = self.kv_server.get_imp_idx(self.layer_id)
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            # logger.info(f"SelfAttention decoding prefill_cache_shape:{self.prefill_cache_shape}, i:{i}")
            h, new_k_cache, new_v_cache = self.compute.mha_gen(h, mask, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config,pos=self.prefill_cache_shape + i, imp_token_idx=imp_token_idx)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h


class MLP:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def set_kv_server(self, kv_server):
        self.kv_server = kv_server

    def _sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
        weight_specs = [
            # wi
            ((4 * h, h), dtype, path + "fc1.weight"),
            # bi
            ((4 * h,), dtype, path + "fc1.bias"),
            # wo
            ((h, 4 * h), dtype, path + "fc2.weight"),
            # bo
            ((h,), dtype, path + "fc2.bias"),
            # w_ln
            ((h,), dtype, path + "final_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "final_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        wi, bi, wo, bo, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                wi.smart_copy(dst1), bi.smart_copy(dst2),
                wo.smart_copy(dst1), bo.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home, max_prompt_len, max_gen_len):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i, j):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, j):
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
             (w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()
        else:
            ((wi, _), (bi, _), (wo, _), (bo, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        fast = getattr(self.kv_server, "fast_retrieval", False)
        _start_gpu_stage("mlp", fast, self._sync)
        h = self.compute.mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
        _stop_gpu_stage("mlp", fast, self._sync)
        hidden.val = h


class TransformerLayer:
    def __init__(self, config, env, policy, i):
        self.attention = SelfAttention(config, env, policy, i)
        self.mlp = MLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)

    def set_kv_server(self, kv_server):
        self.attention.set_kv_server(kv_server)
        self.mlp.set_kv_server(kv_server)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, k):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.attention.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0:
            weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_gpu_batch(self, cache_home, max_prompt_len, max_gen_len):
        self.attention.init_cache_one_gpu_batch(cache_home, max_prompt_len, max_gen_len)

    def load_cache(self, cache_home, cache_read_buf, i, j):
        self.attention.load_cache(cache_home, cache_read_buf, i, j)

    def store_cache(self, cache_home, cache_write_buf, i):
        self.attention.store_cache(cache_home, cache_write_buf, i)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, j):
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val

        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, i, k, j)
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k, j)

class OptLM:
    def __init__(self, config: Union[str, OptConfig], env: ExecutionEnv, path: str, offload_dir: str, policy: Policy, max_prompt_len: int, max_gen_len: int, K: int, L: int, compaction_threshold: float=0.0, pq_num_subvectors: int=2, pq_num_bits: int=6, pq_retention_ratio: float=0.25, pq_query_chunk_size: int=32, pq_kmeans_max_iter: int=20, pq_kmeans_seed: int=0, merge_threshold_bytes: int=4096, merge_extra_read_budget: Optional[float]=None, kv_cache_capacity_bytes: int=0, hot_merge_extra_read_budget: Optional[float]=None, cold_merge_extra_read_budget: Optional[float]=None, hot_growth_config=None):
        self.num_layers = 0
        from flexllmgen.hot_growth import HotGrowthConfig
        hot_growth_config = hot_growth_config or HotGrowthConfig()
        hot_growth_config.validate(pq_retention_ratio)
        if policy.gpu_batch_size != 1 or policy.num_gpu_batches != 1 or max_gen_len != 1:
            raise ValueError('AdaKVs requires one request and one generated token')
        if policy.compress_cache or policy.compress_weight or policy.cpu_cache_compute:
            raise ValueError('AdaKVs currently uses uncompressed GPU attention')
        if policy.cache_gpu_percent != 100 or policy.attn_sparsity != 1:
            raise ValueError('AdaKVs manages prefix selection with GPU request KV')
        if isinstance(config, str):
            config = get_opt_config(config, max_seq_len=8192)
        validate_opt_context_length(max_prompt_len, max_gen_len, config.max_seq_len)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches
        self.max_prompt_len = max_prompt_len
        self.max_gen_len = max_gen_len
        self.persist_strategy = 'guarded_hot_growth'

        layers = []
        layers.append(InputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(SelfAttention(self.config, self.env, self.policy, i))
                layers.append(MLP(self.config, self.env, self.policy, i))
            else:
                layers.append(TransformerLayer(self.config, self.env, self.policy, i))
        layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)
        self.num_hidden_layers = self.config.num_hidden_layers

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.init_all_weights() # 权重全部读入weights_home

        self.radix_tree = RadixTree()
        ### default settings, note that device=cuda:6
        self.kv_store_path = os.path.join(offload_dir, "kv_store")
        if not os.path.exists(self.kv_store_path):
            os.makedirs(self.kv_store_path)
        self.kv_server = create_retrieval_server(
            self.config,
            self.num_hidden_layers,
            self.kv_store_path,
            K=K,
            L=L,
             compaction_threshold=compaction_threshold,
            merge_threshold_bytes=merge_threshold_bytes,
            merge_extra_read_budget=merge_extra_read_budget,
            kv_cache_capacity_bytes=kv_cache_capacity_bytes,
            hot_merge_extra_read_budget=hot_merge_extra_read_budget,
            cold_merge_extra_read_budget=cold_merge_extra_read_budget,

            pq_num_subvectors=pq_num_subvectors,
            pq_num_bits=pq_num_bits,
            pq_retention_ratio=pq_retention_ratio,
            pq_query_chunk_size=pq_query_chunk_size,
            pq_kmeans_max_iter=pq_kmeans_max_iter,
            pq_kmeans_seed=pq_kmeans_seed)
        self.kv_server.reorder_topk_ratio = None
        self.kv_server.hot_growth_config = hot_growth_config
        if hot_growth_config is not None:
            hot_growth_config.configure_store(self.kv_server.kv_store, pq_retention_ratio)
        self.kv_server.incremental_demotion = False
        correction = getattr(self.kv_server, "online_correction", None)
        if correction is not None:
            correction.demotion_enabled = False
        self.kv_server.initial_attention_topk_ratio = (
            self.kv_server.selection_retention_ratio)
        if hot_growth_config is not None:
            self.kv_server.initial_attention_topk_ratio = (
                hot_growth_config.initial_ratio(pq_retention_ratio))
        self.set_kv_server()

        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
                self.init_cache(j, k, max_prompt_len, max_gen_len)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy, max_gen_len, max_prompt_len)


    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def set_kv_server(self):
        for l in self.layers:
            l.set_kv_server(self.kv_server)

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "decoder.embed_positions.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_opt_weights(self.config.name, self.path)

        self.layers[j].init_weight(self.weight_home[j], expanded_path)

    def load_weight(self, i, j, k, overlap=True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    def init_cache(self, j, k, max_prompt_len, max_gen_len):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k], max_prompt_len, max_gen_len)


    def load_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        # if i == 0:  # prefill, no cache
        #     return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i, j)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i, j)

    def store_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1 and i != 0:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:  # load from the last layer
            val = self.hidden[i][j-1][k].pop().move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:  # move to home
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
            self.weight_read_buf[j], self.attention_mask[k],
            self.cache_write_buf[j][k], i, k, j)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True]) # 在后面加一排1
            return
        # prefill阶段, attention_mask把pad的部分置为false
        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)

    def generate(self,
                 inputs: Union[np.array, List[int]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 req_id: int = 0,
                 save_res: bool = False,
                 verbose: int = 0):
        validate_opt_context_length(len(inputs[0]), max_new_tokens,
                                    self.config.max_seq_len)
        matched_prefix = self.radix_tree.match(inputs[0])
        correction = getattr(self.kv_server, 'online_correction', None)
        if correction is not None:
            if max_new_tokens != 1 or len(matched_prefix) > 1:
                raise ValueError('Online correction supports one prefix and one generated token')
            if correction.pending or correction.deferred:
                raise RuntimeError('Call finish_one_query before starting the next request')
        prefix_only = False
        new_prefix_id = 0
        if len(matched_prefix) == 0:
            prefix_only = True
            new_prefix_id = self.radix_tree.insert(inputs[0])
        elif len(matched_prefix) == 1:
            new_prefix_id = list(matched_prefix.keys())[0]
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
            common_prefix_kv_ptr=None,
            common_prefix_len=None,
            matched_prefix=matched_prefix,
            prefix_only=prefix_only,
            new_prefix_id = new_prefix_id,
            req_id = req_id,
            save_res = save_res
        )
        # Narrow fast path: one dense-GPU sparse-prefill request, CPU/GPU
        # weights only. Disk/activation offload, decoding and accuracy runs
        # retain the established synchronized path.
        self.kv_server.fast_retrieval = (
            os.environ.get("FLEXGEN_RETRIEVAL_PIPELINE", "1") != "0"
            and True
            and not prefix_only and len(matched_prefix) == 1
            and max_new_tokens == 1 and cut_gen_len is None
            and debug_mode is None
            and self.policy.overlap
            and self.policy.gpu_batch_size == 1 and self.num_gpu_batches == 1
            and self.policy.w_disk_percent == 0
            and self.policy.act_gpu_percent == 100
            and not self.policy.cpu_cache_compute
            and not self.policy.compress_cache and not self.policy.compress_weight
            and not False
            and not False)
        self.kv_server.report_reorder_status(req_id)
        # logger.info(f"generate: Task={task}")
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        q_len = prompt_len - sum(matched_prefix.values())
        self.warmup_one_query(q_len)

        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()

        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        self.set_task(task)

        # Keep built PQ metadata resident between shared-prefix requests.
        if self.task.prefix_only == False:
            self.kv_server.prepare_retrieval(prefix_id=1)

        # Generate
        if debug_mode is None:
            if not overlap:
                # No overlap, easy to understand, suitable for debugging
                self.generation_loop_normal()
            else:
                # Overlap I/O and compute
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()
                else:
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch":
            # Run fewer layeres and batches for debugging
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            # No overlap, fewer batches, execution time breakdown
            self.generation_loop_debug_normal()
        else:
            raise ValueError("Invalid debug mode: {debug_mode}")

        return self.output_ids

    def warmup_one_query(self, q_len):
        self.kv_server.warmup_retrieval(q_len)

    def final_finish(self):
        self.kv_server.shutdown_reorder_worker()
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        # Delete cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()
        #self.clear_cache_files(self.task.new_prefix_id)

    def finish_one_query(self):
        self.sync()
        correction = getattr(self.kv_server, 'online_correction', None)
        self.correction_update_time = 0.
        if correction is not None:
            update_started = time.perf_counter()
            correction.commit()
            # Account for deferred GPU work here, not in next request's TTFT.
            self.sync()
            self.correction_update_time = time.perf_counter() - update_started
        self.kv_server.set_hot_growth_foreground(False)
        self.reorder_sync_wait_time = 0.
        self.reorder_execution_mode = (
            "sync" if os.environ.get("KVSTORE_REORDER_SYNC", "0") == "1" else "async")
        if not self.kv_server.persisted:
            timer = timers("kv first persist")
            print(
                "Initial KV persist started: "
                f"request={self.task.req_id}, "
                f"prefix={self.task.new_prefix_id}, "
                f"strategy={'guarded_hot_growth'}",
                flush=True,
            )
            timer.start()
            try:
                self.kv_server.reorder_persist(
                    self.task.new_prefix_id)
            finally:
                timer.stop()
            duration = timer.costs[-1]
            print(
                "Initial KV persist completed: "
                f"request={self.task.req_id}, "
                f"prefix={self.task.new_prefix_id}, "
                f"strategy={'guarded_hot_growth'}, "
                f"duration={duration:.6f} s",
                flush=True,
            )
        elif self.task.req_id >= 1:
            timer = timers("kv incremental reorder")
            timer.start()
            try:
                self.kv_server.incremental_reorder_async(
                    self.task.new_prefix_id, self.task.req_id,
                    **({"hot_growth": True}))
            finally:
                timer.stop()
            if self.reorder_execution_mode == "sync":
                wait_started = time.perf_counter()
                self.kv_server.wait_for_reorder()
                self.reorder_sync_wait_time = time.perf_counter() - wait_started
        self.kv_server.reset(switch=False)
        logger.info("query finished , now sync the model")
        if self.kv_server.prefix_lens:
            avg_prefix = (
                sum(self.kv_server.prefix_lens) /
                len(self.kv_server.prefix_lens))
            avg_imp = (
                sum(self.kv_server.imp_lens) /
                len(self.kv_server.imp_lens))
            print(
                f"avg common len: {avg_prefix}; avg imp len: {avg_imp}; "
                f"retention: {avg_imp / avg_prefix}")


    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start(self.sync)
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
            timers("generate").stop(self.sync)
            if self.task.stop is not None and np.all(self.stopped):
                break

    def generation_loop_debug_normal(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()

        timers("load_weight").reset()
        timers("load_cache_prefill").reset()
        timers("load_cache_decoding").reset()
        timers("store_cache_prefill").reset()
        timers("store_cache_decoding").reset()
        timers("compute_layer_prefill").reset()
        timers("compute_layer_decoding").reset()
        load_weight_timer = timers("load_weight")

        for i in range(self.execute_gen_len):
            if i == 0:
                timers("prefill_total").start()
                load_cache_timer = timers("load_cache_prefill")
                store_cache_timer = timers("store_cache_prefill")
                compute_layer_timer = timers("compute_layer_prefill")
            else:
                load_cache_timer = timers("load_cache_decoding")
                store_cache_timer = timers("store_cache_decoding")
                compute_layer_timer = timers("compute_layer_decoding")

            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)

            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()

                load_weight_timer.start(self.sync)
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k)
                load_weight_timer.stop(self.sync)

                for k in range(self.num_gpu_batches):
                    load_cache_timer.start(self.sync)
                    self.load_cache(i, j, k)
                    load_cache_timer.stop(self.sync)
                    self.load_hidden(i, j, k)
                    compute_layer_timer.start(self.sync)
                    self.compute_layer(i, j, k)
                    compute_layer_timer.stop(self.sync)
                    self.store_hidden(i, j, k)
                    store_cache_timer.start(self.sync)
                    self.store_cache(i, j, k)
                    store_cache_timer.stop(self.sync)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill_total").stop(self.sync)

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill_total").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

        # Debug the costs of individual functions
        print(f"#layers: {self.num_layers}")

        print(f"#batches prefill:  "
              f"{self.num_layers * self.num_gpu_batches}")
        print(f"#batches decoding: "
              f"{(self.task.gen_len - 1) * self.num_layers * self.num_gpu_batches}")
        print(f"load_weight            (per-layer)"
              f": {np.mean(timers('load_weight').costs):.6f} s")
        for stage in ["prefill", "decoding"]:
            for func in ["load_cache", "store_cache", "compute_layer"]:
                name = func + "_" + stage
                costs = timers(name).costs
                print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s")

    def generation_loop_overlap_single_batch(self):
        if getattr(self.kv_server, "fast_retrieval", False):
            return self.generation_loop_retrieval_pipeline()
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_retrieval_pipeline(self):
        """Prefetch weights without globally waiting for the next layer."""
        def record_consumption(value, stream):
            # Copied weights were allocated on the transfer stream. Keep
            # their storage alive until the compute stream finishes using it.
            if isinstance(value, ValueHolder):
                record_consumption(value.val, stream)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    record_consumption(item, stream)
            else:
                data = getattr(value, "data", None)
                if isinstance(data, torch.Tensor) and data.is_cuda:
                    data.record_stream(stream)

        self.load_weight(0, 0, 0)
        self.sync()  # Same prologue boundary as the original TTFT timer.
        stream = torch.cuda.current_stream()
        ready = [None] * self.num_layers
        timers("generate").start()
        self.update_attention_mask(0, 0)
        for j in range(self.num_layers):
            if j + 1 < self.num_layers:
                self.load_weight(0, j + 1, 0)
                ready[j + 1] = torch.cuda.Event()
                ready[j + 1].record(self.load_weight_stream)
            if ready[j] is not None:
                stream.wait_event(ready[j])
            record_consumption(self.weight_read_buf[j], stream)
            self.load_hidden(0, j, 0)
            self.compute_layer(0, j, 0)
            # TTFT-only request: discard transient suffix KV, no persistence
            # or asynchronous cache writer consumes these workspace views.
            self.cache_write_buf[j][0].pop()
            self.store_hidden(0, j, 0)
        timers("generate").stop(self.sync)

    def generation_loop_overlap_multi_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

    def generation_loop_debug_single_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def generation_loop_debug_multi_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def __del__(self):
        self.delete_all_weights()
