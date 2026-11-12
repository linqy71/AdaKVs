"""Implement tensor computations with pytorch."""
from enum import Enum, auto
from functools import partial
from itertools import count
import math
import os
import queue
import shutil
import time
import threading
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from flexllmgen.timer import timers

import logging

logging.basicConfig(#filename="test.log", filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%m-%d %H:%M:%S", level=logging.DEBUG)
logger = logging.getLogger(__name__)

from flexllmgen.utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)

general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None
# torch.cuda.set_device(6)


def attention_mass_topk(attention_weights, retention_ratio):
    """Select tokens receiving the most aggregate full-prefill attention."""
    if attention_weights.ndim != 3:
        raise ValueError(
            "attention_weights must have shape [heads, queries, tokens]")
    if not 0 < retention_ratio <= 1:
        raise ValueError("retention_ratio must be in (0, 1]")
    token_count = attention_weights.shape[2]
    if token_count <= 0:
        raise ValueError("attention_weights must contain at least one token")
    topk_count = min(
        token_count, max(1, math.ceil(token_count * retention_ratio)))
    timer = timers("initial attention topk") if attention_weights.is_cuda else None
    if timer is not None:
        timer.start_cuda()
    attention_scores = attention_weights.sum(dim=1)
    indices = torch.topk(attention_scores, k=topk_count, dim=1).indices
    if timer is not None:
        timer.stop_cuda()
    return indices


def chunked_causal_prefill(q, k, v, padding_mask, n_head, chunk_size,
                          topk_ratio=None, score_observer=None,
                          score_query_start=0):
    """Full causal attention with bounded query-axis workspace.

    Q is already scaled. Match the legacy -1e4 mask (including padded query
    rows), and optionally accumulate initial attention importance in FP32.
    Importance only sums query rows at or after score_query_start; the attention
    output and each row's causal softmax still cover the normal full prompt.
    No attention probability tensor spanning all queries is retained.
    """
    if chunk_size <= 0:
        raise ValueError("Prefill query chunk size must be positive")
    if topk_ratio is not None and not 0 < topk_ratio <= 1:
        raise ValueError("Attention top-k ratio must be in (0, 1]")
    heads, length, dim = q.shape
    if not 0 <= score_query_start < length:
        raise ValueError("Score query start must be within the prompt")
    batch = heads // n_head
    output = torch.empty_like(q)
    scores = None
    topk_timer = timers("initial attention topk") if q.is_cuda else None
    if topk_ratio is not None or score_observer is not None:
        if topk_timer is not None:
            topk_timer.start_cuda()
        scores = torch.zeros(
            (heads, length), dtype=torch.float32, device=q.device)
        if topk_timer is not None:
            topk_timer.stop_cuda()
    key_positions = torch.arange(length, device=q.device)
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        logits = torch.bmm(q[:, start:end], k)
        causal = key_positions[None, :] <= key_positions[start:end, None]
        mask = padding_mask.view(batch, 1, 1, length) & causal[None, None]
        logits = logits.view(batch, n_head, end - start, length)
        logits.masked_fill_(~mask, -1e4)
        probabilities = logits.view(heads, end - start, length).softmax(dim=-1)
        output[:, start:end] = torch.bmm(probabilities, v)
        if scores is not None and end > score_query_start:
            if topk_timer is not None:
                topk_timer.start_cuda()
            first_row = max(start, score_query_start) - start
            scores.add_(probabilities[:, first_row:].sum(
                dim=1, dtype=torch.float32))
            if topk_timer is not None:
                topk_timer.stop_cuda()
        del logits, probabilities, mask, causal
    indices = None
    if score_observer is not None:
        score_observer(scores)
    if topk_ratio is not None:
        count = min(length, max(1, math.ceil(length * topk_ratio)))
        # Preserve the original score dtype before top-k. Chunked reductions
        # may still differ at floating-point ties, as with any GEMM tiling.
        if topk_timer is not None:
            topk_timer.start_cuda()
        indices = scores.to(q.dtype).topk(count, dim=1).indices
        if topk_timer is not None:
            topk_timer.stop_cuda()
    return output, indices

def fix_recursive_import():
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    from flexllmgen import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice


class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()

    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        else:
            raise ValueError(f"Invalid name: {name}")


class TorchTensor:
    """
    Wrap pytorch tensors to support
      - Unified representation for normal and compressed tensors on
        GPUs, CPUs, disks and mixed devices.
      - Asynchronous copy between tensors on any formats and any devices.

    This is achieved by implementing the data movement APIs for primitive cases
    and using recursive structures to handle other combinations.

    Note:
    For a tensor on a TorchDevice, self.data is a primitive tensor.
      type: torch.Tensor.
    For a tensor on a TorchDisk, self.data is a filename.
      type: str
    For a tensor on a TorchMixedDevice, self.data is (tensors, segment_points)
      type: Tuple[Tuple[TorchTensor], Tuple[int]]
    For a tensor on a TorchCompressedDevice, self.data is (data, scale, compression_config)
      type: Tuple[TorchTensor, TorchTensor, CompressionConfig]
    """
    name_count = count()

    def __init__(self, shape, dtype, data, device, name=None):
        if isinstance(data, torch.Tensor):
            assert data.device == device.dev

        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.device = device

        # Whether delete the file when the tensor is deleted
        self.delete_file = True

        self.name = name or TorchTensor.next_name()

    @property
    def bytes(self):
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

    @classmethod
    def next_name(cls):
        return f"t_{next(cls.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        self.device = self.data = None

    def load_from_np(self, np_array):
        if self.device.device_type == DeviceType.DISK:
            with open(self.data, "wb") as fout:
                np.save(fout, np_array)
        else:
            if self.device.device_type == DeviceType.COMPRESSED:
                tmp = torch.from_numpy(np_array)
                tmp = global_cpu_device.compressed_device.compress(tmp, self.data[2])
                general_copy(self, None, tmp, None)
            else:
                self.data.copy_(torch.from_numpy(np_array))

    def load_from_np_file(self, filename):
        if self.device.device_type == DeviceType.DISK:
            shutil.copy(filename, self.data)
        else:
            self.load_from_np(np.load(filename))

    def copy(self, dst, src_indices=None):
        if src_indices:
            assert all(x.step is None for x in src_indices)
            shape = tuple(x.stop - x.start for x in src_indices
                ) + self.shape[len(src_indices):]
        else:
            shape = self.shape

        if dst.device_type == DeviceType.COMPRESSED:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
        else:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
        general_copy(ret, None, self, src_indices)
        return ret

    def smart_copy(self, dst, src_indices=None):
        if self.device == dst:
            return self, False
        return self.copy(dst, src_indices=src_indices), True

    def move(self, dst):
        if self.device == dst:
            return self
        ret = self.copy(dst)
        self.delete()
        return ret

    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")


class TorchDevice:
    """Wrap tensor and computation APIs of a single CPU or GPU."""

    def __init__(self, name, mem_capacity=None, flops=None):
        self.name = name
        self.mem_capacity = mem_capacity
        self.flops = flops

        self.dev = torch.device(name)
        self.device_type = DeviceType.convert(self.dev.type)
        self.compressed_device = TorchCompressedDevice(self)

        self.links = {}

        self.attention_compute_workspace = None
        self.workspace_pt = 0

        if self.device_type == DeviceType.CPU:
            global global_cpu_device
            global_cpu_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        if self.device_type == DeviceType.CPU:
            pin_memory = True if pin_memory is None else pin_memory
        else:
            pin_memory = False
        dtype = np_dtype_to_torch_dtype[dtype]
        data = torch.empty(shape, dtype=dtype, pin_memory=pin_memory, device=self.dev)
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, tensor):
        pass

    def init_attention_compute_workspace(self, config, task, policy, max_gen_len, max_prompt_len):
        if self.device_type != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace

        if not policy.compress_cache:
            b = policy.gpu_batch_size
            n_head = config.n_head
            head_dim = config.input_dim // n_head
            max_seq_len = max_gen_len + max_prompt_len - 1
            self.attention_compute_workspace = []
            self.workspace_pt = 0

            # We currently separate SelfAttention and MLP as two layers,
            # so we only need one workspace instead of two.
            for i in range(1 if policy.sep_layer else 2):
                shape = (max_seq_len, b * n_head, head_dim)
                k_cache = self.allocate(shape, np.float32, pin_memory=False)
                v_cache = self.allocate(shape, np.float32, pin_memory=False)
                self.attention_compute_workspace.append((k_cache, v_cache))
        else:
            self.compressed_device.init_attention_compute_workspace(
                config, task, policy, max_gen_len, max_prompt_len)

    def next_attention_compute_workspace(self):
        self.workspace_pt = (self.workspace_pt + 1) % len(
            self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pt]

    def del_attention_compute_workspace(self):
        self.attention_compute_workspace = None

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        data = token_ids.data.ne(pad_token_id)
        if donate[0]: token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def extend_attention_mask(self, attention_mask, donate):
        bs = attention_mask.shape[0]
        data = torch.concat((attention_mask.data,
             torch.ones((bs, 1), dtype=attention_mask.dtype, device=self.dev)), dim=1)
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)

    def opt_input_embed(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        # pos embedding
        positions = torch.cumsum(mask, dim=1).int() * mask + 1

        # cut positions if `past_key_values_length` is > 0
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]

        pos_embed = F.embedding(positions, w_pos.data)

        data = token_embed + pos_embed
        return TorchTensor.create_from_torch(data, self)

    def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate,
                         do_sample, temperature):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        if donate[0]: inputs.delete()

        # output embedding
        logits = F.linear(hidden, w_token.data)
        last_token_logits = logits[:,-1,:]

        # Optional quality-evaluation observer; ordinary generation is unchanged.
        callback = getattr(self, "output_logits_callback", None)
        if callback is not None:
            callback(last_token_logits)

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)

    def init_cache_one_gpu_batch(self, config, task, policy,
                                 max_prompt_len=None, max_gen_len=None):
        """Allocate from the task, or explicit limits before a task exists."""
        prompt_len = task.prompt_len if max_prompt_len is None else max_prompt_len
        gen_len = task.gen_len if max_gen_len is None else max_gen_len
        num_head, hidden_size = config.n_head, config.input_dim
        shape = (prompt_len + gen_len - 1,
                 policy.gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        return k_cache, v_cache


    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache,
            comp_config, K, L, attention_topk_ratio=None,
            prefill_query_chunk_size=None, score_observer=None,
            score_query_start=0):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        if prefill_query_chunk_size is not None:
            value, attention_topk = chunked_causal_prefill(
                q, k, v, attention_mask.data, n_head,
                prefill_query_chunk_size, attention_topk_ratio, score_observer,
                score_query_start)
            value = value.view(b, n_head, s, head_dim)
        else:
            value, attention_topk = self._dense_prefill_attention(
                q, k, v, attention_mask, b, s, n_head, head_dim,
                attention_topk_ratio, score_observer, score_query_start)

        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)
        value.add_(inputs.data)
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)
        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)
        result = (TorchTensor.create_from_torch(value, self), k, v)
        if attention_topk is not None:
            return result + (attention_topk,)
        return result

    def _dense_prefill_attention(self, q, k, v, attention_mask, b, s,
                                 n_head, head_dim, attention_topk_ratio,
                                 score_observer=None, score_query_start=0):
        if not 0 <= score_query_start < s:
            raise ValueError("Score query start must be within the prompt")
        attn_weights = torch.bmm(q, k)

        # self.transform_kernel(attn_weights, q, k, scaling, K, L)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s) # 下三角矩阵
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        attention_topk = None
        if score_observer is not None:
            score_observer(attn_weights[:, score_query_start:].sum(
                dim=1, dtype=torch.float32))
        if attention_topk_ratio is not None:
            # Rank key positions using the requested query rows. The default
            # starts at zero and includes the complete causal prefill.
            attention_topk = attention_mass_topk(
                attn_weights[:, score_query_start:], attention_topk_ratio)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        return value, attention_topk

    def warmup_gpu(self, typical_shapes):
        for shapes in typical_shapes:
            b, s, common_prefix_len, h = shapes
            inputs = torch.randn(b, s, h, device='cuda')
            _ = F.layer_norm(inputs, (h,))

    def prepare_suffix_query_states(
            self, inputs, attention_mask, w_q, b_q, w_ln, b_ln, n_head,
            matched_prefix):
        """Compute the normalized suffix and its Q projection once."""
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)

        b, s, h = inputs.shape
        head_dim = h // n_head
        common_prefix_len = sum(matched_prefix.values())
        total_prompt_len = attention_mask.shape[1]

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        if s == total_prompt_len:
            hidden = hidden[:, common_prefix_len:, :]
        else:
            assert s == total_prompt_len - common_prefix_len, (
                f"Expected suffix-only hidden length {total_prompt_len - common_prefix_len}, got {s}"
            )
        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data)
        q = q.view(b, hidden.shape[1], n_head, head_dim)
        if b != 1:
            raise NotImplementedError(
                "Shared-prefix sparse selection currently requires batch size 1")
        return q[0], hidden

    def get_suffix_query_states(self, inputs, attention_mask, w_q, b_q,
                w_ln, b_ln, n_head, k_cache, donate,
                compress_cache, comp_config, matched_prefix):
        """Return suffix queries while preserving the legacy interface."""
        query_states, _ = self.prepare_suffix_query_states(
            inputs, attention_mask, w_q, b_q, w_ln, b_ln, n_head,
            matched_prefix)
        return query_states

    def mha_prefill_with_kv(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                compress_cache, comp_config, matched_prefix, imp_token_idx, K, L,
                prepared_query_states=None, prepared_suffix_hidden=None,
                kv_workspace=None, attention_observer=None):
        """Multi-head attention (prefill with prefix cache)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            if prepared_query_states is None:
                w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)
        if compress_cache:
            # shape: (n_imp, b * n_head, head_dim)
            k = k_cache.device.decompress(k_cache)
            v = v_cache.device.decompress(v_cache)
        elif kv_workspace is None:
            # shape: (n_imp, b * n_head, head_dim)
            k = k_cache.data
            v = v_cache.data
        # shape b*n_head, n_imp
        n_imp = imp_token_idx.shape[1]
        if kv_workspace is None:
            k = k[:n_imp]
            v = v[:n_imp]

        '''
        整个序列计算q向量(b, s, h)
        计算非前缀部分的k_new,v_new向量(b, suffix_len, h)   suffix_len = s - common_prefix_len
        获取前缀部分的kv缓存: (n_imp, b * n_head, head_dim), 把两部分拼接在一起(n_imp + suffix_len, b * n_head, head_dim)

        计算attn_weights(b * n_head, s, n_imp + suffix_len)
        mask和softmax后计算attn_values(b * n_head, s, head_dim)

        调整维度后和inputs.data相加返回。同时返回拼接后的kv_cache向量(n_imp + suffix_len, b * n_head, head_dim)
        '''

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        total_prompt_len = attention_mask.shape[1]
        common_prefix_len = sum(matched_prefix.values())
        suffix_len = total_prompt_len - common_prefix_len

        if s == total_prompt_len:
            residual = inputs.data[:, common_prefix_len:, :]
        else:
            assert s == suffix_len, f"Expected suffix-only hidden length {suffix_len}, got {s}"
            residual = inputs.data

        if ((prepared_query_states is None) !=
                (prepared_suffix_hidden is None)):
            raise ValueError(
                "prepared_query_states and prepared_suffix_hidden must be "
                "provided together")

        if prepared_query_states is None:
            hidden = F.layer_norm(
                inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
            suffix_hidden = (
                hidden[:, common_prefix_len:, :]
                if s == total_prompt_len else hidden)
            q = F.linear(suffix_hidden, w_q.data, bias=b_q.data)
            q = q.view(b, suffix_hidden.shape[1], n_head, head_dim)
        else:
            suffix_hidden = prepared_suffix_hidden
            expected_hidden_shape = (b, suffix_len, h)
            expected_query_shape = (suffix_len, n_head, head_dim)
            if tuple(suffix_hidden.shape) != expected_hidden_shape:
                raise ValueError(
                    "Prepared suffix hidden shape mismatch: expected "
                    f"{expected_hidden_shape}, got {tuple(suffix_hidden.shape)}")
            if tuple(prepared_query_states.shape) != expected_query_shape:
                raise ValueError(
                    "Prepared query shape mismatch: expected "
                    f"{expected_query_shape}, got "
                    f"{tuple(prepared_query_states.shape)}")
            q = prepared_query_states.unsqueeze(0)

        query_len = suffix_hidden.shape[1]
        query_start = total_prompt_len - query_len

        # shape: (b, query_len, h)
        q = q * scaling
        # shape: (b, suffix_len, h)
        k_new = F.linear(suffix_hidden, w_k.data, bias=b_k.data)
        v_new = F.linear(suffix_hidden, w_v.data, bias=b_v.data)

        # shape: (b, query_len, n_head, head_dim)
        q = q.view(b, query_len, n_head, head_dim)
        # shape: (b, suffix_len, n_head, head_dim)
        k_new = k_new.view(b, suffix_len, n_head, head_dim)
        v_new = v_new.view(b, suffix_len, n_head, head_dim)
        # shape: (b * n_head, query_len, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, query_len, head_dim)

        # shape: (suffix_len, b * n_head, head_dim)
        k_new = k_new.permute(1, 0, 2, 3).reshape(suffix_len, b * n_head, head_dim)
        v_new = v_new.permute(1, 0, 2, 3).reshape(suffix_len, b * n_head, head_dim)
        # origin k_cache shape: (n_imp + suffix_len, b * n_head, head_dim)
        if kv_workspace is not None:
            # Prefix is already in head-major GPU storage. Append only the
            # new suffix, avoiding a full prefix cat/copy on every layer.
            total_len = n_imp + suffix_len
            key_rows = kv_workspace["key"][:, :total_len]
            value_rows = kv_workspace["value"][:, :total_len]
            key_rows[:, n_imp:].copy_(k_new.permute(1, 0, 2))
            value_rows[:, n_imp:].copy_(v_new.permute(1, 0, 2))
            k = key_rows.permute(1, 0, 2)
            v = value_rows.permute(1, 0, 2)
        else:
            k = torch.cat((k, k_new), dim=0)
            v = torch.cat((v, v_new), dim=0)
        # shape: (b * n_head, head_dim, n_imp + suffix_len)
        k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, n_imp + suffix_len)
        # shape: (b * n_head, n_imp + suffix_len, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, n_imp + suffix_len, head_dim)

        # shape: (b * n_head, query_len, n_imp + suffix_len)
        attn_weights = torch.bmm(q, k)
        l = n_imp + suffix_len

        imp_token_idx = imp_token_idx.to(self.dev)
        suffix_token_idx = torch.arange(
            common_prefix_len, total_prompt_len, device=self.dev
        ).unsqueeze(0).expand(b * n_head, -1)
        token_idxs = torch.cat([imp_token_idx, suffix_token_idx], dim=1)
        valid_mask = torch.cat(
            [imp_token_idx.ne(-1),
             torch.ones_like(suffix_token_idx, dtype=torch.bool)],
            dim=1,
        )
        token_idxs = token_idxs.masked_fill(~valid_mask, total_prompt_len)
        token_idxs = token_idxs.view(b, n_head, l)
        valid_mask = valid_mask.view(b, n_head, l)

        query_positions = torch.arange(
            query_start, total_prompt_len, device=self.dev
        ).view(1, 1, query_len, 1)
        mask = valid_mask.unsqueeze(2) & (query_positions >= token_idxs.unsqueeze(2))

        attn_weights = attn_weights.view(b, n_head, query_len, l)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, query_len, l)
        attn_weights = F.softmax(attn_weights, dim=2)
        if attention_observer is not None:
            attention_observer(imp_token_idx, q, k, attn_weights, mask)
        # shape: (b, n_head, query_len, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, query_len, head_dim)
        # shape: (b, query_len, h)
        value = value.transpose(1, 2).reshape(b, query_len, h)
        value = F.linear(value, w_out.data, bias=b_out.data)
        # shape : b, query_len, h
        value.add_(residual)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (n_imp + suffix_len, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        return TorchTensor.create_from_torch(value, self), k, v


    def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config, pos, imp_token_idx):
        """Multi-head attention (decoding phase).
        新增pos参数表示该层当前KVcache的第一维的长度.

        计算部分没有进行mask操作

        """
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        # src_s = attention_mask.shape[1]
        src_s = pos
        # logger.info(f"mha_gen: src_s = {src_s}, mask.shape={attention_mask.shape[1]}")
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        ### generate cache mask
        bn, n_imp = imp_token_idx.shape
        imp_token_idx = imp_token_idx.view(b, n_head, n_imp).to(self.dev)
        cache_mask = torch.ones_like(imp_token_idx, device=self.dev, dtype=bool)
        cache_mask[imp_token_idx == -1] = False
        pad = src_s - n_imp
        padding = torch.ones((b, n_head, pad), device=self.dev, dtype=bool)
        cache_mask = torch.cat([cache_mask, padding], dim=2)


        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

                if k.is_cuda:
                    #logger.info(f"mha_gen: k.dtype{k.dtype}, v.dtype{v.dtype}, q.dtype{q.dtype}")
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim, cache_mask)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)

        return TorchTensor.create_from_torch(value, self), k_new, v_new

    def _attention_weights(self, q, k, mask, b, src_s, n_head, cache_mask):
        # shape: (b * n_head, 1, s)
        attn_weights = torch.bmm(q, k)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        ### shape: b, n_head, src_s
        if cache_mask is not None:
            cache_mask = cache_mask.view(b, n_head, 1, src_s)
            attn_weights = torch.where(cache_mask, attn_weights, -1e4)

        # attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, 1, src_s)
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim, cache_mask=None):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head, cache_mask)
        # shape: (b, n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b,
                                src_s, tgt_s, n_head, head_dim, attn_sparsity):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        topk = int(attn_sparsity * (attn_weights.shape[2] - 1))
        topk_weights, topk_indices = attn_weights[:, :, :-1].topk(
            topk, dim=2, sorted=False)
        topk_indices = topk_indices.view(b * n_head, topk).transpose(0, 1)
        # shape: (b * n_head, 1, topk+1)
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, -1].unsqueeze(-1)], dim=-1)

        if k.is_cuda:
            v_home = v_cache
            v_buf = self.allocate((topk+1, b*n_head, head_dim), np.float16)
            topk_indices = topk_indices.cpu()
        else:
            (v_home, v_buf) = v_cache

        # shape: (s, b * n_head, head_dim)
        indices_src = topk_indices
        indices_tgt = (slice(0, indices_src.shape[0]), slice(0, v_home.shape[1]))
        general_copy(v_buf, indices_tgt, v_home, indices_src)
        v_home.device.synchronize()

        # shape: (topk+1, b * n_head, head_dim)
        v = v_buf.data[:topk+1]
        v[topk:topk+1] = v_new
        # shape: (b * n_head, topk+1, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, topk+1, head_dim)

        # shape: (b * n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new,
            mask, b, src_s, tgt_s, n_head, head_dim):
        # The caches are stored on both gpu and cpu.
        # Compute attention on gpu for caches stored on gpu.
        # Compute attention on cpu for caches stored on cpu.
        k_gpu, k_cpu = k_cache[0].data, k_cache[1].data
        v_gpu, v_cpu = v_cache[0].data, v_cache[1].data
        seg = k_gpu.shape[1]

        # Compute GPU part
        b_gpu = seg // n_head
        q_gpu = q[:seg]
        # shape: (s, b * n_head, head_dim)
        k_gpu = k_gpu[:src_s, :seg, :]
        v_gpu = v_gpu[:src_s, :seg, :]
        k_gpu[src_s-1:src_s, :, :] = k_new[:, :seg, :]
        v_gpu[src_s-1:src_s, :, :] = v_new[:, :seg, :]
        # shape: (b * n_head, head_dim, s)
        k_gpu = k_gpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_gpu = v_gpu.permute(1, 0, 2)

        mask_gpu = mask[:b_gpu].cuda()
        value_gpu = self._attention_value(q_gpu, k_gpu, v_gpu, mask_gpu,
            b_gpu, src_s, tgt_s, n_head, head_dim)

        # Compute CPU Part
        b_cpu = b - b_gpu
        q_cpu = q[seg:].float().cpu()
        # shape: (s, b * n_head, head_dim)
        k_cpu = k_cpu[:src_s, seg:, :]
        v_cpu = v_cpu[:src_s, seg:, :]
        k_cpu[src_s-1:src_s, :, :] = k_new[:, seg:, :]
        v_cpu[src_s-1:src_s, :, :] = v_new[:, seg:, :]
        # shape: (b * n_head, head_dim, s)
        k_cpu = k_cpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_cpu = v_cpu.permute(1, 0, 2)

        mask_cpu = mask[b_gpu:]
        value_cpu = self._attention_value(q_cpu, k_cpu, v_cpu, mask_cpu,
            b_cpu, src_s, tgt_s, n_head, head_dim)

        value = torch.cat([value_gpu, value_cpu.cuda().half()], dim=0)
        return value

    def mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        # decompress weights
        if wi.device.device_type == DeviceType.COMPRESSED:
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape

        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        out = F.linear(out, wi.data, bias=bi.data)
        F.relu(out, inplace=True)
        out = F.linear(out, wo.data, bias=bo.data)

        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)

    def synchronize(self):
        torch.cuda.synchronize()

    def mem_stats(self):
        if self.device_type == DeviceType.CUDA:
            cur_mem = torch.cuda.memory_allocated(self.dev)
            peak_mem = torch.cuda.max_memory_allocated(self.dev)
        elif self.device_type == DeviceType.CPU:
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"


class TorchDisk:
    """Manage tensors stored on a disk."""

    def __init__(self, path, mem_capacity=None, cuda_id=0, num_copy_threads=4):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))
        self.mem_capacity = mem_capacity

        self.device_type = DeviceType.DISK
        self.compressed_device = TorchCompressedDevice(self)

        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        else:
            os.makedirs(self.path)

        self.links = {}

        # Copy threads
        self.copy_queue = queue.Queue()
        self.copy_threads = [
            threading.Thread(
                target=copy_worker_func, args=(self.copy_queue, cuda_id)
            ) for _ in range(num_copy_threads)
        ]
        for t in self.copy_threads:
            t.start()

        global global_disk_device
        global_disk_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           path, self, name=name)

    def delete(self, tensor):
        if os.path.exists(tensor.data) and tensor.delete_file:
            os.remove(tensor.data)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        k_cache = self.allocate(shape, np.float16)
        v_cache = self.allocate(shape, np.float16)
        return k_cache, v_cache

    def submit_copy(self, *args):
        self.copy_queue.put_nowait(args)

    def synchronize(self):
        self.copy_queue.join()

    def close_copy_threads(self):
        for _ in range(len(self.copy_threads)):
            self.copy_queue.put_nowait(None)
        for t in self.copy_threads:
            t.join()
        self.copy_queue.join()
        self.copy_queue = None

    def mem_stats(self):
        raise NotImplementedError()

    def print_stats(self):
        raise NotImplementedError()

    def __del__(self):
        if self.copy_queue:
            self.close_copy_threads()


# Segment dimension for tensors stored on TorchMixedDevice
SEG_DIM = 1

class TorchMixedDevice:
    """Manage tensors stored on multiple physical devices."""

    def __init__(self, base_devices):
        self.name = "mixed"
        self.device_type = DeviceType.MIXED
        self.base_devices = base_devices

    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
        assert sum(seg_lengths) == shape[SEG_DIM]
        assert len(seg_lengths) == len(self.base_devices)
        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.base_devices
        tensors = []
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM+1:]
                tensors.append(devices[i].allocate(seg_shape, dtype,
                    pin_memory=pin_memory))

        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (tensors, seg_points), self, name=name)

    def delete(self, tensor):
        for x in self.tensor.data[0]:
            if x:
                x.delete()

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)

        # We have to round to a multiple of `num_head`
        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = shape[SEG_DIM]  - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache


class TorchLink:
    """An I/O link between two devices."""

    def __init__(self, a, b, a_to_b_bandwidth, b_to_a_bandwidth):
        self.a = a
        self.b = b
        self.a_to_b_bandwidth = a_to_b_bandwidth
        self.b_to_a_bandwidth = b_to_a_bandwidth

        a.add_link(self)
        b.add_link(self)

    def io_time(self, src, dst, size):
        if src == self.a:
            assert dst == self.b
            bandwidth = self.a_to_b_bandwidth
        elif src == self.b:
            assert dst == self.a
            bandwidth = self.b_to_a_bandwidth
        else:
            raise ValueError(f"Invalid source {src}")

        return size / bandwidth


def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    """Launch a general asynchronous copy between two tensors.
    It is equivalent to `dst[dst_indices] = src[src_indices]` in numpy syntax.
    The copy is asynchronous. To wait for the copy to complete, you need to call
    >>> env.disk.synchronize()
    >>> torch.cuda.synchronize()
    """
    if dst.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert src.device.device_type != DeviceType.MIXED
        seg_points = dst.data[1]

        for i in range(len(dst.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            general_copy(dst.data[0][i], tmp_dst_indices, src, tmp_src_indices)
    elif src.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert dst.device.device_type != DeviceType.MIXED
        seg_points = src.data[1]

        for i in range(len(src.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1])
            general_copy(dst, tmp_dst_indices, src.data[0][i], tmp_src_indices)
    elif (src.device.device_type == DeviceType.COMPRESSED or
          dst.device.device_type == DeviceType.COMPRESSED):
        # The tensor is compressed, do recursive calls
        general_copy_compressed(dst, dst_indices, src, src_indices)
    elif src.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CUDA and
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        # The cpu tensor is not pinned, dispatch to copy threads and use pin_memory
        # as a relay
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.CUDA and
          not src.data.is_pinned()):
        # The cpu tensor is not pinned, use pin_memory as a relay
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        src = src.pin_memory()
        dst.copy_(src, non_blocking=True)
    else:
        # The normal path
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)


def cut_indices(indices, start, stop, base=0):
    assert all(x.step is None for x in indices)
    seg = indices[SEG_DIM]
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])


def map_to_torch_tensor(tensor, indices):
    if tensor.device.device_type == DeviceType.DISK:
        data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
    else:
        data = tensor.data

    # BC: this is supposed to only handle the sparse v_cache case
    if torch.is_tensor(indices):
        return vector_gather(data, indices)
    return data[indices] if indices else data


def copy_worker_func(queue, cuda_id):
    """The copy worker thread."""
    torch.cuda.set_device(cuda_id)

    cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
    copy_stream = torch.cuda.Stream()

    with torch.cuda.stream(copy_stream):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)
            # logger.info(f"Copying src_indices = {src_indices}, src_data.shape = {src_data.shape}, "
            #             f"dst_indices = {dst_indices}, dst_data.shape = {dst_data.shape}")
            if (src.device.device_type == DeviceType.CUDA or
                dst.device.device_type == DeviceType.CUDA):
                # Use a pinned cpu buffer as a relay
                size = np.prod(src_data.shape)
                # logger.info(f"Copying src_data.shape = {src_data.shape}, "
                #             f"dst_data.shape = {dst_data.shape}, size = {size}")
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                tmp_cpu_buf.copy_(src_data)
                dst_data.copy_(tmp_cpu_buf) # dst不是pinned memory,所以不用non_blocking
            else:
                dst_data.copy_(src_data)

            queue.task_done()

def sync_general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice],cpu_buf):
    """synchronous copy between two tensors.
    Only supporting copy among pinned tensors on gpu, cpu, or disk.
    """


    src_dev = src.device.device_type
    dst_dev = dst.device.device_type
    if src_dev == DeviceType.DISK or dst_dev == DeviceType.DISK:
        src_data = map_to_torch_tensor(src, src_indices)
        dst_data = map_to_torch_tensor(dst, dst_indices)
        if (src_dev == DeviceType.CUDA or
            dst_dev == DeviceType.CUDA):
            # Use a pinned cpu buffer as a relay
            size = np.prod(src_data.shape)
            tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
            tmp_cpu_buf.copy_(src_data)
            dst_data.copy_(tmp_cpu_buf, non_blocking=True)
        else:
            dst_data.copy_(src_data)
    elif src_dev == DeviceType.CPU and dst_dev == DeviceType.CUDA and not src_data.is_pinned():
        # The cpu tensor is not pinned, use pin_memory as a relay
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        src = src.pin_memory()
        dst.copy_(src, non_blocking=True)
    else:
        # The normal path
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)
