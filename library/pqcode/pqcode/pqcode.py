"""PQCode-based shared-prefix token selection.

The index is constructed per decoder layer and KV head.  Selection aggregates
the approximate attention probability of every suffix query before taking one
fixed-budget top-k token set per attention head.
"""

import math
from typing import Dict, Tuple

import torch



class PQCodeIndex:
    """Product-quantization index for shared-prefix retrieval."""

    FORMAT_VERSION = 1

    def __init__(self):
        self.allocated = False
        self.metadata = {}
        self.device_metadata = {}

    def alloc(self, num_subvectors: int, num_bits: int, num_layers: int, num_attention_heads: int, num_key_value_heads: int, batch_size: int, max_length: int, head_dim: int, retention_ratio: float, query_chunk_size: int=32, kmeans_max_iter: int=20, kmeans_seed: int=0, kmeans_tolerance: float=0.0001, kmeans_group_chunk_size: int=4):
        if num_subvectors <= 0:
            raise ValueError("num_subvectors must be positive")
        if not 1 <= num_bits <= 8:
            raise ValueError("num_bits must be in [1, 8]")
        if num_layers <= 0 or num_attention_heads <= 0:
            raise ValueError("layer and attention-head counts must be positive")
        if num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by "
                "num_key_value_heads")
        if batch_size != 1:
            raise NotImplementedError(
                "PQCode shared-prefix selection currently requires batch_size=1")
        if max_length <= 0 or head_dim <= 0:
            raise ValueError("max_length and head_dim must be positive")
        if head_dim % num_subvectors != 0:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by "
                f"num_subvectors={num_subvectors}")
        if not 0 < retention_ratio <= 1:
            raise ValueError("retention_ratio must be in (0, 1]")
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if kmeans_max_iter <= 0:
            raise ValueError("kmeans_max_iter must be positive")
        if kmeans_tolerance < 0:
            raise ValueError("kmeans_tolerance cannot be negative")
        if kmeans_group_chunk_size <= 0:
            raise ValueError("kmeans_group_chunk_size must be positive")

        self.use_token_groups = False
        self.selected_group_ids = {}
        self.num_subvectors = num_subvectors
        self.num_bits = num_bits
        self.requested_centroid_count = 1 << num_bits
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_groups = (
            num_attention_heads // num_key_value_heads)
        self.batch_size = batch_size
        self.max_length = max_length
        self.head_dim = head_dim
        self.subvector_dim = head_dim // num_subvectors
        self.retention_ratio = retention_ratio
        self.query_chunk_size = query_chunk_size
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_seed = kmeans_seed
        self.kmeans_tolerance = kmeans_tolerance
        self.kmeans_group_chunk_size = kmeans_group_chunk_size
        self.metadata: Dict[Tuple[int, int], Dict[str, object]] = {}
        self.device_metadata: Dict[Tuple[int, int, str], Dict[str, object]] = {}
        self.allocated = True

    def _invalidate_device_metadata(self, metadata_key):
        stale_keys = [
            cache_key for cache_key in self.device_metadata
            if cache_key[:2] == metadata_key
        ]
        for cache_key in stale_keys:
            del self.device_metadata[cache_key]

    def prepare_device_metadata(
        self,
        layer_idx: int,
        request_id: int,
        device,
    ):
        """Cache PQ lookup data, sharing KV-head metadata in grouped mode."""
        self._check_allocated()
        self._validate_layer_request(layer_idx, request_id)
        metadata_key = (layer_idx, request_id)
        if metadata_key not in self.metadata:
            raise ValueError(
                f"Missing PQCode metadata for layer {layer_idx}, "
                f"request {request_id}")

        target = torch.device(device)
        cache_key = (layer_idx, request_id, str(target))
        cached = self.device_metadata.get(cache_key)
        if cached is not None:
            return cached

        metadata = self.metadata[metadata_key]
        centroids = metadata["centroids"].to(
            device=target, dtype=torch.float32, non_blocking=True)
        pass
        # torch.gather requires int64 indices. Convert once when the prefix is
        # built/loaded instead of once per layer for every request.
        codes = metadata["codes"].to(
            device=target, dtype=torch.long, non_blocking=True)
        head_to_kv = (
            torch.arange(self.num_attention_heads, device=target)
            // self.num_attention_groups)
        cached = {
            "head_centroids": centroids.index_select(0, head_to_kv),
            "head_codes": codes.index_select(0, head_to_kv),
        }
        self.device_metadata[cache_key] = cached
        return cached

    def _check_allocated(self):
        if not self.allocated:
            raise RuntimeError("PQCodeIndex.alloc() must be called first")

    def _validate_layer_request(self, layer_idx: int, request_id: int):
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"Invalid layer index {layer_idx}")
        if not 0 <= request_id < self.batch_size:
            raise ValueError(f"Invalid request index {request_id}")

    def _initial_centroids(
        self,
        vectors: torch.Tensor,
        centroid_count: int,
        layer_idx: int,
        request_id: int,
    ) -> torch.Tensor:
        group_count, vector_count, _ = vectors.shape
        device = vectors.device
        generator = torch.Generator(device=device)
        generator.manual_seed(
            self.kmeans_seed + layer_idx * 1000003 + request_id * 9176)
        indices = torch.empty(
            (group_count, centroid_count), dtype=torch.long, device=device)
        for group_idx in range(group_count):
            indices[group_idx] = torch.randperm(
                vector_count,
                generator=generator,
                device=device,
            )[:centroid_count]
        group_indices = torch.arange(
            group_count, device=device).unsqueeze(1)
        return vectors[group_indices, indices].clone()

    @staticmethod
    def _assign(vectors: torch.Tensor, centroids: torch.Tensor):
        vector_norms = vectors.square().sum(dim=-1, keepdim=True)
        centroid_norms = centroids.square().sum(dim=-1).unsqueeze(1)
        distances = vector_norms + centroid_norms
        distances.add_(torch.bmm(vectors, centroids.transpose(1, 2)), alpha=-2)
        return distances.argmin(dim=-1)

    def _lloyd(
        self,
        vectors: torch.Tensor,
        centroid_count: int,
        layer_idx: int,
        request_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run deterministic batched Lloyd iterations on ``vectors.device``."""
        group_count, vector_count, subvector_dim = vectors.shape
        device = vectors.device
        centroids = self._initial_centroids(
            vectors, centroid_count, layer_idx, request_id)
        final_codes = torch.empty(
            (group_count, vector_count), dtype=torch.long, device=device)

        for group_start in range(
                0, group_count, self.kmeans_group_chunk_size):
            group_end = min(
                group_count, group_start + self.kmeans_group_chunk_size)
            group_vectors = vectors[group_start:group_end]
            group_centroids = centroids[group_start:group_end]
            chunk_groups = group_end - group_start

            for _ in range(self.kmeans_max_iter):
                codes = self._assign(group_vectors, group_centroids)
                group_offsets = (
                    torch.arange(chunk_groups, device=device).view(-1, 1)
                    * centroid_count)
                flat_codes = (codes + group_offsets).reshape(-1)
                sums = torch.zeros(
                    (chunk_groups * centroid_count, subvector_dim),
                    dtype=torch.float32,
                    device=device,
                )
                sums.index_add_(
                    0, flat_codes,
                    group_vectors.reshape(-1, subvector_dim))
                counts = torch.bincount(
                    flat_codes, minlength=chunk_groups * centroid_count)
                counts = counts.reshape(chunk_groups, centroid_count)
                updated = group_centroids.clone()
                nonempty = counts > 0
                updated[nonempty] = (
                    sums.reshape(
                        chunk_groups, centroid_count, subvector_dim)[nonempty]
                    / counts[nonempty].unsqueeze(1).float())
                shift = (updated - group_centroids).square().sum(dim=-1)
                group_centroids = updated
                if (float(shift.max().item()) <=
                        self.kmeans_tolerance * self.kmeans_tolerance):
                    break

            centroids[group_start:group_end] = group_centroids
            final_codes[group_start:group_end] = self._assign(
                group_vectors, group_centroids)

        return centroids, final_codes

    def build(
        self,
        layer_idx: int,
        request_id: int,
        keys: torch.Tensor,
    ):
        """Construct centroids and codes from raw prefix Keys.

        Args:
            keys: ``(num_key_value_heads, prefix_len, head_dim)`` tensor.
        """
        self._check_allocated()
        self._validate_layer_request(layer_idx, request_id)
        if keys.ndim != 3:
            raise ValueError(f"Expected 3-D keys, got {tuple(keys.shape)}")
        expected_head_shape = (self.num_key_value_heads, self.head_dim)
        if (keys.shape[0], keys.shape[2]) != expected_head_shape:
            raise ValueError(
                "PQCode key shape mismatch: expected "
                f"({self.num_key_value_heads}, prefix_len, {self.head_dim}), "
                f"got {tuple(keys.shape)}")
        prefix_len = int(keys.shape[1])
        if prefix_len <= 0 or prefix_len > self.max_length:
            raise ValueError(
                f"Invalid prefix length {prefix_len}; max is {self.max_length}")

        build_keys = keys.detach().to(dtype=torch.float32).contiguous()
        vectors = build_keys.reshape(
            self.num_key_value_heads,
            prefix_len,
            self.num_subvectors,
            self.subvector_dim,
        ).permute(0, 2, 1, 3).reshape(
            self.num_key_value_heads * self.num_subvectors,
            prefix_len,
            self.subvector_dim,
        ).contiguous()
        centroid_count = min(self.requested_centroid_count, prefix_len)
        centroids, codes = self._lloyd(
            vectors, centroid_count, layer_idx, request_id)
        centroids = centroids.reshape(
            self.num_key_value_heads,
            self.num_subvectors,
            centroid_count,
            self.subvector_dim,
        ).contiguous()
        codes = codes.reshape(
            self.num_key_value_heads,
            self.num_subvectors,
            prefix_len,
        ).to(torch.uint8).contiguous()
        self.metadata[(layer_idx, request_id)] = {
            "prefix_len": prefix_len,
            "centroid_count": centroid_count,
            "centroids": centroids,
            "codes": codes,
        }
        pass
        self._invalidate_device_metadata((layer_idx, request_id))


    def num_selected_tokens(self, prefix_len: int) -> int:
        return min(
            prefix_len,
            max(1, int(math.ceil(prefix_len * self.retention_ratio))))

    def batch_retrieve_multi(self, layer_idx: int, query_states: torch.Tensor, results: torch.Tensor, nnz: torch.Tensor, max_index: int, request_id: int=0, importance_scores: torch.Tensor=None):
        """Select fixed-budget prefix token sets shared by suffix queries.

        By default each Q head owns a selection (legacy API). With
        group_by_kv=True, sum normalized scores within each GQA group before top-k and
        return fixed-budget KV-head rows to KVStore.
        Optional CPU float32 importance_scores receive mean approximate attention
        probabilities in the same sorted token order, using the existing scores.
        """
        self._check_allocated()
        self._validate_layer_request(layer_idx, request_id)
        metadata_key = (layer_idx, request_id)
        if metadata_key not in self.metadata:
            raise ValueError(
                f"Missing PQCode metadata for layer {layer_idx}, "
                f"request {request_id}")
        if query_states.ndim != 3:
            raise ValueError(
                f"Expected 3-D query_states, got {tuple(query_states.shape)}")
        expected_query_shape = (self.num_attention_heads, self.head_dim)
        if (query_states.shape[1], query_states.shape[2]) != expected_query_shape:
            raise ValueError(
                "PQCode query shape mismatch: expected "
                f"(suffix_len, {self.num_attention_heads}, {self.head_dim}), "
                f"got {tuple(query_states.shape)}")
        if query_states.shape[0] <= 0:
            raise ValueError("PQCode requires at least one suffix query")
        if results.device.type != "cpu" or nnz.device.type != "cpu":
            raise ValueError("results and nnz must be CPU tensors")
        if results.dtype != torch.int32 or nnz.dtype != torch.int32:
            raise ValueError("results and nnz must use torch.int32")
        selection_heads = self.num_attention_heads
        if results.shape != (
                self.batch_size * selection_heads, self.max_length):
            raise ValueError(f"Unexpected results shape {tuple(results.shape)}")
        if nnz.shape != (self.batch_size * selection_heads,):
            raise ValueError(f"Unexpected nnz shape {tuple(nnz.shape)}")
        if importance_scores is not None and (
                importance_scores.device.type != "cpu"
                or importance_scores.dtype != torch.float32
                or importance_scores.shape != results.shape
                or not importance_scores.is_contiguous()):
            raise ValueError("importance_scores must be contiguous CPU float32 with results shape")

        metadata = self.metadata[metadata_key]
        stored_len = int(metadata["prefix_len"])
        prefix_len = int(max_index)
        if prefix_len <= 0 or prefix_len > stored_len:
            raise ValueError(
                f"Invalid PQCode prefix cap {prefix_len}; stored length is "
                f"{stored_len}")

        pass

        device = query_states.device
        device_metadata = self.prepare_device_metadata(
            layer_idx, request_id, device)
        head_centroids = device_metadata["head_centroids"]
        head_codes = device_metadata["head_codes"][..., :prefix_len]
        queries = query_states.transpose(0, 1).contiguous().float()
        aggregate = torch.zeros(
            (self.num_attention_heads, prefix_len),
            dtype=torch.float32,
            device=device,
        )
        scale = self.head_dim ** -0.5

        for query_start in range(
                0, queries.shape[1], self.query_chunk_size):
            query = queries[
                :, query_start:query_start + self.query_chunk_size]
            query_len = query.shape[1]
            query = query.reshape(
                self.num_attention_heads,
                query_len,
                self.num_subvectors,
                self.subvector_dim,
            ).permute(0, 2, 1, 3)
            lookup = torch.matmul(
                query, head_centroids.transpose(2, 3))
            approximate_logits = torch.zeros(
                (self.num_attention_heads, query_len, prefix_len),
                dtype=torch.float32,
                device=device,
            )
            for subvector_idx in range(self.num_subvectors):
                gather_index = head_codes[:, subvector_idx].unsqueeze(1)
                gather_index = gather_index.expand(
                    self.num_attention_heads, query_len, prefix_len)
                approximate_logits.add_(torch.gather(
                    lookup[:, subvector_idx], 2, gather_index))
            approximate_logits.mul_(scale)
            aggregate.add_(
                torch.softmax(approximate_logits, dim=-1).sum(dim=1))

        pass
        recovery = getattr(self, 'hot_recovery_observer', None)
        if recovery is not None:
            recovery.capture_scores(layer_idx, aggregate, False)
        token_budget = self.num_selected_tokens(prefix_len)
        correction = getattr(self, 'online_correction', None)
        if correction is not None:
            aggregate = correction.apply(layer_idx, aggregate)
        selected = torch.topk(
            aggregate, k=token_budget, dim=1, sorted=False).indices
        selected = selected.sort(dim=1).values
        if importance_scores is not None:
            selected_scores = aggregate.gather(1, selected)
            divisor = query_states.shape[0] * (1)
            selected_scores.mul_(1.0 / divisor)
            # Enqueue only selected scores before the already-required blocking
            # ID transfer. That transfer also makes this pinned copy ready.
            importance_scores[:, :token_budget].copy_(selected_scores, non_blocking=True)
        selected = selected.to(device="cpu", dtype=torch.int32)
        results.zero_()
        nnz.fill_(token_budget)
        results[:, :token_budget].copy_(selected)
        return selected

    def state_dict(self, layer_idx: int, request_id: int = 0):
        self._check_allocated()
        self._validate_layer_request(layer_idx, request_id)
        metadata_key = (layer_idx, request_id)
        if metadata_key not in self.metadata:
            raise ValueError(
                f"Missing PQCode metadata for layer {layer_idx}, "
                f"request {request_id}")
        metadata = self.metadata[metadata_key]
        state = {
            "format_version": self.FORMAT_VERSION,
            "layer_idx": layer_idx,
            "request_id": request_id,
            "prefix_len": int(metadata["prefix_len"]),
            "centroid_count": int(metadata["centroid_count"]),
            "num_subvectors": self.num_subvectors,
            "num_bits": self.num_bits,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "centroids": metadata["centroids"].detach().cpu(),
        }
        state["codes"] = metadata["codes"].detach().cpu()
        return state

    def load_state_dict(self, state):
        self._check_allocated()
        version = state.get("format_version")
        if version != self.FORMAT_VERSION:
            raise ValueError('Only plain PQ metadata format version 1 is supported')
        required = {
            "format_version", "layer_idx", "request_id", "prefix_len",
            "centroid_count", "num_subvectors", "num_bits",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "centroids", "codes",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"PQCode state is missing fields: {sorted(missing)}")
        if int(state["format_version"]) not in (1, 2):
            raise ValueError(
                f"Unsupported PQCode format version {state['format_version']}")
        expected_config = {
            "num_subvectors": self.num_subvectors,
            "num_bits": self.num_bits,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
        }
        for field, expected in expected_config.items():
            actual = int(state[field])
            if actual != expected:
                raise ValueError(
                    f"PQCode {field} mismatch: expected {expected}, got "
                    f"{actual}")

        layer_idx = int(state["layer_idx"])
        request_id = int(state["request_id"])
        self._validate_layer_request(layer_idx, request_id)
        prefix_len = int(state["prefix_len"])
        centroid_count = int(state["centroid_count"])
        if prefix_len <= 0 or prefix_len > self.max_length:
            raise ValueError(f"Invalid persisted prefix length {prefix_len}")
        if not 1 <= centroid_count <= min(
                self.requested_centroid_count, prefix_len):
            raise ValueError(
                f"Invalid persisted centroid count {centroid_count}")
        expected_centroid_shape = (
            self.num_key_value_heads,
            self.num_subvectors,
            centroid_count,
            self.subvector_dim,
        )
        expected_code_shape = (
            self.num_key_value_heads,
            self.num_subvectors,
            prefix_len,
        )
        centroids = state["centroids"].detach().to(
            device="cpu", dtype=torch.float32).contiguous()
        groups = None
        codes = state["codes"].detach().to(device="cpu", dtype=torch.uint8).contiguous()
        if tuple(centroids.shape) != expected_centroid_shape:
            raise ValueError(
                f"Unexpected centroid shape {tuple(centroids.shape)}; "
                f"expected {expected_centroid_shape}")
        if codes is not None and tuple(codes.shape) != expected_code_shape:
            raise ValueError(
                f"Unexpected code shape {tuple(codes.shape)}; "
                f"expected {expected_code_shape}")
        if codes is not None and codes.numel() and int(codes.max().item()) >= centroid_count:
            raise ValueError("Persisted PQCode contains an invalid centroid id")
        self.metadata[(layer_idx, request_id)] = {
            "prefix_len": prefix_len,
            "centroid_count": centroid_count,
            "centroids": centroids,
            "codes": codes,
        }
        pass
        self._invalidate_device_metadata((layer_idx, request_id))

    def metadata_bytes(self):
        def size(value):
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(size(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return sum(size(v) for v in value)
            return 0
        def resident(value, device):
            if isinstance(value, torch.Tensor):
                return size(value) if value.device.type == device else 0
            if isinstance(value, dict):
                return sum(resident(v, device) for v in value.values())
            if isinstance(value, (list, tuple)):
                return sum(resident(v, device) for v in value)
            return 0
        return dict(persisted_tensor_bytes=size(self.metadata),
                    host_tensor_bytes=resident(self.metadata, "cpu"),
                    device_cache_tensor_bytes=resident(self.device_metadata, "cuda"),
                    host_cache_tensor_bytes=resident(self.device_metadata, "cpu"))

    def clear(self):
        self.metadata.clear()
        self.device_metadata.clear()
        self.selected_group_ids.clear()
