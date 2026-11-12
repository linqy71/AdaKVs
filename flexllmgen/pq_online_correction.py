"""Request-delayed PQ calibration for fixed shared prefixes."""
import math
import threading

import torch


class PQOnlineCorrection:
    def __init__(self, alpha=.2, epsilon=1e-12, beta=1.):
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError('Correction alpha must be in (0, 1]')
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('Correction epsilon must be positive')
        if not math.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError('Correction beta must be in [0, 1]')
        self.alpha, self.epsilon = alpha, epsilon
        self.beta = beta
        self.weights = {}
        self.pending = {}
        self.deferred = {}
        self.completed_requests = 0
        self.demotion_enabled = False
        self._weight_lock = threading.Lock()
        self._weights_ready = None
        self._demotion_read_done = None
        self._demotion_stream = None

    def apply(self, layer, scores):
        if layer not in self.weights:
            self.weights[layer] = torch.ones_like(scores, dtype=torch.float32)
        weights = self.weights[layer]
        if weights.shape != scores.shape or weights.device != scores.device:
            raise ValueError('Prefix/heads/device changed without resetting correction state')
        if self.beta == 0:
            return scores
        return scores * (weights if self.beta == 1 else weights.pow(self.beta))


    def discard_pending(self):
        """Drop a failed request without updating learned correction weights."""
        self.pending.clear()
        self.deferred.clear()

    @torch.inference_mode()
    def observe(self, layer, metadata, ids, scaled_q, keys, actual, mask):
        """Batch PQ lookup on the selector's resident metadata, without K decoding.

        Q is already scaled. Predict on the same loaded prefix plus causal
        suffix as actual attention. Only weights persist across requests;
        selected IDs and ratios are temporary GPU buffers until commit.
        """
        if layer in self.deferred:
            raise RuntimeError('A layer was observed twice before request commit')
        if keys.shape != (scaled_q.shape[0], scaled_q.shape[2], actual.shape[-1]):
            raise ValueError('Invalid MHA correction key shape')
        self._observe_suffix(layer, metadata, ids, scaled_q,
                             keys[:, :, ids.shape[1]:], actual, mask)

    @torch.inference_mode()
    def defer_observation(self, layer, metadata, ids, scaled_q, keys, actual, mask):
        """Retain request-owned attention and only clone the reusable suffix.

        No PQ reconstruction, softmax, ratio calculation or EMA is launched
        here. The large prefix KV workspace may be reused immediately after
        attention; correction only needs exact suffix Keys, not prefix Keys.
        Q/probabilities/mask/IDs are fresh tensors in the attention backend and
        are not subsequently mutated. Keeping references preserves their
        lifetime without cloning every attention matrix.
        """
        if layer in self.pending or layer in self.deferred:
            raise RuntimeError('A layer was observed twice before request commit')
        self.deferred[layer] = (metadata, ids.detach(), scaled_q.detach(),
                                keys[:, :, ids.shape[1]:].detach().clone(),
                                actual.detach(), mask.detach())

    @torch.inference_mode()
    def _observe_suffix(self, layer, metadata, ids, scaled_q, suffix_keys, actual, mask):
        if layer in self.pending:
            raise RuntimeError('A layer was observed twice before request commit')
        centroids, codes = metadata['head_centroids'], metadata['head_codes']
        heads, queries, dim = scaled_q.shape
        width = ids.shape[1]
        if (ids.shape[0] != heads or centroids.shape[0] != heads or
                codes.shape[:2] != centroids.shape[:2] or
                centroids.shape[1] * centroids.shape[-1] != dim or
                suffix_keys.shape != (heads, dim, actual.shape[-1] - width) or
                actual.shape[:2] != (heads, queries) or queries == 0):
            raise ValueError('Invalid MHA correction shapes')
        if any(t.device != scaled_q.device for t in (centroids, codes, ids, suffix_keys, actual, mask)):
            raise ValueError('Correction must reuse metadata and attention on the same device')
        if codes.dtype != torch.long:
            raise ValueError('Reuse the selector int64 device codes')
        valid = ids >= 0
        tokens = ids.long().clamp_min(0)
        selected_codes = codes.gather(2, tokens[:, None].expand(-1, codes.shape[1], -1))
        mask = mask.reshape(actual.shape)
        predicted = torch.zeros((heads, width), device=scaled_q.device, dtype=torch.float32)
        suffix_keys = suffix_keys.float()
        subdim = centroids.shape[-1]
        for start in range(0, queries, 32):
            query = scaled_q[:, start:start + 32].float()
            logits = torch.zeros((heads, query.shape[1], actual.shape[-1]),
                                 device=query.device, dtype=torch.float32)
            for sub in range(centroids.shape[1]):
                lookup = torch.bmm(query[:, :, sub * subdim:(sub + 1) * subdim],
                                   centroids[:, sub].transpose(1, 2))
                logits[:, :, :width].add_(lookup.gather(
                    2, selected_codes[:, sub, None].expand(-1, query.shape[1], -1)))
            logits[:, :, width:] = torch.bmm(query, suffix_keys)
            logits.masked_fill_(~mask[:, start:start + query.shape[1]], -1e4)
            predicted.add_(logits.softmax(-1)[:, :, :width].sum(1))
        predicted /= queries
        truth = actual[:, :, :width].float().mean(1)
        ratio = (truth / predicted.clamp_min(self.epsilon)).masked_fill(~valid, 0)
        self.pending[layer] = (tokens, valid, ratio)


    @torch.inference_mode()
    def commit(self):
        # Called by finish_one_query after generation, before the next request
        # can select tokens. Preserve the original one-request EMA causality.
        for layer in list(self.deferred):
            self._observe_suffix(layer, *self.deferred[layer])
            del self.deferred[layer]
        if not self.pending:
            return
        return self._commit()

    @torch.inference_mode()
    def _commit(self):
        """Validate once per request, then update only observed GPU weights."""
        if not self.pending:
            return
        finite = torch.stack([torch.isfinite(ratio).all()
                              for _, _, ratio in self.pending.values()]).all()
        if not bool(finite):
            raise ValueError('Non-finite attention correction ratio')
        for layer, (tokens, valid, ratio) in self.pending.items():
            weights = self.weights[layer]
            old = weights.gather(1, tokens)
            updated = (1 - self.alpha) * old + self.alpha * ratio
            # Padding maps to token 0. Make every such write identical to the
            # genuine token-0 update (or its old weight when not observed).
            # Fixed shapes avoid GPU nonzero/boolean-index host synchronization.
            zero_slot = valid & (tokens == 0)
            zero_weight = torch.where(zero_slot.any(1),
                                      updated.masked_fill(~zero_slot, 0).sum(1),
                                      weights[:, 0])
            weights.scatter_(1, tokens, torch.where(valid, updated, zero_weight[:, None]))
        self.pending.clear()
        self.completed_requests += 1

    def clear(self):
        # Runtime drains the reorder worker before resetting a prefix.
        self._weights_ready = None
        self._demotion_read_done = None
        self.weights.clear()
        self.pending.clear()
        self.completed_requests = 0
        self.deferred.clear()
