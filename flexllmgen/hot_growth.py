"""Configuration for native, foreground-priority guarded hot growth."""
from dataclasses import asdict, dataclass
import math
import os


@dataclass(frozen=True)
class HotGrowthConfig:
    budget_multiplier: float = 1.5
    initial_multiplier: float = .5
    warm_min_multiplier: float = .2
    candidate_retention: float = .6
    growth_multiplier: float = .2
    half_life: float = 8.
    admission_margin: float = .02
    warmup: int = 1
    max_write_bytes: int = 1024 * 1024

    def initial_ratio(self, retention):
        return retention * self.budget_multiplier

    def initial_count(self, length, retention):
        return math.floor(length * self.initial_ratio(retention) + 1e-9)

    def validate(self, retention):
        if not all(math.isfinite(x) for x in [retention, *asdict(self).values()]):
            raise ValueError('Hot growth parameters must be finite')
        if (not 0 < retention <= 1 or self.budget_multiplier <= 0 or
                self.budget_multiplier * retention > 1 or
                self.initial_multiplier < 0 or self.warm_min_multiplier <= 0 or
                self.initial_multiplier + self.warm_min_multiplier > self.budget_multiplier or
                not 0 <= self.candidate_retention <= 1 or self.growth_multiplier <= 0 or
                self.half_life <= 0 or self.admission_margin < 0 or
                self.warmup < 1 or self.max_write_bytes < 1):
            raise ValueError('Invalid hot growth budgets, retention gate or work limits')
        return self

    def configure_store(self, store, retention):
        self.validate(retention)
        if not all(hasattr(store, name) for name in ('configure_hot_growth', 'get_initial_key_view')):
            raise RuntimeError('Rebuild library/kvstore for guarded_hot_growth')
        store.configure_hot_growth(
            retention, self.budget_multiplier, self.initial_multiplier,
            self.warm_min_multiplier, self.candidate_retention,
            self.growth_multiplier, self.half_life, self.admission_margin,
            self.warmup, self.max_write_bytes)
        configure_read_mode(store)


def configure_read_mode(store):
    mode = os.environ.get('HOT_GROWTH_READ_MODE', 'hybrid')
    if mode != 'hybrid':
        raise ValueError('AdaKVs requires HOT_GROWTH_READ_MODE=hybrid')
    if hasattr(store, 'set_hot_growth_read_mode'):
        store.set_hot_growth_read_mode(
            mode, int(os.environ.get('HOT_GROWTH_FIXED_BLOCK_BYTES', '16384') if mode == 'fixed'
                      else os.environ.get('HOT_GROWTH_COLD_MERGE_BYTES', '8192')))
    elif mode != 'adaptive':
        raise RuntimeError('Rebuild library/kvstore for hybrid hot growth reads')


def add_arguments(parser, include_adaptive=True):
    defaults = HotGrowthConfig()
    for name, value in asdict(defaults).items():
        parser.add_argument('--hot-growth-' + name.replace('_', '-'),
                            type=type(value), default=value,
                            help=f'guarded_hot_growth {name}; default: {value}')


def from_arguments(args):
    return HotGrowthConfig(**{
        name: getattr(args, 'hot_growth_' + name, value)
        for name, value in asdict(HotGrowthConfig()).items()})


def initial_attention_topk(query, keys, count, chunk_size=128):
    """Rank prefix KV using the first real request, once before persistence.

    query: [queries, attention heads, dim]; keys: [KV heads, prefix, dim].
    Keep dense scores on the query device and transfer only the sorted top-k.
    """
    import torch

    with torch.inference_mode():
        if (query.ndim != 3 or keys.ndim != 3 or query.shape[0] < 1 or
                keys.shape[0] < 1 or query.shape[1] % keys.shape[0] or
                query.shape[2] != keys.shape[2] or not 1 <= count <= keys.shape[1] or
                chunk_size < 1):
            raise ValueError('Invalid initial hot growth attention shapes/budget')
        heads, length, dim = keys.shape
        groups = query.shape[1] // heads
        q = query.transpose(0, 1)
        result = torch.empty((heads, count), dtype=torch.int32)
        for head in range(heads):
            key = keys[head].to(device=q.device, dtype=torch.float32).T
            scores = torch.zeros(length, device=q.device, dtype=torch.float64)
            for start in range(0, q.shape[1], chunk_size):
                chunk = q[head*groups:(head+1)*groups, start:start+chunk_size].float()
                probabilities = ((chunk @ key) * dim**-.5).softmax(-1)
                scores.add_(probabilities.sum((0, 1), dtype=torch.float64))
            result[head].copy_(scores.topk(count, sorted=True).indices)
        return torch.full((heads,), count, dtype=torch.int32), result
