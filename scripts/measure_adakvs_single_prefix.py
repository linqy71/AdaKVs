"""Measure one shared prefix with the retained LongBench timing boundaries.

TTFT = sum(timers('generate').costs), with one output token per request.
KV I/O = sum(timers('io part test').costs): native KV collection, including
cache lookup, read planning, host scatter and admission, excluding GPU copy.
Request 0 initializes the layout and is excluded from the reuse summary.
Prefix construction and finish_one_query maintenance are reported separately.
TTFT follows the model's internal loop boundary; generate_wall_seconds also
covers request setup and weight preloading outside that boundary. Synchronous
maintenance is the default; in async mode finish_seconds excludes background
work after submission. No OS page-cache flushing or read bandwidth cap is set.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'library/kvstore', ROOT / 'library/pqcode', ROOT):
    sys.path.insert(0, str(directory))

REFERENCE_ENV = {
    'KVSTORE_REORDER_SYNC': '1', 'HOT_GROWTH_READ_MODE': 'hybrid',
    'HOT_GROWTH_COLD_MERGE_BYTES': '8192',
    'HYBRID_CACHE_MISS_READ_MODE': 'auto', 'HYBRID_CACHE_HOT_OVERLAP': '0',
    'ACCESS_CONTROL_HOT_BYPASS': '0', 'ACCESS_CONTROL_ALPHA': '1',
    'ACCESS_CONTROL_L0_BYTES': '56000', 'PQ_CORRECTION_ALPHA': '.2',
    'PQ_CORRECTION_BETA': '.5', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--longbench-file', type=Path,
                        help='LongBench JSONL with context and input fields')
    source.add_argument('--prefix-file', type=Path, help='Shared UTF-8 context')
    parser.add_argument('--suffix', action='append',
                        help='Repeat for local requests; used with --prefix-file')
    parser.add_argument('--prefix-index', type=int, default=0,
                        help='Index among contexts with at least --min-questions')
    parser.add_argument('--min-questions', type=int, default=10)
    parser.add_argument('--max-requests', type=int, default=10,
                        help='Includes initialization request 0; minimum 2')
    parser.add_argument('--prefix-len', type=int, default=8000,
                        help='Maximum shared-prefix tokens; short contexts stay short')
    parser.add_argument('--model', default='facebook/opt-30b')
    parser.add_argument('--path', required=True, help='FlexLLMGen weight root')
    parser.add_argument('--tokenizer-path', required=True)
    parser.add_argument('--offload-dir', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--position-weights')
    parser.add_argument('--percent', type=int, nargs=6,
                        default=[100, 0, 100, 0, 100, 0])
    parser.add_argument('--no-overlap', action='store_true')
    parser.add_argument('--combined-layers', action='store_true')
    parser.add_argument('--adakvs-retention', type=float, default=.25)
    parser.add_argument('--adakvs-cache-gib', type=float, default=1.)
    parser.add_argument('--prefill-query-chunk-size', type=int, default=1024)
    parser.add_argument('--output', type=Path, default=Path('adakvs_timing.jsonl'),
                        help='New JSONL file; existing files are never overwritten')
    args = parser.parse_args()
    if args.max_requests < 2 or args.min_questions < 1 or args.prefix_index < 0:
        parser.error('Require max-requests >= 2, min-questions >= 1, prefix-index >= 0')
    if not 0 < args.prefix_len < args.max_length - 1:
        parser.error('Require 0 < prefix-len < max-length - 1')
    if args.prefix_file and (not args.suffix or len(args.suffix) < 2):
        parser.error('--prefix-file requires at least two --suffix values')
    if args.longbench_file and args.suffix:
        parser.error('--suffix is only used with --prefix-file')
    if not 0 < args.adakvs_retention <= 1:
        parser.error('Require 0 < adakvs-retention <= 1')
    if not math.isfinite(args.adakvs_cache_gib) or args.adakvs_cache_gib < 0:
        parser.error('Cache capacity must be finite and nonnegative')
    if args.prefill_query_chunk_size < 1:
        parser.error('prefill-query-chunk-size must be positive')
    if (any(x < 0 or x > 100 for x in args.percent)
            or any(sum(args.percent[i:i + 2]) > 100 for i in (0, 2, 4))
            or args.percent[2:4] != [100, 0]):
        parser.error('Require valid GPU/CPU percentage pairs and cache placement 100 0')
    return args


def load_context(args):
    if args.prefix_file:
        return args.prefix_file.expanduser().read_text(encoding='utf-8'), args.suffix[:args.max_requests]
    # Same exact-context grouping and source order as load_longbench_requests.
    grouped = {}
    with args.longbench_file.expanduser().open(encoding='utf-8') as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            context, question = row['context'], row['input']
            if not isinstance(context, str) or not isinstance(question, str):
                raise ValueError('LongBench context and input must be strings')
            grouped.setdefault(context, []).append(question)
    candidates = [(context, questions) for context, questions in grouped.items()
                  if len(questions) >= args.min_questions]
    if args.prefix_index >= len(candidates):
        raise ValueError(f'Only {len(candidates)} contexts meet --min-questions')
    context, questions = candidates[args.prefix_index]
    questions = questions[:args.max_requests]
    if len(questions) < 2:
        raise ValueError('Need initialization request 0 and at least one reuse request')
    return context, questions


def prepare_inputs(tokenizer, context, questions, prefix_limit, max_length):
    # Match _full_prefix_inputs: tokenize complete prompts, find their common
    # token prefix, truncate only its left side, and preserve all suffix tokens.
    inputs = tokenizer([context + question + '\n' for question in questions],
                       truncation=False).input_ids
    common_len = 0
    for tokens in zip(*inputs):
        if len(set(tokens)) != 1:
            break
        common_len += 1
    common_len = min(common_len, min(map(len, inputs)) - 1)
    suffix_max = max(len(tokens) - common_len for tokens in inputs)
    prefix_len = min(common_len, prefix_limit, max_length - 1 - suffix_max)
    if prefix_len < 2:
        raise ValueError('Insufficient shared prefix or suffix exceeds context budget')
    prefix = inputs[0][common_len - prefix_len:common_len]
    return prefix, [prefix + tokens[common_len:] for tokens in inputs]


def emit(output, record):
    output.write(json.dumps(record, ensure_ascii=False) + '\n')
    output.flush()


def percentile95(values):
    ordered = sorted(values)
    position = (len(ordered) - 1) * .95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def measure_requests(model, prefix, inputs, output):
    from flexllmgen.timer import timers

    model.sync()
    for timer in timers.timers.values():
        timer.reset()
    initial_ratio = model.kv_server.initial_attention_topk_ratio
    model.kv_server.initial_attention_topk_ratio = None
    started = time.perf_counter()
    try:
        model.generate([prefix], max_new_tokens=1, req_id=-1)
        model.sync()
    finally:
        model.kv_server.initial_attention_topk_ratio = initial_ratio
    prefix_seconds = time.perf_counter() - started
    emit(output, {'type': 'prefix_build', 'seconds': prefix_seconds,
                  'pq_build_seconds': timers('build pq meta').elapsed('sum')})
    print(f'Prefix build: {prefix_seconds:.6f} s (excluded from reuse statistics)', flush=True)
    records = []
    for request_id, tokens in enumerate(inputs):
        model.sync()
        for timer in timers.timers.values():
            timer.reset()
        started = time.perf_counter()
        generated = model.generate([tokens], max_new_tokens=1, req_id=request_id)
        model.sync()
        generate_wall = time.perf_counter() - started
        if len(timers('generate').costs) != 1:
            raise RuntimeError('Expected one generate timer sample for one output token')
        ttft = timers('generate').elapsed('sum')
        io_seconds = timers('io part test').elapsed('sum')
        # finish_one_query clears native counters and commits correction/layout.
        # Read TTFT and collection time before that boundary.
        finish_started = time.perf_counter()
        model.finish_one_query()
        finish_seconds = time.perf_counter() - finish_started
        record = {
            'type': 'request', 'request_id': request_id,
            'phase': 'initialization' if request_id == 0 else 'reuse',
            'prefix_len': len(prefix), 'suffix_len': len(tokens) - len(prefix),
            'prefill_time': ttft, 'kv_io_time': io_seconds,
            'generate_wall_seconds': generate_wall,
            'finish_seconds': finish_seconds,
            'request_seconds_including_reorder': time.perf_counter() - started,
            'correction_update_time': model.correction_update_time,
            'reorder_sync_wait_time': model.reorder_sync_wait_time,
            'reorder_execution_mode': model.reorder_execution_mode,
            'output_token_id': int(generated[0, -1]),
        }
        emit(output, record)
        records.append(record)
        print(f"request={request_id:02d} phase={record['phase']} "
              f"TTFT={ttft * 1000:.3f} ms KV_IO={io_seconds * 1000:.3f} ms "
              f"finish={finish_seconds * 1000:.3f} ms", flush=True)
    model.kv_server.wait_for_reorder()
    measured = records[1:]
    summary = {'type': 'summary', 'reuse_requests': len(measured),
               'excluded_request_ids': [0]}
    for source, name in (('prefill_time', 'ttft'), ('kv_io_time', 'kv_io')):
        samples = [record[source] * 1000 for record in measured]
        summary[f'mean_{name}_ms'] = statistics.mean(samples)
        summary[f'p95_{name}_ms'] = percentile95(samples)
    emit(output, summary)
    print('REUSE SUMMARY ' + json.dumps(summary), flush=True)


def run(args):
    for name, value in REFERENCE_ENV.items():
        os.environ.setdefault(name, value)
    import torch
    from transformers import AutoTokenizer
    from flexllmgen.compression import CompressionConfig
    from flexllmgen.opt_config import get_opt_config
    from flexllmgen.retrieval_opt_adakvs import OptLM, Policy
    from flexllmgen.pytorch_backend_adakvs import TorchDevice, TorchDisk, TorchMixedDevice
    from flexllmgen.utils import ExecutionEnv

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for TTFT measurement')
    context, questions = load_context(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    prefix, inputs = prepare_inputs(tokenizer, context, questions,
                                    args.prefix_len, args.max_length)
    config = get_opt_config(args.model, max_seq_len=args.max_length,
                            position_weights_path=args.position_weights)
    if any(config.pad_token_id in tokens for tokens in inputs):
        raise ValueError('AdaKVs requires unpadded inputs')
    policy = Policy(1, 1, *args.percent, not args.no_overlap,
                    not args.combined_layers, True, False, 1., False,
                    CompressionConfig(4, 64, 0, False), False,
                    CompressionConfig(4, 64, 2, False))
    args.offload_dir = args.offload_dir.expanduser()
    args.output = args.output.expanduser()
    args.offload_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        emit(output, {'type': 'config',
                      'arguments': {k: str(v) if isinstance(v, Path) else v
                                    for k, v in vars(args).items()},
                      'prefix_len': len(prefix), 'requests': len(inputs),
                      'environment': {k: os.environ.get(k) for k in (
                          *REFERENCE_ENV, 'KVSTORE_READ_BPS',
                          'FLEXGEN_RETRIEVAL_PIPELINE')},
                      'torch_version': torch.__version__,
                      'gpu': torch.cuda.get_device_name(0)})
        print(f'Prefix tokens={len(prefix)}, requests={len(inputs)}, output={args.output}', flush=True)
        with tempfile.TemporaryDirectory(prefix='adakvs-timing-', dir=args.offload_dir) as directory:
            gpu, cpu = TorchDevice('cuda:0'), TorchDevice('cpu')
            disk = TorchDisk(directory)
            env = ExecutionEnv(gpu, cpu, disk, TorchMixedDevice([gpu, cpu, disk]))
            model = None
            try:
                model = OptLM(config, env, args.path, directory, policy,
                              max(map(len, inputs)), 1, 10, 100,
                              compaction_threshold=1.,
                              pq_retention_ratio=args.adakvs_retention,
                              kv_cache_capacity_bytes=int(args.adakvs_cache_gib * (1 << 30)),
                              merge_extra_read_budget=.15)
                model.kv_server.prefill_query_chunk_size = args.prefill_query_chunk_size
                with torch.inference_mode():
                    measure_requests(model, prefix, inputs, output)
            finally:
                try:
                    if model is not None:
                        try:
                            model.final_finish()
                        finally:
                            model.delete_all_weights()
                finally:
                    env.close_copy_threads()


if __name__ == '__main__':
    run(parse_args())
