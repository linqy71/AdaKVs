"""AdaKVs options and local-text runner for the FlexLLMGen entry point."""
import os
from pathlib import Path
import tempfile


def add_parser_arguments(parser):
    group = parser.add_argument_group('AdaKVs')
    group.add_argument('--adakvs', action='store_true')
    group.add_argument('--prefix-file', type=Path)
    group.add_argument('--suffix', action='append', help='Repeat for shared-prefix requests')
    group.add_argument('--tokenizer-path')
    group.add_argument('--max-length', type=int, default=8192)
    group.add_argument('--position-weights', help='OPT position table matching --max-length')
    group.add_argument('--adakvs-retention', type=float, default=.25)
    group.add_argument('--adakvs-cache-gib', type=float, default=1.)
    group.add_argument('--prefill-query-chunk-size', type=int, default=1024)


def run(args):
    # These are the retained reference-path defaults, not experiment switches.
    for name, value in {
        'KVSTORE_REORDER_SYNC': '1', 'HOT_GROWTH_READ_MODE': 'hybrid',
        'HOT_GROWTH_COLD_MERGE_BYTES': '8192',
        'HYBRID_CACHE_MISS_READ_MODE': 'auto', 'HYBRID_CACHE_HOT_OVERLAP': '0',
        'ACCESS_CONTROL_HOT_BYPASS': '0', 'ACCESS_CONTROL_ALPHA': '1',
        'ACCESS_CONTROL_L0_BYTES': '56000', 'PQ_CORRECTION_ALPHA': '.2',
        'PQ_CORRECTION_BETA': '.5',
    }.items():
        os.environ.setdefault(name, value)

    import math
    import torch
    from transformers import AutoTokenizer
    from flexllmgen.compression import CompressionConfig
    from flexllmgen.opt_config import get_opt_config, validate_opt_context_length
    from flexllmgen.retrieval_opt_adakvs import OptLM, Policy
    from flexllmgen.pytorch_backend_adakvs import TorchDevice, TorchDisk, TorchMixedDevice
    from flexllmgen.utils import ExecutionEnv

    if args.prefix_file is None or not args.suffix:
        raise ValueError('--adakvs requires --prefix-file and at least one --suffix')
    if args.gen_len != 1 or args.debug_mode is not None or args.cut_gen_len is not None:
        raise ValueError('The retained online-correction path requires --gen-len 1 and complete requests')
    if args.gpu_batch_size != 1 or args.num_gpu_batches != 1:
        raise ValueError('AdaKVs requires --gpu-batch-size 1 --num-gpu-batches 1')
    if (len(args.percent) != 6 or any(x < 0 or x > 100 for x in args.percent)
            or any(sum(args.percent[i:i + 2]) > 100 for i in (0, 2, 4))):
        raise ValueError('--percent requires three valid GPU/CPU percentage pairs')
    if not math.isfinite(args.adakvs_cache_gib) or args.adakvs_cache_gib < 0:
        raise ValueError('Cache capacity must be finite and nonnegative')
    if args.prefill_query_chunk_size < 1:
        raise ValueError('Prefill query chunk size must be positive')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model)
    prefix = tokenizer.encode(args.prefix_file.read_text(encoding='utf-8'), add_special_tokens=True)
    suffixes = [tokenizer.encode(text, add_special_tokens=False) for text in args.suffix]
    if not prefix or any(not suffix for suffix in suffixes):
        raise ValueError('Prefix and suffix token sequences must be nonempty')
    prompt_len = max(len(prefix) + len(suffix) for suffix in suffixes)
    validate_opt_context_length(prompt_len, 1, args.max_length)
    config = get_opt_config(args.model, max_seq_len=args.max_length,
                            position_weights_path=args.position_weights)
    if config.pad_token_id in prefix or any(config.pad_token_id in s for s in suffixes):
        raise ValueError('AdaKVs expects unpadded requests')
    policy = Policy(args.gpu_batch_size, args.num_gpu_batches, *args.percent,
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity, args.compress_weight,
                    CompressionConfig(4, 64, 0, False), args.compress_cache,
                    CompressionConfig(4, 64, 2, False))
    offload = Path(args.offload_dir).expanduser()
    offload.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='adakvs-', dir=offload) as directory:
        gpu, cpu = TorchDevice('cuda:0'), TorchDevice('cpu')
        disk = TorchDisk(directory)
        env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk,
                           mixed=TorchMixedDevice([gpu, cpu, disk]))
        model = None
        try:
            model = OptLM(config=config, env=env, path=args.path, offload_dir=directory,
                          policy=policy, max_prompt_len=prompt_len, max_gen_len=1,
                          K=10, L=100, compaction_threshold=1.,
                          pq_retention_ratio=args.adakvs_retention,
                          kv_cache_capacity_bytes=int(args.adakvs_cache_gib * (1 << 30)),
                          merge_extra_read_budget=.15)
            model.kv_server.prefill_query_chunk_size = args.prefill_query_chunk_size
            # Preserve the original prefix-only build. The first real suffix
            # supplies initial ranking before finish_one_query persists KV.
            initial_ratio = model.kv_server.initial_attention_topk_ratio
            model.kv_server.initial_attention_topk_ratio = None
            with torch.inference_mode():
                model.generate([prefix], max_new_tokens=1, req_id=-1)
                model.sync()
                model.kv_server.initial_attention_topk_ratio = initial_ratio
                for request_id, suffix in enumerate(suffixes):
                    output = model.generate([prefix + suffix], max_new_tokens=1, req_id=request_id)
                    model.finish_one_query()
                    print(tokenizer.decode(output[0, -1:], skip_special_tokens=True))
            model.kv_server.wait_for_reorder()
        finally:
            try:
                if model is not None:
                    try:
                        model.final_finish()
                    finally:
                        model.delete_all_weights()
            finally:
                env.close_copy_threads()
