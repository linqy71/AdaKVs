"""Basic AdaKVs checks without model downloads or external datasets.

Install the checkout, library/pqcode and library/kvstore first.
Use --require-cuda to require the tiny OPT runtime check.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kvstore import KVStore
from pqcode import PQCodeIndex
from flexllmgen.pq_online_correction import PQOnlineCorrection
from flexllmgen.hot_growth import HotGrowthConfig

WORK_DIR = None


class CoreChecks(unittest.TestCase):
    def test_original_entry(self):
        from flexllmgen.flex_opt import add_parser_arguments
        parser = argparse.ArgumentParser()
        add_parser_arguments(parser)
        self.assertFalse(parser.parse_args([]).adakvs)
        self.assertTrue(parser.parse_args(['--adakvs']).adakvs)
        from flexllmgen.retrieval_opt_adakvs import OptLM
        self.assertTrue(callable(OptLM.generation_loop_retrieval_pipeline))

    def test_pq_roundtrip_and_deferred_correction(self):
        torch.manual_seed(7)
        index = PQCodeIndex()
        index.alloc(2, 6, 1, 2, 2, 1, 64, 8, .25)
        keys = torch.randn(2, 32, 8)
        index.build(0, 0, keys)
        query = torch.randn(2, 2, 8)
        results = torch.zeros(2, 64, dtype=torch.int32)
        counts = torch.zeros(2, dtype=torch.int32)
        index.batch_retrieve_multi(0, query, results, counts, 32)
        self.assertEqual(counts.tolist(), [8, 8])
        expected = ((query.transpose(0, 1) @ keys.transpose(1, 2)) * 8 ** -.5).softmax(-1).sum(1)
        for head in range(2):
            self.assertEqual(set(results[head, :8].tolist()), set(expected[head].topk(8).indices.tolist()))
        state = index.state_dict(0)
        index.clear()
        index.load_state_dict(state)
        metadata = index.prepare_device_metadata(0, 0, 'cpu')
        self.assertIs(metadata, index.prepare_device_metadata(0, 0, 'cpu'))
        correction = PQOnlineCorrection(alpha=.2, beta=.5)
        correction.apply(0, torch.ones(2, 32))
        ids = results[:, :8].long()
        selected_keys = keys.gather(1, ids[..., None].expand(-1, -1, 8)).transpose(1, 2)
        q = query.transpose(0, 1) * 8 ** -.5
        truth = (q @ selected_keys).softmax(-1)
        mask = torch.ones(2, 2, 8, dtype=torch.bool)
        correction.defer_observation(0, metadata, ids, q, selected_keys, truth, mask)
        self.assertEqual(correction.completed_requests, 0)
        self.assertTrue(correction.deferred)
        correction.commit()
        self.assertEqual(correction.completed_requests, 1)
        torch.testing.assert_close(correction.weights[0], torch.ones(2, 32), rtol=1e-5, atol=1e-5)

    def test_native_direct_output_cache_and_reorder(self):
        torch.manual_seed(19)
        keys = torch.randn(2, 128, 16).half()
        values = torch.randn_like(keys)
        for capacity in (0, 32768):
            with self.subTest(cache_bytes=capacity), tempfile.TemporaryDirectory(dir=WORK_DIR) as directory:
                store = KVStore()
                store.alloc(1, 2, 2, 16, 128)
                store.configure_cache(capacity, 'access_control')
                HotGrowthConfig().configure_store(store, .25)
                store.set_merge_extra_read_budget(.15)
                store.fill(0, keys, values)
                initial = torch.tensor([[h, t] for h in range(2) for t in range(48)])
                store.reorder_persist(directory, 1, 0, initial)
                store.finish_persist()
                cold = Path(directory) / '1_layer0_part1.bin'
                cold_bytes = cold.read_bytes()
                ids = torch.zeros(2, 128, dtype=torch.int32)
                slabs = [torch.full((2 * 130 * 16 + 11,), -77., dtype=torch.float16) for _ in range(2)]
                for request in range(1, 9):
                    rows = [[0, request + 48, 127, 0], [17, request + 80]]
                    counts = torch.tensor([len(row) for row in rows], dtype=torch.int32)
                    for h, row in enumerate(rows):
                        ids[h, :len(row)] = torch.tensor(row, dtype=torch.int32)
                    for slab in slabs:
                        slab.fill_(-77.)
                    selected, width = store.merge_collect_into(1, 0, ids, counts, *slabs, 2)
                    self.assertEqual(width, 4)
                    live = 2 * (width + 2) * 16
                    for slab, reference in zip(slabs, (keys, values)):
                        packed = slab[:live].reshape(2, width + 2, 16)
                        for h, row in enumerate(rows):
                            torch.testing.assert_close(packed[h, :len(row)], reference[h, row], rtol=0, atol=0)
                            self.assertTrue(bool((packed[h, len(row):] == 0).all()))
                        self.assertTrue(bool((slab[live:] == -77).all()))
                    before = store.get_hot_growth_layout(0)
                    for slab in slabs:
                        slab.fill_(float('nan'))  # Snapshot must own its payload.
                    selection = torch.tensor([[h, t] for h, row in enumerate(rows) for t in row])
                    store.hot_growth_update(directory, 1, 0, selection, request)
                    after = store.get_hot_growth_layout(0)
                    for h in range(2):
                        hot = int(before[h, 0])
                        torch.testing.assert_close(after[h, 1:hot + 1], before[h, 1:hot + 1])
                    self.assertEqual(cold.read_bytes(), cold_bytes)
                for partition in store.get_cache_partition_stats():
                    self.assertLessEqual(partition['used_bytes'], partition['capacity_bytes'])
                del store

    def test_native_cache_policy(self):
        with tempfile.TemporaryDirectory(dir=WORK_DIR) as directory:
            binary = Path(directory) / 'cache-check'
            subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(ROOT / 'library/kvstore'),
                            str(ROOT / 'scripts/test_adakvs_cache.cc'), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for the FlexLLMGen runtime')
class RuntimeCheck(unittest.TestCase):
    def test_tiny_opt_shared_prefix(self):
        from flexllmgen.compression import CompressionConfig
        from flexllmgen.opt_config import OptConfig
        from flexllmgen.retrieval_opt_adakvs import OptLM, Policy
        from flexllmgen.pytorch_backend_adakvs import TorchDevice, TorchDisk, TorchMixedDevice
        from flexllmgen.utils import ExecutionEnv

        with tempfile.TemporaryDirectory(dir=WORK_DIR) as directory:
            weights = Path(directory) / 'opt-adakvs-smoke-np'
            weights.mkdir()
            rng = np.random.default_rng(11)

            def save(name, shape, norm=False):
                array = np.ones(shape) if norm else rng.normal(0, .08, shape)
                with (weights / name).open('wb') as output:
                    np.save(output, array.astype(np.float16))

            save('decoder.embed_tokens.weight', (256, 32))
            save('decoder.et_embed_positions.weight', (162, 32))
            save('decoder.layer_norm.weight', (32,), True)
            save('decoder.layer_norm.bias', (32,))
            for layer in range(2):
                base = f'decoder.layers.{layer}.'
                for projection in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
                    save(base + 'self_attn.' + projection + '.weight', (32, 32))
                    save(base + 'self_attn.' + projection + '.bias', (32,))
                for norm in ('self_attn_layer_norm', 'final_layer_norm'):
                    save(base + norm + '.weight', (32,), True)
                    save(base + norm + '.bias', (32,))
                save(base + 'fc1.weight', (128, 32))
                save(base + 'fc1.bias', (128,))
                save(base + 'fc2.weight', (32, 128))
                save(base + 'fc2.bias', (32,))
            # Original initialization probes this filename before loading.
            (weights / 'decoder.embed_positions.weight').write_bytes(
                (weights / 'decoder.et_embed_positions.weight').read_bytes())
            config = OptConfig(name='opt-adakvs-smoke', num_hidden_layers=2, max_seq_len=160,
                               hidden_size=32, n_head=2, input_dim=32, ffn_embed_dim=128, vocab_size=256)
            for overlap, separate, cpu_weights in ((False, True, False), (True, False, True)):
                gpu, cpu = TorchDevice('cuda:0'), TorchDevice('cpu')
                disk = TorchDisk(str(Path(directory) / 'offload'))
                env = ExecutionEnv(gpu, cpu, disk, TorchMixedDevice([gpu, cpu, disk]))
                policy = Policy(1, 1, 0 if cpu_weights else 100, 100 if cpu_weights else 0,
                                100, 0, 100, 0, overlap, separate, True, False, 1., False,
                                CompressionConfig(4, 64, 0, False), False, CompressionConfig(4, 64, 2, False))
                model = None
                try:
                    model = OptLM(config, env, directory, disk.path, policy, 132, 1, 10, 100,
                                  kv_cache_capacity_bytes=32768, merge_extra_read_budget=.15)
                    prefix = list(range(2, 130))
                    ratio = model.kv_server.initial_attention_topk_ratio
                    model.kv_server.initial_attention_topk_ratio = None
                    with torch.inference_mode():
                        model.generate([prefix], max_new_tokens=1, req_id=-1)
                        model.sync()
                        model.kv_server.initial_attention_topk_ratio = ratio
                        for request in range(3):
                            output = model.generate([prefix + [140 + request]], max_new_tokens=1, req_id=request)
                            model.finish_one_query()
                            self.assertEqual(output.shape, (1, 130))
                        model.kv_server.wait_for_reorder()
                        self.assertTrue(model.kv_server.persisted)
                finally:
                    try:
                        if model is not None:
                            model.final_finish()
                            model.delete_all_weights()
                    finally:
                        env.close_copy_threads()


if __name__ == '__main__':
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir')
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    WORK_DIR = args.work_dir
    if args.require_cuda and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    os.environ['HOT_GROWTH_READ_MODE'] = 'hybrid'
    os.environ['HYBRID_CACHE_HOT_OVERLAP'] = '0'
    os.environ['KVSTORE_REORDER_SYNC'] = '1'
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
