"""Correctness checks for the PLE experiment; synthetic data is not a benchmark."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from ple_features import GGUFTable, hash_rows
from ple_experiment import partition, load_rows, vote_features, network, warm_start, replay_positions
import ngram_study as ns


class PLETests(unittest.TestCase):
    def test_hash_against_scalar_runtime_rules(self):
        rng = np.random.default_rng(14)
        ctx = rng.integers(-1, 41, (100, 3))
        ctx[:, 0] = rng.integers(0, 41, 100)
        ctx[:3] = [[3, 9, 4], [9, 3, 4], [3, -1, 4]]
        mult = [2**63 + 13, 2**62 + 17, 2**61 + 19]
        vocabs, offsets = [5, 7, 11, 13], [0, 5, 12, 23]
        expected = []
        for row in ctx:
            c, cut = [int(row[0])], False
            for t in row[1:]:
                cut = cut or t < 0 or t == 9
                c.append(9 if cut else int(t))
            heads = []
            mixed = (c[0] * mult[0]) & ((1 << 64) - 1)
            for j in (1, 2):
                mixed ^= (c[j] * mult[j]) & ((1 << 64) - 1)
                for h in range(2 * (j - 1), 2 * j):
                    heads.append(mixed % vocabs[h] + offsets[h])
            expected.append(heads)
        np.testing.assert_array_equal(hash_rows(ctx, mult, offsets, vocabs, 2, 9), expected)

    def test_duplicate_prompts_share_split(self):
        a = dict(id='a', prompt_ids=[1, 2, 3])
        self.assertEqual(partition(a), partition(dict(a, id='b')))

    def test_wrong_target_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'rows.jsonl'
            p.write_text(json.dumps(dict(target_id='wrong', temperature=0, prompt_ids=[1], answer_ids=[2])) + '\n')
            with self.assertRaises(ValueError):
                load_rows(p, 'right')

    def test_replay_limit_keeps_complete_requests(self):
        rows = [dict(prompt_ids=[1, 2, 3], answer_ids=[4] * 5, amax=[4] * 5,
                     source='synthetic', thinking=False) for _ in range(3)]
        data = ns.Data(rows)
        idx = replay_positions(data, 3, 7)
        self.assertEqual(len(idx), 5)
        self.assertEqual(data.pos[idx[-1]], 7)

    def test_actual_reranker_warm_start(self):
        import torch
        path = Path(__file__).parent / 'data/code/9b-code-pod-c3.mdr'
        if not path.exists():
            self.skipTest('local trained reranker is unavailable')
        width = 8 * 12 + 9 + 7 * 64
        base, extra = network(width), network(width + 640)
        warm_start(base, path, 6, width)
        warm_start(extra, path, 6, width)
        x = torch.randn(4, width)
        with torch.no_grad():
            torch.testing.assert_close(base(x), extra(torch.cat([x, torch.randn(4, 640)], 1)))
        # Hidden layers are copied verbatim; only the first layer changes shape.
        self.assertEqual(base[2].weight.shape, (256, 512))
        self.assertGreater(float(base[2].weight.detach().abs().sum()), 1)

    def test_empty_candidates(self):
        K = 3
        cands = [('static', k, np.array([False]), np.array([0]), (np.array([0]), np.array([0]))) for k in range(1, K + 1)]
        ii, tok, X = vote_features(cands, ns.Calibration(K), np.array([[1, 2, 3]]), K, np.zeros(10))
        self.assertEqual(len(ii), 0)
        self.assertEqual(X.shape, (0, 8 * 2 * K + 9))

    def test_gguf_gather_and_training_smoke(self):
        from gguf import GGUFWriter
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            path = td / 'tiny.gguf'
            w = GGUFWriter(str(path), 'qwen4exp')
            for key, values in [('layer_multipliers', [3, 7, 13]), ('head_offsets', [0, 5, 12, 23]),
                                ('head_vocab_sizes', [5, 7, 11, 13])]:
                w.add_array('qwen4exp.ple.' + key, values)
            w.add_uint32('qwen4exp.ple.heads_per_ngram', 2)
            w.add_uint32('qwen4exp.ple.eos_token_id', 9)
            rng = np.random.default_rng(11)
            table = rng.standard_normal((36, 32)).astype(np.float32)
            w.add_tensor('per_layer_token_embd.weight', table)
            w.add_tensor('token_embd.weight', rng.standard_normal((64, 32)).astype(np.float32))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            ple = GGUFTable(path, [0, 2])
            C = np.array([[3, 4, 5], [9, 3, 4]])
            ids = hash_rows(C, [3, 7, 13], [0, 5, 12, 23], [5, 7, 11, 13], 2, 9)
            np.testing.assert_allclose(ple.features(C), table[ids[:, [0, 2]]].reshape(2, -1))
            self.assertEqual(ple.candidate_features(C, np.array([6, 7])).shape, (2, 128))
            # Cover all prompt-disjoint splits with repeated continuations for lookup coverage.
            rows = []
            for i in range(200):
                rows.append(dict(id=str(i), target_id='synthetic', temperature=0,
                                 prompt_ids=[10 + i // 50, 10 + i % 50, 1, 2, 3, 1, 2, 3],
                                 answer_ids=[1, 2, 3] * 4))
            rpath = td / 'rows.jsonl'
            rpath.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            done = subprocess.run([sys.executable, str(Path(__file__).with_name('ple_experiment.py')),
                                   '--gguf', str(path), '--heads', '0,2', '--rows', str(rpath),
                                   '--target-id', 'synthetic', '--out', str(td / 'out'), '--K', '3',
                                   '--epochs', '1', '--max-positions', '50', '--threads', '1'],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            result = json.loads((td / 'out/results.json').read_text())
            self.assertEqual(len(result['arms']), 4)
            for arm in result['arms'].values():
                self.assertGreater(arm['depth_3']['tokens'], 0)
            binary = os.environ.get('MEMDRAFT_PLE_REPLAY')
            if binary:
                test_rows = [dict(r, amax=r['answer_ids'], source='synthetic', thinking=False)
                             for r in rows if partition(r) == 'test']
                data = ns.Data(test_rows)
                idx = replay_positions(data, 3, 50)
                selected = test_rows[:int(data.seq[idx[-1]]) + 1]
                ns.write_stream(ns.Data(selected), str(td / 'test.stream'))
                for arm in ('reranker', 'reranker_ple'):
                    got = subprocess.run([binary, str(td / 'out/lookup.mdt'), str(td / 'test.stream'),
                                          '--pad', '1', '--n-max', '3', '--ple-reranker', str(td / f'out/{arm}.mdp')],
                                         capture_output=True, text=True)
                    self.assertEqual(got.returncode, 0, got.stderr)
                    native = json.loads(got.stdout)
                    for key in ns.COUNTS:
                        self.assertEqual(native[key], result['arms'][arm]['depth_3'][key], (arm, key, native))

    def test_explicit_head_surgery_layout(self):
        spec = importlib.util.spec_from_file_location('surgery', Path(__file__).parent / 'research/qwen38-head-selection/surgery_qwen38.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Field:
            def __init__(self, value): self.value = value
            def contents(self): return self.value
        class Reader:
            fields = {module.K_HEAD_OFFSETS: Field([0, 5, 12, 23]),
                      module.K_HEAD_VOCABS: Field([5, 7, 11, 13]),
                      'qwen4exp.ple.heads_per_ngram': Field(2)}
        offsets, vocabs, segments, n = module.plan_ple([Reader()], 1, [1, 3])
        self.assertEqual(vocabs, [1, 7, 1, 13])
        self.assertEqual(offsets, [20, 0, 20, 7])
        self.assertEqual(segments, [(5, 7), (23, 13), ('zero', 1)])
        self.assertEqual(n, 21)
        with self.assertRaises(ValueError):
            module.plan_ple([Reader()], 1, [4])


if __name__ == '__main__':
    unittest.main()
