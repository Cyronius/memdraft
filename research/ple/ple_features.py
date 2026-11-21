"""Read Qwen4exp ngram features from mmap'd GGUF rows, without copying the table.

Hashing follows llama.cpp src/models/qwen4exp.cpp set_input: newest token first,
uint64 wrapping products, XOR, per-head modulus, EOS-padded predecessors.
"""
from functools import lru_cache
from pathlib import Path
import re

import numpy as np


def hash_rows(tokens, multipliers, offsets, vocabs, per_gram, eos):
    """tokens is [batch, newest-first window]; negative predecessors mean missing.

    The current token itself may be EOS; only predecessors trigger a reset.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[1] != len(multipliers):
        raise ValueError('one newest-first column per multiplier is required')
    if np.any(tokens[:, 0] < 0):
        raise ValueError('current token must be present')
    if len(vocabs) != (len(multipliers) - 1) * per_gram or len(offsets) != len(vocabs):
        raise ValueError('invalid head layout')
    if np.any(np.asarray(vocabs) <= 0) or np.any(np.asarray(offsets) < 0):
        raise ValueError('invalid head offsets or vocab sizes')
    ctx = tokens.copy()
    cut = np.zeros(len(ctx), bool)
    for j in range(1, ctx.shape[1]):
        cut |= (ctx[:, j] < 0) | (ctx[:, j] == eos)
        ctx[cut, j] = eos
    with np.errstate(over='ignore'):
        products = ctx.astype(np.uint64) * np.asarray(multipliers, np.uint64)
    mixed = products[:, 0].copy()
    result = []
    for j in range(1, ctx.shape[1]):
        mixed ^= products[:, j]
        for h in range((j - 1) * per_gram, j * per_gram):
            result.append(mixed % np.uint64(vocabs[h]) + np.uint64(offsets[h]))
    return np.stack(result, axis=1).astype(np.int64)


class GGUFTable:
    """Lazy, bounded dequantization cache. The original GGUF remains mmap'd.

    No projected copy of the multi-GB PLE table is materialized. Quantized rows
    are dequantized on demand; this Python implementation is for offline study.
    """
    def __init__(self, path, heads=None, cache_rows=32768):
        from gguf import GGUFReader
        path = Path(path)
        m = re.match(r'(.*)-(\d{5})-of-(\d{5})\.gguf$', path.name)
        paths = ([path.parent / f'{m[1]}-{i:05d}-of-{m[3]}.gguf'
                  for i in range(1, int(m[3]) + 1)] if m else [path])
        self.paths = paths
        self.readers = [GGUFReader(str(p)) for p in paths]
        f = self.readers[0].fields
        def get(key):
            return f['qwen4exp.ple.' + key].contents()
        if f['general.architecture'].contents() != 'qwen4exp':
            raise ValueError('expected qwen4exp architecture')
        self.multipliers = [int(v) for v in get('layer_multipliers')]
        self.offsets = [int(v) for v in get('head_offsets')]
        self.vocabs = [int(v) for v in get('head_vocab_sizes')]
        self.per_gram, self.eos = int(get('heads_per_ngram')), int(get('eos_token_id'))
        self.heads = list(heads) if heads is not None else [h for h, v in enumerate(self.vocabs) if v > 1]
        if not self.heads or len(set(self.heads)) != len(self.heads):
            raise ValueError('select distinct retained heads')
        if any(h < 0 or h >= len(self.vocabs) or self.vocabs[h] <= 1 for h in self.heads):
            raise ValueError('selected head is absent or pruned')
        tensors = [t for r in self.readers for t in r.tensors if t.name == 'per_layer_token_embd.weight']
        if len(tensors) != 1:
            raise ValueError('expected exactly one PLE tensor')
        self.tensor = tensors[0]
        self.width = int(self.tensor.shape[0])
        if len(self.tensor.shape) != 2:
            raise ValueError('expected a matrix')
        self.n_rows = int(self.tensor.shape[1])
        if any(o + v > self.n_rows for o, v in zip(self.offsets, self.vocabs)):
            raise ValueError('head range exceeds tensor')
        self._row = lru_cache(maxsize=cache_rows)(self._read_row)

    def _read_row(self, row):
        from gguf.quants import dequantize
        raw = self.tensor.data.reshape(self.n_rows, -1)[row:row + 1]
        return dequantize(raw, self.tensor.tensor_type).reshape(self.width).astype(np.float32)

    def features(self, windows):
        rows = hash_rows(windows, self.multipliers, self.offsets, self.vocabs, self.per_gram, self.eos)
        rows = rows[:, self.heads]
        if not len(rows):
            return np.empty((0, len(self.heads) * self.width), np.float32)
        return np.stack([self._row(int(r)) for r in rows.flat]).reshape(len(rows), -1)

    def candidate_features(self, contexts, candidates):
        """Last committed ngram and hypothetical candidate-ending ngram.

        These use known context and the proposal only, never the target's future.
        Context columns are newest first; missing history must be -1.
        """
        n = len(self.multipliers)
        if contexts.shape[1] < n:
            raise ValueError('context is shorter than PLE window')
        proposed = np.column_stack([candidates, contexts[:, :n - 1]])
        return np.concatenate([self.features(contexts[:, :n]), self.features(proposed)], axis=1)
