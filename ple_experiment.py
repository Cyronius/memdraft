#!/usr/bin/env python3
"""Offline combined lookup/reranker/PLE experiment; no speed claims from replay.

Input JSONL is greedy target output, with prompt_ids, answer_ids, id, target_id
and temperature=0. Split by prompt identity before table building or training.
The target_id must identify the exact pruned quant, not just the model family.
Run --help for arguments. See PLE_EXPERIMENT.md for the remaining live work.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

import ngram_study as ns
from ple_features import GGUFTable


def partition(row):
    # Group duplicates (including different IDs for identical tokenized prompts).
    raw = np.asarray(row['prompt_ids'], dtype='<i4').tobytes()
    b = int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little') % 10
    return 'table' if b < 3 else 'cal' if b == 3 else 'train' if b < 8 else 'val' if b == 8 else 'test'


def load_rows(path, target_id):
    rows, seen = [], set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('target_id') != target_id or r.get('temperature') != 0:
                raise ValueError('every row must identify this exact target and temperature=0')
            p, a = r['prompt_ids'], r['answer_ids']
            if not p or not a:
                continue
            if any(not isinstance(t, int) or t < 0 for t in p + a):
                raise ValueError('token IDs must be nonnegative integers')
            if len(p) + len(a) >= 8192:
                raise ValueError('request lookup index requires fewer than 8192 tokens')
            key = tuple(p)
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(r, amax=a, source=r.get('source', 'unknown'), thinking=r.get('thinking', False)))
    return rows


def positions(data, K):
    # ns.Data.context indexes flattened arrays; exclude short windows so they
    # cannot wrap into another request. All arms use the same eligible positions.
    return np.flatnonzero(data.written & (data.amax >= 0) & (data.pos >= K))


def replay_positions(data, K, limit):
    """Keep whole requests: truncating in mid-answer can overcount acceptance."""
    idx = positions(data, K)
    if not len(idx):
        return idx
    ends = np.flatnonzero(np.r_[data.seq[idx][1:] != data.seq[idx][:-1], True]) + 1
    eligible = ends[ends <= limit]
    end = int(eligible[-1] if len(eligible) else ends[0])
    return idx[:end]


def vote_features(cands, calib, contexts, K, logfreq):
    """Existing reranker's vote/count features, with variable K and vocabulary.

    Token-embedding features are added by TokenFeatures. No imports from the
    earlier scratch training scripts (which load datasets at import time).
    """
    S, n = 2 * K, len(contexts)
    H, T = np.zeros((n, S), bool), np.full((n, S), -1, np.int32)
    CF, A, B = [np.zeros((n, S), np.float32) for _ in range(3)]
    for kind, k, hit, tok, cell in cands:
        s = 2 * (k - 1) + (kind == 'context')
        cf = calib.confidence(kind, k, cell)
        hit = hit & (cf > 0)
        H[:, s], T[:, s], CF[:, s] = hit, np.where(hit, tok, -1), np.where(hit, cf, 0)
        if kind == 'static':
            A[:, s] = np.where(hit, (cell[0] + 1) / len(ns.CT_EDGES), 0)
            B[:, s] = np.where(hit, (cell[1] + 1) / ns.N_RATIO, 0)
        else:
            A[:, s] = np.where(hit, cell[0] / ns.CC_MAX, 0)
    agree = H[:, :, None] & H[:, None, :] & (T[:, :, None] == T[:, None, :])
    ii, ss = np.nonzero(H & ~(agree & np.tril(np.ones((S, S), bool), -1)[None]).any(2))
    ag, cf = agree[ii, ss].astype(np.float32), CF[ii]
    top = CF.max(1)
    base = S - 1 - np.argmax(CF[:, ::-1] >= top[:, None], axis=1)
    own, rival = (ag * cf).max(1), ((1 - ag) * cf).max(1)
    tok, p0, p1 = T[ii, ss], contexts[ii, 0], contexts[ii, 1]
    X = np.concatenate([ag, ag * cf, ag * A[ii], ag * B[ii], H[ii], cf, A[ii], B[ii],
                        np.stack([own, rival, ag.sum(1) / S, own - rival, ag[np.arange(len(ii)), base[ii]]], 1),
                        np.stack([logfreq[tok], logfreq[p0], tok == p0, tok == p1], 1)], 1).astype(np.float32)
    return ii, tok, X


class TokenFeatures:
    """Bounded-cache 64-wide fixed projections of input/output embeddings.

    This is a retrainable baseline from the existing reranker's MLP family,
    not a claim that the 9B checkpoint's learned coordinates transfer to Qwen3.8.
    Both experimental arms receive identical token features.
    """
    def __init__(self, ple, seed=17, d=64):
        from functools import lru_cache
        self.tables = {t.name: t for r in ple.readers for t in r.tensors}
        self.inp = self.tables['token_embd.weight']
        self.out = self.tables.get('output.weight', self.inp)
        rng = np.random.default_rng(seed)
        self.projection = {}
        for t in (self.inp, self.out):
            width = int(t.shape[0])
            if width not in self.projection:
                self.projection[width] = (rng.standard_normal((width, d)) / np.sqrt(width)).astype(np.float32)
        self.row = lru_cache(maxsize=32768)(self._row)
        self.d = d

    def _row(self, name, token):
        from gguf.quants import dequantize
        t = self.tables[name]
        if token < 0 or token >= int(t.shape[1]):
            raise ValueError('token out of vocabulary')
        raw = t.data.reshape(int(t.shape[1]), -1)[token:token + 1]
        return dequantize(raw, t.tensor_type).reshape(-1) @ self.projection[int(t.shape[0])]

    def features(self, C, tok):
        if not len(tok):
            return np.empty((0, (C.shape[1] + 1) * self.d), np.float32)
        return np.stack([np.concatenate([self.row(self.out.name, int(t))] +
                         [self.row(self.inp.name, int(c)) for c in ctx]) for ctx, t in zip(C, tok)])


class Combined:
    def __init__(self, static, calib, K, freq, ple, token_features):
        self.static, self.calib, self.K, self.freq = static, calib, K, freq
        self.ple, self.token_features = ple, token_features

    def features(self, C, cands):
        ii, tok, votes = vote_features(cands, self.calib, C, self.K, self.freq)
        basic = np.concatenate([votes, self.token_features.features(C[ii], tok)], 1)
        extra = self.ple.candidate_features(C[ii], tok)
        return ii, tok, basic, extra

    def pool(self, data, limit):
        idx = positions(data, self.K)[:limit]
        if not len(idx):
            raise ValueError('split has no eligible positions')
        inreq = ns.InRequest(data, self.K)
        chunks = []
        for b in range(0, len(idx), 1024):
            ix = idx[b:b + 1024]
            C, avail = data.context(ix, self.K)
            cands = ns.candidates(C, avail, data.seq[ix], data.pos[ix], self.static, inreq, self.K)
            ii, tok, base, ple = self.features(C, cands)
            chunks.append((base, ple, (tok == data.amax[ix][ii]).astype(np.float32), b + ii))
        base, ple, y, group = [np.concatenate(z) for z in zip(*chunks)]
        if not len(y):
            raise ValueError('split has no lookup proposals')
        return dict(base=base, ple=ple, y=y, group=group, positions=len(idx))

    def drafts(self, data, idx, model, norm, with_ple, depth=3, policy='both'):
        C, avail = data.context(idx, self.K)
        inreq = ns.InRequest(data, self.K)
        drafts = np.full((len(idx), depth), -1, np.int32)
        for d in range(depth):
            cands = ns.candidates(C, avail, data.seq[idx], data.pos[idx],
                                  self.static if policy != 'context' else None,
                                  inreq if policy != 'static' else None, self.K)
            if model is None:
                tok, _ = ns.choose(cands, self.calib)
            else:
                ii, candidates, base, extra = self.features(C, cands)
                X = np.concatenate([base, extra], 1) if with_ple else base
                scores = predict(model, X, norm)
                tok, best = np.full(len(idx), -1, np.int32), np.full(len(idx), -np.inf)
                for i, t, score in zip(ii, candidates, scores):
                    if score >= best[i]:
                        tok[i], best[i] = t, score
            # Same fixed-depth fallback as memdraft.h --pad.
            tok = np.where(tok >= 0, tok, C[:, 0])
            drafts[:, d] = tok
            C = np.column_stack([tok, C[:, :-1]])
            avail = np.minimum(avail + 1, self.K)
        return drafts


def network(width):
    # Existing production reranker uses the same 512 -> 256 -> 128 MLP.
    return torch.nn.Sequential(torch.nn.Linear(width, 512), torch.nn.ReLU(),
                               torch.nn.Linear(512, 256), torch.nn.ReLU(),
                               torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))


def warm_start(model, path, K, base_width):
    """Reuse the trained .mdr MLP, dropping token-class columns only.

    Target-specific projected embeddings are rebuilt from the new GGUF. This
    initializes a new scorer; it is not a compatible deployment of the 9B model.
    New PLE columns start at zero so both arms begin with the same scores.
    """
    with open(path, 'rb') as f:
        head = f.read(64)
        if len(head) != 64:
            raise ValueError('truncated reranker')
        magic, version, V, S, nc, nctx, d, h1, h2, h3, nx = struct.unpack('<8s10I16x', head)
        if (magic, version, S, nctx, d, h1, h2, h3) != (b'MDRERANK', 1, 2 * K, K, 64, 512, 256, 128):
            raise ValueError('warm-start reranker has incompatible architecture or context length')
        if nx != 8 * S + 5 + 3 * nc + 4 or base_width != 8 * S + 9 + (K + 1) * d:
            raise ValueError('incompatible feature layout')
        f.seek((V + 7) // 8 * 8 + 4 * V + 8 * V * d, 1)
        def array(shape):
            n = int(np.prod(shape))
            raw = f.read(n * 4)
            if len(raw) != n * 4:
                raise ValueError('truncated reranker weights')
            return np.frombuffer(raw, '<f4').reshape(shape).copy()
        old = array((nx + (K + 1) * d, h1)).T
        vote = 8 * S + 5
        mapped = np.concatenate([old[:, :vote], old[:, nx - 4:nx], old[:, nx:]], 1)
        with torch.no_grad():
            model[0].weight.zero_()
            model[0].weight[:, :base_width].copy_(torch.from_numpy(mapped))
            model[0].bias.copy_(torch.from_numpy(array((h1,))))
            for layer, ni, no in [(2, h1, h2), (4, h2, h3), (6, h3, 1)]:
                model[layer].weight.copy_(torch.from_numpy(array((ni, no)).T))
                model[layer].bias.copy_(torch.from_numpy(array((no,))))


def export_native(path, combined, model, norm, with_ple):
    """Export native scorer weights plus references to the original GGUF bytes.

    The GGUF files must remain at their absolute paths and must be immutable.
    No ngram or token embedding table is copied into the scorer file.
    """
    ple, tf = combined.ple, combined.token_features
    width = len(norm[0])
    extra = 2 * len(ple.heads) * ple.width if with_ple else 0
    with open(path, 'wb') as f:
        f.write(struct.pack('<8s11I', b'MDPLE001', combined.K, tf.d, int(with_ple), len(ple.heads),
                            len(ple.multipliers), ple.per_gram, ple.eos, width - extra, width,
                            len(combined.freq), len(ple.vocabs)))
        def arr(a, dtype='<f4'):
            f.write(np.asarray(a, dtype=dtype).tobytes())
        arr(ple.multipliers, '<u8'); arr(ple.offsets, '<u8'); arr(ple.vocabs, '<u8'); arr(ple.heads, '<u4')
        for tensor in (tf.inp, tf.out, ple.tensor):
            source = next(p for p, r in zip(ple.paths, ple.readers) if any(t is tensor for t in r.tensors))
            absolute = str(source.resolve()).encode()
            f.write(struct.pack('<I', len(absolute)))
            f.write(absolute)
            f.write(struct.pack('<QQIII', source.stat().st_size, tensor.data_offset,
                                int(tensor.shape[0]), int(tensor.shape[1]), int(tensor.tensor_type)))
        arr(combined.freq)
        arr(tf.projection[int(tf.inp.shape[0])]); arr(tf.projection[int(tf.out.shape[0])])
        arr(norm[0]); arr(norm[1])
        for layer in (0, 2, 4, 6):
            arr(model[layer].weight.detach().numpy()); arr(model[layer].bias.detach().numpy())


def predict(model, X, norm):
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 4096):
            out.append(model(torch.from_numpy((X[b:b + 4096] - norm[0]) / norm[1])).flatten().numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


def fit(train, val, extra, epochs, seed, warm_path=None, K=6):
    torch.manual_seed(seed)
    X = np.concatenate([train['base'], train['ple']], 1) if extra else train['base']
    V = np.concatenate([val['base'], val['ple']], 1) if extra else val['base']
    norm = X.mean(0), np.maximum(X.std(0), 1e-4)
    model = network(X.shape[1])
    if warm_path:
        warm_start(model, warm_path, K, train['base'].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.from_numpy((X - norm[0]) / norm[1])
    y = torch.from_numpy(train['y'])
    best, saved, bad = float('inf'), None, 0
    for epoch in range(epochs):
        order = torch.randperm(len(y))
        model.train()
        for b in range(0, len(y), 2048):
            j = order[b:b + 2048]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(X[j]).flatten(), y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        z = predict(model, V, norm)
        loss = float(np.mean(np.logaddexp(0, z) - val['y'] * z))
        if loss < best:
            best, bad = loss, 0
            saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(saved)
    return model, norm, dict(validation_bce=best, epochs=epoch + 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', required=True, help='exact target pruned Q4_K_XL, or first shard')
    p.add_argument('--rows', required=True, help='greedy target output JSONL')
    p.add_argument('--target-id', required=True, help='immutable identifier of exact pruned target')
    p.add_argument('--heads', default='0,8', help='global retained head IDs')
    p.add_argument('--out', required=True)
    p.add_argument('--K', type=int, default=6, choices=range(3, 9))
    p.add_argument('--max-positions', type=int, default=100000, help='per split')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--warm-start-reranker', help='reuse trained .mdr MLP; rebuild token features and fine-tune both arms')
    args = p.parse_args()
    if min(args.epochs, args.max_positions, args.threads) < 1:
        p.error('epochs, max-positions, threads must be positive')
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.rows, args.target_id)
    parts = {s: [r for r in rows if partition(r) == s] for s in ('table', 'cal', 'train', 'val', 'test')}
    if any(not r for r in parts.values()):
        raise ValueError('need nonempty prompt-disjoint table/cal/train/val/test splits')
    data = {s: ns.Data(r) for s, r in parts.items()}
    ix = positions(data['table'], args.K)
    if not len(ix):
        raise ValueError('table split has no eligible positions')
    static = ns.Static.build(data['table'], ix, data['table'].amax[ix], args.K)
    cal, ci = data['cal'], positions(data['cal'], args.K)
    if not len(ci):
        raise ValueError('calibration split has no eligible positions')
    C, avail = cal.context(ci, args.K)
    calib = ns.Calibration(args.K)
    calib.add(ns.candidates(C, avail, cal.seq[ci], cal.pos[ci], static, ns.InRequest(cal, args.K), args.K), cal.amax[ci])
    ple = GGUFTable(args.gguf, [int(h) for h in args.heads.split(',')])
    freq = np.log1p(np.bincount(data['table'].amax[ix], minlength=int(next(
        t for r in ple.readers for t in r.tensors if t.name == 'token_embd.weight').shape[1]))).astype(np.float32) / 16
    combined = Combined(static, calib, args.K, freq, ple, TokenFeatures(ple, args.seed))
    pools = {s: combined.pool(data[s], args.max_positions) for s in ('train', 'val')}
    report = dict(args=vars(args), split_requests={s: len(r) for s, r in parts.items()},
                  measurement='offline greedy replay; no wall-clock speed measured',
                  baseline='retrained 512-256-128 MLP with fixed token projections; optional .mdr MLP initialization', arms={})
    models = {}
    for name, extra in [('reranker', False), ('reranker_ple', True)]:
        model, norm, result = fit(pools['train'], pools['val'], extra, args.epochs, args.seed,
                                  args.warm_start_reranker, args.K)
        models[name] = model, norm, extra
        torch.save(dict(state=model.state_dict(), mean=torch.from_numpy(norm[0]), scale=torch.from_numpy(norm[1]),
                        target_id=args.target_id, heads=ple.heads, K=args.K, with_ple=extra, seed=args.seed), out / f'{name}.pt')
        export_native(out / f'{name}.mdp', combined, model, norm, extra)
        report['arms'][name] = result
    ns.export_table2(static, calib, str(out / 'lookup.mdt'))
    test = data['test']
    ti = replay_positions(test, args.K, args.max_positions)
    if not len(ti):
        raise ValueError('test split has no eligible positions')
    C, avail = test.context(ti, args.K)
    cands = ns.candidates(C, avail, test.seq[ti], test.pos[ti], static, ns.InRequest(test, args.K), args.K)
    coverage, oracle = np.zeros(len(ti), bool), np.zeros(len(ti), bool)
    for kind, k, hit, token, cell in cands:
        hit = hit & (calib.confidence(kind, k, cell) > 0)
        coverage |= hit
        oracle |= hit & (token == test.amax[ti])
    report['candidate_coverage'] = dict(positions=len(ti), any_proposal=float(coverage.mean()),
                                        correct_token_proposed=float(oracle.mean()))
    # Keep whole requests, including the target's final bonus token.
    for name, setting in [('prompt_lookup', (None, None, False)), ('lookup_both', (None, None, False)), *models.items()]:
        model, norm, extra = setting
        drafts = combined.drafts(test, ti, model, norm, extra, policy='context' if name == 'prompt_lookup' else 'both')
        accepted, judged = ns.score(test, ti, drafts, np.ones_like(drafts), 'greedy')
        for depth in (1, 2, 3):
            proposed = np.minimum(judged, depth)
            run = ns.walk(test, ti, np.minimum(accepted, proposed), proposed)
            counts = {k: int(run[k].sum()) for k in ns.COUNTS}
            counts['tokens_per_step'] = counts['tokens'] / counts['steps']
            counts['acceptance'] = counts['accepted'] / max(1, counts['drafted'])
            report['arms'].setdefault(name, {})[f'depth_{depth}'] = counts
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
