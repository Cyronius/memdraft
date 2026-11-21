"""Two questions about the request index and the table, asked offline on the code test answers.

1. Recency and agreement for the reranker. The request index proposes what followed the latest
   earlier occurrence of a context, and the reranker sees only how many earlier occurrences there
   were. Two more features per request slot (contexts of 1..6 tokens):
     agreement  how many of the most recent occurrences in a row (up to 4) were followed by that
                same token: (run - 1) / 3, so 0 when only the latest one was
     recency    tokens since the latest occurrence, log1p(distance) / log1p(4096), at most 1
   each as is and masked by "this slot proposes this candidate" (as the other slot features are).
   Three rerankers fine-tuned the same way from rerank-big.standins.pt (new input weights start at
   zero): without them, with agreement, with agreement and recency.

   To stay small, the rerankers learn on the 4,200 answers written on this machine (all positions)
   and the calibration answers, with proposals from one table that leaves those answers (and the
   gpt-3.5 solutions to their prompts) out: shipped text, gpt-3.5 code, yesterday's own answers in the
   table part, and the pod's answers. The same table at test time for every arm. Every tenth answer
   is validation for early stopping. Scored through whole draft chains (rr.chain, GPU step costs),
   with a paired bootstrap over test answers against the reranker without the new features.

2. Runner-ups in the table. The table keeps one next token per context. The table here also keeps
   the second most frequent one, and at the first drafted token of every test position counts:
   how often the right token is a runner-up and in none of the 12 proposals (what keeping
   runner-ups could add at most), how often that happens where the reranker's pick is wrong, per
   context length and by how dominant the top token is, and how much bigger the table would get.
   Contexts seen fewer than 3 times are left out, as in the shipped table.

Run from the repo root under a memory limit, at low priority (run_rerank8.sh). Argument: output prefix.
THREADS (default 4), EPOCHS (default 8); POD_LIMIT and MORE_LIMIT make a small trial run."""
import gc
import json
import os
import resource
import sys

import numpy as np
import torch

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools'))
import code_study as cs   # noqa: E402
import gen_code           # noqa: E402
import ngram_study as ns  # noqa: E402
import rerank as rr       # noqa: E402
import rerank2 as r2      # noqa: E402
import rerank7 as r7      # noqa: E402

THREADS = int(os.environ.get('THREADS', 4))
torch.set_num_threads(THREADS)
torch.manual_seed(0)
K, S, V = rr.K, rr.S, rr.V
EPOCHS = int(os.environ.get('EPOCHS', 8))
HIDDEN = (512, 256, 128)
BEST = 'embeddings, learned from 512'
MIN_COUNT = 3
N_BASE = 8 * S + 5 + 3 * rr.NC + 4          # rr.build's columns
ARMS = {'without the new features': N_BASE, 'with agreement': N_BASE + 2 * K, 'with agreement and recency': N_BASE + 4 * K}
BASE = 'without the new features'


def peak():
    return f'peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB'


class Static2(ns.Static):
    """ns.Static that also keeps each context's second most frequent next token and its count."""

    @classmethod
    def build(cls, data, idx, labels, K):
        self = cls(K)
        self.second, self.csecond = [], []
        h = np.zeros(len(idx), np.uint64)
        for k in range(1, K + 1):
            h = ns.mix(h, data.tok[idx - k])
            ok = data.pos[idx] >= k
            packed = ((h[ok] >> ns.LBITS) << ns.LBITS) | labels[ok].astype(np.uint64)
            packed.sort()
            first = np.flatnonzero(np.r_[True, packed[1:] != packed[:-1]])
            count = np.diff(np.r_[first, len(packed)])
            pair_key, pair_lab = packed[first] >> ns.LBITS, (packed[first] & ns.LMASK).astype(np.int32)
            del packed, first
            head = np.flatnonzero(np.r_[True, pair_key[1:] != pair_key[:-1]])
            total = np.add.reduceat(count, head)
            top = np.maximum.reduceat(count, head)
            group = np.repeat(np.arange(len(head)), np.diff(np.r_[head, len(pair_key)]))
            winners = np.flatnonzero(count == top[group])
            winners = winners[np.r_[True, group[winners][1:] != group[winners][:-1]]]
            c2 = count.copy()
            c2[winners] = 0
            sec = np.maximum.reduceat(c2, head)
            runner = np.flatnonzero((c2 == sec[group]) & (c2 > 0))
            runner = runner[np.r_[True, group[runner][1:] != group[runner][:-1]]]
            second = np.full(len(head), -1, np.int32)
            second[group[runner]] = pair_lab[runner]
            self.keys.append(pair_key[head])
            self.best.append(pair_lab[winners])
            self.cbest.append(top.astype(np.int64))
            self.ctot.append(total.astype(np.int64))
            self.second.append(second)
            self.csecond.append(sec.astype(np.int64))
            del pair_key, pair_lab, count, c2, group, winners, runner
        return self

    def runner_up(self, k, h):
        """(kept hit, runner-up token or -1, best count, total count) at order k."""
        keys = self.keys[k - 1]
        q = h >> ns.LBITS
        i = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
        hit = (keys[i] == q) & (self.ctot[k - 1][i] >= self.min_count)
        return hit, np.where(hit, self.second[k - 1][i], -1), self.cbest[k - 1][i], self.ctot[k - 1][i]


def extras(data, inreq, C, avail, seq, pos):
    """Per position and order k: agreement run and recency of the request proposal (0 when none)."""
    n = len(C)
    run, rec = np.zeros((n, K), np.float32), np.zeros((n, K), np.float32)
    h = np.zeros(n, np.uint64)
    base = data.start[seq]
    for k in range(1, K + 1):
        h = ns.mix(h, C[:, k - 1])
        index = inreq.index[k - 1]
        x = inreq.pack(h, seq, pos)
        key = x >> ns.PBITS
        i = np.searchsorted(index, x) - 1
        j = np.maximum(i, 0)
        valid = (i >= 0) & ((index[j] >> ns.PBITS) == key) & (avail >= k)
        at = lambda m: data.tok[np.minimum(base + (index[m] & ns.PMASK).astype(np.int64), len(data.tok) - 1)]   # junk where the key differs
        latest = at(j)
        latest_pos = (index[j] & ns.PMASK).astype(np.int64)
        r, alive = np.ones(n, np.float32), valid.copy()
        for back in (1, 2, 3):
            m = i - back
            mm = np.maximum(m, 0)
            alive = alive & (m >= 0) & ((index[mm] >> ns.PBITS) == key) & (at(mm) == latest)
            r += alive
        run[:, k - 1] = np.where(valid, (r - 1) / 3, 0)
        rec[:, k - 1] = np.where(valid, np.minimum(np.log1p(np.maximum(pos - latest_pos, 0)) / np.log1p(4096), 1), 0)
    return run, rec


def build(data, inreq, cands, calib, C, avail, seq, pos, logfreq, cls):
    """rr.build's rows with 4K more columns: agreement, agreement if this slot proposes it, recency, the same."""
    ii, tok, X, ids = rr.build(cands, calib, C, logfreq, cls)
    run, rec = extras(data, inreq, C, avail, seq, pos)
    ag = X[:, 1:S:2]                                   # the request slots' "proposes this token"
    E = np.concatenate([run[ii], ag * run[ii], rec[ii], ag * rec[ii]], 1)
    return ii, tok, np.concatenate([X, E], 1).astype(np.float32), ids


def features(rows, static, calib, logfreq, cls, chunk=200_000):
    data = ns.Data(rows)
    li, inreq, out = ns.labelled(data), ns.InRequest(data, K), []
    for b in range(0, len(li), chunk):
        idx = li[b:b + chunk]
        C, avail = data.context(idx, K)
        seq, pos = data.seq[idx], data.pos[idx]
        cands = ns.candidates(C, avail, seq, pos, static, inreq, K)
        ii, tok, X, ids = build(data, inreq, cands, calib, C, avail, seq, pos, logfreq, cls)
        out.append((X, ids, (tok == data.amax[idx][ii]).astype(np.float32), seq[ii]))
    X, ids, y, s = (np.concatenate(x) for x in zip(*out))
    return {'X': X, 'ids': ids, 'y': y, 'seq': s, 'positions': len(li)}


def widened(state, n_in):
    """standins state dict with zero input weights for the new columns (inserted before the embeddings)."""
    state = {k: v.clone() for k, v in state.items()}
    w = state['net.0.weight']
    state['net.0.weight'] = torch.cat([w[:, :N_BASE], torch.zeros(w.shape[0], n_in - N_BASE), w[:, N_BASE:]], 1)
    return state


def main():
    prefix = sys.argv[1]
    pool = gen_code.pick(rr.dataset, 2000, gen_code.SEED, gen_code.MAX_CHARS)
    held_ids = {p['id'] for p in pool if cs.part(p['id']) != 'table'}
    own = cs.own_rows('data/code/rows-9b.jsonl')
    more = cs.own_rows('data/code/rows-9b-more.jsonl')[:int(os.environ.get('MORE_LIMIT', 10**9))]
    pod = r7.light('data/code/rows-9b-pod.jsonl', int(os.environ['POD_LIMIT']) if 'POD_LIMIT' in os.environ else None)
    more_ids = {r['id'] for r in more}
    test = ns.Data([r for r in own if cs.part(r['id']) == 'test' and r['finished']])
    own_table = [r for r in own if cs.part(r['id']) == 'table']
    cal_rows = [r for r in own if cs.part(r['id']) == 'cal']
    other = [r for r in cs.other_rows(rr.dataset, rr.tokenizer, own[0], held_ids) if r['id'] not in more_ids]
    chat = [r for s in cs.SHARDS for r in ns.load_shard(s, None, 'data/cache') if r['finished']]
    cls = rr.token_classes()
    emb = {'pca': None, 'full': None, 'p512': tuple(torch.from_numpy(e) for e in rr.embeddings_pca(512))}
    result = {'threads': THREADS, 'epochs': EPOCHS, 'min_count': MIN_COUNT}

    def save():
        json.dump(result, open(f'{prefix}.json', 'w'), indent=1)

    table_rows = chat + other + own_table + pod
    data = ns.Data(table_rows)
    idx = ns.labelled(data)
    static = Static2.build(data, idx, data.amax[idx], K)
    counts = np.bincount(data.amax[idx], minlength=V)
    result['table_tokens'] = int(len(idx))
    del data, idx, table_rows, pod, chat, other
    gc.collect()
    static.min_count = MIN_COUNT
    logfreq = (np.log1p(counts) / 16).astype(np.float32)
    ns.log(f"table: {result['table_tokens']:,} tokens (without this machine's 4,200 answers); {peak()}")

    cal = ns.Data([r for r in cal_rows if r['finished']])
    ci = ns.labelled(cal)
    calib = ns.Calibration(K)
    calib.add(r2.proposals(cal, ci, static, ns.InRequest(cal, K))[1], cal.amax[ci])

    # Rows to learn from: this machine's answers and the calibration answers, every tenth answer for validation.
    learn = features(more + [r for r in cal_rows if r['finished']], static, calib, logfreq, cls)
    is_val = learn['seq'] % 10 == 0
    train = {k: learn[k][~is_val] for k in ('X', 'ids', 'y')}
    val = {k: learn[k][is_val] for k in ('X', 'ids', 'y')}
    del learn
    gc.collect()
    ns.log(f"learning rows {len(train['y']):,}, validation {len(val['y']):,}; {peak()}")
    for name, cols, lo in (('agreement', slice(N_BASE, N_BASE + K), 'request slots'), ('recency', slice(N_BASE + 2 * K, N_BASE + 3 * K), '')):
        v = train['X'][:, cols]
        result.setdefault('feature_spread', {})[name] = {'share_nonzero': float((v > 0).mean()), 'mean_when_nonzero': float(v[v > 0].mean()) if (v > 0).any() else 0}

    start = torch.load('data/code/rerank-big.standins.pt')[BEST]
    models = {}
    for name, n_in in ARMS.items():
        model = r2.Net(n_in, 'p512', emb, HIDDEN)
        model.load_state_dict(widened(start, n_in))
        model, info = r2.fit(name, [train], val, n_in, model=model, lr=5e-4, epochs=EPOCHS)
        models[name] = model
        result.setdefault('training', {})[name] = info
        torch.save({k: m.state_dict() for k, m in models.items()}, f'{prefix}.pt')
        save()
    del train, val
    gc.collect()

    # Scoring on the test answers.
    ti = ns.labelled(test)
    inreq_test = ns.InRequest(test, K)
    draws = np.random.default_rng(1).integers(0, len(test.start), (4000, len(test.start)))
    runs = {}

    def picker(model):
        def pick(cands, C, d):
            seq, pos = test.seq[ti], test.pos[ti]
            avail = np.minimum(test.pos[ti] + d, K).astype(np.int32)
            ii, tok, X, ids = build(test, inreq_test, cands, calib, C, avail, seq, pos, logfreq, cls)
            with torch.no_grad():
                p = np.concatenate([torch.sigmoid(model(torch.from_numpy(X[b:b + 16384]), torch.from_numpy(ids[b:b + 16384])).double()).numpy()
                                       for b in range(0, len(X), 16384)]) if len(X) else np.zeros(0)
            token, conf = np.full(len(C), -1, np.int32), np.zeros(len(C), np.float64)
            np.maximum.at(conf, ii, p)
            win = p >= conf[ii]
            token[ii[win]] = tok[win]
            has = np.zeros(len(C), bool)
            has[ii] = True
            return token, np.where(has, np.maximum(conf, 1e-30), 0)
        return pick

    def score(name, pick):
        r = rr.chain(test, ti, inreq_test, static, pick)
        per = r.pop('per answer, pad 3')
        tok, spent = runs[name] = (np.array(per['tokens']), np.array(per['spent']))
        line = ''
        if name != BASE and BASE in runs:
            b = runs[BASE]
            diff = np.sort(tok[draws].sum(1) / spent[draws].sum(1) - b[0][draws].sum(1) / b[1][draws].sum(1))
            gain = float(tok.sum() / spent.sum() - b[0].sum() / b[1].sum())
            r['gain over the reranker without the new features'] = {'pad 3': gain, '95%': [float(diff[100]), float(diff[3899])],
                                                                    'answers_faster': int((tok / spent > b[0] / b[1]).sum())}
            line = f'  {gain:+.3f} ({diff[100]:+.3f} to {diff[3899]:+.3f}) over the reranker without them'
        result.setdefault('scores', {})[name] = r
        ns.log(f"{name:40s} first guess {r['first_guess_right']:.3f}  " + '  '.join(
            f"pad {n} {r[f'pad {n}']['speedup']:.3f}x ({r[f'pad {n}']['tokens_per_step']:.2f})" for n in (3, 4, 5)) + line)
        save()

    score('the drafter', lambda cands, C, d: ns.choose(cands, calib))
    for name, model in models.items():
        score(f'reranker {name}' if name != BASE else BASE, picker(model))

    # Runner-ups, at the first drafted token of every test position.
    C, avail = test.context(ti, K)
    seq, pos = test.seq[ti], test.pos[ti]
    truth = test.amax[ti]
    cands = ns.candidates(C, avail, seq, pos, static, inreq_test, K)
    listed = np.zeros(len(ti), bool)
    for kind, k, hit, tok, cell in cands:
        listed |= hit & (tok == truth)
    rerank_pick = picker(models[BASE])(cands, C, 0)[0]
    h = np.zeros(len(ti), np.uint64)
    runner_any = np.zeros(len(ti), bool)
    per_k, by_ratio = {}, {}
    for k in range(1, K + 1):
        h = ns.mix(h, C[:, k - 1])
        hit, second, cbest, ctot = static.runner_up(k, h)
        hit &= avail >= k
        best = static.lookup(k, h)[1]
        right2 = hit & (second == truth)
        runner_any |= right2
        wrong1 = hit & (best != truth)
        per_k[k] = {'kept_hits': int(hit.sum()), 'top_right': float((hit & (best == truth)).sum() / max(hit.sum(), 1)),
                    'runner_up_right_when_top_wrong': float(right2.sum() / max(wrong1.sum(), 1)),
                    'has_runner_up': float((hit & (second >= 0)).sum() / max(hit.sum(), 1))}
        ratio = cbest / np.maximum(ctot, 1)
        for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
            m = wrong1 & (ratio >= lo) & (ratio < hi)
            cell = by_ratio.setdefault(f'top share {lo:.2f}-{min(hi, 1):.2f}', [0, 0])
            cell[0] += int(m.sum())
            cell[1] += int((m & right2).sum())
    new = runner_any & ~listed
    result['runner_ups'] = {
        'positions': int(len(ti)),
        'right token among the 12 proposals': float(listed.mean()),
        'right token a runner-up and not among them': float(new.mean()),
        'reranker pick wrong': float((rerank_pick != truth).mean()),
        'reranker pick wrong and right token only a runner-up': float((new & (rerank_pick != truth)).mean()),
        'per context length': per_k,
        'when the top token is wrong, runner-up right, by top token share': {k: {'top wrong': v[0], 'runner-up right': v[1] / max(v[0], 1)} for k, v in by_ratio.items()},
    }
    sizes = {}
    for k in range(1, K + 1):
        kept = static.ctot[k - 1] >= MIN_COUNT
        sizes[k] = {'kept': int(kept.sum()), 'with runner-up': int((kept & (static.second[k - 1] >= 0)).sum()),
                    'with runner-up, top share under 0.8': int((kept & (static.second[k - 1] >= 0) & (static.cbest[k - 1] < 0.8 * static.ctot[k - 1])).sum())}
    tot = {key: sum(s[key] for s in sizes.values()) for key in sizes[1]}
    result['runner_ups']['table entries'] = dict(sizes, total=tot)
    ru = result['runner_ups']
    ns.log(f"runner-ups: right token among the proposals {ru['right token among the 12 proposals']:.3f}; a runner-up and not among them "
           f"{ru['right token a runner-up and not among them']:.3f}; reranker wrong and right token only a runner-up "
           f"{ru['reranker pick wrong and right token only a runner-up']:.3f}; entries with a runner-up {tot['with runner-up'] / tot['kept']:.0%} "
           f"(top share under 0.8: {tot['with runner-up, top share under 0.8'] / tot['kept']:.0%})")
    save()
    ns.log(f'done; {peak()}')


if __name__ == '__main__':
    main()
