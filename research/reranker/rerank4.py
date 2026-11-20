"""Does the reranker do better trained to choose (a softmax over each position's proposals) than
trained to say whether each proposal is right (what rerank.py does)? Set-up B of rerank2.py: the
table is shipped + gpt-3.5 code, the reranker learns on the own-code answers outside the test part.
Run from the repo root with python3."""
import json
import os
import sys

import numpy as np
import torch

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import code_study as cs
import gen_code
import ngram_study as ns
import rerank as rr
import rerank2 as r2

K, S = rr.K, rr.S


def padded(pool):
    """Rows grouped by position: X [P, S, F], mask, right?, ids, answer number."""
    grp = pool['grp']
    first = np.r_[True, grp[1:] != grp[:-1]]
    g, start = np.cumsum(first) - 1, np.flatnonzero(first)
    rank, P = np.arange(len(grp)) - start[g], int(first.sum())
    X = np.zeros((P, S, pool['X'].shape[1]), np.float32)
    M, Y, ids = np.zeros((P, S), bool), np.zeros((P, S), np.float32), np.zeros((P, S, pool['ids'].shape[1]), np.int64)
    X[g, rank], M[g, rank], Y[g, rank], ids[g, rank] = pool['X'], True, pool['y'], pool['ids']
    return {'X': torch.from_numpy(X), 'M': torch.from_numpy(M), 'Y': torch.from_numpy(Y), 'ids': torch.from_numpy(ids), 'seq': pool['seq'][start]}


def losses(model, d, j, kind):
    X, M, Y, ids = d['X'][j], d['M'][j], d['Y'][j], d['ids'][j]
    logits = model(X.flatten(0, 1), ids.flatten(0, 1)).view(len(X), S)
    out = 0.0
    if kind in ('each', 'both'):
        out = out + torch.nn.functional.binary_cross_entropy_with_logits(logits[M], Y[M])
    if kind in ('choose', 'both'):
        has = Y.sum(1) > 0
        logp = torch.log_softmax(logits.masked_fill(~M, -1e9), 1)
        out = out + -(logp * Y).sum(1)[has].mean()
    picked = logits.masked_fill(~M, -1e9).argmax(1)
    return out, float(Y[torch.arange(len(Y)), picked].sum()), len(X)


def fit(name, train, val, n_in, kind, hidden=(256, 128), epochs=40, lr=2e-3, batch=2048, patience=4, seed=0):
    torch.manual_seed(seed)
    model = r2.Net(n_in, None, None, hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(train['X'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((n + batch - 1) // batch), pct_start=0.05)
    best, state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, batch):
            loss, _, _ = losses(model, train, perm[b:b + batch], kind)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            right = sum(losses(model, val, torch.arange(b, min(b + 8192, len(val['X']))), kind)[1] for b in range(0, len(val['X']), 8192))
        acc = right / len(val['X'])
        if acc > best + 1e-4:
            best, state, bad = acc, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad == patience:
                break
    model.load_state_dict(state)
    model.eval()
    ns.log(f'   trained {name}: {n:,} positions, {epoch + 1} epochs, validation first guess {best:.4f}')
    return model, {'positions': n, 'epochs': epoch + 1, 'validation_first_guess': best}


def main():
    pool = gen_code.pick(rr.dataset, 2000, gen_code.SEED, gen_code.MAX_CHARS)
    held_ids = {p['id'] for p in pool if cs.part(p['id']) != 'table'}
    own = cs.own_rows('data/code/rows-9b.jsonl')
    test = ns.Data([r for r in own if cs.part(r['id']) == 'test' and r['finished']])
    learn_rows = [r for r in own if cs.part(r['id']) != 'test']
    other = cs.other_rows(rr.dataset, rr.tokenizer, own[0], held_ids)
    shards = [r for s in cs.SHARDS for r in ns.load_shard(s, None, 'data/cache') if r['finished']]
    cls = rr.token_classes()
    static, counts, n = r2.table(shards + other)
    logfreq = (np.log1p(counts) / 16).astype(np.float32)
    learn = ns.Data(learn_rows)
    li = ns.labelled(learn)
    calib = ns.Calibration(K)
    calib.add(r2.proposals(learn, li, static, ns.InRequest(learn, K))[1], learn.amax[li])
    own_pool = padded(r2.features(learn_rows, static, calib, logfreq, cls))
    is_val = torch.from_numpy(own_pool['seq'] % 10 == 0)
    val = {k: v[is_val] for k, v in own_pool.items() if k != 'seq'}
    train = {k: v[~is_val] for k, v in own_pool.items() if k != 'seq'}
    n_in = own_pool['X'].shape[2]
    ns.log(f"table {n:,} tokens; {len(train['X']):,} positions to learn on, {len(val['X']):,} for validation; "
           f"validation first guess of the drafter now: see below")

    ti = ns.labelled(test)
    inreq_test = ns.InRequest(test, K)
    result, base = {}, None

    def score(name, pick, info=None):
        nonlocal base
        r = rr.chain(test, ti, inreq_test, static, pick)
        per = r.pop('per answer, pad 3')
        tok, spent = np.array(per['tokens']), np.array(per['spent'])
        if base is None:
            base = (tok, spent)
        draws = np.random.default_rng(1).integers(0, len(tok), (4000, len(tok)))
        diff = np.sort(tok[draws].sum(1) / spent[draws].sum(1) - base[0][draws].sum(1) / base[1][draws].sum(1))
        r.update(info or {}, gain_pad3=float(r['pad 3']['speedup'] - base[0].sum() / base[1].sum()), gain_95=[float(diff[100]), float(diff[3899])])
        result[name] = r
        ns.log(f"{name:44s} first guess {r['first_guess_right']:.3f}  pad 3 {r['pad 3']['speedup']:.3f}x  gain {r['gain_pad3']:+.3f} "
               f"({diff[100]:+.3f} to {diff[3899]:+.3f})  pad 4 {r['pad 4']['speedup']:.3f}x")
        json.dump(result, open(sys.argv[1], 'w'), indent=1)

    def reranked(model):
        def pick(cands, C, d):
            ii, tok, X, ids = rr.build(cands, calib, C, logfreq, cls)
            with torch.no_grad():
                p = np.concatenate([torch.sigmoid(model(torch.from_numpy(X[b:b + 16384]), torch.from_numpy(ids[b:b + 16384])).double()).numpy()
                                    for b in range(0, len(X), 16384)])
            token, conf = np.full(len(C), -1, np.int32), np.zeros(len(C), np.float64)
            np.maximum.at(conf, ii, p)
            win = p >= conf[ii]
            token[ii[win]] = tok[win]
            return token, np.maximum(conf, 1e-30)
        return pick

    score('most confident (the drafter now)', lambda cands, C, d: ns.choose(cands, calib))
    for kind, label in (('each', 'is each proposal right?'), ('choose', 'choose among the proposals'), ('both', 'both')):
        for seed in (0, 1):
            model, info = fit(f'{label}, seed {seed}', train, val, n_in, kind, seed=seed)
            score(f'{label}, seed {seed}', reranked(model), info)
    model, info = fit('both, wider (512-256-128)', train, val, n_in, 'both', hidden=(512, 256, 128))
    score('both, wider (512-256-128)', reranked(model), info)


if __name__ == '__main__':
    main()
