"""The reranker with the largest table (shipped + own + gpt-3.5 code, the 1.43x set-up), trained on
every own answer outside the test part.

The own-code answers are in the table, so their proposals would be too good to learn from. Each
quarter of them gets its proposals from a table built without that quarter (the other three
quarters stay in). The calibration answers are not in the table and are used as they are.
Optionally the reranker first learns on the gpt-3.5 and chat stand-ins of rerank2.py (STAGE=pre),
with token embeddings too (STAGE=pre,p512).
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

K = rr.K


def main():
    stage = os.environ.get('STAGE', '').split(',')
    pool = gen_code.pick(rr.dataset, 2000, gen_code.SEED, gen_code.MAX_CHARS)
    held_ids = {p['id'] for p in pool if cs.part(p['id']) != 'table'}
    own = cs.own_rows('data/code/rows-9b.jsonl')
    test = ns.Data([r for r in own if cs.part(r['id']) == 'test' and r['finished']])
    own_table = [r for r in own if cs.part(r['id']) == 'table']
    cal_rows = [r for r in own if cs.part(r['id']) == 'cal']
    other = cs.other_rows(rr.dataset, rr.tokenizer, own[0], held_ids)
    shards = [[r for r in ns.load_shard(s, None, 'data/cache') if r['finished']] for s in cs.SHARDS]
    chat = shards[0] + shards[1] + shards[2]
    cls = rr.token_classes()
    emb = {'pca': None, 'full': None,
           'p512': tuple(torch.from_numpy(e) for e in rr.embeddings_pca(512)) if 'p512' in stage else None}

    static, counts, n = r2.table(chat + own_table + other)
    logfreq = (np.log1p(counts) / 16).astype(np.float32)
    cal = ns.Data([r for r in cal_rows if r['finished']])
    ci = ns.labelled(cal)
    calib = ns.Calibration(K)
    calib.add(r2.proposals(cal, ci, static, ns.InRequest(cal, K))[1], cal.amax[ci])

    cal_pool = r2.features(cal_rows, static, calib, logfreq, cls)
    folds = []
    for f in range(4):
        s, _, _ = r2.table(chat + other + [r for i, r in enumerate(own_table) if i % 4 != f])
        folds.append(r2.features(own_table[f::4], s, calib, logfreq, cls))
    parts = [cal_pool] + folds
    val = {k: np.concatenate([p[k][p['seq'] % 10 == 0] for p in parts]) for k in ('X', 'ids', 'y')}
    train = [r2.take(p, p['seq'] % 10 != 0) for p in parts]
    ns.log(f"table {n:,} tokens. calibration answers {len(train[0]['y']):,} rows, own-table answers (cross-fitted) "
           f"{sum(len(p['y']) for p in train[1:]):,} rows, validation {len(val['y']):,} rows")
    n_in = cal_pool['X'].shape[1]

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
        best = max((f'pad {k}' for k in range(1, rr.DEPTH + 1)), key=lambda p: r[p]['speedup'])
        r.update(info or {}, gain_pad3=float(r['pad 3']['speedup'] - base[0].sum() / base[1].sum()), gain_95=[float(diff[100]), float(diff[3899])],
                 best_rule=best)
        result[name] = r
        ns.log(f"{name:58s} first guess {r['first_guess_right']:.3f}  pad 3 {r['pad 3']['speedup']:.3f}x  gain {r['gain_pad3']:+.3f} "
               f"({diff[100]:+.3f} to {diff[3899]:+.3f})  best {best} {r[best]['speedup']:.3f}x "
               f"({r[best]['tokens_per_step']:.2f} tok/step, acc {r[best]['acceptance']:.0%})")
        json.dump(result, open(sys.argv[1], 'w'), indent=1)

    def reranked(model):
        def pick(cands, C, d):
            ii, tok, X, ids = rr.build(cands, calib, C, logfreq, cls)
            with torch.no_grad():
                p = np.concatenate([torch.sigmoid(model(torch.from_numpy(X[b:b + 16384]), torch.from_numpy(ids[b:b + 16384]))).numpy()
                                    for b in range(0, len(X), 16384)])
            token, conf = np.full(len(C), -1, np.int32), np.zeros(len(C), np.float32)
            np.maximum.at(conf, ii, p)
            win = p >= conf[ii]
            token[ii[win]] = tok[win]
            return token, conf
        return pick

    def run(name, parts, **kw):
        model, info = r2.fit(name, parts, val, n_in, **kw)
        score(name, reranked(model), info)
        return model

    score('most confident (the drafter now)', lambda cands, C, d: ns.choose(cands, calib))
    run('calibration answers only (as before)', train[:1])
    run('all own answers', train)
    if 'pre' in stage:
        s, _, _ = r2.table(chat + own_table + [r for i, r in enumerate(other) if i % 10])
        gpt_pool = r2.features(other[::10], s, calib, logfreq, cls)
        s, _, _ = r2.table(shards[0] + shards[1] + own_table + other)
        chat_pool = r2.features(shards[2][::3], s, calib, logfreq, cls)
        del s
        ns.log(f"stand-ins: gpt-3.5 code {len(gpt_pool['y']):,} rows, chat {len(chat_pool['y']):,} rows")
        m, _ = r2.fit('gpt-3.5 code + chat (before own)', [gpt_pool, chat_pool], val, n_in, epochs=8, patience=2, hidden=(512, 256, 128))
        run('wider: gpt-3.5 code + chat, then all own', train, model=m, lr=5e-4, epochs=20)
        if 'p512' in stage:
            m, _ = r2.fit('embeddings: gpt-3.5 code + chat (before own)', [gpt_pool, chat_pool], val, n_in, mode='p512', emb=emb, epochs=8,
                          patience=2, hidden=(512, 256, 128))
            run('embeddings: gpt-3.5 code + chat, then all own', train, model=m, lr=5e-4, epochs=20)


if __name__ == '__main__':
    main()
