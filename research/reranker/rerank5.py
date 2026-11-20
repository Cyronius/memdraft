"""The reranker trained on everything there is, with the largest table (the 1.43x set-up).

First one pass over all the stand-in text, each piece with proposals from a table built without it:
  gpt-3.5 code  all 74K solutions, in ten folds (labels are gpt-3.5's tokens)
  chat          all three shards of the 9B's chat answers, in nine pieces (labels are its top choice)
about 38M positions. Then fine-tuning on the 9B's own code answers outside the test part (the
own-code answers that are in the table get their proposals from a table without their quarter).
Three rerankers learn side by side on the same rows: without token embeddings, with 64 fixed
principal components per token, and with 64 numbers per token learned from 512 components.
Run from the repo root with python3; argument: output prefix."""
import json
import os
import sys
import time

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
TOTAL_STEPS = 24500             # about 99M rows in batches of 4096


def main():
    prefix = sys.argv[1]
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
    emb = {'pca': tuple(torch.from_numpy(e) for e in rr.embeddings()), 'full': None,
           'p512': tuple(torch.from_numpy(e) for e in rr.embeddings_pca(512))}

    static, counts, n = r2.table(chat + own_table + other)
    logfreq = (np.log1p(counts) / 16).astype(np.float32)
    cal = ns.Data([r for r in cal_rows if r['finished']])
    ci = ns.labelled(cal)
    calib = ns.Calibration(K)
    calib.add(r2.proposals(cal, ci, static, ns.InRequest(cal, K))[1], cal.amax[ci])

    parts = [r2.features(cal_rows, static, calib, logfreq, cls)]
    for f in range(4):
        s, _, _ = r2.table(chat + other + [r for i, r in enumerate(own_table) if i % 4 != f])
        parts.append(r2.features(own_table[f::4], s, calib, logfreq, cls))
    val = {k: np.concatenate([p[k][p['seq'] % 10 == 0] for p in parts]) for k in ('X', 'ids', 'y')}
    own_train = [r2.take(p, p['seq'] % 10 != 0) for p in parts]
    n_in = parts[0]['X'].shape[1]
    ns.log(f"table {n:,} tokens. own answers to fine-tune on: {sum(len(p['y']) for p in own_train):,} rows; validation {len(val['y']):,} rows")

    ti = ns.labelled(test)
    inreq_test = ns.InRequest(test, K)
    result, base = {}, None
    table_now = lambda: static

    def score(name, pick, info=None):
        nonlocal base
        r = rr.chain(test, ti, inreq_test, table_now(), pick)
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
        json.dump(result, open(f'{prefix}.json', 'w'), indent=1)

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
    del static, parts                             # one table in memory at a time; built again for the scoring

    torch.manual_seed(0)
    models = {'no embeddings': r2.Net(n_in, None, emb, (512, 256, 128)),
              'embeddings, 64 fixed': r2.Net(n_in, 'pca', emb, (512, 256, 128)),
              'embeddings, learned from 512': r2.Net(n_in, 'p512', emb, (512, 256, 128))}
    opts = {k: torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4) for k, m in models.items()}
    scheds = {k: torch.optim.lr_scheduler.LambdaLR(o, lambda s: min(1.0, (s + 1) / 200) * max(0.1, 1 - s / TOTAL_STEPS)) for k, o in opts.items()}
    seen = 0

    def learn(what, rows, tab):
        nonlocal seen
        t0 = time.time()
        p = r2.features(rows, tab, calib, logfreq, cls)
        X, I, y = torch.from_numpy(p['X']), torch.from_numpy(p['ids']), torch.from_numpy(p['y'])
        t1, line = time.time(), []
        for k, m in models.items():
            m.train()
            perm = torch.randperm(len(y))
            for b in range(0, len(perm), 4096):
                j = perm[b:b + 4096]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(m(X[j], I[j]), y[j])
                opts[k].zero_grad()
                loss.backward()
                opts[k].step()
                scheds[k].step()
            m.eval()
            line.append(f'{r2.loss_on(m, val):.4f}')
        seen += len(y)
        ns.log(f"   {what}: {p['positions']:,} positions, {len(y):,} rows ({seen / 1e6:.1f}M so far); features {t1 - t0:.0f} s, "
               f"training {time.time() - t1:.0f} s; own-validation loss " + ' / '.join(line))

    fold = 0

    def gpt_fold():
        nonlocal fold
        tab, _, _ = r2.table(chat + own_table + [r for i, r in enumerate(other) if i % 10 != fold])
        learn(f'gpt-3.5 code fold {fold}', other[fold::10], tab)
        fold += 1

    for s in range(3):
        for piece in range(3):
            tab, _, _ = r2.table(shards[(s + 1) % 3] + shards[(s + 2) % 3] + own_table + other)
            learn(f'chat shard {s} piece {piece}', shards[s][piece::3], tab)
            del tab
            gpt_fold()
    gpt_fold()
    torch.save({k: m.state_dict() for k, m in models.items()}, f'{prefix}.standins.pt')
    static, _, _ = r2.table(chat + own_table + other)

    for k, m in models.items():
        score(f'{k}: stand-ins only', reranked(m), {'rows': seen})
        m, info = r2.fit(f'{k}: stand-ins, then own', own_train, val, n_in, model=m, lr=5e-4, epochs=20)
        score(f'{k}: stand-ins, then own', reranked(m), info)
    torch.save({k: m.state_dict() for k, m in models.items()}, f'{prefix}.final.pt')
    ns.log('done')


if __name__ == '__main__':
    main()
