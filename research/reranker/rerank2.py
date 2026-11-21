"""Train the reranker of rerank.py on more: how does it do with more answers to learn from?

The target has written only 433K tokens of code, so beyond that the extra answers are stand-ins:
  own   the 9B's code answers outside the test part (table + calibration parts)
  gpt   a tenth of the gpt-3.5 solutions, proposals from a table without that tenth
        (labels are gpt-3.5's next token, not the 9B's)
  chat  a third of the last shipped-text shard, proposals from a table without that shard
        (labels are the 9B's top choice, but on chat)
The table at test time is shipped + gpt-3.5 code (no own code, set-up B of rerank.py), so the own
answers are free to learn on. Validation for early stopping is always a tenth of the own answers.
Run from the repo root with python3; STAGE picks what is trained."""
import json
import os
import random
import sys

import numpy as np
import torch

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools'))
import code_study as cs
import gen_code
import ngram_study as ns
import rerank as rr

K, V = rr.K, rr.V
torch.set_num_threads(int(os.environ.get('THREADS', 8)))
torch.manual_seed(0)


def table(rows):
    data = ns.Data(rows)
    idx = ns.labelled(data)
    static = ns.Static.build(data, idx, data.amax[idx], K)
    return static, np.bincount(data.amax[idx], minlength=V), len(idx)


def proposals(data, idx, static, inreq):
    C, avail = data.context(idx, K)
    return C, ns.candidates(C, avail, data.seq[idx], data.pos[idx], static, inreq, K)


def features(rows, static, calib, logfreq, cls, chunk=300_000):
    """Rows of rr.build for every answer position of `rows`: X, ids, right?, answer number."""
    data = ns.Data(rows)
    li, inreq, out = ns.labelled(data), ns.InRequest(data, K), []
    for b in range(0, len(li), chunk):
        idx = li[b:b + chunk]
        C, cands = proposals(data, idx, static, inreq)
        ii, tok, X, ids = rr.build(cands, calib, C, logfreq, cls)
        out.append((X, ids, (tok == data.amax[idx][ii]).astype(np.float32), data.seq[idx][ii], b + ii))
    X, ids, y, seq, grp = (np.concatenate(x) for x in zip(*out))
    return {'X': X, 'ids': ids, 'y': y, 'seq': seq, 'grp': grp, 'positions': len(li)}


def take(pool, mask):
    return {k: (v[mask] if k != 'positions' else int(len(np.unique(pool['seq'][mask])))) for k, v in pool.items()}


class Net(torch.nn.Module):
    def __init__(self, n_in, mode, emb, hidden):
        super().__init__()
        self.mode, self.emb, self.n_in = mode, emb, n_in
        d = n_in + {None: 0, 'pca': 64 * 7, 'full': 64 * 7, 'p512': 64 * 7}[mode]
        if mode in ('full', 'p512'):
            self.p_in, self.p_out = (torch.nn.Linear(4096 if mode == 'full' else 512, 64, bias=False) for _ in range(2))
            self.drop = torch.nn.Dropout(0.1)
        layers = []
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            d = h
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1))

    def forward(self, x, ids):
        x = x[:, :self.n_in]
        if self.mode == 'pca':
            x = torch.cat([x, self.emb['pca'][1][ids[:, 0]], self.emb['pca'][0][ids[:, 1:]].flatten(1)], 1)
        if self.mode in ('full', 'p512'):
            x = torch.cat([x, self.drop(self.p_out(self.emb[self.mode][1][ids[:, 0]])),
                           self.drop(self.p_in(self.emb[self.mode][0][ids[:, 1:]])).flatten(1)], 1)
        return self.net(x).squeeze(1)


def loss_on(model, part, batch=16384):
    total = 0.0
    with torch.no_grad():
        for b in range(0, len(part['y']), batch):
            s = slice(b, b + batch)
            total += float(torch.nn.functional.binary_cross_entropy_with_logits(
                model(torch.from_numpy(part['X'][s]), torch.from_numpy(part['ids'][s])), torch.from_numpy(part['y'][s]), reduction='sum'))
    return total / len(part['y'])


def fit(name, parts, val, n_in, mode=None, emb=None, hidden=(256, 128), epochs=30, lr=2e-3, model=None, patience=3, batch=4096):
    """Train on the rows of `parts` (a list of pools), stopping on the loss on `val`."""
    X = np.concatenate([p['X'] for p in parts]) if len(parts) > 1 else parts[0]['X']
    ids = np.concatenate([p['ids'] for p in parts]) if len(parts) > 1 else parts[0]['ids']
    y = np.concatenate([p['y'] for p in parts]) if len(parts) > 1 else parts[0]['y']
    Xt, It, yt = torch.from_numpy(X), torch.from_numpy(ids), torch.from_numpy(y)
    model = model or Net(n_in, mode, emb, hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(y) + batch - 1) // batch), pct_start=0.05)
    model.eval()
    best, state, bad = loss_on(model, val) if any(p.requires_grad for p in model.parameters()) else 1e9, None, 0
    state = {k: t.clone() for k, t in model.state_dict().items()}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(yt))
        for b in range(0, len(perm), batch):
            j = perm[b:b + batch]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xt[j], It[j]), yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        v = loss_on(model, val)
        if v < best - 1e-4:
            best, state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad == patience:
                break
    model.load_state_dict(state)
    model.eval()
    weights = sum(p.numel() for p in model.parameters())
    ns.log(f'   trained {name}: {len(y):,} rows, {epoch + 1} epochs, own-validation loss {best:.4f}, {weights:,} weights')
    return model, {'rows': int(len(y)), 'epochs': epoch + 1, 'validation_loss': best, 'weights': weights}


def main():
    stage = os.environ.get('STAGE', 'data').split(',')
    pool = gen_code.pick(rr.dataset, 2000, gen_code.SEED, gen_code.MAX_CHARS)
    held_ids = {p['id'] for p in pool if cs.part(p['id']) != 'table'}
    own = cs.own_rows('data/code/rows-9b.jsonl')
    test = ns.Data([r for r in own if cs.part(r['id']) == 'test' and r['finished']])
    learn_rows = [r for r in own if cs.part(r['id']) != 'test']           # table and calibration parts, cut-off answers too
    other = cs.other_rows(rr.dataset, rr.tokenizer, own[0], held_ids)
    shards = [[r for r in ns.load_shard(s, None, 'data/cache') if r['finished']] for s in cs.SHARDS]
    cls = rr.token_classes()
    emb = {'pca': tuple(torch.from_numpy(e) for e in rr.embeddings()),
           'full': rr.full_embeddings() if 'full' in stage else None,
           'p512': tuple(torch.from_numpy(e) for e in rr.embeddings_pca(512)) if 'p512' in stage else None}

    static, counts, n = table(shards[0] + shards[1] + shards[2] + other)
    logfreq = (np.log1p(counts) / 16).astype(np.float32)
    learn = ns.Data(learn_rows)
    li = ns.labelled(learn)
    calib = ns.Calibration(K)
    calib.add(proposals(learn, li, static, ns.InRequest(learn, K))[1], learn.amax[li])
    own_pool = features(learn_rows, static, calib, logfreq, cls)
    is_val = own_pool['seq'] % 10 == 0
    val, own_train = take(own_pool, is_val), take(own_pool, ~is_val)
    ns.log(f"table {n:,} tokens. own: {len(learn_rows)} answers, {own_pool['positions']:,} positions, {len(own_pool['y']):,} rows "
           f"({int(is_val.sum()):,} of them validation)")

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
        rng = np.random.default_rng(1)
        draws = rng.integers(0, len(tok), (4000, len(tok)))
        diff = np.sort(tok[draws].sum(1) / spent[draws].sum(1) - base[0][draws].sum(1) / base[1][draws].sum(1))
        best = max((f'pad {k}' for k in range(1, rr.DEPTH + 1)), key=lambda p: r[p]['speedup'])
        r.update(info or {}, gain_pad3=float(r['pad 3']['speedup'] - base[0].sum() / base[1].sum()), gain_95=[float(diff[100]), float(diff[3899])],
                 best_rule=best)
        result[name] = r
        ns.log(f"{name:58s} first guess {r['first_guess_right']:.3f}  pad 3 {r['pad 3']['speedup']:.3f}x  gain {r['gain_pad3']:+.3f} "
               f"({diff[100]:+.3f} to {diff[3899]:+.3f})  best {best} {r[best]['speedup']:.3f}x")
        json.dump(result, open(sys.argv[1], 'w'), indent=1)

    def reranked(model):
        def pick(cands, C, d):
            ii, tok, X, ids = rr.build(cands, calib, C, logfreq, cls)
            p = np.concatenate([torch.sigmoid(model(torch.from_numpy(X[b:b + 16384]), torch.from_numpy(ids[b:b + 16384]))).detach().numpy()
                                for b in range(0, len(X), 16384)])
            token, conf = np.full(len(C), -1, np.int32), np.zeros(len(C), np.float32)
            np.maximum.at(conf, ii, p)
            win = p >= conf[ii]
            token[ii[win]] = tok[win]
            return token, conf
        return pick

    def run(name, parts, **kw):
        with torch.no_grad():
            pass
        model, info = fit(name, parts, val, own_pool['X'].shape[1], **kw)
        with torch.no_grad():
            score(name, reranked(model), info)
        return model

    score('most confident (the drafter now)', lambda cands, C, d: ns.choose(cands, calib))

    if 'curve' in stage:                                   # more of the target's own code
        seqs = np.unique(own_train['seq'])
        random.Random(0).shuffle(seqs := list(seqs))
        for share in (1 / 16, 1 / 8, 1 / 4, 1 / 2):
            some = take(own_train, np.isin(own_train['seq'], seqs[:int(len(seqs) * share)]))
            run(f"own code, {share:.4g} of it ({some['positions']} answers)", [some])
    own_model = run(f"own code, all ({own_train['positions']} answers)", [own_train])

    gpt_pool = chat_pool = None
    if {'data', 'full', 'pca', 'p512'} & set(stage):
        fold = other[::10]
        s, _, n = table(shards[0] + shards[1] + shards[2] + [r for i, r in enumerate(other) if i % 10])
        gpt_pool = features(fold, s, calib, logfreq, cls)
        ns.log(f"gpt: table {n:,} tokens; {len(fold)} solutions, {gpt_pool['positions']:,} positions, {len(gpt_pool['y']):,} rows")
        some = shards[2][::3]
        s, _, n = table(shards[0] + shards[1] + other)
        chat_pool = features(some, s, calib, logfreq, cls)
        ns.log(f"chat: table {n:,} tokens; {len(some)} answers, {chat_pool['positions']:,} positions, {len(chat_pool['y']):,} rows")
        del s
    if 'data' in stage:
        run('gpt-3.5 code only', [gpt_pool], epochs=8, patience=2)
        run('chat only', [chat_pool], epochs=8, patience=2)
        run('own + gpt-3.5 code', [own_train, gpt_pool], epochs=10, patience=2)
        run('own + gpt-3.5 code + chat', [own_train, gpt_pool, chat_pool], epochs=10, patience=2)
        m, _ = fit('gpt-3.5 code + chat (before fine-tuning)', [gpt_pool, chat_pool], val, own_pool['X'].shape[1], epochs=8, patience=2)
        with torch.no_grad():
            score('gpt-3.5 code + chat', reranked(m))
        run('gpt-3.5 code + chat, then own', [own_train], model=m, lr=5e-4, epochs=20)
        m, _ = fit('wider, gpt-3.5 code + chat (before fine-tuning)', [gpt_pool, chat_pool], val, own_pool['X'].shape[1], epochs=8, patience=2,
                   hidden=(512, 256, 128))
        run('wider (512-256-128): gpt-3.5 code + chat, then own', [own_train], model=m, lr=5e-4, epochs=20)
    for mode in ('pca', 'p512', 'full'):
        if mode in stage:
            run(f'embeddings ({mode}): own code', [own_train], mode=mode, emb=emb)
            m, _ = fit(f'embeddings ({mode}): gpt-3.5 code + chat (before fine-tuning)', [gpt_pool, chat_pool], val, own_pool['X'].shape[1],
                       mode=mode, emb=emb, epochs=4 if mode == 'full' else 8, patience=2, hidden=(512, 256, 128))
            with torch.no_grad():
                score(f'embeddings ({mode}): gpt-3.5 code + chat', reranked(m))
            run(f'embeddings ({mode}): gpt-3.5 code + chat, then own', [own_train], model=m, lr=5e-4, epochs=20)


if __name__ == '__main__':
    main()
