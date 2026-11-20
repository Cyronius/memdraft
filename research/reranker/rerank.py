"""Would a small reranker over the lookups' proposals beat taking the most confident one?

At each position the lookups give up to 12 proposals (table and request, contexts of 1..6 tokens).
The drafter takes the one with the highest calibrated confidence. Here a small MLP scores each
distinct proposed token instead, from:
  votes       which lookups propose it, their confidences and counts, and what the rivals look like
  +classes    the kind of token proposed and of the two tokens before it, and how common it is
  +embeddings the target's own embeddings (64 principal components) of the proposed token and the
              four tokens before it
and the highest score is drafted. Scored on the code test answers through whole draft chains, with
the 9B's GPU step costs. `oracle` takes the right token whenever any lookup proposes it: the most
any reranker over this list could do.

Two set-ups, because the reranker must learn on answers that are not in the table:
  A  the largest table (shipped + own + gpt-3.5 code); reranker learns on the calibration answers
  B  shipped + gpt-3.5 code (no own code); reranker learns on the own-code answers + calibration
Run from the repo root with python3."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import code_study as cs
import gen_code
import ngram_study as ns

torch.set_num_threads(8)
torch.manual_seed(0)
K, S, DEPTH, NC, V = 6, 12, 6, 8, 248320
COST = cs.COSTS[0]
SP = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(gen_code.CACHE, 'models/Qwen3.5-9B')
EARLIER = np.tril(np.ones((S, S), bool), -1)
dataset = os.path.join(gen_code.CACHE, 'code-prompts-src/data-oss_instruct-decontaminated.jsonl')
tokenizer = f'{MODEL}/tokenizer.json'


def embeddings():
    path = f'{SP}/emb64.npz'
    if os.path.exists(path):
        z = np.load(path)
        return z['inp'], z['out']
    from safetensors import safe_open
    out = {}
    with safe_open(f'{MODEL}/model.safetensors-00001-of-00004.safetensors', 'pt') as f:
        for name, key in (('inp', 'model.language_model.embed_tokens.weight'), ('out', 'lm_head.weight')):
            E = f.get_tensor(key).float()
            E -= E.mean(0)
            _, s, Vt = torch.pca_lowrank(E, q=64, center=False, niter=3)
            P = E @ Vt
            out[name] = (P / P.std(0)).numpy().astype(np.float32)
            ns.log(f'{key}: 64 components hold {float((s ** 2).sum() / (E ** 2).sum()):.0%} of the variance')
            del E
    np.savez(path, **out)
    return out['inp'], out['out']


def embeddings_pca(q):
    """The first q principal components of the input and output embeddings, each scaled to unit spread overall."""
    path = f'{SP}/emb{q}.npz'
    if os.path.exists(path):
        z = np.load(path)
        return z['inp'], z['out']
    from safetensors import safe_open
    out = {}
    with safe_open(f'{MODEL}/model.safetensors-00001-of-00004.safetensors', 'pt') as f:
        for name, key in (('inp', 'model.language_model.embed_tokens.weight'), ('out', 'lm_head.weight')):
            E = f.get_tensor(key).float()
            E -= E.mean(0)
            _, s, Vt = torch.pca_lowrank(E, q=q, center=False, niter=3)
            P = E @ Vt
            out[name] = (P / P.std()).numpy().astype(np.float32)
            ns.log(f'{key}: {q} components hold {float((s ** 2).sum() / (E ** 2).sum()):.0%} of the variance')
            del E
    np.savez(path, **out)
    return out['inp'], out['out']


def full_embeddings():
    from safetensors import safe_open
    with safe_open(f'{MODEL}/model.safetensors-00001-of-00004.safetensors', 'pt') as f:
        inp, out = (f.get_tensor(k).float() for k in ('model.language_model.embed_tokens.weight', 'lm_head.weight'))
    return inp / inp.std(), out / out.std()


def token_classes():
    path = f'{SP}/tokcls.npy'
    if os.path.exists(path):
        return np.load(path)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer)
    cls = np.zeros(V, np.int8)
    for i in range(min(V, tok.get_vocab_size())):
        t = tok.decode([i], skip_special_tokens=False)
        core = t.strip()
        ident = core.replace('_', 'a').isalnum()
        if not t or t.startswith('<|') or t.startswith('<'):
            c = 0
        elif '\n' in t:
            c = 1
        elif not core:
            c = 2
        elif not any(ch.isalnum() or ch == '_' for ch in core):
            c = 3
        elif core.isdigit():
            c = 4
        elif ident and t[0] == ' ':
            c = 5
        elif ident:
            c = 6
        else:
            c = 7
        cls[i] = c
    np.save(path, cls)
    return cls


def slots(cands, calib):
    n = len(cands[0][2])
    H, T = np.zeros((n, S), bool), np.full((n, S), -1, np.int32)
    CF, A, B = (np.zeros((n, S), np.float32) for _ in range(3))
    for kind, k, hit, tok, cell in cands:
        s = 2 * (k - 1) + (kind == 'context')
        H[:, s], T[:, s] = hit, np.where(hit, tok, -1)
        CF[:, s] = np.where(hit, calib.confidence(kind, k, cell), 0)
        if kind == 'static':
            A[:, s], B[:, s] = np.where(hit, (cell[0] + 1) / 10, 0), np.where(hit, (cell[1] + 1) / 12, 0)
        else:
            A[:, s] = np.where(hit, cell[0] / ns.CC_MAX, 0)
    return H, T, CF, A, B


def build(cands, calib, C, logfreq, cls):
    """One row per position and distinct proposed token: (position, token, features, token ids)."""
    H, T, CF, A, B = slots(cands, calib)
    agree = H[:, :, None] & H[:, None, :] & (T[:, :, None] == T[:, None, :])
    ii, ss = np.nonzero(H & ~(agree & EARLIER[None]).any(2))
    ag, cf = agree[ii, ss].astype(np.float32), CF[ii]
    top = CF.max(1)
    base = np.where(top >= 1e-9, S - 1 - np.argmax(CF[:, ::-1] >= top[:, None], axis=1), -1)[ii]   # ns.choose's slot
    own, rival = (ag * cf).max(1), ((1 - ag) * cf).max(1)
    is_base = ag[np.arange(len(ii)), np.maximum(base, 0)] * (base >= 0)
    tok, p0, p1 = T[ii, ss], C[ii, 0], C[ii, 1]
    one = np.eye(NC, dtype=np.float32)
    X = np.concatenate([ag, ag * cf, ag * A[ii], ag * B[ii], H[ii].astype(np.float32), cf, A[ii], B[ii],
                        np.stack([own, rival, ag.sum(1) / S, own - rival, is_base], 1),
                        one[cls[tok]], one[cls[p0]], one[cls[p1]],
                        np.stack([logfreq[tok], logfreq[p0], tok == p0, tok == p1], 1).astype(np.float32)], 1)
    ids = np.concatenate([tok[:, None], C[ii, :6]], 1).astype(np.int64)
    return ii, tok, X, ids


N_VOTES = 8 * S + 5
VARIANTS = {'votes': (N_VOTES, None), 'votes + classes': (None, None), 'votes + classes + embeddings (64 PCA)': (None, 'pca'),
            'votes + classes + embeddings (learned projection)': (None, 'full')}


class Net(torch.nn.Module):
    def __init__(self, n_in, mode, emb):
        super().__init__()
        self.mode, self.emb = mode, emb
        d = n_in + {None: 0, 'pca': 64 * 5, 'full': 64 * 7}[mode]
        if mode == 'full':                       # a table of 64 numbers per token once trained: a lookup at run time
            self.p_in, self.p_out = (torch.nn.Linear(4096, 64, bias=False) for _ in range(2))
            self.drop = torch.nn.Dropout(0.1)
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(),
                                       torch.nn.Linear(128, 1))

    def forward(self, x, ids):
        if self.mode == 'pca':
            x = torch.cat([x, self.emb['pca'][1][ids[:, 0]], self.emb['pca'][0][ids[:, 1:5]].flatten(1)], 1)
        if self.mode == 'full':
            x = torch.cat([x, self.drop(self.p_out(self.emb['full'][1][ids[:, 0]])),
                           self.drop(self.p_in(self.emb['full'][0][ids[:, 1:]])).flatten(1)], 1)
        return self.net(x).squeeze(1)


def predict(model, X, ids):
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 8192):
            out.append(torch.sigmoid(model(torch.from_numpy(X[b:b + 8192]), torch.from_numpy(ids[b:b + 8192]))))
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


def fit(name, X, ids, y, val, emb):
    n_in, mode = VARIANTS[name]
    X = X[:, :n_in]
    model = Net(X.shape[1], mode, emb)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt, It, yt = torch.from_numpy(X[~val]), torch.from_numpy(ids[~val]), torch.from_numpy(y[~val].astype(np.float32))
    Xv, Iv, yv = torch.from_numpy(X[val]), torch.from_numpy(ids[val]), torch.from_numpy(y[val].astype(np.float32))
    best, state, bad = 1e9, None, 0
    for epoch in range(30):
        model.train()
        perm = torch.randperm(len(Xt))
        for b in range(0, len(perm), 2048):
            j = perm[b:b + 2048]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xt[j], It[j]), yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(sum(float(torch.nn.functional.binary_cross_entropy_with_logits(model(Xv[b:b + 8192], Iv[b:b + 8192]), yv[b:b + 8192],
                                                                                     reduction='sum')) for b in range(0, len(Xv), 8192)) / len(Xv))
        if v < best - 1e-4:
            best, state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad == 3:
                break
    model.load_state_dict(state)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    ns.log(f'   {name}: {len(Xt):,} rows, {epoch + 1} epochs, validation loss {best:.4f}, {params:,} weights')
    return model, n_in, params


def steps(data, idx, accepted, drafted):
    where = np.full(len(data.tok), -1, np.int64)
    where[idx] = np.arange(len(idx))
    d_all, a_all, s_all = [], [], []
    for s in range(len(data.start)):
        j, last = data.ans[s], data.end[s] - 1
        while j <= last:
            i = where[j]
            d_all.append(drafted[i])
            a_all.append(accepted[i])
            s_all.append(s)
            j += accepted[i] + 1
    return np.array(d_all), np.array(a_all), np.array(s_all)


def speed(d, a):
    shape = np.r_[True, d[1:] != d[:-1]]
    return float((a.sum() + len(d)) / (len(d) + COST['beta'] * d.sum() + COST['accept'] * a.sum() + COST['reshape'] * shape.sum()))


def chain(data, idx, inreq, static, pick):
    """Draft DEPTH tokens at every position with `pick`, then decode with each fixed draft length."""
    C, avail = data.context(idx, K)
    seq, pos = data.seq[idx], data.pos[idx]
    tokens = np.full((len(idx), DEPTH), -1, np.int32)
    for d in range(DEPTH):
        token, conf = pick(ns.candidates(C, avail, seq, pos, static, inreq, K), C, d)
        token = np.where(conf > 0, token, C[:, 0])
        tokens[:, d] = token
        C = np.concatenate([token[:, None], C[:, :-1]], 1)
        avail = np.minimum(avail + 1, K)
    accepted, judged = ns.score(data, idx, tokens, None, 'greedy')
    out = {'first_guess_right': float((tokens[:, 0] == data.amax[idx]).mean())}
    for n in range(1, DEPTH + 1):
        d = np.minimum(np.int32(n), judged)
        sd, sa, ss = steps(data, idx, np.minimum(accepted, d), d)
        if n == 3:                                # per answer, for a paired bootstrap over answers
            shape = np.r_[True, (sd[1:] != sd[:-1]) | (ss[1:] != ss[:-1])]
            spent = 1 + COST['beta'] * sd + COST['accept'] * sa + COST['reshape'] * shape
            m = len(data.start)
            out['per answer, pad 3'] = {'tokens': np.bincount(ss, sa + 1, m).tolist(), 'spent': np.bincount(ss, spent, m).tolist()}
        out[f'pad {n}'] = {'tokens_per_step': float((sa.sum() + len(sd)) / len(sd)), 'acceptance': float(sa.sum() / sd.sum()),
                           'speedup': speed(sd, sa)}
    return out


def regime(name, table_rows, learn_rows, test, emb, cls):
    ns.log(f'{name}')
    table = ns.Data(table_rows)
    idx = ns.labelled(table)
    static = ns.Static.build(table, idx, table.amax[idx], K)
    logfreq = (np.log1p(np.bincount(table.amax[idx], minlength=V)) / 16).astype(np.float32)
    del table
    learn = ns.Data(learn_rows)
    li = ns.labelled(learn)
    C, avail = learn.context(li, K)
    cands = ns.candidates(C, avail, learn.seq[li], learn.pos[li], static, ns.InRequest(learn, K), K)
    calib = ns.Calibration(K)
    calib.add(cands, learn.amax[li])
    ii, tok, X, ids = build(cands, calib, C, logfreq, cls)
    y = tok == learn.amax[li][ii]
    val = learn.seq[li][ii] % 10 == 0
    ns.log(f'   table {len(idx):,} tokens; learning on {len(learn.start)} answers, {len(li):,} positions, {len(ii):,} proposals '
           f'({len(ii) / len(li):.1f} per position)')

    ti = ns.labelled(test)
    inreq = ns.InRequest(test, K)

    def oracle(cands, C, d):
        truth = test.amax[np.minimum(ti + d, len(test.tok) - 1)]
        token, conf = ns.choose(cands, calib)
        for kind, k, hit, tok, cell in cands:
            ok = hit & (tok == truth)
            token, conf = np.where(ok, tok, token), np.where(ok, 1, conf)
        return token, conf

    def reranked(model, n_in):
        def pick(cands, C, d):
            ii, tok, X, ids = build(cands, calib, C, logfreq, cls)
            p = predict(model, X[:, :n_in], ids)
            token, conf = np.full(len(C), -1, np.int32), np.zeros(len(C), np.float32)
            np.maximum.at(conf, ii, p)
            win = p >= conf[ii]
            token[ii[win]] = tok[win]
            return token, conf
        return pick

    out = {'most confident (the drafter now)': chain(test, ti, inreq, static, lambda cands, C, d: ns.choose(cands, calib))}
    for variant in [v for v in VARIANTS if not os.environ.get('ONLY') or v in os.environ['ONLY'].split(',')]:
        model, n_in, params = fit(variant, X, ids, y, val, emb)
        out[variant] = dict(chain(test, ti, inreq, static, reranked(model, n_in)), weights=params)
    out['oracle'] = chain(test, ti, inreq, static, oracle)
    for who, r in out.items():
        best = max((f'pad {n}' for n in range(1, DEPTH + 1)), key=lambda p: r[p]['speedup'])
        ns.log(f"   {who:52s} first guess {r['first_guess_right']:.3f}  " + '  '.join(
            f"{p} {r[p]['speedup']:.3f}x" for p in ('pad 2', 'pad 3', 'pad 4')) + f"  best {best} {r[best]['speedup']:.3f}x "
            f"({r[best]['tokens_per_step']:.2f} tok/step, acc {r[best]['acceptance']:.0%})")
    return out


def main():
    pool = gen_code.pick(dataset, 2000, gen_code.SEED, gen_code.MAX_CHARS)
    held_ids = {p['id'] for p in pool if cs.part(p['id']) != 'table'}
    own = cs.own_rows('data/code/rows-9b.jsonl')
    held = [r for r in own if cs.part(r['id']) != 'table']
    cal_rows = [r for r in held if cs.part(r['id']) == 'cal' and r['finished']]
    test = ns.Data([r for r in held if cs.part(r['id']) == 'test' and r['finished']])
    own_table = [r for r in own if cs.part(r['id']) == 'table']
    other = cs.other_rows(dataset, tokenizer, own[0], held_ids)
    shards = [r for s in cs.SHARDS for r in ns.load_shard(s, None, 'data/cache') if r['finished']]
    emb = {'pca': tuple(torch.from_numpy(e) for e in embeddings()), 'full': None if os.environ.get('ONLY') else full_embeddings()}
    cls = token_classes()
    ns.log('token classes: ' + ' '.join(f'{c}:{int((cls == c).sum())}' for c in range(NC)))
    result = {'A: largest table, learns on calibration answers': regime('A', shards + own_table + other, cal_rows, test, emb, cls)}
    if not os.environ.get('ONLY'):
        result['B: shipped + gpt-3.5 code, learns on own code + calibration'] = regime('B', shards + other, own_table + cal_rows, test, emb, cls)
    json.dump(result, open(sys.argv[1], 'w'), indent=1)


if __name__ == '__main__':
    main()
