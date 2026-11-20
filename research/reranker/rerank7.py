"""The reranker on the pod's answers: does fifteen times more own code teach it more?

rerank6.py fine-tuned the reranker on yesterday's own answers and on up to 4,200 more written on
this machine (+0.012x over none). The pod (run_gen_pod.py) wrote about 69,000 more. Here they all go
into the table, and the reranker learns on them the same way: an own answer gets its proposals from
a table built without it (FOLDS parts) and without the gpt-3.5 solution to its prompt.

Every position of every answer would be about 90M rows, 50 GB: too much for this machine's share.
Every position of yesterday's and this machine's answers is kept, as in rerank6.py, and a random
RATE of the positions of the pod's answers (default 0.2: about 17M rows, from every one of the
answers; neighbouring positions say much the same, so a fifth of the positions of every answer is
worth more than all the positions of a fifth of the answers). Each fold's rows go to disk, and are
put together once no table is in memory, sorted so that each training set is a slice (no copies).

Scored on the same test answers as before, pad 3, all with the full table:
  the drafter; rerank-more/600.pt and 4200.pt as they are; the stand-in reranker fine-tuned on
  yesterday's and this machine's answers only, with the first quarter of the pod's answers too,
  and with all of them.
Run from the repo root with python3, under a memory limit (systemd-run
--user --scope -p MemoryMax=30G). Argument: output prefix. POD_ROWS: the pod's rows, from
collect_pod_rows.py. POD_LIMIT, FOLDS, EPOCHS, RATE make a smaller run."""
import gc
import json
import os
import resource
import shutil
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
FOLDS = int(os.environ.get('FOLDS', 8))
RATE = float(os.environ.get('RATE', 0.2))
EPOCHS = int(os.environ.get('EPOCHS', 20))
HIDDEN = (512, 256, 128)
BEST = 'embeddings, learned from 512'
KEYS = ('X', 'ids', 'y', 'when', 'answer')
DRAFTER = 'full table, the drafter'
SAME = "full table, reranker fine-tuned on yesterday's and this machine's answers only"
SHORT = {DRAFTER: 'the drafter', SAME: 'the reranker without the pod'}


def peak():
    return f'peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB'


def light(path, limit=None):
    """cs.own_rows with the token lists as arrays and no text: 69,000 rows of Python ints take gigabytes."""
    rows = []
    for line in open(path):
        if limit is not None and len(rows) >= limit or not line.endswith('\n'):
            break
        r = json.loads(line)
        r.pop('text', None)
        r['prompt_ids'] = np.array(r['prompt_ids'], np.int32)
        r['answer_ids'] = r['amax'] = np.array(r['answer_ids'], np.int32)
        r['source'] = 'python' if r['lang'] == 'python' else 'other-languages'
        rows.append(r)
    return rows


def sampled(rows, static, calib, logfreq, cls, rate, rng, chunk=300_000):
    """r2.features at a random share of the answer positions: rate[i] of those of rows[i]."""
    data = ns.Data(rows)
    li = ns.labelled(data)
    li = li[rng.random(len(li)) < rate[data.seq[li]]]
    inreq, out = ns.InRequest(data, K), []
    for b in range(0, len(li), chunk):
        idx = li[b:b + chunk]
        C, cands = r2.proposals(data, idx, static, inreq)
        ii, tok, X, ids = rr.build(cands, calib, C, logfreq, cls)
        out.append((X, ids, (tok == data.amax[idx][ii]).astype(np.float32), data.seq[idx][ii]))
    X, ids, y, seq = (np.concatenate(x) for x in zip(*out))
    return {'X': X, 'ids': ids, 'y': y, 'seq': seq, 'positions': len(li)}


def main():
    prefix = sys.argv[1]
    spill = prefix + '.features'
    pool = gen_code.pick(rr.dataset, 2000, gen_code.SEED, gen_code.MAX_CHARS)
    pool_ids = {p['id'] for p in pool}
    held_ids = {p['id'] for p in pool if cs.part(p['id']) != 'table'}
    own = cs.own_rows('data/code/rows-9b.jsonl')
    more = cs.own_rows('data/code/rows-9b-more.jsonl')
    pod = light(os.environ['POD_ROWS'], int(os.environ['POD_LIMIT']) if 'POD_LIMIT' in os.environ else None)
    new_ids = [r['id'] for r in more + pod]
    assert len(set(new_ids)) == len(new_ids), 'a prompt has two answers'
    assert not set(new_ids) & pool_ids, 'a new answer is to a prompt inside the pool'
    test = ns.Data([r for r in own if cs.part(r['id']) == 'test' and r['finished']])
    own_table = [r for r in own if cs.part(r['id']) == 'table']
    cal_rows = [r for r in own if cs.part(r['id']) == 'cal']
    other = cs.other_rows(rr.dataset, rr.tokenizer, own[0], held_ids)
    chat = [r for s in cs.SHARDS for r in ns.load_shard(s, None, 'data/cache') if r['finished']]
    cls = rr.token_classes()
    emb = {'pca': None, 'full': None, 'p512': tuple(torch.from_numpy(e) for e in rr.embeddings_pca(512))}
    tokens = lambda rows: int(sum(len(r['answer_ids']) for r in rows))
    ns.log(f"answers: yesterday's outside the test part {len(own_table) + len(cal_rows)} ({tokens(own_table) + tokens(cal_rows):,} tokens), "
           f"this machine's new ones {len(more)} ({tokens(more):,}), the pod's {len(pod)} ({tokens(pod):,}); "
           f'pod positions kept for learning: {RATE:.0%}; {peak()}')

    cal = ns.Data([r for r in cal_rows if r['finished']])
    ci = ns.labelled(cal)
    inreq_cal = ns.InRequest(cal, K)
    ti = ns.labelled(test)
    inreq_test = ns.InRequest(test, K)
    draws = np.random.default_rng(1).integers(0, len(test.start), (4000, len(test.start)))
    result = {'answers': {'yesterday': len(own_table) + len(cal_rows), 'this machine': len(more), 'pod': len(pod)},
              'tokens': {'this machine': tokens(more), 'pod': tokens(pod)}, 'pod_rate': RATE}
    runs = {}

    def score(name, static, pick, against, info=None):
        r = rr.chain(test, ti, inreq_test, static, pick)
        per = r.pop('per answer, pad 3')
        tok, spent = runs[name] = (np.array(per['tokens']), np.array(per['spent']))
        r.update(info or {}, per_answer_pad3={'tokens': tok.tolist(), 'spent': spent.tolist()})
        line = ''
        for ref in against:
            b = runs[ref]
            diff = np.sort(tok[draws].sum(1) / spent[draws].sum(1) - b[0][draws].sum(1) / b[1][draws].sum(1))
            gain = float(tok.sum() / spent.sum() - b[0].sum() / b[1].sum())
            r[f'gain over: {ref}'] = {'pad 3': gain, '95%': [float(diff[100]), float(diff[3899])],
                                      'answers_faster': int((tok / spent > b[0] / b[1]).sum())}
            line += f'  {gain:+.3f} ({diff[100]:+.3f} to {diff[3899]:+.3f}) over {SHORT[ref]}'
        best = max((f'pad {k}' for k in range(1, rr.DEPTH + 1)), key=lambda p: r[p]['speedup'])
        r['best_rule'] = best
        result[name] = r
        ns.log(f"{name:80s} first guess {r['first_guess_right']:.3f}  pad 3 {r['pad 3']['speedup']:.3f}x "
               f"({r['pad 3']['tokens_per_step']:.2f} tok/step, acc {r['pad 3']['acceptance']:.0%})  best {best} {r[best]['speedup']:.3f}x{line}")
        json.dump(result, open(f'{prefix}.json', 'w'), indent=1)

    def reranked(model, calib, logfreq):
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

    def net(state=None, path=None):
        model = r2.Net(n_in, 'p512', emb, HIDDEN)
        model.load_state_dict(state if state is not None else torch.load(path)[BEST])
        model.eval()
        return model

    # The full table: what the drafter and the rerankers we have make of it.
    in_table = own_table + more + pod
    everything_rows = chat + in_table + other
    static, counts, n = r2.table(everything_rows)
    calib = ns.Calibration(K)
    calib.add(r2.proposals(cal, ci, static, inreq_cal)[1], cal.amax[ci])
    logfreq = (np.log1p(counts) / 16).astype(np.float32)
    n_in = rr.build(r2.proposals(cal, ci[:1000], static, inreq_cal)[1], calib, cal.context(ci[:1000], K)[0], logfreq, cls)[2].shape[1]
    ns.log(f'full table: {n:,} tokens; {peak()}')
    result['table_tokens'] = n
    score(DRAFTER, static, lambda cands, C, d: ns.choose(cands, calib), [])
    for f in ('600', '4200'):
        score(f'full table, rerank-more/{f}.pt as it is', static, reranked(net(path=f'data/code/rerank-more/{f}.pt'), calib, logfreq), [DRAFTER])

    # Rows to learn from, each fold's to disk. `when` says how early an answer came: yesterday's 0,
    # this machine's 1 and up, the pod's after those, each in the order written.
    when = {r['id']: 0 for r in cal_rows + own_table}
    when.update({r['id']: 1 + i for i, r in enumerate(more + pod)})
    rate = {r['id']: RATE for r in pod}
    rng = np.random.default_rng(7)
    shutil.rmtree(spill, ignore_errors=True)
    os.makedirs(spill)
    files, counted = [], [0]

    def add(rows, table, tag):
        p = sampled(rows, table, calib, logfreq, cls, np.array([rate.get(r['id'], 1.0) for r in rows]), rng)
        p['when'] = np.array([when[r['id']] for r in rows], np.int32)[p['seq']]
        p['answer'] = p['seq'].astype(np.int64) + counted[0]
        counted[0] += len(rows)
        np.savez(f'{spill}/{tag}.npz', **{k: p[k] for k in KEYS})
        files.append(f'{spill}/{tag}.npz')
        ns.log(f"   {tag}: {len(rows)} answers, {p['positions']:,} positions, {len(p['y']):,} rows; {peak()}")

    add(cal_rows, static, 'cal')
    del static
    gc.collect()
    for f in range(FOLDS):
        rows = in_table[f::FOLDS]
        gone = {r['id'] for r in rows}
        table, _, _ = r2.table(chat + [r for r in other if r['id'] not in gone] + [r for i, r in enumerate(in_table) if i % FOLDS != f])
        add(rows, table, f'fold{f}')
        del table, rows
        gc.collect()

    # Put them together: training rows first, sorted by `when`, then the validation rows (every tenth
    # answer), so that the rows of each training set are a slice of the arrays.
    small = {k: [] for k in ('when', 'answer')}
    for path in files:
        with np.load(path) as z:
            for k in small:
                small[k].append(z[k])
    small = {k: np.concatenate(v) for k, v in small.items()}
    total = len(small['when'])
    is_val = small['answer'] % 10 == 0
    order = np.lexsort((small['when'], is_val))
    where = np.empty(total, np.int64)
    where[order] = np.arange(total)
    with np.load(files[0]) as z:
        everything = {k: np.empty((total,) + z[k].shape[1:], z[k].dtype) for k in ('X', 'ids', 'y')}
    at = 0
    for path in files:
        with np.load(path) as z:
            m = len(z['y'])
            for k in everything:
                everything[k][where[at:at + m]] = z[k]
        at += m
    shutil.rmtree(spill)
    n_train = int((~is_val).sum())
    sorted_when = small['when'][order][:n_train]
    val = {k: v[n_train:] for k, v in everything.items()}
    ns.log(f"rows to learn from: {n_train:,} ({int((sorted_when == 0).sum()):,} from yesterday's answers, "
           f"{int(((sorted_when > 0) & (sorted_when <= len(more))).sum()):,} from this machine's, "
           f"{int((sorted_when > len(more)).sum()):,} from the pod's); validation {total - n_train:,}; {peak()}")

    quarter = len(pod) // 4
    plan = [(SAME, len(more)),
            (f"full table, reranker fine-tuned with the first {quarter} of the pod's answers too", len(more) + quarter),
            (f"full table, reranker fine-tuned with all {len(pod)} of the pod's answers too", len(more) + len(pod))]
    trained = {}
    for name, upto in plan:
        m = int(np.searchsorted(sorted_when, upto, 'right'))
        some = {k: v[:m] for k, v in everything.items()}
        start = torch.load('data/code/rerank-big.standins.pt')[BEST]
        model, info = r2.fit(f'{BEST}, answers up to {upto}', [some], val, n_in, model=net(start), lr=5e-4, epochs=EPOCHS)
        trained[name] = (model.state_dict(), dict(info, answers_used=upto))
        torch.save({k: v[0] for k, v in trained.items()}, f'{prefix}.pt')
        ns.log(f'   {peak()}')
    del everything, val, some
    gc.collect()

    static, _, _ = r2.table(everything_rows)
    for name, (state, info) in trained.items():
        score(name, static, reranked(net(state), calib, logfreq), [DRAFTER] + [SAME] * (name != SAME), info)
    ns.log(f'done; {peak()}')


if __name__ == '__main__':
    main()
