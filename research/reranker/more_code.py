"""What is text from elsewhere worth in the table, next to the target's own code?

The reference is the largest table: shipped text + the 9B's own code (the first MORE_LIMIT new
answers included) + all the gpt-3.5 solutions. Against it, on the same test answers:
  less gpt-3.5 code        a quarter and a half of the solutions: what one doubling of them buys
  less own code            without the new answers, and with half of them
  more text from elsewhere the local K2-Horizon sample (horizon_tokens.py): the instruction rows
                           that hold a code block, and one file of codeio rows
Each table is scored with the drafter as it is (most confident proposal) and with the reranker of
rerank-more/600.pt as it is. Run from the repo root with python3;
argument: output prefix. PARTS=fractions or PARTS=horizon runs a part.
PARTS=pod asks whether answers written on a rented GPU (llama.cpp with CUDA; POD_ROWS, from
collect_pod_rows.py) are worth what answers written on this machine's CPU are: the table without the
new answers, and the table with as many tokens of the pod's answers in their place. The test answers
were written on the CPU, and the two builds part ways at near-ties (8 of 12 answers, 2026-11-11).
PARTS=curve adds the first POD_LIMIT of the pod's answers to the largest table (run_curve.sh does
this each time their number doubles).
Give the job a memory limit (systemd-run --user --scope -p MemoryMax=30G): the machine is shared."""
import gc
import json
import os
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
import rerank2 as r2

K = rr.K
BEST = 'embeddings, learned from 512'
REF = 'shipped + own + gpt-3.5 code (the largest table so far)'


def horizon(name):
    z = np.load(f'data/cache/horizon-{name}.npz')
    ids, lens = z['ids'], z['lens']                     # once: every z[...] reads the whole array again
    cut = np.r_[0, np.cumsum(lens)]
    return [{'id': f'horizon-{name}-{i}', 'source': 'horizon', 'thinking': False, 'prompt_ids': [],
             'answer_ids': ids[cut[i]:cut[i + 1]], 'amax': ids[cut[i]:cut[i + 1]]} for i in range(len(lens))]


def main():
    prefix = sys.argv[1]
    pool = gen_code.pick(rr.dataset, 2000, gen_code.SEED, gen_code.MAX_CHARS)
    held_ids = {p['id'] for p in pool if cs.part(p['id']) != 'table'}
    own = cs.own_rows('data/code/rows-9b.jsonl')
    more = cs.own_rows('data/code/rows-9b-more.jsonl')[:int(os.environ['MORE_LIMIT'])]
    test = ns.Data([r for r in own if cs.part(r['id']) == 'test' and r['finished']])
    own_table = [r for r in own if cs.part(r['id']) == 'table']
    cal_rows = [r for r in own if cs.part(r['id']) == 'cal']
    other = cs.other_rows(rr.dataset, rr.tokenizer, own[0], held_ids)
    chat = [r for s in cs.SHARDS for r in ns.load_shard(s, None, 'data/cache') if r['finished']]
    cls = rr.token_classes()
    emb = {'pca': None, 'full': None, 'p512': tuple(torch.from_numpy(e) for e in rr.embeddings_pca(512))}
    cal = ns.Data([r for r in cal_rows if r['finished']])
    ci = ns.labelled(cal)
    inreq_cal = ns.InRequest(cal, K)
    ti = ns.labelled(test)
    inreq_test = ns.InRequest(test, K)
    draws = np.random.default_rng(1).integers(0, len(test.start), (4000, len(test.start)))
    tokens = lambda rows: int(sum(len(r['answer_ids']) for r in rows))
    result, runs, model = {'new_answers': len(more)}, {}, None

    def score(name, static, pick, ref):
        r = rr.chain(test, ti, inreq_test, static, pick)
        per = r.pop('per answer, pad 3')
        tok, spent = runs[name] = (np.array(per['tokens']), np.array(per['spent']))
        line = ''
        if ref in runs and ref != name:
            b = runs[ref]
            diff = np.sort(tok[draws].sum(1) / spent[draws].sum(1) - b[0][draws].sum(1) / b[1][draws].sum(1))
            gain = float(tok.sum() / spent.sum() - b[0].sum() / b[1].sum())
            r['gain over the largest table so far'] = {'pad 3': gain, '95%': [float(diff[100]), float(diff[3899])]}
            line = f'  {gain:+.3f} ({diff[100]:+.3f} to {diff[3899]:+.3f})'
        best = max((f'pad {k}' for k in range(1, rr.DEPTH + 1)), key=lambda p: r[p]['speedup'])
        result[name] = r
        ns.log(f"{name:74s} first guess {r['first_guess_right']:.3f}  pad 3 {r['pad 3']['speedup']:.3f}x  best {best} {r[best]['speedup']:.3f}x{line}")
        json.dump(result, open(f'{prefix}.json', 'w'), indent=1)

    def reranked(calib, logfreq):
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

    def table(name, rows):
        nonlocal model
        static, counts, n = r2.table(rows)
        calib = ns.Calibration(K)
        calib.add(r2.proposals(cal, ci, static, inreq_cal)[1], cal.amax[ci])
        logfreq = (np.log1p(counts) / 16).astype(np.float32)
        if model is None:
            n_in = rr.build(r2.proposals(cal, ci[:1000], static, inreq_cal)[1], calib, cal.context(ci[:1000], K)[0], logfreq, cls)[2].shape[1]
            model = r2.Net(n_in, 'p512', emb, (512, 256, 128))
            model.load_state_dict(torch.load('data/code/rerank-more/600.pt')[BEST])
            model.eval()
        entries = int(sum(len(k) for k in static.keys))
        ns.log(f'{name}: {n:,} tokens, {entries:,} entries')
        result[f'{name} | size'] = {'tokens': n, 'entries': entries}
        score(f'{name} | drafter', static, lambda cands, C, d: ns.choose(cands, calib), f'{REF} | drafter')
        score(f'{name} | reranker', static, reranked(calib, logfreq), f'{REF} | reranker')
        del static
        gc.collect()

    base = chat + own_table
    parts = os.environ.get('PARTS', 'fractions,horizon').split(',')
    ns.log(f'own code in the table: {tokens(own_table) + tokens(more):,} tokens ({tokens(more):,} new); gpt-3.5 code {tokens(other):,}; shipped text {tokens(chat):,}')
    table(REF, base + more + other)
    if 'fractions' in parts:
        table('gpt-3.5 code: half of it', base + more + other[::2])
        table('gpt-3.5 code: a quarter of it', base + more + other[::4])
        table('own code: half of the new answers', base + more[::2] + other)
        table('own code: none of the new answers', base + other)
    if 'pod' in parts:
        pod, need = [], tokens(more)
        for r in cs.own_rows(os.environ['POD_ROWS']):
            if tokens(pod) >= need:
                break
            pod.append(r)
        assert tokens(pod) >= need and not {r['id'] for r in pod} & ({r['id'] for r in more} | {p['id'] for p in pool})
        ns.log(f"the pod's answers: {len(pod)}, {tokens(pod):,} tokens, in place of {len(more)} from this machine, {need:,} tokens")
        table('own code: none of the new answers', base + other)
        table("own code: the pod's answers in place of this machine's new ones", base + pod + other)
    if 'curve' in parts:
        pod = cs.own_rows(os.environ['POD_ROWS'])[:int(os.environ['POD_LIMIT'])]
        assert not {r['id'] for r in pod} & ({r['id'] for r in more} | {p['id'] for p in pool})
        result['pod_answers'], result['pod_tokens'] = len(pod), tokens(pod)
        table(f"+ {len(pod)} of the pod's answers ({tokens(pod) / 1e6:.2f}M tokens)", base + more + other + pod)
    for names in (['instr-code'], ['codeio'], ['instr-code', 'codeio']) if 'horizon' in parts else ():
        extra = [r for n in names for r in horizon(n)]
        table(f'+ K2-Horizon {" + ".join(names)} ({tokens(extra) / 1e6:.1f}M tokens)', base + more + other + extra)
        del extra
    ns.log('done')


if __name__ == '__main__':
    main()
