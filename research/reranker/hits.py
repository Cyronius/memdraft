"""On the code test answers: is the drafter short of hits, or are its hits wrong?
For every test position: the longest context any lookup matched, the proposal chosen, its
confidence, whether it was right, and whether the right token was among the proposals at all.
Then draft chains: how far a chain runs once its first token is right, and what the confidence
rule buys against always drafting. Run from the repo root with python3."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools'))
import code_study as cs
import gen_code
import ngram_study as ns

K, N = 6, 8
COST = cs.COSTS[0]
dataset = os.path.join(gen_code.CACHE, 'code-prompts-src/data-oss_instruct-decontaminated.jsonl')
tokenizer = os.path.join(gen_code.CACHE, 'models/Qwen3.5-9B/tokenizer.json')

pool = gen_code.pick(dataset, 2000, gen_code.SEED, gen_code.MAX_CHARS)
held_ids = {p['id'] for p in pool if cs.part(p['id']) != 'table'}
own = cs.own_rows('data/code/rows-9b.jsonl')
held = [r for r in own if cs.part(r['id']) != 'table']
cal = ns.Data([r for r in held if cs.part(r['id']) == 'cal' and r['finished']])
test_rows = [r for r in held if cs.part(r['id']) == 'test' and r['finished']]
test = ns.Data(test_rows)
own_table = [r for r in own if cs.part(r['id']) == 'table']
other = cs.other_rows(dataset, tokenizer, own[0], held_ids)
shards = [r for s in cs.SHARDS for r in ns.load_shard(s, None, 'data/cache') if r['finished']]

ti = ns.labelled(test)
seq = test.seq[ti]
first = np.r_[0, np.cumsum([len(r['answer_ids']) for r in test_rows])]
in_code = np.concatenate(cs.fenced(test_rows, tokenizer))[first[seq] + ti - test.ans[seq]]
truth = test.amax[ti]
ci = ns.labelled(cal)
inreq_cal, inreq_test = ns.InRequest(cal, K), ns.InRequest(test, K)


def chosen(cands, calib):
    """ns.choose, also returning the context length and the source of the proposal it took."""
    n = len(cands[0][2])
    token, conf = np.full(n, -1, np.int32), np.zeros(n, np.float32)
    length, source = np.zeros(n, np.int8), np.zeros(n, np.int8)
    for kind, k, hit, tok, cell in cands:
        c = np.where(hit, calib.confidence(kind, k, cell), 0)
        better = c >= np.maximum(conf, 1e-9)
        token, conf = np.where(better, tok, token), np.where(better, c, conf)
        length, source = np.where(better, k, length), np.where(better, 1 if kind == 'static' else 2, source)
    return token, conf, length, source


def steps(accepted, drafted):
    """Decode each test answer step by step; per step: tokens drafted and accepted."""
    where = np.full(len(test.tok), -1, np.int64)
    where[ti] = np.arange(len(ti))
    d_all, a_all = [], []
    for s in range(len(test.start)):
        j, last = test.ans[s], test.end[s] - 1
        while j <= last:
            i = where[j]
            d_all.append(drafted[i])
            a_all.append(accepted[i])
            j += accepted[i] + 1
    return np.array(d_all), np.array(a_all)


def speed(d, a):
    shape = np.r_[True, d[1:] != d[:-1]]              # close enough: ignores answer boundaries
    spent = len(d) + COST['beta'] * d.sum() + COST['accept'] * a.sum() + COST['reshape'] * shape.sum()
    return float((a.sum() + len(d)) / spent)


def study(name, static, use_request):
    calib = ns.Calibration(K)
    C, avail = cal.context(ci, K)
    calib.add(ns.candidates(C, avail, cal.seq[ci], cal.pos[ci], static, inreq_cal if use_request else None, K), cal.amax[ci])
    inreq = inreq_test if use_request else None
    C, avail = test.context(ti, K)
    cands = ns.candidates(C, avail, seq, test.pos[ti], static, inreq, K)
    token, conf, length, source = chosen(cands, calib)
    right = (conf > 0) & (token == truth)
    longest = {'table': np.zeros(len(ti), np.int8), 'request': np.zeros(len(ti), np.int8)}
    offered = np.zeros(len(ti), bool)
    for kind, k, hit, tok, cell in cands:
        which = 'table' if kind == 'static' else 'request'
        longest[which] = np.where(hit, k, longest[which])
        offered |= hit & (tok == truth)
    any_longest = np.maximum(longest['table'], longest['request'])

    def rows(key, groups):
        out = {}
        for label, mask in groups:
            if mask.any():
                out[label] = {'share': float(mask.mean()), 'right': float(right[mask].mean()),
                              'right_token_offered': float(offered[mask].mean()),
                              'share_of_right': float(right[mask].sum() / max(1, right.sum())),
                              'share_of_wrong': float((~right)[mask].sum() / max(1, (~right).sum()))}
        return out

    bands = [(0, 1e-9), (1e-9, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    out = {'positions': int(len(ti)), 'right': float(right.mean()), 'no_proposal': float((conf <= 0).mean()),
           'right_token_offered': float(offered.mean()),
           'by_longest_context_matched': rows('k', [(f'{k}', any_longest == k) for k in range(K + 1)]),
           'by_longest_table_context': rows('k', [(f'{k}', longest['table'] == k) for k in range(K + 1)]),
           'by_longest_request_context': rows('k', [(f'{k}', longest['request'] == k) for k in range(K + 1)]),
           'by_confidence': rows('c', [(f'{lo:.1f}-{min(hi, 1):.1f}', (conf >= lo) & (conf < hi)) for lo, hi in bands]),
           'by_chosen_source': rows('s', [('none', source == 0), ('table', source == 1), ('request', source == 2)]),
           'in_code': rows('w', [('in code blocks', in_code), ('outside', ~in_code)])}
    for where, mask in (('in code blocks', in_code), ('outside', ~in_code)):
        out[f'by_longest_context_matched, {where}'] = {
            f'{k}': {'share': float((any_longest[mask] == k).mean()), 'right': float(right[mask & (any_longest == k)].mean())}
            for k in range(K + 1) if (mask & (any_longest == k)).any()}

    tokens, chance = ns.draft(test, ti, static, inreq, calib, K, N, ns.TAUS[0], pad_to=N)     # always N tokens
    accepted, judged = ns.score(test, ti, tokens, chance, 'greedy')
    out['chain'] = {f'first {j} right, of positions with {j} to judge': float((accepted[judged >= j] >= j).mean()) for j in range(1, N + 1)}
    rules = {}
    for n in (1, 2, 3, 4):
        d = np.minimum(np.int32(n), judged)
        sd, sa = steps(np.minimum(accepted, d), d)
        rules[f'pad {n}'] = (sd, sa)
    for tau in (0.1, 0.3, 0.5, 0.7, 0.9):
        d = np.minimum((chance >= tau).sum(1).astype(np.int32), judged)
        sd, sa = steps(np.minimum(accepted, d), d)
        rules[f'tau {tau}'] = (sd, sa)
    out['rules'] = {r: {'steps_with_a_draft': float((sd > 0).mean()), 'drafted_per_drafting_step': float(sd[sd > 0].mean()) if (sd > 0).any() else 0.0,
                        'acceptance': float(sa.sum() / max(1, sd.sum())), 'steps_all_accepted_of_drafting': float((sa == sd)[sd > 0].mean()) if (sd > 0).any() else 0.0,
                        'steps_none_accepted_of_drafting': float((sa == 0)[sd > 0].mean()) if (sd > 0).any() else 0.0,
                        'tokens_per_step': float((sa.sum() + len(sd)) / len(sd)), 'speedup_gpu9b': speed(sd, sa)}
                    for r, (sd, sa) in rules.items()}
    ns.log(f"{name}: first guess right {out['right']:.3f}, no proposal {out['no_proposal']:.3f}, right token offered {out['right_token_offered']:.3f}")
    return out


result = {'request only': study('request only', None, True)}
for name, rows in (('shipped', shards), ('own code', own_table), ('other code, all', other), ('shipped + own + other all', shards + own_table + other)):
    train = ns.Data(rows)
    idx = ns.labelled(train)
    static = ns.Static.build(train, idx, train.amax[idx], K)
    del train
    result[f'{name} | table only'] = study(f'{name} | table only', static, False)
    result[f'{name} | with request'] = study(f'{name} | with request', static, True)
json.dump(result, open(sys.argv[1], 'w'), indent=1)
