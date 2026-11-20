"""More of the target's own code: what does it buy the table, and what does it buy the reranker?

rows-9b-more.jsonl holds the 9B's answers to prompts past the 2000-prompt pool (run_gen_more.sh),
so none of them is a calibration or test prompt. They go into the table, and the reranker learns
on them. The rerankers start from rerank5.py's three after their pass over the stand-in text
(rerank-big.standins.pt).

An own answer that is in the table gets its proposals from a table built without it: the answers
in the table are cut in FOLDS parts, and each part is looked up in a table without that part's
answers and without the gpt-3.5 solutions to the same prompts. (At test time neither is in the
table. rerank5.py left the gpt-3.5 solutions in.)

Scored on the same test answers as before, pad 3:
  yesterday's table     the drafter, and yesterday's best reranker (repeats rerank5.py's numbers)
  the larger table      the drafter; yesterday's reranker as it is; then rerankers fine-tuned on
                        yesterday's own answers only, on those plus a part of the new ones, and
                        on all of them (the table is the same for all, so only the reranker differs)
Run from the repo root with python3; argument: output prefix.
MORE_LIMIT uses only the first N new answers."""
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
FOLDS = int(os.environ.get('FOLDS', 8))
HIDDEN = (512, 256, 128)
MODES = {'no embeddings': None, 'embeddings, 64 fixed': 'pca', 'embeddings, learned from 512': 'p512'}
BEST = 'embeddings, learned from 512'
OLD, NEW = "yesterday's table, the drafter", 'larger table, the drafter'
SAME = f"larger table, {BEST}: yesterday's own answers only"      # the reranker without the new answers, same table
SHORT = {OLD: "yesterday's drafter", NEW: 'the drafter', SAME: 'the reranker without the new answers'}


def main():
    prefix = sys.argv[1]
    pool = gen_code.pick(rr.dataset, 2000, gen_code.SEED, gen_code.MAX_CHARS)
    pool_ids = {p['id'] for p in pool}
    held_ids = {p['id'] for p in pool if cs.part(p['id']) != 'table'}
    own = cs.own_rows('data/code/rows-9b.jsonl')
    more = cs.own_rows('data/code/rows-9b-more.jsonl')[:int(os.environ.get('MORE_LIMIT', 10**9))]
    assert not any(r['id'] in pool_ids for r in more), 'a new answer is to a prompt inside the pool'
    test = ns.Data([r for r in own if cs.part(r['id']) == 'test' and r['finished']])
    own_table = [r for r in own if cs.part(r['id']) == 'table']
    cal_rows = [r for r in own if cs.part(r['id']) == 'cal']
    other = cs.other_rows(rr.dataset, rr.tokenizer, own[0], held_ids)
    chat = [r for s in cs.SHARDS for r in ns.load_shard(s, None, 'data/cache') if r['finished']]
    cls = rr.token_classes()
    emb = {'pca': tuple(torch.from_numpy(e) for e in rr.embeddings()), 'full': None,
           'p512': tuple(torch.from_numpy(e) for e in rr.embeddings_pca(512))}
    tokens = lambda rows: int(sum(len(r['answer_ids']) for r in rows))
    ns.log(f'new answers: {len(more)}, {tokens(more):,} tokens ({sum(not r["finished"] for r in more)} hit the length limit). '
           f"yesterday's own answers outside the test part: {len(own_table) + len(cal_rows)}, {tokens(own_table) + tokens(cal_rows):,} tokens")

    cal = ns.Data([r for r in cal_rows if r['finished']])
    ci = ns.labelled(cal)
    inreq_cal = ns.InRequest(cal, K)
    ti = ns.labelled(test)
    inreq_test = ns.InRequest(test, K)
    draws = np.random.default_rng(1).integers(0, len(test.start), (4000, len(test.start)))
    result, runs = {'new_answers': len(more), 'new_tokens': tokens(more)}, {}

    def setting(rows):
        """A table and what goes with it: confidences from the calibration answers, token frequencies."""
        static, counts, n = r2.table(rows)
        calib = ns.Calibration(K)
        calib.add(r2.proposals(cal, ci, static, inreq_cal)[1], cal.amax[ci])
        return static, calib, (np.log1p(counts) / 16).astype(np.float32), n

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
        ns.log(f"{name:62s} first guess {r['first_guess_right']:.3f}  pad 3 {r['pad 3']['speedup']:.3f}x "
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

    def loaded(path, name, n_in):
        model = r2.Net(n_in, MODES[name], emb, HIDDEN)
        model.load_state_dict(torch.load(path)[name])
        model.eval()
        return model

    # Yesterday's table: must repeat rerank5.py.
    static, calib, logfreq, n = setting(chat + own_table + other)
    n_in = rr.build(r2.proposals(cal, ci[:1000], static, inreq_cal)[1], calib, cal.context(ci[:1000], K)[0], logfreq, cls)[2].shape[1]
    score(OLD, static, lambda cands, C, d: ns.choose(cands, calib), [], {'table_tokens': n})
    yesterday = loaded('data/code/rerank-big.final.pt', BEST, n_in)
    score("yesterday's table, yesterday's reranker", static, reranked(yesterday, calib, logfreq), [OLD])
    del static

    # The larger table.
    in_table = own_table + more
    static, calib, logfreq, n = setting(chat + in_table + other)
    ns.log(f'larger table: {n:,} tokens')
    score(NEW, static, lambda cands, C, d: ns.choose(cands, calib), [OLD], {'table_tokens': n})
    score("larger table, yesterday's reranker as it is", static, reranked(yesterday, calib, logfreq), [OLD, NEW])

    # What the reranker learns on: every own answer outside the test part, each with a number that
    # says how early it came (yesterday's answers 0, the new ones 1 and up in the order they were written).
    when = {r['id']: 0 for r in cal_rows + own_table}
    when.update({r['id']: 1 + i for i, r in enumerate(more)})
    parts = []

    def add(rows, table):
        p = r2.features(rows, table, calib, logfreq, cls)
        p['when'] = np.array([when[r['id']] for r in rows], np.int32)[p['seq']]
        p['answer'] = p['seq'] + sum(len(q) for q in sizes)
        sizes.append(rows)
        parts.append(p)

    sizes = []
    add(cal_rows, static)
    for f in range(FOLDS):
        rows = in_table[f::FOLDS]
        gone = {r['id'] for r in rows}
        table, _, _ = r2.table(chat + [r for r in other if r['id'] not in gone] + [r for i, r in enumerate(in_table) if i % FOLDS != f])
        add(rows, table)
        del table
    everything = {k: np.concatenate([p[k] for p in parts]) for k in ('X', 'ids', 'y', 'when', 'answer')}
    del parts
    is_val = everything['answer'] % 10 == 0
    val = {k: everything[k][is_val] for k in ('X', 'ids', 'y')}
    ns.log(f"own rows to learn from: {int((~is_val).sum()):,} ({int((~is_val & (everything['when'] == 0)).sum()):,} from yesterday's answers); "
           f"validation {int(is_val.sum()):,}")

    def tuned(name, upto):
        """The stand-in reranker `name`, fine-tuned on the own answers that came no later than `upto`."""
        use = ~is_val & (everything['when'] <= upto)
        some = {k: everything[k][use] for k in ('X', 'ids', 'y')}
        model = loaded('data/code/rerank-big.standins.pt', name, n_in)
        return r2.fit(f'{name}, own answers up to {upto}', [some], val, n_in, model=model, lr=5e-4, epochs=20)

    shares = [s for s in (0, 0.25, 0.5) if s == 0 or len(more) * s >= 100]
    saved = {}
    for share in shares + [1]:
        upto = int(len(more) * share)
        what = "yesterday's own answers only" if share == 0 else f'and the first {upto} new answers' if share < 1 else f'and all {upto} new answers'
        for name in (MODES if share == 1 else [BEST]):
            if share == 1 and not more:
                continue
            model, info = tuned(name, upto)
            score(f'larger table, {name}: {what}', static, reranked(model, calib, logfreq), [OLD, NEW] + [SAME] * (share > 0),
                  dict(info, new_answers_used=upto))
            if share == 1:
                saved[name] = model.state_dict()
    torch.save(saved, f'{prefix}.pt')
    ns.log('done')


if __name__ == '__main__':
    main()
