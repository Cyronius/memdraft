#!/usr/bin/env python3
"""Rank head pairs on calibration perplexity without rewriting the source GGUF.

leave-one-head-out losses, then pairs among the top heads per order (or all
64 pairs with --all-pairs). Optionally evaluate the winner and both controls
on held-out text. Interrupted runs can resume in the same output directory.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--perplexity', required=True)
    p.add_argument('--model', required=True, help='full source Q4_K_XL GGUF, first shard')
    p.add_argument('--corpus', required=True, help='calibration text, never the final evaluation corpus')
    p.add_argument('--held-out', help='separate text for evaluating the selected pair and controls')
    p.add_argument('--out', required=True)
    p.add_argument('--top-per-order', type=int, default=2, choices=range(1, 9))
    p.add_argument('--all-pairs', action='store_true', help='test all 64 bigram/trigram head pairs')
    p.add_argument('--batch-runtime', action='store_true', help='reuse one model load for multiple masks')
    p.add_argument('runtime_args', nargs=argparse.REMAINDER, help='after --: GPU/context/chunk options')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extra = args.runtime_args[1:] if args.runtime_args[:1] == ['--'] else args.runtime_args
    if args.held_out and Path(args.corpus).samefile(args.held_out):
        p.error('--held-out must be a different file from --corpus')
    binary = shutil.which(args.perplexity)
    if binary is None:
        p.error('--perplexity must point to an executable')

    def fingerprint(filename):
        path = Path(filename).resolve()
        stat = path.stat()
        return [str(path), stat.st_size, stat.st_mtime_ns]

    model = Path(args.model)
    shard = re.fullmatch(r'(.*)-00001-of-(\d{5})\.gguf', model.name)
    shards = [model]
    if shard:
        stem, count = shard.groups()
        shards = [model.with_name(f'{stem}-{i:05d}-of-{count}.gguf') for i in range(1, int(count) + 1)]
    inputs = {
        'model': [fingerprint(path) for path in shards],
        'binary': fingerprint(binary),
        'corpus': fingerprint(args.corpus),
        'held_out': fingerprint(args.held_out) if args.held_out else None,
        'runtime_args': extra,
        'batch_runtime': args.batch_runtime,
    }
    results_path = out / 'results.json'
    results = {}
    if results_path.exists():
        previous = json.loads(results_path.read_text())
        if previous.get('inputs') != inputs:
            p.error('existing results came from different inputs; use a new --out directory')
        results = previous['runs']

    yield_file = os.environ.get('ARBITER_YIELD_FILE')

    def record(name, heads, ppl, command):
        if not math.isfinite(ppl) or ppl <= 0:
            raise ValueError(f'{name}: invalid perplexity {ppl}')
        results[name] = dict(heads=list(heads), ppl=ppl, command=command)
        write_json(results_path, dict(inputs=inputs, args=vars(args), runs=results))
        print(name, ppl, flush=True)

    def check_yield():
        if yield_file and Path(yield_file).exists():
            raise SystemExit(75)

    def measure(name, heads, corpus=args.corpus):
        if name in results:
            if results[name]['heads'] != list(heads):
                raise ValueError(f'{name}: cached head mask differs from requested mask')
            return results[name]['ppl']
        check_yield()
        env = dict(os.environ, QWEN4EXP_PLE_KEEP_HEADS=','.join(map(str, heads)))
        command = [binary, '-m', args.model, '-f', corpus, *extra]
        with (out / f'{name}.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        text = (out / f'{name}.log').read_text()
        matches = re.findall(r'Final estimate: PPL = ([0-9.eE+-]+)', text)
        if not matches:
            raise ValueError(f'{name}: missing final perplexity')
        record(name, heads, float(matches[-1]), command)
        return results[name]['ppl']

    def batch_measure(cases, corpus, log_name):
        pending = {','.join(map(str, heads)): (name, heads)
                   for name, heads in cases if name not in results}
        if not pending:
            return
        check_yield()
        env = dict(os.environ, QWEN4EXP_PLE_SWEEP=';'.join(pending))
        command = [binary, '-m', args.model, '-f', corpus, *extra]
        log_path = out / log_name
        with log_path.open('a') as log:
            with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as process:
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if line.startswith('PLE_SWEEP_RESULT '):
                            _, mask, ppl = line.split()
                            name, heads = pending.pop(mask)
                            record(name, heads, float(ppl), command)
                            if name == 'original_0_8' and abs(results['full']['ppl'] - float(ppl)) < 1e-4:
                                raise RuntimeError('full and masked controls are identical; verify the ablation patch and corpus')
                except BaseException:
                    process.terminate()
                    raise
                code = process.wait()
        if code == 75:
            raise SystemExit(75)
        if code or pending:
            raise RuntimeError(f'batched perplexity exited with status {code}, {len(pending)} masks pending, see {log_path}')

    all_heads = list(range(16))
    ablations = [(f'without_{h}', [i for i in all_heads if i != h]) for h in all_heads]
    if args.batch_runtime:
        cases = [('full', all_heads), ('original_0_8', [0, 8]), *ablations]
        if args.all_pairs:
            cases.extend((f'pair_{a}_{b}', [a, b]) for a in range(8) for b in range(8, 16)
                         if (a, b) != (0, 8))
        batch_measure(cases, args.corpus, 'calibration-batch.log')
    full = measure('full', all_heads)
    original = measure('original_0_8', [0, 8])
    # Catch a missing runtime patch instead of selecting heads from identical runs.
    if abs(full - original) < 1e-4:
        raise RuntimeError('full and masked controls are identical; verify the ablation patch and corpus')
    damage = {h: math.log(measure(name, heads) / full) for h, (name, heads) in enumerate(ablations)}
    limit = 8 if args.all_pairs else args.top_per_order
    best = [sorted(range(g * 8, (g + 1) * 8), key=lambda h: damage[h], reverse=True)[:limit] for g in range(2)]
    candidates = [(f'pair_{a}_{b}', [a, b]) for a in best[0] for b in best[1] if (a, b) != (0, 8)]
    if args.batch_runtime:
        batch_measure(candidates, args.corpus, 'calibration-batch.log')
    pairs = [dict(heads=[0, 8], ppl=original)]
    pairs.extend(dict(heads=heads, ppl=measure(name, heads)) for name, heads in candidates)
    pairs.sort(key=lambda result: result['ppl'])
    selected = pairs[0]['heads']
    summary = dict(leave_one_out_log_loss_damage=damage, candidate_pairs=pairs,
                   selected_heads=selected, selection_metric='calibration perplexity')
    if args.held_out:
        selected_name = 'held_out_pair_' + '_'.join(map(str, selected))
        if args.batch_runtime:
            controls = [('held_out_full', all_heads), ('held_out_original_0_8', [0, 8])]
            if selected != [0, 8]:
                controls.append((selected_name, selected))
            batch_measure(controls, args.held_out, 'held-out-batch.log')
        held_out_full = measure('held_out_full', all_heads, args.held_out)
        held_out_original = measure('held_out_original_0_8', [0, 8], args.held_out)
        held_out_selected = held_out_original
        if selected != [0, 8]:
            held_out_selected = measure(selected_name, selected, args.held_out)
        summary['held_out_perplexity'] = dict(full=held_out_full, original=held_out_original,
                                             selected=held_out_selected)
    write_json(out / 'selection.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
