#!/usr/bin/env python3
"""Collect greedy output for ple_experiment.py from a running llama-server.

Start the exact pruned target with --alias TARGET_ID and no speculative decoder.
Input prompts are JSONL objects with id and prompt. Output is resumable JSONL.
"""
import argparse
import json
from pathlib import Path

from gen_code import answer, call


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', default='http://127.0.0.1:8455')
    p.add_argument('--target-id', required=True)
    p.add_argument('--prompts', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--limit', type=int, default=2000)
    args = p.parse_args()
    if args.max_tokens < 1 or args.limit < 1:
        p.error('max-tokens and limit must be positive')
    models = call(args.server, '/v1/models')
    if args.target_id not in [m['id'] for m in models['data']]:
        raise ValueError('server alias must match --target-id; verify exact GGUF before collecting')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as f:
            for line in f:
                row = json.loads(line)
                if row.get('target_id') != args.target_id or row.get('temperature') != 0:
                    raise ValueError('output already contains another target or sampling configuration')
                done.add(row['id'])
    with open(args.prompts) as f, out.open('a') as sink:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            prompt = json.loads(line)
            if prompt['id'] in done:
                continue
            row = answer(args.server, dict(prompt, lang=prompt.get('lang', '')), args.max_tokens)
            if len(row['prompt_ids']) + len(row['answer_ids']) >= 8192:
                raise ValueError('request exceeds the offline lookup index limit; shorten prompts or answers')
            row.update(target_id=args.target_id, temperature=0)
            sink.write(json.dumps(row) + '\n')
            sink.flush()
            done.add(row['id'])
            print(f"{i + 1}: {row['id']}: {len(row['answer_ids'])} tokens", flush=True)


if __name__ == '__main__':
    main()
