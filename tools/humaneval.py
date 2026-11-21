#!/usr/bin/env python3
"""HumanEval for the relaxed-acceptance runs (run_accept_live.sh): prompts in, pass@1 out.

    humaneval.py prompts PARQUET OUT.jsonl     the 164 problems as live_bench prompts (domain humaneval)
    humaneval.py score PARQUET RESULTS_DIR     pass@1 per arm over live_bench's answers

The model is asked for the whole function in one ```python block; the program run is the problem's
prompt (its imports and a docstring-only definition), then the answer's code (which redefines the
function), then the problem's tests. When the answer has no definition of the function, its code is
taken as the body. Each program runs in bubblewrap with no network, a read-only system, a private
/tmp and a 10-second limit, at low priority. The dataset file is data: read with pyarrow, never run.
"""
import collections
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ASK = ('Complete the following Python function. Reply with the complete function, including any imports it '
       'needs, in one ```python code block.\n\n```python\n{prompt}```')


def problems(parquet):
    import pyarrow.parquet as pq
    return {p['task_id']: p for p in pq.read_table(parquet).to_pylist()}


def program(problem, text):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
    code = max(blocks, key=len) if blocks else text
    if f"def {problem['entry_point']}" not in code:
        code = problem['prompt'] + code
    return f"{problem['prompt']}\n\n{code}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"


def run(source):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'prog.py'
        path.write_text(source)
        cmd = ['nice', '-n', '19', 'bwrap', '--unshare-all', '--die-with-parent', '--ro-bind', '/usr', '/usr',
               '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib', '/lib64', '--symlink', 'usr/bin', '/bin',
               '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--ro-bind', str(path), '/tmp/prog.py',
               '--chdir', '/tmp', 'timeout', '10', '/usr/bin/python3', '-I', '/tmp/prog.py']
        try:
            return subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def main():
    mode, parquet = sys.argv[1], sys.argv[2]
    probs = problems(parquet)
    if mode == 'prompts':
        Path(sys.argv[3]).write_text(''.join(json.dumps({'id': t, 'domain': 'humaneval', 'lang': 'python',
                                                         'prompt': ASK.format(prompt=p['prompt'])}) + '\n' for t, p in probs.items()))
        return
    rows = []
    for path in sorted(Path(sys.argv[3]).glob('*.json')):
        rows += [r for r in json.load(open(path)).get('rows', []) if r.get('domain') == 'humaneval' and r.get('prompt') and not r.get('shared_gpu')]
    latest = {(r['arm'], r['prompt']): r for r in rows}           # a rerun after an interrupted turn replaces it
    with ThreadPoolExecutor(4) as pool:
        passed = dict(zip(latest, pool.map(lambda key: run(program(probs[key[1]], latest[key]['text'])), latest)))
    arms = list(dict.fromkeys(a for a, _ in latest))
    base = arms[0]
    out = {}
    print(f'{"arm":28s} {"problems":>8s} {"pass@1":>7s} {"same text as plain":>19s} {"pass changed vs plain":>22s}')
    for arm in arms:
        mine = [p for a, p in latest if a == arm]
        ok = sum(passed[arm, p] for p in mine)
        same = sum(latest[arm, p]['text'] == latest[base, p]['text'] for p in mine if (base, p) in latest)
        flips = collections.Counter((passed[base, p], passed[arm, p]) for p in mine if (base, p) in latest)
        out[arm] = {'problems': len(mine), 'passed': ok, 'pass@1': ok / len(mine), 'same_text_as_plain': same,
                    'gained': flips[False, True], 'lost': flips[True, False]}
        print(f'{arm:28s} {len(mine):8d} {ok / len(mine):7.3f} {same:13d}/{len(mine):<5d} '
              f'{"+" + str(flips[False, True]):>10s} {"-" + str(flips[True, False]):>6s}')
    Path(sys.argv[3], 'humaneval.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
