#!/bin/bash
# Relaxed acceptance (--spec-accept-ratio, patch_llama.py) on the GPU: Qwen3.5-9B through Vulkan, the
# table + reranker drafting 3 tokens (pad), greedy decoding. A drafted token is also kept when the model
# gives it at least R times the probability of its own top token; R = 1.0 is the lossless rule.
#   arms      plain decoding, then R = 1.0, 0.9, 0.8, 0.5, 0.3, 0.1
#   prompts   the 164 HumanEval problems (prompts/humaneval-v1.jsonl, 512 tokens; humaneval.py scores
#             pass@1 from the answers) and the 12 coding prompts of the 1.93x pilot (code-v1, 192 tokens)
# Turns of at most 9 minutes (12 prompts, every arm, plain first) as in run_wide_live.sh: each when the
# GPU is ours alone and the load average is below LOAD (or after LOAD_WAIT seconds), GAP seconds apart.
# Afterwards show_wide.py (speed) and humaneval.py (pass@1) write summary.txt.
#
#   setsid nohup ./run_accept_live.sh > data/code/accept.log 2>&1 < /dev/null &
#   cancel: pkill -f '[r]un_accept_live.sh'; pkill -f '[l]ive_bench.py.*accept-'
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/mnt/fedora/home/cyrus/code/model-experiments/target/manifold-audit/env/bin/python}
S=$HOME/.cache/model-experiments/memdraft
NEW=$PWD/data/code/9b-code-pod-c3
STAMP=$(date +%Y%m%d-%H%M)
OUT=results/accept-9b-vk-$STAMP
WORK=data/code/accept-$STAMP
mkdir -p "$OUT" "$WORK"
D="--spec-type ngram-table --spec-table-pad --spec-table-n-max 3 --spec-table-file $NEW.mdt --spec-table-reranker $NEW.mdr"
declare -A ARM=(
  [r1.0]="$D"
  [r0.9]="$D --spec-accept-ratio 0.9"
  [r0.8]="$D --spec-accept-ratio 0.8"
  [r0.5]="$D --spec-accept-ratio 0.5"
  [r0.3]="$D --spec-accept-ratio 0.3"
  [r0.1]="$D --spec-accept-ratio 0.1"
)
ALL=(r1.0 r0.9 r0.8 r0.5 r0.3 r0.1)

$PY - "$WORK" <<'EOF'
import json, sys
work = sys.argv[1]
he = [json.loads(l) for l in open('prompts/humaneval-v1.jsonl')]
for n in range(0, len(he), 12):
    open(f'{work}/he-{n // 12:02d}.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in he[n:n + 12]))
rows = [json.loads(l) for l in open('prompts/code-v1.jsonl')]
seen = {}
pick = [r for r in rows if seen.setdefault(r['domain'], []).append(r) or len(seen[r['domain']]) <= 6]
open(f'{work}/oss.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in pick))
EOF

JOBS=(oss $(cd "$WORK" && ls he-*.jsonl | sed 's/\.jsonl$//'))
turn=0
for part in "${JOBS[@]}"; do
  case $part in oss) TOKENS=192;; *) TOKENS=512;; esac
  LEFT=("${ALL[@]}")
  for attempt in 1 2 3; do
    turn=$((turn + 1))
    [ $turn -gt 1 ] && sleep "${GAP:-300}"
    waited=0
    while awk -v m="${LOAD:-8}" '{exit !($1 >= m)}' /proc/loadavg && [ $waited -lt "${LOAD_WAIT:-2700}" ]; do
      sleep 60; waited=$((waited + 60))
    done
    # A preemptible job counts as leaving only while arbiter holds the GPU for it (then our `arbiter run gpu` makes it yield).
    until $PY gpu_alone.py > /dev/null || $PY gpu_alone.py --preemptible-ok > /dev/null; do
      sleep 60
    done
    name="$part-r1-a$attempt"
    echo "$(date +%T) turn $turn: $name, load $(cut -d' ' -f1 /proc/loadavg) after waiting ${waited}s, arms ${LEFT[*]}"
    ARMS=(--arm none=)
    for a in "${LEFT[@]}"; do ARMS+=(--arm "$a=${ARM[$a]}"); done
    arbiter run gpu port-8455 --label "MemDraft: relaxed acceptance, turn $turn of about 15 (9B, at most 9 min, stops if the GPU gets company)" --eta 10m -- \
      $PY live_bench.py --gpu-guard --guard-checks 2 --guard-gap 5 --retries 1 --time-limit "${LIMIT:-540}" --arm-seconds 75 \
      --server "$S/llama.cpp/build-vk/bin/llama-server" --model "$S/models/Qwen3.5-9B-Q4_K_M.gguf" \
      --prompts "$WORK/$part.jsonl" --max-tokens $TOKENS --out "$OUT/$name.json" -- -ngl 99 -c 8192 "${ARMS[@]}"
    echo "$(date +%T) turn $turn: exit $?"
    LEFT=($($PY -c 'import json, os, sys
done = json.load(open(sys.argv[1])).get("complete_arms", []) if os.path.exists(sys.argv[1]) else []
print(" ".join(a for a in sys.argv[2:] if a not in done))' "$OUT/$name.json" "${LEFT[@]}"))
    [ ${#LEFT[@]} -eq 0 ] && break
    echo "$(date +%T) still to run on $part: ${LEFT[*]}"
  done
done
{ $PY show_wide.py "$OUT"; echo; $PY humaneval.py score "$S/humaneval/test.parquet" "$OUT"; } > "$OUT/summary.txt" 2>&1
touch "$OUT.done"
echo "$(date +%T) all turns done"
