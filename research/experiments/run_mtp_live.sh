#!/bin/bash
# The table drafter against the model's own multi-token-prediction head (llama.cpp's --spec-type draft-mtp)
# on the GPU: Qwen3.5-9B Q4_K_M through Vulkan, greedy decoding, lossless acceptance. The head is the one
# Qwen ships in the 9B's weights, exported alone with `convert_hf_to_gguf.py --mtp --outtype q8_0`.
#   table-3              table + reranker, 3 tokens (pad)
#   mtp-1, mtp-2, mtp-3  the head drafting up to 1, 2, 3 tokens
#   mtp-0.75+table-3     the head drafts while it gives its token at least 0.75; when it drafts nothing, the table does
#   mtp-0.75x3+table-3   the head drafts only when all 3 tokens reach 0.75, else the table: every step has 3 draft tokens
#   prompts   HumanEval problems 0-23 (512 tokens) and the 12 coding prompts of code-v1 (192 tokens)
# Waits for run_prior_live.sh to finish when given its results folder. The first turn is a probe (one prompt):
# if the head does not draft on this build, the run ends there. Turns as in run_accept_live.sh.
#
#   setsid nohup ./run_mtp_live.sh [research/results/prior-9b-vk-STAMP] > data/code/mtp.log 2>&1 < /dev/null &
#   cancel: pkill -f '[r]un_mtp_live.sh'; pkill -f '[l]ive_bench.py.*mtp-9b'
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
S=${MEMDRAFT_CACHE:-$HOME/.cache/model-experiments/memdraft}
NEW=$PWD/data/code/9b-code-pod-c3
HEAD=$S/mtp-head/Qwen3.5-9B-q8_0.gguf
STAMP=$(date +%Y%m%d-%H%M)
OUT=research/results/mtp-9b-vk-$STAMP
WORK=data/code/mtp-$STAMP
mkdir -p "$OUT" "$WORK"
if [ $# -gt 0 ]; then
  until [ -e "$1.done" ]; do sleep 60; done
  echo "$(date +%T) $1 has finished"
fi
T="--spec-table-pad --spec-table-n-max 3 --spec-table-file $NEW.mdt --spec-table-reranker $NEW.mdr"
M="--spec-draft-model $HEAD"
declare -A ARM=(
  [table-3]="--spec-type ngram-table $T"
  [mtp-3]="--spec-type draft-mtp $M --spec-draft-n-max 3"
  [mtp-2]="--spec-type draft-mtp $M --spec-draft-n-max 2"
  [mtp-1]="--spec-type draft-mtp $M --spec-draft-n-max 1"
  [mtp-0.75+table-3]="--spec-type draft-mtp,ngram-table $M --spec-draft-n-max 3 --spec-draft-p-min 0.75 $T"
  [mtp-0.75x3+table-3]="--spec-type draft-mtp,ngram-table $M --spec-draft-n-max 3 --spec-draft-n-min 3 --spec-draft-p-min 0.75 $T"
)
ALL=(table-3 mtp-3 mtp-0.75+table-3 mtp-2 mtp-0.75x3+table-3 mtp-1)   # most wanted first

$PY - "$WORK" <<'EOF'
import json, sys
work = sys.argv[1]
he = [json.loads(l) for l in open('bench/prompts/humaneval-v1.jsonl')][:24]
for n in range(0, len(he), 12):
    open(f'{work}/he-{n // 12:02d}.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in he[n:n + 12]))
open(f'{work}/probe.jsonl', 'w').write(json.dumps(he[0]) + '\n')
rows = [json.loads(l) for l in open('bench/prompts/code-v1.jsonl')]
seen = {}
pick = [r for r in rows if seen.setdefault(r['domain'], []).append(r) or len(seen[r['domain']]) <= 6]
open(f'{work}/oss.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in pick))
EOF

gpu_ours() {
  waited=0
  while awk -v m="${LOAD:-8}" '{exit !($1 >= m)}' /proc/loadavg && [ $waited -lt "${LOAD_WAIT:-2700}" ]; do
    sleep 60; waited=$((waited + 60))
  done
  # A preemptible job counts as leaving only while arbiter holds the GPU for it (then our `arbiter run gpu` makes it yield).
  until $PY tools/gpu_alone.py > /dev/null || $PY tools/gpu_alone.py --preemptible-ok > /dev/null; do
    sleep 60
  done
}
bench() {   # prompts, tokens, result name, arms...
  local prompts=$1 tokens=$2 name=$3; shift 3
  arbiter run gpu port-8455 --label "MemDraft: against the model's MTP head, $name (9B, at most 9 min, stops if the GPU gets company)" --eta 10m -- \
    $PY tools/live_bench.py --gpu-guard --guard-checks 2 --guard-gap 5 --retries 1 --time-limit "${LIMIT:-540}" --arm-seconds 75 \
    --server "$S/llama.cpp/build-vk/bin/llama-server" --model "$S/models/Qwen3.5-9B-Q4_K_M.gguf" \
    --prompts "$prompts" --max-tokens "$tokens" --out "$name" -- -ngl 99 -c 8192 "$@"
}
complete() {   # result file, arms...: the arms not complete in it
  $PY -c 'import json, os, sys
done = json.load(open(sys.argv[1])).get("complete_arms", []) if os.path.exists(sys.argv[1]) else []
print(" ".join(a for a in sys.argv[2:] if a not in done))' "$@"
}

gpu_ours
echo "$(date +%T) probe"
bench "$WORK/probe.jsonl" 48 "$WORK/probe.json" --arm none= --arm "mtp-3=${ARM[mtp-3]}"
if [ -n "$(complete "$WORK/probe.json" mtp-3)" ] || ! grep -q 'draft acceptance' "$WORK/probe.server.log"; then
  echo "$(date +%T) the head did not draft in the probe (see $WORK/probe.server.log); stopping"
  touch "$OUT.failed"
  exit 1
fi

turn=0
for part in he-00 oss he-01; do
  case $part in oss) TOKENS=192;; *) TOKENS=512;; esac
  LEFT=("${ALL[@]}")
  for attempt in 1 2; do
    turn=$((turn + 1))
    sleep "${GAP:-300}"
    gpu_ours
    name="$part-r1-a$attempt"
    echo "$(date +%T) turn $turn: $name, load $(cut -d' ' -f1 /proc/loadavg) after waiting ${waited}s, arms ${LEFT[*]}"
    ARMS=(--arm none=)
    for a in "${LEFT[@]}"; do ARMS+=(--arm "$a=${ARM[$a]}"); done
    bench "$WORK/$part.jsonl" $TOKENS "$OUT/$name.json" "${ARMS[@]}"
    echo "$(date +%T) turn $turn: exit $?"
    LEFT=($(complete "$OUT/$name.json" "${LEFT[@]}"))
    [ ${#LEFT[@]} -eq 0 ] && break
    echo "$(date +%T) still to run on $part: ${LEFT[*]}"
  done
done
$PY tools/show_wide.py "$OUT" > "$OUT/summary.txt" 2>&1
touch "$OUT.done"
echo "$(date +%T) all turns done"
