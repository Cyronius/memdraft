#!/bin/bash
# Making head-first drafting cheaper (2026-11-18). The head's output projection (248k x 4096) is most of its
# cost, so smaller-precision heads are tried; head-first now asks the head for 1 token by itself.
# Qwen3.5-9B Q4_K_M through Vulkan, greedy, lossless. All arms but table-3 put ngram-simple in front ("copy").
#   copy+table-3            ngram-simple, else table + reranker, 3 tokens (pad)
#   copy+head8+table-4      ngram-simple, else the head's token (q8_0 head, output 1081 MB) then 3 from the table
#   copy+head8+table-5      the same, 4 from the table
#   copy+head6+table-4      head re-quantized Q4_K_M: layer Q4_K, output Q6_K (834 MB)
#   copy+head4+table-4      head re-quantized Q4_K_M with output Q4_K (572 MB)
#   copy+head4+table-5      the same, 4 from the table
#   prompts   HumanEval problems 0-23 (512 tokens); the 12 coding prompts of code-v1 (192 tokens), two rounds
# Turns as in run_accept_live.sh.
#
#   setsid nohup ./run_headfirst_live.sh > data/code/headfirst.log 2>&1 < /dev/null &
#   cancel: pkill -f '[r]un_headfirst_live.sh'; pkill -f '[l]ive_bench.py.*headfirst-9b'
set -u
cd "$(dirname "$0")"
PY=${PY:-python3}
S=${MEMDRAFT_CACHE:-$HOME/.cache/model-experiments/memdraft}
NEW=$PWD/data/code/9b-code-pod-c3
STAMP=$(date +%Y%m%d-%H%M)
OUT=results/headfirst-9b-vk-$STAMP
WORK=data/code/headfirst-$STAMP
mkdir -p "$OUT" "$WORK"
T="--spec-table-pad --spec-table-file $NEW.mdt --spec-table-reranker $NEW.mdr"
C="--spec-type ngram-simple,draft-mtp,ngram-table --spec-draft-p-min 0 --spec-table-head-first"
declare -A ARM=(
  [copy+table-3]="--spec-type ngram-simple,ngram-table --spec-table-n-max 3 $T"
  [copy+head8+table-4]="$C --spec-draft-model $S/mtp-head/Qwen3.5-9B-q8_0.gguf --spec-table-n-max 4 $T"
  [copy+head8+table-5]="$C --spec-draft-model $S/mtp-head/Qwen3.5-9B-q8_0.gguf --spec-table-n-max 5 $T"
  [copy+head6+table-4]="$C --spec-draft-model $S/mtp-head/Qwen3.5-9B-q4_k_m.gguf --spec-table-n-max 4 $T"
  [copy+head4+table-4]="$C --spec-draft-model $S/mtp-head/Qwen3.5-9B-q4_k_m-out-q4k.gguf --spec-table-n-max 4 $T"
  [copy+head4+table-5]="$C --spec-draft-model $S/mtp-head/Qwen3.5-9B-q4_k_m-out-q4k.gguf --spec-table-n-max 5 $T"
)
ALL=(copy+table-3 copy+head8+table-4 copy+head4+table-4 copy+head6+table-4 copy+head8+table-5 copy+head4+table-5)   # most wanted first

$PY - "$WORK" <<'EOF'
import json, sys
work = sys.argv[1]
he = [json.loads(l) for l in open('prompts/humaneval-v1.jsonl')][:24]
for n in range(0, len(he), 12):
    open(f'{work}/he-{n // 12:02d}.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in he[n:n + 12]))
rows = [json.loads(l) for l in open('prompts/code-v1.jsonl')]
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
  until $PY gpu_alone.py > /dev/null || $PY gpu_alone.py --preemptible-ok > /dev/null; do
    sleep 60
  done
}
bench() {   # prompts, tokens, result name, arms...
  local prompts=$1 tokens=$2 name=$3; shift 3
  arbiter run gpu port-8455 --label "MemDraft: MTP head drafts first, table continues, $name (9B, at most 9 min, stops if the GPU gets company)" --eta 10m -- \
    $PY live_bench.py --gpu-guard --guard-checks 2 --guard-gap 5 --retries 1 --time-limit "${LIMIT:-540}" --arm-seconds 75 \
    --server "$S/llama.cpp/build-vk/bin/llama-server" --model "$S/models/Qwen3.5-9B-Q4_K_M.gguf" \
    --prompts "$prompts" --max-tokens "$tokens" --out "$name" -- -ngl 99 -c 8192 "$@"
}
complete() {   # result file, arms...: the arms not complete in it
  $PY -c 'import json, os, sys
done = json.load(open(sys.argv[1])).get("complete_arms", []) if os.path.exists(sys.argv[1]) else []
print(" ".join(a for a in sys.argv[2:] if a not in done))' "$@"
}

turn=0
for pr in he-00:1 oss:1 he-01:1 oss:2; do
  part=${pr%:*} round=${pr#*:}
  case $part in oss) TOKENS=192;; *) TOKENS=512;; esac
  LEFT=("${ALL[@]}")
  for attempt in 1 2; do
    turn=$((turn + 1))
    sleep "${GAP:-300}"
    gpu_ours
    name="$part-r$round-a$attempt"
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
$PY show_wide.py "$OUT" > "$OUT/summary.txt" 2>&1
touch "$OUT.done"
echo "$(date +%T) all turns done"
