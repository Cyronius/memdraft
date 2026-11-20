#!/bin/bash
# The table drafter against llama.cpp's own lookup drafters on the GPU, and three cheap changes to it
# (Qwen3.5-9B through Vulkan, greedy decoding, lossless acceptance).
#   llama-cache    --spec-type ngram-cache: the classic lookup decoding, as shipped
#   llama-mod      --spec-type ngram-mod: the drafter --spec-default turns on, as shipped
#   llama-simple   --spec-type ngram-simple, as shipped
#   table-3        table + reranker, 3 tokens (pad): the setup of the 2026-11-17 run
#   table-5, -8    the same drafting 5 and 8 tokens
#   copy+table-3   ngram-simple in front: a 12-token match earlier in the request drafts up to 48 tokens
#   mod+table-3    ngram-mod in front
# The stock arms get llama.cpp's checkpoint path on Qwen3.5 (restore and decode again after a rejection);
# arms with the table get per-token snapshots for drafts up to its length, the checkpoint for longer ones.
#   prompts   the 12 coding prompts of code-v1 (192 tokens), the first 36 HumanEval problems (512 tokens),
#             the 6 whole-file edits of code-edit-v1 (640 tokens)
# Turns of at most 9 minutes as in run_accept_live.sh: each when the GPU is ours alone and the load
# average is below LOAD (or after LOAD_WAIT seconds), GAP seconds apart. show_wide.py writes summary.txt.
#
#   setsid nohup ./run_prior_live.sh > data/code/prior.log 2>&1 < /dev/null &
#   cancel: pkill -f '[r]un_prior_live.sh'; pkill -f '[l]ive_bench.py.*prior-'
set -u
cd "$(dirname "$0")"
PY=${PY:-python3}
S=${MEMDRAFT_CACHE:-$HOME/.cache/model-experiments/memdraft}
NEW=$PWD/data/code/9b-code-pod-c3
STAMP=$(date +%Y%m%d-%H%M)
OUT=results/prior-9b-vk-$STAMP
WORK=data/code/prior-$STAMP
mkdir -p "$OUT" "$WORK"
T="--spec-table-pad --spec-table-file $NEW.mdt --spec-table-reranker $NEW.mdr"
declare -A ARM=(
  [table-3]="--spec-type ngram-table --spec-table-n-max 3 $T"
  [table-5]="--spec-type ngram-table --spec-table-n-max 5 $T"
  [table-8]="--spec-type ngram-table --spec-table-n-max 8 $T"
  [copy+table-3]="--spec-type ngram-simple,ngram-table --spec-table-n-max 3 $T"
  [mod+table-3]="--spec-type ngram-mod,ngram-table --spec-table-n-max 3 $T"
  [llama-cache]="--spec-type ngram-cache"
  [llama-mod]="--spec-type ngram-mod"
  [llama-simple]="--spec-type ngram-simple"
)
ALL=(table-3 copy+table-3 llama-mod llama-cache table-5 mod+table-3 llama-simple table-8)   # most wanted first

$PY - "$WORK" <<'EOF'
import json, sys
work = sys.argv[1]
he = [json.loads(l) for l in open('prompts/humaneval-v1.jsonl')][:36]
for n in range(0, len(he), 12):
    open(f'{work}/he-{n // 12:02d}.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in he[n:n + 12]))
rows = [json.loads(l) for l in open('prompts/code-v1.jsonl')]
seen = {}
pick = [r for r in rows if seen.setdefault(r['domain'], []).append(r) or len(seen[r['domain']]) <= 6]
open(f'{work}/oss.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in pick))
edit = [r for r in map(json.loads, open('prompts/code-edit-v1.jsonl')) if r['domain'] == 'edit']
for n in range(0, len(edit), 3):
    open(f'{work}/edit-{n // 3}.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in edit[n:n + 3]))
EOF

JOBS=(he-00 edit-0 oss he-01 edit-1 he-02)
turn=0
for part in "${JOBS[@]}"; do
  case $part in oss) TOKENS=192;; he-*) TOKENS=512;; *) TOKENS=640;; esac
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
    arbiter run gpu port-8455 --label "MemDraft: against llama.cpp's lookup drafters, turn $turn of about 8 (9B, at most 9 min, stops if the GPU gets company)" --eta 10m -- \
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
$PY show_wide.py "$OUT" > "$OUT/summary.txt" 2>&1
touch "$OUT.done"
echo "$(date +%T) all turns done"
