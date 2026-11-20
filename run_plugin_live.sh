#!/bin/bash
# The drafter-plugin interface on current llama.cpp master, against llama.cpp's built-in drafters and the
# patched 7d7f028 build (2026-11-18). Qwen3.5-9B Q4_K_M through Vulkan, greedy, lossless. Each turn runs both
# builds back to back, each with its own plain baseline.
#   plugin build (master + llama-plugin-hook.patch):
#     copy+plugin-3         ngram-simple, else MemDraft (plugin/libmemdraft.so), 3 tokens
#     copy+head4+plugin-4   ngram-simple, else the MTP head's token then 3 from MemDraft (--spec-plugin-prefix 1)
#     mtp-builtin-3         --spec-type draft-mtp, 3 tokens (head re-quantized Q4_K_M, Q4_K output)
#     mtp-plugin-3          the same head through plugin/examples/libmtp.so
#     draft-builtin-3       --spec-type draft-simple with Qwen3.5-2B Q4_K_M, 3 tokens
#     draft-plugin-3        the same draft model through plugin/examples/libdraft_model.so
#     mod-builtin           --spec-type ngram-mod
#     mod-plugin            ngram-mod through plugin/examples/libbuiltin_ngram.so (drafts proven identical offline)
#   patched build (7d7f028 + patch_llama.py, MemDraft compiled in):
#     copy+table-3, copy+head4+table-4   the same as copy+plugin-3 and copy+head4+plugin-4
#   prompts   HumanEval problems 0-11 (512 tokens); the 12 coding prompts of code-v1 (192 tokens), two rounds
# Turns as in run_accept_live.sh.
#
# Overrides: PARTS (default "he-00:1 oss:1 oss:2"), PLUGIN_ARMS and PATCHED_ARMS (arm names; empty skips
# that build), ATTEMPTS (turns per part and build, default 2).
#
#   setsid nohup ./run_plugin_live.sh > data/code/plugin.log 2>&1 < /dev/null &
#   cancel: pkill -f '[r]un_plugin_live.sh'; pkill -f '[l]ive_bench.py.*plugin-9b'
set -u
cd "$(dirname "$0")"
PY=${PY:-python3}
S=${MEMDRAFT_CACHE:-$HOME/.cache/model-experiments/memdraft}
NEW=$PWD/data/code/9b-code-pod-c3
STAMP=$(date +%Y%m%d-%H%M)
OUT=results/plugin-9b-vk-$STAMP
WORK=data/code/plugin-$STAMP
mkdir -p "$OUT" "$WORK"
T="--spec-table-pad --spec-table-file $NEW.mdt --spec-table-reranker $NEW.mdr"
E=$PWD/plugin/examples
P="--spec-plugin $PWD/plugin/libmemdraft.so --spec-plugin-args table=$NEW.mdt,reranker=$NEW.mdr,pad=1,vocab_hash=c52114e74bc3a20f"
HEAD=$S/mtp-head/Qwen3.5-9B-q4_k_m-out-q4k.gguf
D2=$S/models/Qwen3.5-2B-Q4_K_M.gguf
H4="--spec-draft-model $HEAD --spec-draft-p-min 0"
declare -A ARM=(
  [copy+table-3]="--spec-type ngram-simple,ngram-table --spec-table-n-max 3 $T"
  [copy+head4+table-4]="--spec-type ngram-simple,draft-mtp,ngram-table $H4 --spec-table-head-first --spec-table-n-max 4 $T"
  [copy+plugin-3]="--spec-type ngram-simple,plugin --spec-plugin-n-max 3 $P"
  [copy+head4+plugin-4]="--spec-type ngram-simple,draft-mtp,plugin $H4 --spec-plugin-prefix 1 --spec-plugin-n-max 4 $P"
  [mtp-builtin-3]="--spec-type draft-mtp $H4 --spec-draft-n-max 3"
  [mtp-plugin-3]="--spec-type plugin --spec-plugin $E/libmtp.so --spec-plugin-n-max 3 --spec-plugin-args model=$HEAD"
  [draft-builtin-3]="--spec-type draft-simple -md $D2 --spec-draft-n-max 3"
  [draft-plugin-3]="--spec-type plugin --spec-plugin $E/libdraft_model.so --spec-plugin-n-max 3 --spec-plugin-args model=$D2"
  [mod-builtin]="--spec-type ngram-mod"
  [mod-plugin]="--spec-type plugin --spec-plugin $E/libbuiltin_ngram.so --spec-plugin-n-max 64 --spec-plugin-args type=ngram-mod"
)
declare -A SERVER=(
  [patched]=$S/llama.cpp/build-vk/bin/llama-server
  [plugin]=$S/llama.cpp-plugin/build-vk/bin/llama-server
)
declare -A ARMS_OF=(
  [patched]="${PATCHED_ARMS-copy+table-3 copy+head4+table-4}"
  [plugin]="${PLUGIN_ARMS-copy+plugin-3 copy+head4+plugin-4 mtp-builtin-3 mtp-plugin-3 draft-builtin-3 draft-plugin-3 mod-builtin mod-plugin}"
)

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
  # A preemptible job counts as leaving only while arbiter lists it as preemptible (then our `arbiter run gpu` makes it yield).
  until $PY gpu_alone.py > /dev/null || $PY gpu_alone.py --preemptible-ok > /dev/null; do
    sleep 60
  done
}
bench() {   # server, prompts, tokens, result name, arms...
  local server=$1 prompts=$2 tokens=$3 name=$4; shift 4
  arbiter run gpu port-8455 --label "MemDraft: drafter plugins against built-in drafters, $name (9B, at most 9 min, stops if the GPU gets company)" --eta 10m -- \
    $PY live_bench.py --gpu-guard --guard-checks 2 --guard-gap 5 --retries 1 --time-limit "${LIMIT:-540}" --arm-seconds 75 \
    --server "$server" --model "$S/models/Qwen3.5-9B-Q4_K_M.gguf" \
    --prompts "$prompts" --max-tokens "$tokens" --out "$name" -- -ngl 99 -c 8192 "$@"
}
complete() {   # result file, arms...: the arms not complete in it
  $PY -c 'import json, os, sys
done = json.load(open(sys.argv[1])).get("complete_arms", []) if os.path.exists(sys.argv[1]) else []
print(" ".join(a for a in sys.argv[2:] if a not in done))' "$@"
}

turn=0
for pr in ${PARTS:-he-00:1 oss:1 oss:2}; do
  part=${pr%:*} round=${pr#*:}
  case $part in oss) TOKENS=192;; *) TOKENS=512;; esac
  for build in plugin patched; do
    LEFT=(${ARMS_OF[$build]})
    [ ${#LEFT[@]} -eq 0 ] && continue
    for attempt in $(seq "${ATTEMPTS:-2}"); do
      turn=$((turn + 1))
      sleep "${GAP:-120}"
      gpu_ours
      name="$part-$build-r$round-a$attempt"
      echo "$(date +%T) turn $turn: $name, load $(cut -d' ' -f1 /proc/loadavg) after waiting ${waited}s, arms ${LEFT[*]}"
      ARMS=(--arm none=)
      for a in "${LEFT[@]}"; do ARMS+=(--arm "$a=${ARM[$a]}"); done
      bench "${SERVER[$build]}" "$WORK/$part.jsonl" $TOKENS "$OUT/$name.json" "${ARMS[@]}"
      echo "$(date +%T) turn $turn: exit $?"
      LEFT=($(complete "$OUT/$name.json" "${LEFT[@]}"))
      [ ${#LEFT[@]} -eq 0 ] && break
      echo "$(date +%T) still to run on $part with $build: ${LEFT[*]}"
    done
  done
done
$PY show_wide.py "$OUT" > "$OUT/summary.txt" 2>&1
touch "$OUT.done"
echo "$(date +%T) all turns done"
