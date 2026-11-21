#!/bin/bash
# Tokens per second on coding prompts, Qwen3.5-9B: plain decoding against the table drafter with the
# 2026-11-10 code table (9b-code-800-c3, 0.4M tokens of the 9B's own code), and with the table that
# holds the pod's answers too (export_reranker.py: 39M tokens of own code), the drafter alone and
# with the reranker. Same prompts and settings as run_code_live.sh: the 12 test prompts of
# code_study.py (in no table, and not trained on), 192 tokens.
#
#   ./run_rerank_live.sh cpu   # 8 CPU threads, arms in turns every two prompts, about 35 minutes
#   ./run_rerank_live.sh vk    # GPU through Vulkan, in turns of at most 9 minutes between other
#                                     # sessions' GPU jobs (as run_lookup.sh vk), plain decoding again each turn
#
#   setsid nohup ./run_rerank_live.sh cpu > data/code/live-rerank.log 2>&1 < /dev/null &
#   cancel: pkill -f '[r]un_rerank_live.sh'; pkill -f '[l]ive_bench.py.*rerank-9b'
set -u
cd "$(dirname "$0")/../.."
BACKEND=${1:?cpu or vk}
PY=${PY:-python3}
S=${MEMDRAFT_CACHE:-$HOME/.cache/model-experiments/memdraft}
OLD=$PWD/data/code/9b-code-800-c3.mdt
NEW=$PWD/${NEW:-data/code/9b-code-pod-c3}
OUT=research/results/rerank-9b-$BACKEND-$(date +%Y%m%d-%H%M)
T="--spec-type ngram-table --spec-table-pad"
R="--spec-table-file $NEW.mdt --spec-table-reranker $NEW.mdr"
declare -A ARM=(
  [old-table]="$T --spec-table-n-max 2 --spec-table-file $OLD"
  [pod-table]="$T --spec-table-n-max 2 --spec-table-file $NEW.mdt"
  [pod-table-3]="$T --spec-table-n-max 3 --spec-table-file $NEW.mdt"
  [pod-table-reranker]="$T --spec-table-n-max 2 $R"
  [pod-table-reranker-3]="$T --spec-table-n-max 3 $R"
)

case $BACKEND in
  cpu)
    arbiter run --cpu 8 port-8455 --label "MemDraft: table + reranker on coding prompts (9B on 8 CPU threads, arms in turns, 35 min)" --eta 45m -- \
      $PY tools/live_bench.py --chunk 2 \
      --server "$S/llama.cpp/build-cpu/bin/llama-server" --model "$S/models/Qwen3.5-9B-Q4_K_M.gguf" \
      --prompts bench/prompts/code-v1.jsonl --per-domain "${PER_DOMAIN:-6}" --max-tokens "${MAX_TOKENS:-192}" --out "$OUT.json" -- -ngl 0 -t 8 -c 8192 \
      --arm none= \
      --arm "old-table=${ARM[old-table]}" \
      --arm "pod-table=${ARM[pod-table]}" \
      --arm "pod-table-reranker=${ARM[pod-table-reranker]}" \
      --arm "pod-table-reranker-3=${ARM[pod-table-reranker-3]}"
    echo "$(date +%T) exit $?"
    ;;
  vk)
    LEFT=(pod-table-reranker-3 pod-table-3 pod-table-reranker pod-table old-table)   # most wanted first
    turn=0
    while [ ${#LEFT[@]} -gt 0 ] && [ $turn -lt "${TURNS:-8}" ]; do
      turn=$((turn + 1))
      $PY tools/gpu_alone.py --wait --preemptible-ok --checks 2 --gap 5
      ARMS=(--arm none=)
      for a in "${LEFT[@]}"; do ARMS+=(--arm "$a=${ARM[$a]}"); done
      arbiter run gpu port-8455 --label "MemDraft: table + reranker on coding prompts (9B, at most 9 min, stops if the GPU gets company)" --eta 10m -- \
        $PY tools/live_bench.py --gpu-guard --guard-checks 2 --guard-gap 5 --retries 1 --time-limit "${LIMIT:-520}" --arm-seconds 75 \
        --server "$S/llama.cpp/build-vk/bin/llama-server" --model "$S/models/Qwen3.5-9B-Q4_K_M.gguf" \
        --prompts bench/prompts/code-v1.jsonl --per-domain "${PER_DOMAIN:-6}" --max-tokens "${MAX_TOKENS:-192}" \
        --out "$OUT-turn$turn.json" -- -ngl 99 -c 8192 "${ARMS[@]}"
      echo "$(date +%T) turn $turn: exit $?"
      LEFT=($($PY -c 'import json, os, sys
done = json.load(open(sys.argv[1])).get("complete_arms", []) if os.path.exists(sys.argv[1]) else []
print(" ".join(a for a in sys.argv[2:] if a not in done))' "$OUT-turn$turn.json" "${LEFT[@]}"))
      touch "$OUT-turn$turn.done"
      echo "$(date +%T) turn $turn over; still to run: ${LEFT[*]:-nothing}"
      [ ${#LEFT[@]} -gt 0 ] && sleep "${GAP:-900}"
    done
    ;;
  *) echo "unknown backend $BACKEND"; exit 2;;
esac
touch "$OUT.done"
