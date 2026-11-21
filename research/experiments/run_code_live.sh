#!/bin/bash
# Tokens per second on coding prompts, Qwen3.5-9B on 8 CPU threads: plain decoding against the
# table drafter with a table that has code in its text (code_study.py --export), with the shipped
# table (no code in its text), with the in-request lookup alone, and llama.cpp's own lookup
# drafters. The prompts are code_study.py's test prompts: their solutions are in neither table.
# Arms take turns every two prompts, because CPU speed drifts within minutes on this machine.
#
# If tools/gen_code.py is running it is stopped first (two 9B servers on the CPU would time each other)
# and started again afterwards; it continues where it stopped. About 45 minutes.
#
#   setsid nohup ./run_code_live.sh > data/code/live-cpu.log 2>&1 < /dev/null &
#   cancel: pkill -f '[r]un_code_live.sh'; pkill -f '[l]ive_bench.py.*code-9b'
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
S=${MEMDRAFT_CACHE:-$HOME/.cache/model-experiments/memdraft}
SHIPPED=$PWD/data/9b-all-c3.mdt
CODE=${CODE_TABLE:-$PWD/data/code/9b-code-c3.mdt}
OUT=research/results/code-9b-cpu-$(date +%Y%m%d-%H%M)
PAD2="--spec-type ngram-table --spec-table-n-max 2 --spec-table-pad --spec-table-file"

GEN='^[^ ]*python[0-9.]* tools/gen_code.py'
COUNT=$(pgrep -af "$GEN" | sed -n 's/.* --count \([0-9]*\).*/\1/p' | head -1)
if [ -n "$COUNT" ]; then
  echo "$(date +%T) stopping tools/gen_code.py (--count $COUNT) for the timing"
  pkill -TERM -f "$GEN"
  for _ in $(seq 60); do pgrep -f '[r]un_gen_code.sh' > /dev/null || break; sleep 2; done
  pkill -f '[l]lama-server.* --port 8455 ' && echo "the generation server was still up: stopped it"
  rm -f "data/code/gen-$COUNT.done"
fi

arbiter run --cpu 8 port-8455 --label "MemDraft on coding prompts (9B on 8 CPU threads, arms in turns, 45 min)" --eta 50m -- \
  $PY tools/live_bench.py --chunk 2 \
  --server "$S/llama.cpp/build-cpu/bin/llama-server" --model "$S/models/Qwen3.5-9B-Q4_K_M.gguf" \
  --prompts bench/prompts/code-v1.jsonl --per-domain "${PER_DOMAIN:-6}" --max-tokens "${MAX_TOKENS:-192}" --out "$OUT.json" -- -ngl 0 -t 8 -c 8192 \
  --arm none= \
  --arm "code-table=$PAD2 $CODE" \
  --arm "shipped-table=$PAD2 $SHIPPED" \
  --arm "request-only=$PAD2 $CODE --spec-table-source context" \
  --arm "ngram-cache=--spec-type ngram-cache" \
  --arm "ngram-mod=--spec-type ngram-mod"
rc=$?
echo "$(date +%T) exit $rc"

if [ -n "$COUNT" ]; then
  echo "$(date +%T) starting tools/gen_code.py again"
  setsid nohup ./run_gen_code.sh "$COUNT" >> data/code/gen.log 2>&1 < /dev/null &
fi
touch "$OUT.done"
