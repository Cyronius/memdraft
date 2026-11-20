#!/bin/bash
# A short GPU pilot on Qwen3.5-9B (Vulkan): what a decode step costs with drafts on board, and a
# first tokens/second for the table drafter. About 6 minutes of GPU.
#
# It starts only when no other process has the GPU (gpu_alone.py; arbiter alone is not enough:
# idle servers never book the GPU), holds arbiter's gpu while it runs, and stops its server within seconds if
# another process shows up on the GPU. Other work on the machine still runs beside it, so these
# are pilot numbers; run_measure.sh is the clean version.
#
#   setsid nohup ./run_pilot.sh > data/pilot.log 2>&1 < /dev/null &
#   cancel: pkill -f run_pilot.sh; pkill -f 'live_bench.py.*pilot'
set -u
cd "$(dirname "$0")/.."
PY=target/manifold-audit/env/bin/python
S=$HOME/.cache/model-experiments/memdraft
T=$PWD/data/9b-all-c3.mdt
OUT=results/pilot-9b-vk-$(date +%Y%m%d-%H%M)
TABLE="--spec-type ngram-table --spec-table-file $T"
CONST="--spec-type ngram-table --spec-table-const 2 --spec-table-n-max 2"

$PY gpu_alone.py --wait --checks 2 --gap 10
arbiter run gpu port-8455 --label "MemDraft GPU pilot (9B, 6 min, stops if the GPU gets company)" --eta 8m -- \
  $PY live_bench.py --gpu-guard --guard-checks 1 --guard-gap 5 \
  --server "$S/llama.cpp/build-vk/bin/llama-server" --model "$S/models/Qwen3.5-9B-Q4_K_M.gguf" \
  --prompts prompts/live-v1.jsonl --per-domain 1 --max-tokens 96 --out "$OUT.json" -- -ngl 99 -c 8192 \
  --arm none= \
  --arm "accept2=$CONST --spec-synth-rates 1,1" \
  --arm "reject2=$CONST --spec-synth-rates 0,0" \
  --arm "reject2-every-other-step=$CONST --spec-synth-rates 0,0 --spec-table-const-period 2" \
  --arm "both=$TABLE --spec-table-tau 0.3 --spec-table-n-max 4" \
  --arm "pad2=$TABLE --spec-table-n-max 2 --spec-table-pad"
echo "exit $?"
touch "$OUT.done"
