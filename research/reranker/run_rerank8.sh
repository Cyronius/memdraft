#!/bin/bash
# rerank8.py at low priority: 4 CPU threads, nice 19, idle I/O, under a 28 GB memory limit (no dibs
# cpu resource on this install; low priority is the courtesy).
# Results: data/code/rerank-more/feat.{log,json,pt,done} (or the prefix given).
#
#   setsid nohup research/reranker/run_rerank8.sh > /dev/null 2>&1 < /dev/null &
#   trial: POD_LIMIT=2000 MORE_LIMIT=400 EPOCHS=1 run_rerank8.sh data/code/rerank-more/feat-trial
#   cancel: pkill -f '[r]un_rerank8.sh'; pkill -f '[r]erank8.py'
set -u
cd "$(dirname "$0")/../.."
O=${1:-data/code/rerank-more/feat}
PY=${PY:-python3}
mkdir -p "$(dirname "$O")"
OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4} \
  systemd-run --user --scope -q -p MemoryMax=28G -p MemorySwapMax=0 -- \
  nice -n 19 ionice -c3 $PY research/reranker/rerank8.py "$O" > "$O.log" 2>&1
rc=$?
echo "$(date '+%F %T') exit $rc" >> "$O.log"
[ "$rc" -eq 0 ] && touch "$O.done"
