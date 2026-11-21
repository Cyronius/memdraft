#!/bin/bash
# When the pod job has ended (its pod deleted), put its rows in data/code/rows-9b-pod.jsonl
# and train and score the reranker on them (rerank7.py: about two hours on 8 CPU threads at low
# priority, under a 30 GB memory limit). Results: data/code/rerank-more/pod.{log,json,pt,done}.
#
#   setsid nohup research/reranker/run_pod_rerank.sh data/code/pod/gen-XXXX \
#       >> data/code/rerank-more/pod-driver.log 2>&1 < /dev/null &
#   cancel: pkill -f '[r]un_pod_rerank.sh'; pkill -f '[r]erank7.py'
set -u
cd "$(dirname "$0")/../.."
JOB=$1
D=research/reranker
O=data/code/rerank-more
PY=${PY:-python3}
ROWS=data/code/rows-9b-pod.jsonl
until grep -q "Deletion confirmed" data/code/pod/launch.log 2>/dev/null || [ -e "$JOB/job-lifecycle.json" ]; do
  sleep 120
done
sleep 60                                   # the launcher's last copy of the rows
echo "$(date '+%F %T') the pod job has ended"
$PY research/code-generation/collect_pod_rows.py "$JOB" --out "$ROWS" || exit 1
echo "$(date '+%F %T') training"
POD_ROWS="$ROWS" OMP_NUM_THREADS=8 \
  dibs run cpu --label "MemDraft: reranker on the pod's code answers (8 CPU threads, low priority)" --eta 3h -- \
  systemd-run --user --scope -q -p MemoryMax=30G -p MemorySwapMax=0 -- \
  nice -n 10 $PY $D/rerank7.py "$O/pod" > "$O/pod.log" 2>&1
rc=$?
echo "$(date '+%F %T') exit $rc"
[ "$rc" -eq 0 ] && touch "$O/pod.done"
