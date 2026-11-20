#!/bin/bash
# As a pod (run_gen_pod.py) writes code answers: each time their number reaches one of the given
# counts, score the table with that many of them added (more_code.py, PARTS=curve: about 10 minutes
# on 8 CPU threads at low priority, under a 30 GB memory limit). The reference table and the 1700
# answers from this machine in it are the same every time, so the lines can be read as one curve.
# Results land in data/code/rerank-more/curve-N.{log,json,done}.
#
#   setsid nohup research/reranker/run_curve.sh data/code/pod/gen-XXXX 4000 8000 16000 32000 64000 \
#       >> data/code/rerank-more/curve-driver.log 2>&1 < /dev/null &
#   cancel: pkill -f '[r]un_curve.sh'; pkill -f '[m]ore_code.py'
set -u
cd "$(dirname "$0")/../.."
JOB=$1; shift
D=research/reranker
O=data/code/rerank-more
PY=${PY:-python3}
for N in "$@"; do
  [ -e "$O/curve-$N.done" ] && continue
  until [ "$(cat $JOB/rows/rows-*.jsonl 2>/dev/null | wc -l)" -ge "$((N + 200))" ] || [ -e "$JOB/job-lifecycle.json" ]; do
    sleep 120
  done
  $PY collect_pod_rows.py "$JOB" --out "$O/curve-$N.rows.jsonl"
  if [ "$(wc -l < "$O/curve-$N.rows.jsonl")" -lt "$N" ]; then
    echo "$(date '+%F %T') the pod ended with fewer than $N answers: stopping"
    rm -f "$O/curve-$N.rows.jsonl"
    exit 0
  fi
  echo "$(date '+%F %T') $N of the pod's answers are there: scoring"
  MORE_LIMIT=1700 PARTS=curve POD_LIMIT=$N POD_ROWS="$O/curve-$N.rows.jsonl" OMP_NUM_THREADS=8 \
    dibs run cpu --label "MemDraft: table with $N of the pod's code answers (8 CPU threads, low priority)" --eta 30m -- \
    systemd-run --user --scope -q -p MemoryMax=30G -p MemorySwapMax=0 -- \
    nice -n 10 $PY $D/more_code.py "$O/curve-$N" > "$O/curve-$N.log" 2>&1
  rc=$?
  rm -f "$O/curve-$N.rows.jsonl"
  echo "$(date '+%F %T') $N: exit $rc"
  [ "$rc" -eq 0 ] && touch "$O/curve-$N.done"
done
