#!/bin/bash
# As run_gen_more.sh writes new code answers: each time their number reaches one of the given
# counts, train and score the reranker with that many (rerank6.py, about 15 minutes on 8 CPU
# threads at low priority). Results land in data/code/rerank-more/N.{log,json,pt,done}.
#
#   setsid nohup research/reranker/run_more.sh 600 1800 4200 >> data/code/rerank-more/driver.log 2>&1 < /dev/null &
#   cancel: pkill -f '[r]un_more.sh'; pkill -f '[r]erank6.py'
set -u
PY=${PY:-python3}
cd "$(dirname "$0")/../.."
D=research/reranker
O=data/code/rerank-more
ROWS=data/code/rows-9b-more.jsonl
mkdir -p $O
for N in "$@"; do
  [ -e "$O/$N.done" ] && continue
  until [ "$(wc -l < $ROWS 2>/dev/null || echo 0)" -ge "$N" ]; do
    sleep 120
  done
  echo "$(date '+%F %T') $N new answers are there: training"
  MORE_LIMIT=$N dibs run cpu --label "MemDraft: reranker with $N new code answers (8 CPU threads, low priority)" --eta 1h -- \
    nice -n 10 $PY $D/rerank6.py "$O/$N" > "$O/$N.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') $N: exit $rc"
  [ "$rc" -eq 0 ] && touch "$O/$N.done"
done
