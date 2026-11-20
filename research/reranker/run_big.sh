#!/bin/bash
# The reranker trained on all the stand-in text, then on own code (rerank5.py). CPU only, about 1.5 hours.
#   setsid nohup research/reranker/run_big.sh > data/code/rerank-big.log 2>&1 < /dev/null &
#   cancel: pkill -f '[r]erank5.py'
set -u
PY=${PY:-python3}
cd "$(dirname "$0")/../.."
D=research/reranker
dibs run cpu --label "MemDraft: reranker trained on all stand-in text (8 CPU threads, low priority, 1.5 h)" --eta 2h -- \
  nice -n 10 $PY $D/rerank5.py data/code/rerank-big
rc=$?
echo "$(date +%T) exit $rc"
[ "$rc" -eq 0 ] && touch data/code/rerank-big.done
