#!/usr/bin/env bash
# The sprint's CPU queue, one job at a time (two at once is 2.7x slower each), in the pre-registered order.
# Every step skips work already saved, so the queue can be stopped and started again.
#   bash sprint/queue.sh > sprint/results/logs/queue.log 2>&1
set -e
cd "$(dirname "$0")"
PY=/c/300_apps/Python/Python311/python
SCREEN="5gad2022 mcfp_strict mcfp"
step() { echo "=== $(date '+%m-%d %H:%M') $*"; "$PY" -u "$@"; }

step e0_baselines.py random 0
step c1_length_view.py random 0
step analysis_saved.py
for idea in B_A B_S; do for run in $SCREEN; do step train.py $idea $run; done; done
step score_trained.py random 0
for run in $SCREEN; do step train.py nand $run; done
step score_trained.py random 0
for run in $SCREEN; do step train.py ce_s1 $run; done
for run in $SCREEN; do step train.py adv8 $run; done
step score_trained.py random 0
step e0_baselines.py time 0
step train.py B_dro ciciot2022 time 0
step score_trained.py time 0 ciciot2022
for s in 1 2; do step e0_baselines.py random $s; done
for s in "random 1" "random 2" "time 0"; do step c1_length_view.py $s; done
step analysis_saved.py
echo "=== $(date '+%m-%d %H:%M') queue done"
