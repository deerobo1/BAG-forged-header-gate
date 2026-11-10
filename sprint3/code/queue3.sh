#!/bin/sh
# Sprint 3 queue, restarted at E7 after the empty-batch fix.
cd "$(dirname "$0")/../.."
python sprint3/code/e6_packets.py --full > sprint3/results/logs/e7.log 2>&1
python -c "import sys; sys.path.insert(0, 'sprint3/code'); import e9_curve; e9_curve.cost()" > sprint3/results/logs/e9.log 2>&1
python sprint3/code/e8_group.py 0 1 2 > sprint3/results/logs/e8.log 2>&1
echo queue done > sprint3/results/logs/queue.done
