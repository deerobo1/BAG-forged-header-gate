#!/bin/sh
# Sprint 3 queue, one job at a time (CPU). Waits for e124.py, then runs the rest in PREREG order.
cd "$(dirname "$0")/../.."
while ! grep -q "^ustc time 0" sprint3/results/logs/e124.log; do sleep 30; done
python sprint3/code/e3_flags.py      > sprint3/results/logs/e3.log 2>&1
python sprint3/code/e6_packets.py    > sprint3/results/logs/e6.log 2>&1
python sprint3/code/e6_packets.py --full > sprint3/results/logs/e7.log 2>&1
python -c "import sys; sys.path.insert(0, 'sprint3/code'); import e9_curve; e9_curve.cost()" > sprint3/results/logs/e9.log 2>&1
python sprint3/code/e8_group.py 0 1 2 > sprint3/results/logs/e8.log 2>&1
echo queue done > sprint3/results/logs/queue.done
