#!/usr/bin/env bash
# Runs the three versions on Hopper medium (3 runs in parallel on one GPU)
# and writes the learning curves to runs/. Usage: bash run_experiments.sh [python]
PY=${1:-python}
STEPS=300000
mkdir -p runs
for task in hopper; do
  for v in 0 1 2; do
    nohup $PY version$v.py --task $task --steps $STEPS \
      --log runs/v${v}_${task}.json > runs/v${v}_${task}.log 2>&1 &
  done
done
wait
