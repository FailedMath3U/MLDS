#!/usr/bin/env bash
# Start this script with nohup. It computes only: no reports, plots or chat callbacks.
set -euo pipefail
cd /nfs/home/lvj1668/mlds_assignment2_mspacman_compute
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
PY=/nfs/home/lvj1668/mlds_assignment1_pong_env/bin/python
exec 9>pipeline.lock
flock -n 9 || { echo 'Another MsPacman pipeline already holds the lock.'; exit 1; }
echo "pipeline PID: $$"
"$PY" -u mspacman_dqn.py train --out runs/seed1668 --seed 1668 --episodes 5000 &
trainer_pid=$!
echo "$trainer_pid" > trainer.pid
wait "$trainer_pid"
"$PY" -u mspacman_dqn.py evaluate --checkpoint runs/seed1668/best.pt \
    --out evaluation --episodes 500 --seed-start 9000000 --epsilon .01
echo 'MSPACMAN COMPUTATION COMPLETE'
