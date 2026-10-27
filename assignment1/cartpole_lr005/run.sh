#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${CARTPOLE_PYTHON:-$HOME/mlds_assignment1_cartpole_env/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo 'Set CARTPOLE_PYTHON to the Python executable with PyTorch and Gymnasium.' >&2
  exit 1
fi
if [[ -d "$task_dir/runs" ]]; then
  echo 'Runs already exist in this folder. Use a new copy to preserve them.' >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
for seed in 1668 1669 1670; do
  "$python_bin" "$task_dir/cartpole.py" train --mode no_baseline --seed "$seed" --episodes 8000 --batch-size 32 --lr 0.005
  "$python_bin" "$task_dir/cartpole.py" evaluate --checkpoint "$task_dir/runs/no_baseline_seed$seed/last.pt"
done
"$python_bin" "$task_dir/plot_results.py"
