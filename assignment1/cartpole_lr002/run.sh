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
"$python_bin" "$task_dir/cartpole.py" suite --episodes 8000 --batch-size 32 --lr 0.002 | tee "$task_dir/suite.log"
"$python_bin" "$task_dir/validate.py" | tee "$task_dir/validation.log"
"$python_bin" "$task_dir/plot_results.py"
