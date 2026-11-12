# Assignment 2 — CartPole

## Recorded results

The selected model is `results/runs/symmetric_ddqn_seed1668/best.pt`, saved at
190,000 training steps. A single frozen, greedy evaluation on 500 separate seeds
produced mean reward **499.176**, population standard deviation **13.738305**, and
**497/500** episodes lasting 500 steps. Failures lasted 462, 205, and 421 steps.
The original evaluation files and weights are retained unchanged.

The network is 4 → 128 ReLU → 128 ReLU → 2 linear outputs, with symmetry
averaging, Double DQN targets, gamma .95, uniform replay, Huber loss, Adam, and
an EMA target network. Exact settings and training logs are in each run folder.
The comparison changed symmetry, Double DQN, and update frequency together.

## Code

| File | Purpose |
|---|---|
| `source/dqn.py` | Final-network training, validation, checkpoint selection, evaluation, and first-ten-step data. |
| `source/archive/dqn_initial.py` | Original DQN run used in the comparison plot. |
| `source/validate.py` | Initial-state stress grid, action maps, and a successful game trace. |
| `source/analyze_failures.py` | Reproduce the three recorded failures using the original evaluation batch arrangement. |
| `source/plot_results.py` | Generate all nine plots from recorded results. |

## Regenerate plots locally

```sh
python3 -m venv .venv-plots
.venv-plots/bin/python -m pip install -r requirements-plots.txt
.venv-plots/bin/python source/plot_results.py
```

Plots cover training rewards, final reward histogram, validation, initial-state
stress, action margins, a separate green trajectory, winning/failing pole angles,
algorithm comparison, and maximum Q-values. The first-ten-step data are in
`results/evaluation/first_ten_steps.json`. No model or GPU is needed for plotting.

## Reproduce computation with one Deepdish GPU

Install `requirements-compute.txt` with the appropriate CUDA build of PyTorch.
The recorded run used Python 3.12.6 and one visible CUDA device. From a fresh
compute directory containing the files in `source/`:

```sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python dqn.py train --out runs/repeat --seed 1668 --steps 200000 \
  --lr 0.0003 --train-every 1 --symmetry --double-dqn
python dqn.py evaluate --checkpoint runs/repeat/best.pt \
  --out evaluation_repeat --episodes 500 --seed-start 9000000
python validate.py --checkpoint runs/repeat/best.pt --out validation_repeat
python analyze_failures.py --checkpoint runs/repeat/best.pt --out failures_repeat
```

The final command diagnoses the original three failures; it is not a new held-out
score. Expected failure lengths apply to the supplied checkpoint. Fresh training
can produce different weights/results. Use new output folders, and new training
and evaluation seeds for independent statistical replication.

To reproduce the original comparison, copy `source/archive/dqn_initial.py` into
the compute directory and run:

```sh
python dqn_initial.py train --out runs/original_repeat --seed 1668 --steps 200000
```

## Data

- `results/runs/`: both training runs, configuration, histories, and checkpoints.
- `results/evaluation/`: original 500-game evaluation, summary, and first ten decisions.
- `results/validation/`: phase maps, successful game trace, and 81-state stress grid.
- `results/failure_analysis/`: three DQN failure trajectories.
- `results/selected_model.json`: selected checkpoint and SHA-256.
- `figures/`: plots as PDF and PNG.
