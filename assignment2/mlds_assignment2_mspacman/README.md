# Assignment 2 — MsPacman

## Recorded result

Training completed 5,000 full games, 3,777,202 decisions, and 939,301 updates in
about 3.03 hours. Validation selected the checkpoint saved after game 4,750.
The original 500-game evaluation used epsilon .01 and produced mean original
score **2,033.58**, population SD **486.316135**, median 1,960, and range 750–4,340.
The matched 500-game random baseline averaged 257.76. These results come from
one training seed and do not establish optimal play or robustness to retraining.

Selected model: `results/seed1668/best.pt`.
SHA-256: `5f3f9e59a5335c1a61519fd1d4d44848fa428ea278acd810ae7ad0bafe042fe3`.

## Implementation

The environment is `ALE/MsPacman-v5` with nine actions, default four-frame skip,
.25 sticky-action probability, and a 108,000-emulator-frame game limit. A life loss
does not reset the full game. Gamma is .99. Four preprocessed 88×80 frames feed
a CNN with 32/64/64 filters, a 512-unit dense layer, and dueling value/advantage
heads. Double DQN, Huber loss, Adam, and gradient norm clipping are used.

Replay stores 200,000 transitions; warmup is 20,000 decisions, batch size 32,
one update every four decisions, and a target copy every 10,000 decisions.
Epsilon decreases from 1 to .05 over 1,000,000 decisions; learning rate follows
a cosine schedule from .0001 to .000025 over training.

Training clips rewards to [-1,1]. All score plots and final evaluation statistics
use original arcade points. Clipping changes the learning objective. Four frames
were not compared with two in this experiment. Final evaluation uses 1% random
actions and is not fully greedy.

## Local plots and checks

```sh
python3 -m venv .venv-plots
.venv-plots/bin/python -m pip install -r requirements-plots.txt
.venv-plots/bin/python source/plot_results.py
python3 source/audit_artifacts.py
python3 source/mathematical_checks.py
```

The plotting script generates eight figures and numerical metrics from saved
data. The audit checks source/model hashes, score statistics, recorded action
choices, and plot files. The mathematical checker writes numerical bounds and
consistency results to `results/mathematical_checks.json`. None reruns gameplay.

## Reproduce computation using one Deepdish GPU

Use Python 3.12 and `requirements-compute.txt`, choosing a compatible CUDA build
of PyTorch. From a new compute directory containing the source files:

```sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python preflight.py --out preflight_repeat
python mspacman_dqn.py train --out runs/repeat --seed 1668 --episodes 5000
python mspacman_dqn.py evaluate --checkpoint runs/repeat/best.pt \
  --out evaluation_repeat --episodes 500 --seed-start 9000000 --epsilon .01
python postrun_checks.py --checkpoint runs/repeat/best.pt --out checks_repeat
```

The historical `source/run_nohup.sh` records the original remote paths and launch; adjust paths
and use fresh output directories for a new run. Its pipeline lock and training
commands protect existing outputs. For independent replication, use new training
and evaluation seeds. Saved replay is not included, so exact training resumption
is not promised.

## Files

- `source/`: training, preprocessing/replay checks, numerical diagnostics, and plots.
- `results/seed1668/`: training data, configuration, validation history, and models.
- `results/evaluation/`: original 500 trained-policy games and summary.
- `results/report_checks/`: matched random games and diagnostic images/trajectory.
- `results/preflight.json`: preliminary checks from the recorded computation.
- `figures/`: the eight figures, each as PNG and vector PDF.
