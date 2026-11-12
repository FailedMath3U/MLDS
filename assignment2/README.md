# Assignment 2 — Deep Q-Network

GitHub username: **FailedMath3U**. Repository: **MLDS**.
Public source: <https://github.com/FailedMath3U/MLDS/tree/main/assignment2>.

| Component | Location |
|---|---|
| CartPole training and diagnostics | [Instructions](mlds_assignment2_cartpole/README.md), [source](mlds_assignment2_cartpole/source/) |
| MsPacman training and diagnostics | [Instructions](mlds_assignment2_mspacman/README.md), [source](mlds_assignment2_mspacman/source/) |
| CartPole plots | [Figures](mlds_assignment2_cartpole/figures/) |
| MsPacman plots | [Figures](mlds_assignment2_mspacman/figures/) |

Both projects retain training logs, configurations, validation histories,
evaluated models, and the original 500-game evaluation records. The CartPole
code includes only the experiments discussed in the current local report. Its
original DQN source is needed for the algorithm-comparison plot.

## Regenerate plots locally

From this directory:

```sh
python3 -m venv .venv-plots
.venv-plots/bin/python -m pip install -r mlds_assignment2_mspacman/requirements-plots.txt
.venv-plots/bin/python mlds_assignment2_cartpole/source/plot_results.py
.venv-plots/bin/python mlds_assignment2_mspacman/source/plot_results.py
```

These commands read recorded data and generate plots without training or
repeating held-out evaluation. They contain no written-report renderer.

## Computation

Use the project READMEs for training arguments and diagnostics. Computation
requires `requirements-compute.txt` and exactly one visible CUDA GPU. Generate
figures locally, not on Deepdish. Use new output folders for repeat experiments.
Reusing saved evaluation seeds checks reproducibility; use new seeds for
independent statistical replication.

CartPole uses gamma .95 and raw +1 rewards per step. Final evaluation is greedy:
mean 499.176, population SD 13.738305, and 497/500 full-length games. MsPacman
uses gamma .99 and four input frames. Rewards are clipped only for learning;
score plots and final evaluation use original arcade points. Its mean score is
2,033.58 and population SD is 486.316135 over 500 games with 1% random actions.
Clipping changes the learning objective; the assignment does not explicitly
discuss this implementation choice.
