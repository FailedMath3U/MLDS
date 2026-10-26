# CartPole actor learning rate 0.005

Completed on deepdish5, CPU only. Both methods use seeds 1668, 1669 and 1670, 8,000 training episodes each, and 500 final-policy evaluations. Actor rate is 0.005; critic rate stays 0.01. All other settings match the original experiment.

Results: `results/summary.csv`; plots: `results/figures/`; numerical checks: `validation/validation.json`. Run `bash run.sh` in a fresh copy without existing runs, using the Python environment specified by `CARTPOLE_PYTHON` (default: the original deepdish environment). Original files and earlier runs were preserved.
