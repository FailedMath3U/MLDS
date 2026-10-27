# CartPole actor learning rate 0.002

These are the three basic-REINFORCE runs cited in the final report: seeds 1668, 1669 and 1670, 8,000 training episodes each and 500 stochastic final-policy evaluations. They used CPU only; all settings except actor rate match the original basic method.

Raw logs, configs and actors are in `runs/`. Evaluation means are in `results/summary.csv`. Run `bash run.sh` in a source-only copy without existing runs; set `CARTPOLE_PYTHON` to a Python executable with the project dependencies. The launcher refuses to overwrite saved runs and trains only the reported no-baseline method.
