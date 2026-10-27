# Pong-v5: reported training and validation

The final report is `../mlds_assignment1_submission/ASSIGNMENT1_REPORT.pdf`.

The original training implementation is retained: `pong_env.py`, `pong_core.py`, `run_pong.py`, `config.json` and `supervise.sh`. It uses ALE/Pong-v5, only actions 2/3, binary frame differences, full-game Monte Carlo returns with gamma 0.99, a 6400–200 ReLU–2 actor, Adam 0.001, batch eight and a gradient-norm cap of five. Part 2 uses the previous-batch scalar EMA. One GPU hosts both actors; CPU workers emulate games.

Use a fresh source-only directory and a Python environment with `requirements.txt` installed. Set a single GPU UUID consistently in the config, supervisor and `CUDA_VISIBLE_DEVICES`, and configure the supervisor's Python path. Run:

```bash
python run_pong.py preflight
nohup setsid bash supervise.sh > nohup.log 2>&1 < /dev/null &
```

Preflight checks hand-computed returns across points, analytic gradients, action restrictions, signed/reset image differences, seeded sticky-action replay, matched initial actors, single-GPU execution and optimizer recovery. The original code/source hashes remain valid. Full runs resume atomic checkpoints; see the source for persistence behavior.

Retained numerical evidence:

- `runs/*/training.csv`, `initial.pt`, `final.pt`, and `evaluation/test_final*`: the two reported 20,000-game runs and 500 final-policy evaluations.
- `preflight/results.json` and `initial_gradient_variance.json`: the preflight and initial candidate-baseline variance study.
- `references/untrained.json` and its rewards CSV: the 32-game reference.
- `diagnostics/baseline_*.json`: the four actual EMA variance probes described in the report.

The original runner internally performs auxiliary checkpoint validation as part of its training workflow. Its saved auxiliary selections and evaluations are not included in this submission. Only the final policies supply the reported results. `plot_results.py` plots final-policy results only; final report figures are rebuilt from raw data by the submission's builder.
