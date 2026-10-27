# Assignment 1: revised user draft

`ASSIGNMENT1_REPORT.tex` and `ASSIGNMENT1_REPORT.pdf` follow the supplied draft's section structure and wording. `USER_DRAFT.tex` preserves the submitted text; `DRAFT_CHANGES.diff` shows every edit. Changes are limited to factual/mathematical corrections, required figures and statistics, missing discussion sections, and execution/validation instructions. `revise_draft.py` reproduces the edits.

The report includes all required plots plus both **cost and reward value (negative cost)** for the finite- and infinite-horizon shuttle solutions. Main game plots use final policies, training seed 1668, complete training logs and 500 stochastic evaluations per model. Pong's failure to learn winning play is stated explicitly.

## Build the PDF without training

Keep this folder beside the existing assignment projects in `assignment1/`.

```bash
python -m pip install -r requirements.txt
python build_assets.py
python revise_draft.py
tectonic ASSIGNMENT1_REPORT.tex
```

The LaTeX ZIP includes generated inputs and vector figures. Select `ASSIGNMENT1_REPORT.tex` as the main file; no Python or sibling folders are needed merely to compile that package. Rerunning algorithms or validation requires the full repository.

## Added tests for claims in the draft

```bash
python validate_report_claims.py
python validate_shuttle_and_images.py
```

Both are CPU-only and write numerical results under `validation/`. Original training algorithms, checkpoints and result files remain read-only. Reports were not copied to deepdish; only the additional numerical validator was run there.

`validate_report_claims.py` imports the original CartPole/Pong implementations and checks:

- A zero baseline reproduces the no-baseline gradient on identical data in both implementations (maximum difference 0).
- With all rewards overridden to zero, a CartPole baseline fitted on 64 episodes predicts held-out zero targets with MSE 1.306e-6 after 400 fitting steps; Pong's actual EMA remains zero from zero and decays as predicted from a nonzero start.
- For one-step reward-1 episodes, the learned CartPole predictor approaches 1 (held-out mean 1.000019, MSE 6.542e-9); Pong's EMA approaches 0.999727 after 160 updates from zero.
- With an exact baseline 1 for those one-step returns, both actor-gradient norms are exactly zero. Exact action enumeration also checks that the no-baseline gradient has zero expectation despite nonzero individual samples.
- The actual Pong collector finishes a real normal game (1,008 decisions) and a separate externally truncated game (one decision). Its test pool rejects any attempt to step after ending; final return equals final reward in each case.

The EMA convergence probes deliberately reuse fixed synthetic data to isolate update mechanics. They are not new on-policy performance experiments. The constant-return-one condition applies to one-step episodes; rewards of 1 throughout a longer episode do not imply return 1.

`validate_shuttle_and_images.py` independently checks all 2,010 state/action/arrival combinations, all 402 transition-probability rows, underfilled service and capped queues. Synthetic moving pixels produce -1/+1 differences, a stationary paddle block cancels, and preprocessing leaves its inputs unchanged.

## Existing validation retained

The original shuttle `shuttle.py` runs its 14 analytic/DP/tree/simulation experiments. CartPole `validate.py`, `validate_task_edges.py` and `instability_experiment.py` provide the original analytic gradients, boundary tests, mixed-length collector test, variance experiment and frozen-policy controls. Pong `run_pong.py preflight` and saved `diagnostics/` supply point-boundary returns, restricted-action full games, sticky-action replay, GPU/recovery tests and variance probes. See `VALIDATION_NOTES.md` for the claim-to-evidence mapping.
