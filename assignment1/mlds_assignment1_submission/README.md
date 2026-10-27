# Assignment 1: final submission

`ASSIGNMENT1_REPORT.pdf` is the 12-page submission. `ASSIGNMENT1_REPORT.tex` is its editable source. It includes the required game plots and both cost and negative-cost value plots for the shuttle. All main game plots use final policies, seed 1668, complete training logs and 500 stochastic evaluations.

## Rebuild from saved raw results

Keep this folder beside the three experiment projects in `assignment1/`:

```bash
python -m pip install -r requirements.txt
python build_assets.py
tectonic ASSIGNMENT1_REPORT.tex
```

The builder reads the reported raw CSV/JSON/NPZ files directly and regenerates every figure and numerical macro. It has no dependency on a separate report folder. To compile a standalone LaTeX package, select `ASSIGNMENT1_REPORT.tex` as the main file; the package includes generated inputs and vector figures.

## Run the additional numerical checks

```bash
python validate_report_claims.py
python validate_shuttle_and_images.py
```

Both are CPU-only and save numerical evidence under `validation/`. They preserve the original algorithm sources and training results.

`validate_report_claims.py` checks zero-baseline gradient equivalence, zero/unit-return baseline fitting and gradient cancellation, prior-batch EMA use, and real Pong collection under termination and an external time limit. The fixed-data EMA probes test baseline mechanics, not new on-policy performance. Unit-return cancellation applies to one-step episodes; unit rewards in longer episodes have larger returns.

`validate_shuttle_and_images.py` checks all 2,010 shuttle state/action/arrival combinations, probability pooling at the cap, signed -1/+1 motion differences, cancellation of a stationary paddle block and non-mutation of raw frames.

The original reported validations live in `shuttle.py`, CartPole's `validate.py`/`validate_task_edges.py` and `instability_experiment.py`, and Pong's preflight and retained EMA diagnostic records. `VALIDATION_NOTES.md` maps report claims to these files. Detailed run instructions and remaining limitations are also in the report.
