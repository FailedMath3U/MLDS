# Assignment 1: CartPole and Pong report

`POLICY_GRADIENT_REPORT.pdf` is the nine-page report for policy-gradient Parts 1 and 2. `ASSIGNMENT1_FULL_REPORT.pdf` joins it to the revised four-page shuttle report for a single 13-page submission.

The report follows the assigned tasks: Part 1(a) CartPole, Part 1(b) Pong, and both repeated with explicit baselines. All eight required plots use final fixed-budget policies: raw training rewards plus 100-episode means, and histograms of 500 stochastic evaluations. GitHub identity, run instructions, model choices, quantitative validation and the three-month experiment plan are included.

## Rebuild without training

Keep this folder beside the original CartPole/Pong projects and the `cartpole_lr005`/`cartpole_lr002` projects. For the original Downloads layout, the builder also recognizes `cartpole_lr005_executed`.

```bash
python -m pip install -r requirements.txt
python build_assets.py
tectonic POLICY_GRADIENT_REPORT.tex
python combine_submission.py
```

Wait for the asset builder to finish before compiling. Alternatively use pdfLaTeX twice. The builder recomputes all main means and population SDs, audits all eighteen CartPole learning-rate runs, reads task validation and instability experiments, and generates numerical macros/tables and eight vector figures. Derived evidence is saved to `metrics.json`. Repository identity is read from `submission.json`.

The standalone LaTeX ZIP includes generated tables, macros and figures and compiles without Python. Rebuilding numerical assets needs the full repository.

## Validation evidence emphasized in this revision

- Task-specific signed boundary crossings, simultaneous failure/time-limit flags, shortest and unequal-length episodes, and positive/negative advantage signs: `../mlds_assignment1_cartpole/validate_task_edges.py` and `task_validation/`.
- Fixed-direction negative controls and targeted recovery conditions, explicitly evaluated greedily and kept separate from required stochastic tests.
- Exact original-run replay, paired frozen-checkpoint evaluations, and 2,000-episode fixed-policy controls for training dips.
- Controlled actor-rate comparisons at 0.01, 0.005 and 0.002, retaining all three seeds.
- Exact-gradient predictions, complete-game collection with actions 2/3, return calculation across Pong scoring events, and paired baseline-gradient variance.

The report distinguishes a local independent-fit variance benefit from variance reduction during actual training. It treats Pong as partially observed and makes no claim that FIRE is redundant. **Neither final Pong policy won an evaluation game.** Correctness controls do not establish successful task learning or adequate tuning.

All original training/evaluation artifacts were preserved. Reports are local and in the private GitHub repository, not copied to deepdish. The marker needs access to that repository.
