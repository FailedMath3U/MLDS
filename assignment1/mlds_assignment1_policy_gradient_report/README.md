# Combined CartPole and Pong report

`POLICY_GRADIENT_REPORT.pdf` is the combined report for Assignment 1's policy-gradient Parts 1 and 2. The shuttle section remains separately editable in `../mlds_assignment1_shuttle/SHUTTLE_REPORT.pdf`. `ASSIGNMENT1_FULL_REPORT.pdf` joins it with this report into one PDF for the full assignment.

The report contains eight required plots: training rewards with 100-episode moving averages, and 500-episode evaluation histograms, for both environments with and without baselines. Main figures use final fixed-budget checkpoints; The concise report also includes the controlled CartPole dip experiment and learning-rate comparisons at 0.01, 0.005 and 0.002. Auxiliary validation-selected Pong results remain in the saved experiment data.

## Build without training

With the experiment folders beside this folder:

```bash
python -m pip install numpy matplotlib
python build_assets.py
tectonic POLICY_GRADIENT_REPORT.tex
```

Alternatively run `pdflatex POLICY_GRADIENT_REPORT.tex` twice, after generating the assets. Wait for `build_assets.py` to finish before compiling: it writes all eight figure files.

`submission.json` contains the actual GitHub identity and repository URL. `build_assets.py` writes numerical macros, tables and metadata from recorded results, independently verifies per-episode means and population standard deviations, and regenerates figures. `metrics.json` preserves the derived values and source validation evidence.

The standalone LaTeX ZIP includes all generated tables, figures and metadata, so it compiles without Python or the experiment directories. To rebuild numerical assets, use the full repository.

## Scope and interpretation

CartPole's primary basic model reached 500 in all 500 evaluations, but other seeds varied. Pong training completed without either method learning to win: all 500 final-policy evaluations per method were losses. The report does not substitute job completion for task success or selectively omit unfavorable variance measurements.

The GitHub repository is private. Ensure the marker can access it before Canvas submission. Include the separate shuttle section when assembling the complete assignment submission.

To recreate the complete single-PDF assignment, install pypdf and run `python combine_submission.py`. This preserves vector figures and resolves internal named links before joining the two reports.
