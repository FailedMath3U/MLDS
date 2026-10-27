# Assignment 1: shuttle dispatch

Algorithms are in `shuttle.py`; plots are in the separate, executed `shuttle_plots.ipynb`.
Read `SOLUTION.md` for the mathematical solution, assumptions and validation evidence.
No GPU or deepdish connection is needed for this problem.

## Run locally

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python shuttle.py
jupyter lab shuttle_plots.ipynb
```

Run all notebook cells to recreate the plots. The notebook loads saved results rather
than rerunning the algorithms. Open it from this folder or its parent Downloads folder.
Outputs are written beside the solver regardless of the shell working directory.

`python shuttle.py` runs all three solvers AND the 14 validation experiments. Failed
checks raise an error. Detailed numerical evidence is saved in `results/summary.json`.
The CSV contains all 201 states; the NPZ also stores the full finite-horizon policy.
PNG/PDF figures are under `results/figures/`.

For automated notebook regeneration/execution, run `python build_notebook.py`.
That development utility requires `nbformat`, `nbclient` and `ipykernel`, included
in requirements.txt. It uses temporary Jupyter directories under /tmp.

The assignment's final submission additionally requires a PDF report and a GitHub
repository link. This folder solves the shuttle section only. Review the assumptions,
especially the interpretation of finite-horizon enumeration, before submitting.

## Report and numerical audit

`SHUTTLE_REPORT.pdf` is the revised four-page report, organized around the MDP,
parts (a)–(c), task-specific edge cases, quantitative validation, and the additional
three-month experiment plan. It includes GitHub identity, run instructions, all
three required plots, and definitions for the notation.

To rebuild from saved results without running the solvers:

```bash
python make_report_figures.py
tectonic SHUTTLE_REPORT.tex
```

The figure builder audits the saved arrays and writes `report_metrics.tex` and
`results/report_metrics.json`. It independently evaluates the feasible policy
that dispatches only when at least 15 passengers are waiting. Its discounted
cost from an empty queue is 597.355975, versus 576.670168 for the optimal threshold
of 13. The report also shows the two action costs at states 12 and 13, explaining
where the optimal decision switches. Original solver source and result arrays
are preserved.

`SHUTTLE_LATEX_SOURCE.zip` is a standalone LaTeX package with generated inputs and
vector figures. The full-assignment PDF is in the sibling
`mlds_assignment1_policy_gradient_report` folder and now contains 13 pages
(four shuttle plus nine policy-gradient pages). Reports are not copied to deepdish.
