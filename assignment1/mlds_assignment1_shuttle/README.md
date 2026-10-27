# Shuttle dispatch

`shuttle.py` implements the reported finite-horizon enumeration, infinite-horizon value iteration and policy iteration. The model uses capacity 15, dispatch cost 100, holding cost 2, uniform arrivals 1–5, queue cap 200 and gamma 0.95. The finite horizon is 500 decisions with zero terminal cost. Overflow is discarded; arrivals enter after service and holding cost.

```bash
python -m pip install -r requirements.txt
python shuttle.py
jupyter lab shuttle_plots.ipynb
```

Use a fresh copy for solver reproduction to preserve the recorded arrays. The solver runs its numerical/analytic checks and saves `results/summary.json`, `state_results.csv` and `shuttle_results.npz`. The notebook displays only the final submission's two cost curves, their negative-cost value curves and the optimal policy. `python build_notebook.py` rebuilds and executes it.

The final report and its standalone figure builder are in `../mlds_assignment1_submission/`. Further exhaustive transition and image checks are implemented there in `validate_shuttle_and_images.py`. No GPU is needed.
