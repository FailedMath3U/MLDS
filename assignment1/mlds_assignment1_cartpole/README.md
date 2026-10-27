# CartPole-v1: reported experiments

The final report is `../mlds_assignment1_submission/ASSIGNMENT1_REPORT.pdf`.

`cartpole.py` implements the original CPU-only PyTorch REINFORCE and learned-baseline runs. Policy: 4–64 tanh–2 logits; gamma 0.95; Adam 0.01; batch 32 complete episodes; 8,000 episodes per seed. Baseline: separate state/time–64 tanh–1 predictor, fitted for five MSE steps after each actor update. Returns are unnormalized and retain outer discounting. Evaluation samples actions for 500 fresh episodes per final actor.

Use a fresh source copy for training:

```bash
python -m pip install -r requirements.txt
python cartpole.py suite --episodes 8000 --lr 0.01
python validate.py
python validate_task_edges.py
python instability_experiment.py
python plot_instability_experiment.py
jupyter lab cartpole_plots.ipynb
```

`runs/` retains all three seeds per method because their averages are discussed. `validation/` holds the analytic-gradient, return, terminal/truncation, random-reference, frozen-gradient-variance and ten-step trace evidence. `task_validation/` holds the reported boundary, mixed-length and advantage-sign checks. `instability_experiment/` contains the basic-policy before/during dip and frozen control.

The notebook plots saved data without PyTorch or training. `python build_notebook.py` rebuilds and executes it. Additional baseline/real-Pong checks and final report plots are generated from the submission folder. The core algorithm and original numerical records are preserved.
