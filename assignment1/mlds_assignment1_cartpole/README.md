# Assignment 1: CartPole-v1

This folder completes the CartPole portion of **Part 1 (basic policy gradient)**
and **Part 2 (policy gradient with a baseline)**. Pong is not included.

- `cartpole.py`: PyTorch REINFORCE, training and 500-episode stochastic evaluation.
- `validate.py`: analytic gradients, predictable environment experiments, frozen-policy gradient variance, random reference and ten-step trace.
- `cartpole_plots.ipynb`: separate, executed plotting notebook reading saved results.
- `CARTPOLE_REPORT.tex` / `.pdf`: report with notation, methods, results and validation.
- `runs/`: all six final/initial checkpoints, configurations, 8,000-episode training logs and evaluation rewards.
- `validation/`: numerical validation evidence, paired gradients and the ten-decision trace.
- `results/figures/`: PNG and vector PDF plots.
- `pilot_runs/`: archived 1,024-episode functionality pilot, separate from the full experiments.

## Results

Each row is a final model evaluated over 500 fresh episodes using **sampled actions**.
The reported standard deviation is the population SD of those 500 scores.
The primary seed, 1668, was fixed before the full runs; every seed is reported.

| Method | Seed | Mean | SD |
|---|---:|---:|---:|
| No baseline | 1668 | 500.000 | 0.000 |
| No baseline | 1669 | 271.824 | 28.856 |
| No baseline | 1670 | 483.886 | 65.298 |
| Learned baseline | 1668 | 260.308 | 57.013 |
| Learned baseline | 1669 | 360.516 | 79.141 |
| Learned baseline | 1670 | 349.940 | 76.321 |

Uniform random policy: mean 22.782, SD 12.058 (500 episodes).
At one frozen initial policy, a learned baseline reduced the trace of the sample
gradient covariance from 5.338262 to 0.842421, a reduction of 84.22%.
This does not guarantee better training: the baseline method had lower final
scores in these runs. Three seeds are insufficient for broad statistical claims.

## Environment and GPU usage

Experiments ran on **deepdish5, CPU only, zero GPUs**. The code clears
`CUDA_VISIBLE_DEVICES` and creates all tensors/models on CPU. Thus it cannot use
multiple GPUs. One CPU thread was used for PyTorch and BLAS.

The exact environment ID is **CartPole-v1**, using Gymnasium 1.2.2 (the maintained
Gym successor), default +1 rewards and 500-step limit. No custom dynamics, reward
shaping or alternative version is substituted. The original assignment calls
this OpenAI Gym; the library choice is disclosed in the report.

Observed software: PyTorch 2.14.0+cu130, NumPy 2.5.3, Python 3.12, Gymnasium 1.2.2.
The CUDA-enabled PyTorch installation was reused, but no CUDA device was used.
Each run's `config.json` contains the actual versions and hyperparameters.

## Reproduce on deepdish5

The environment and source are already present:

```bash
cd ~/mlds_assignment1_cartpole
source ~/mlds_assignment1_cartpole_env/bin/activate
python cartpole.py suite --episodes 8000 --lr 0.01
python validate.py
```

This regenerates all six training/evaluation runs and overwrites their output
files. Preserve `runs/` first if you want to retain the original experiments.

To run only the primary basic method and evaluate it:

```bash
python cartpole.py train --mode no_baseline --seed 1668 --episodes 8000
python cartpole.py evaluate --checkpoint runs/no_baseline_seed1668/last.pt
```

Use `--mode baseline` for the baseline experiment. A suite uses seeds 1668–1670.
Training resets use seed*100000 + episode index. Evaluation resets use
9000000–9000499. Evaluation does not update model weights and is not used to
select a checkpoint. Always evaluate the final checkpoint from the fixed budget.

## Run locally / create plots

In a Python virtual environment with PyTorch available:

```bash
python -m pip install -r requirements.txt
jupyter lab cartpole_plots.ipynb
```

Run all cells. Plotting needs only NumPy, Matplotlib and Jupyter; it does not need
PyTorch or Gymnasium when using the saved results. For automatic regeneration:

```bash
python build_notebook.py
```

That utility additionally uses nbformat, nbclient and ipykernel (listed in the
requirements). It creates temporary Jupyter directories under /tmp.

Compile the report with `tectonic CARTPOLE_REPORT.tex`, or run
`pdflatex CARTPOLE_REPORT.tex` twice. The source ZIP includes all report figures
and is suitable for Overleaf. The notebook's plots use undiscounted episode scores;
the training objective uses discounted reward with gamma=0.95.

## Objective and baseline

Policy: 4 inputs → 64 tanh units → 2 categorical logits. Adam lr=0.01; batch=32
complete episodes; 8,000 episodes (250 updates); gamma=0.95. The loss uses the
exact discounted reward-to-go estimator, including the outer gamma**t factor.
Both methods use a fixed scale 1-gamma and gradient norm cap 10. No gradients
actually reached that cap in the recorded runs. Returns/advantages are not
normalized, so the basic method does not hide a baseline in normalization.

The baseline is a separate 5 → 64 tanh → 1 network, taking the four state
components and elapsed time/500. Its prediction is detached and calculated
before fitting on the current batch. Five critic MSE/Adam steps at lr=0.01
follow each actor update. Targets are full Monte Carlo reward-to-go, with no
TD bootstrapping. Both termination and truncation finish an episode.

## Validation experiments

`python validate.py` checks discounted returns against a hand calculation,
enumerates an exactly solvable two-step MDP to check the expected gradient and
baseline invariance, verifies that a rewarded action increases in probability,
replays deterministic environment dynamics with fixed seeds/actions, and probes
cart/pole failure boundaries and a one-step TimeLimit.

The frozen-policy experiment fits a critic on 512 trajectories and estimates
paired gradients on 512 different trajectories. It measures the sum of
per-parameter sample gradient variances, not episode-score variance. Its measured
outcome is recorded rather than forced to pass a predetermined improvement test.

`validation/first_ten_steps.csv` exposes all state components, right-action
probability, sampled action, reward and end flags for ten decisions. Use it to
inspect behavior and make calculations or predictions of your own.
