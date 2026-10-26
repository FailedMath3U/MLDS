# Controlled CartPole experiment: episode noise versus policy changes

This is a follow-up experiment; original training/evaluation data and the published reports were not overwritten. It uses CPU only.

## Protocol

1. Replay both original 8,000-episode training runs with seed 1668 and unchanged hyperparameters. Both reproduced every original reward and their final policy weights exactly.
2. Recover four actor checkpoints per method, selected from the already observed training curves. This checkpoint selection is exploratory, not a prospective comparison of algorithm performance.
3. Freeze each checkpoint and evaluate it on 500 fresh episodes with common reset seeds 9400000–9400499 and per-episode action RNG seeds 34000000–34000499. A separate sequential evaluator is used; no model updates occur.
4. Compare before/during mean scores using paired differences and 5,000 paired percentile-bootstrap replicates. Intervals are conditional on these fixed checkpoints and reset distribution, not on the population of training runs.
5. Evaluate each pre-dip checkpoint for another 2,000 fresh episodes with no learning, comparing its rolling 100-episode means to the original next 2,000 training episodes. The samples differ; rolling windows overlap, so their ranges are descriptive rather than independent trials or a p-value.

## Results

Each checkpoint mean below is from 500 new episodes. Mean change is during minus before; brackets give its paired 95% bootstrap interval.

| Method | Before checkpoint: mean | During checkpoint: mean | Mean change [95% CI] |
|---|---:|---:|---:|
| no_baseline | 5120: 453.108 | 6400: 83.114 | -369.994 [-377.430, -362.038] |
| baseline | 3072: 440.466 | 6144: 215.686 | -224.780 [-231.886, -217.436] |

The basic model's frozen 100-episode means ranged from 430.11 to 472.29, while continued training reached 74.64. The learned-baseline model's frozen means ranged from 413.00 to 453.82, while continued training reached 211.37.

## Interpretation

These selected dips reflect real changes in expected policy performance, beyond the episode noise measured under fixed policies. Sampling noise can still be an upstream cause of bad gradient updates. This experiment does not identify whether learning rate, gradient-estimator variance, baseline fitting, or another optimization mechanism caused the deterioration, and it does not establish how often this happens across training seeds.

## Files and reproduction

- `results.json`: full numerical results, intervals and replay checks.
- `plan.json`: seeds, sample sizes and selection disclosure, saved before new evaluations.
- `*_evaluation.csv`: every checkpoint evaluation outcome and its seeds.
- `*_frozen_control.csv`: every fixed-policy control outcome.
- `*_replay.json`: exact replay verification.
- `*.pt`: recovered fixed actor checkpoints.
- `controlled_comparison.png` / `.pdf`: plot comparing checkpoint quality and fixed-policy sampling noise.

From the parent CartPole folder, run `python instability_experiment.py`, then `python plot_instability_experiment.py`. The two original helpers are imported for exact replay; the new evaluator does not call the original collector. A fresh reproduction overwrites this experiment folder's outputs, not the original run folders.
