# Supplementary CartPole task validation

Run `python validate_task_edges.py` from the parent CartPole project. It requires the original two primary final policies and writes only this output folder. Existing `cartpole.py`, `validate.py`, and primary actor hashes are checked before and after the experiment. CPU only; no retraining.

## Predictions and observed evidence

- **Signed physical boundaries:** test both actions at each exact signed cart/angle threshold, and crossing from just inside with outward velocity. The default Euler dynamics leave the exact zero-velocity boundary nonterminal for that step; crossing terminates, with terminal reward 1. All 16 trials matched. `results.json` records every initial and next state.
- **Simultaneous failure/time limit:** a boundary crossing with a one-step time limit sets both termination and truncation flags and awards exactly one reward of 1. Both flags were observed.
- **Mixed episode lengths:** temporarily substitute controlled real Gymnasium environments into the original collector (in-memory only). Episodes end after 1, 1 and 3 transitions. Their independent returns and the average of their separate actor gradients must match the batched calculation. Maximum component gradient difference was 1.863e-9. The one-step return was exactly 1. See `mixed_length_trajectories.csv`.
- **Both advantage signs:** for a sampled action with return 1, a two-logit Adam update at rate 0.01 raises its probability from 0.5 to 0.504999876 with baseline 0 and lowers it to 0.495000213 with baseline 2. This checks the actor-loss sign on positive and negative advantages.

## Task behavior: negative controls and stress starts

Frozen primary actors (seed 1668) use **greedy** actions, separately from the main report's stochastic 500-episode evaluations. Constant-left/right controls and both actors receive the same 64 reset seeds per condition (9800000–9800063).

- Default: all observations use Gymnasium's default seeded initialization.
- Tilted outward: pole angle ±0.15 rad and angular velocity ±0.5 rad/s, with alternating signs.
- Near track edge: cart position ±2.3 and velocity ±0.5 m/s, with alternating signs.
- Components not explicitly overwritten retain their default seeded values.

| Policy | Default mean | Tilted mean | Near-edge mean |
|---|---:|---:|---:|
| Always left | 9.344 | 9.500 | 7.750 |
| Always right | 9.375 | 9.500 | 7.781 |
| Basic, greedy | 469.766 | 488.438 | 8.703 |
| Learned baseline, greedy | 283.609 | 284.766 | 13.328 |

Constant-force controls fail quickly as expected. Trained actors improve default and tilted performance but perform poorly near the edge. This measures limits of these policies; it does not prove that every stress start is recoverable, establish robustness over the whole state space, or compare greedy and stochastic action selection on matched samples. Conditions were selected exploratorily after seeing the original results. All individual outcomes and flags are in `control_rollouts.csv`; summary SDs and scores-at-cap are in `results.json`.
