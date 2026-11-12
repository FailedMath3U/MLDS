# MLDS

## Assignment 2

- **[Code, results, models, and reproduction instructions](assignment2/README.md)**
- [CartPole project](assignment2/mlds_assignment2_cartpole/README.md)
- [MsPacman project](assignment2/mlds_assignment2_mspacman/README.md)

| Environment | Mean original episode reward | Population SD | Evaluation episodes |
|---|---:|---:|---:|
| CartPole-v1 | 499.176 | 13.738305 | 500 |
| MsPacman-v5 | 2,033.58 | 486.316135 | 500 |

CartPole reached 500 steps in 497/500 games. MsPacman evaluation used 1% random
actions; all reported scores use original game points. Both models were selected
using separate validation seeds. The training code, saved models, numerical
evidence, and plotting code are included.

## Assignment 1

- **[Final report (PDF)](assignment1/mlds_assignment1_submission/ASSIGNMENT1_REPORT.pdf)**
- [Editable LaTeX source](assignment1/mlds_assignment1_submission/ASSIGNMENT1_REPORT.tex)
- [Build and validation instructions](assignment1/mlds_assignment1_submission/README.md)
- [Assignment checklist](assignment1/mlds_assignment1_submission/REQUIREMENTS_CHECKLIST.md)

| Report component | Source and recorded evidence |
|---|---|
| Shuttle enumeration, value iteration and policy iteration | [Shuttle project](assignment1/mlds_assignment1_shuttle/README.md) |
| CartPole basic and learned-baseline methods | [CartPole project](assignment1/mlds_assignment1_cartpole/README.md) |
| CartPole basic learning-rate comparisons | [0.005](assignment1/cartpole_lr005/README.md), [0.002](assignment1/cartpole_lr002/README.md) |
| Pong basic and constant-baseline methods | [Pong project](assignment1/mlds_assignment1_pong/RUN_NOTES.md) |
| Additional numerical validation | [Evidence map](assignment1/mlds_assignment1_submission/VALIDATION_NOTES.md) |

Each primary final policy was evaluated on 500 stochastic episodes:

| Environment | Basic mean ± population SD | Baseline mean ± population SD |
|---|---:|---:|
| CartPole-v1 | 500.000 ± 0.000 | 260.308 ± 57.013 |
| Pong-v5 | -18.610 ± 1.681 | -18.446 ± 1.851 |

CartPole used CPU only; Pong used one GPU. Both final Pong policies lost every evaluation game. The repository retains the final submission and the source, raw model results and numerical evidence supporting it. Use fresh source copies for training; numerical validators and plotting instructions are in the project READMEs.
