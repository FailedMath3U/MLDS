# MLDS

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
