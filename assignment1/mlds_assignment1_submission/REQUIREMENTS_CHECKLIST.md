# Assignment checklist for the user-draft report

| Required item | Where addressed |
|---|---|
| GitHub username, repository name and link | Introduction |
| Shuttle MDP, assumptions and discount 0.95 | Shuttle Model |
| Finite T=500 enumeration and time-zero plot | Shuttle (a); Figure 1, with both cost and negative-cost value |
| Infinite-horizon value iteration and plot | Shuttle (b); Figure 2, with both cost and negative-cost value |
| Policy iteration and policy plot | Shuttle (c); Figure 3 |
| Shuttle edge cases and quantitative validation | Edge Cases and Validation and Correctness Tests; transition audit, independent known-answer changes and exhaustive small tree |
| PyTorch REINFORCE, Adam and stated learning rates | Policy Gradient overview and each game's results |
| CartPole-v1, discount 0.95 | CartPole Model |
| Pong-v5, visual input, actions 2/3, discount 0.99 | Pong Model/Policy; one-GPU setup stated |
| Episode rewards plus trailing 100-episode average | Top panels of Figures 4–7 |
| 500 evaluation episodes, histogram, mean and SD | Bottom panels of Figures 4–7 and corresponding results paragraphs |
| Repeat both games with an explained baseline | Policy Gradient with Baseline; CartPole and Pong subsections |
| Tests of baseline correctness and variance | Baseline Edge Cases, Correctness Checks and Validation; exact zero/unit-return controls and paired variance evidence |
| Quantitative evidence beyond format tests | Known-answer MDP changes; analytic expected gradients; frozen-policy episode-noise comparison; held-out baseline fits; actual full-game/end-condition checks |
| Three additional months, for each part | Shuttle, CartPole, Pong and baseline future-experiment subsections |
| How to run code and each validation | Running the Code; README and VALIDATION_NOTES.md give the evidence mapping |
| Source code provided | Full repository includes original implementations, added validators, report builders, plots and raw numerical evidence |

Main results retain the original policies. Neither final Pong policy won a test game; the report states this and does not equate implementation checks with successful task learning. Fixed-data baseline-mechanics probes and separate baseline-fitting studies are distinguished from original on-policy training.
