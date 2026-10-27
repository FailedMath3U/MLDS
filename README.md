# MLDS

Machine Learning and Decision Science coursework.

## Assignment 1 — reinforcement learning

- **[Full assignment report, revised from the user draft (PDF)](assignment1/mlds_assignment1_submission/ASSIGNMENT1_REPORT.pdf)**
- [Full report LaTeX source](assignment1/mlds_assignment1_submission/ASSIGNMENT1_REPORT.tex)
- [Draft changes and added validation](assignment1/mlds_assignment1_submission/README.md)
- **[Combined CartPole + Pong report (PDF)](assignment1/mlds_assignment1_policy_gradient_report/POLICY_GRADIENT_REPORT.pdf)**
- [Combined report LaTeX source](assignment1/mlds_assignment1_policy_gradient_report/POLICY_GRADIENT_REPORT.tex)
- [Specification checklist](assignment1/mlds_assignment1_submission/REQUIREMENTS_CHECKLIST.md)
- [Shuttle report (PDF)](assignment1/mlds_assignment1_shuttle/SHUTTLE_REPORT.pdf)

The combined report covers basic policy gradient and baseline experiments for **both** CartPole-v1 and Pong-v5. The shuttle report remains separately editable; the full-assignment PDF joins both sections into one file for Canvas.

| Project | Code and documentation |
|---|---|
| Shuttle dispatch | [Solvers, 14 validation experiments, notebook and results](assignment1/mlds_assignment1_shuttle/README.md) |
| CartPole-v1 | [PyTorch REINFORCE, learned baseline, three seeds and evaluations](assignment1/mlds_assignment1_cartpole/README.md) |
| Pong-v5 | [Single-GPU training, constant EMA baseline, preflight and recovery protocol](assignment1/mlds_assignment1_pong/RUN_NOTES.md) |
| Combined report | [Rebuild instructions, audited metrics and figures](assignment1/mlds_assignment1_policy_gradient_report/README.md) |

### Recorded outcomes

Each primary final policy was evaluated with stochastic actions for 500 complete episodes. Values are mean ± population standard deviation of undiscounted reward.

| Environment | Basic policy gradient | With baseline |
|---|---:|---:|
| CartPole-v1, seed 1668 | 500.000 ± 0.000 | 260.308 ± 57.013 |
| Pong-v5, seed 1668 | -18.610 ± 1.681 | -18.446 ± 1.851 |

Other CartPole seeds are included. **Neither Pong policy learned to win:** all 500 final-policy evaluation games per method were losses. Completed training, validation passes, and reduced gradient variance at some policies do not establish task success or universal variance improvement.

### Reproducibility

All algorithm, validation, notebook, plotting and report-generation code is included. Saved configurations specify the actual software and hyperparameters. CSV/JSON files retain individual training/evaluation outcomes and diagnostic evidence, and final/initial policy files are provided. Large rolling optimizer checkpoints and redundant ZIP archives are omitted; they remain in the original deepdish experiment directory.

CartPole used CPU only. Pong used **one physical GPU**, with both methods alternating batches in a single process. Refer to each project's instructions before retraining: original run folders are preserved and can be overwritten by reproduction commands.

The repository is private. The assignment marker needs repository access to review the required code.
