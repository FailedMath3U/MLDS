# Assignment specification audit

Source: the user-provided mlds_assignment1.pdf. This combined document covers policy-gradient Parts 1 and 2; the shuttle report is separate.

| Requirement | Evidence / report location |
|---|---|
| GitHub username, repository name and link | Front page; FailedMath3U / MLDS / https://github.com/FailedMath3U/MLDS |
| All code in GitHub | Repository includes all three assignment implementations, validation, plotting and report-generation source |
| PyTorch policy-gradient implementation | CartPole cartpole.py; Pong pong_core.py and run_pong.py |
| CartPole-v1, gamma 0.95 | Methods table and recorded config.json files |
| Pong-v5, gamma 0.99, frames as input | Methods, Pong metadata, preflight; ALE/Pong-v5 canonical namespaced ID |
| Pong actions restricted to 2 and 3 | Environment/worker assertions, preflight complete games, trace |
| Adam 0.01 or explained alternative | CartPole 0.01; Pong 0.001; recorded one-update KL comparison, no claim 0.01 was proven unstable |
| Training reward and 100-episode mean for each part | Four required training plots, complete logs, episodes start at 1 and mean at episode 100 |
| 500 evaluation episodes, histogram, mean and SD | Four required histograms; means/SDs recomputed from individual rewards |
| Part 2 repeats both environments with an explained baseline | Learned state/time baseline for CartPole; lagged constant EMA for Pong |
| At most one GPU on deepdish | Pong single UUID and one main process; CartPole CPU with zero GPUs |
| Edge cases per environment/method | Termination/truncation, reset leakage, preprocessing, point boundaries, baseline independence and clipping |
| Algorithm validation beyond format/unit tests | Exact toy gradients, predictable modified returns, random/untrained comparisons, paired frozen-policy gradient variances, behavior traces |
| Quantitative evidence and test instructions | Dedicated validation sections and reproducibility commands; JSON/CSV and model artifacts in repository |
| Additional three months of experiments | Separate CartPole, Pong, and baseline investigation plans |
| Labeled/formatted plots | Vector PDFs, units/score convention, captions, episode counts, moving-average window |

## Remaining submission actions

- Give the marker access to the private GitHub repository.
- Use ASSIGNMENT1_FULL_REPORT.pdf for a single PDF including the shuttle section, or use the policy-gradient-only PDF if submitting that section separately.
- Review the notation, modeling choices, and saved traces before submission.

The observed failure to learn winning Pong control is disclosed, not marked as successfully solving the task. Computational checks support specified algorithm behavior but cannot guarantee full marks or empirical optimality.
