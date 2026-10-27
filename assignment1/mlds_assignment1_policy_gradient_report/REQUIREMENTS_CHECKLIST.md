# Assignment specification audit

Source: `mlds_assignment1.pdf`. The policy-gradient PDF is nine pages; the full PDF adds the revised four-page shuttle report.

| Assignment requirement | Concrete report evidence |
|---|---|
| GitHub username, repository name and link | Page 1: FailedMath3U / MLDS / repository URL |
| PyTorch basic policy gradient and discounted episodic objective | Section 1: objective, return-to-go, exact loss, batching and optimizer settings |
| CartPole-v1, discount 0.95 | Section 2 and saved configs |
| Pong-v5, frames, discount 0.99, actions 2/3 only | Section 3: namespaced ID, visual processing, partial observation and restrictions |
| Adam 0.01 or explained alternative | CartPole 0.01; Pong 0.001 with measured one-update comparison and limited interpretation |
| Training reward + 100-episode moving mean | Figures 1, 3, 5 and 7, generated from complete raw logs |
| 500 evaluations, histogram, mean and SD | Figures 2, 4, 6 and 8; final fixed-budget policies; stochastic action selection stated |
| Repeat both tasks with an explained baseline | Sections 4 and 5: learned state/time critic and prior-batch scalar EMA |
| One GPU on deepdish | Section 1 and execution instructions; CPU-only CartPole and one-GPU Pong |
| Edge cases and how handled | Section 6: signed physical crossings, both end flags, shortest/mixed episodes; Section 7: scoring/serving, sticky actions, signed differencing/reset |
| Quantitative validation beyond basic format checks | Fixed-direction controls, targeted recovery starts, frozen-policy noise experiment, exact-gradient predictions, paired baseline variance, learning-rate experiments |
| Baseline validation for Part 2 | Both advantage signs; exact expected-gradient invariance; independent-fit versus actual-training variance clearly distinguished |
| Directions explaining each validation experiment | Section 9: exact commands, purpose, output folders; source READMEs provide seed/sample details |
| Additional three months | Section 10: concrete questions and controlled comparisons for both games and baselines |
| All code used provided | Repository includes original algorithms, supplementary validators, plotting, all lower-rate suites and report builders |

## Limits stated in the report

- Neither Pong final policy won any of its 500 evaluation games.
- One Pong training seed cannot establish method superiority.
- A one-update KL comparison does not prove long-run learning-rate stability.
- Baseline variance reduction at an initial actor does not prove it throughout training; actual Pong EMA probes did not consistently reduce variance.
- CartPole stress checks use greedy actors and exploratory modified starts, separately from the required stochastic evaluations.
- Frozen-dip bootstrap intervals condition on selected checkpoints, not uncertainty across training seeds.

The private GitHub repository must be accessible to the marker. Use `ASSIGNMENT1_FULL_REPORT.pdf` for the full submission or the policy-gradient PDF for that section alone.
