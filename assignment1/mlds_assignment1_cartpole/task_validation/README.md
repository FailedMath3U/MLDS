# CartPole task edge cases cited in the report

Run `python validate_task_edges.py` from the parent folder. CPU only; writes this folder and checks hashes of the original source and primary actors.

- All 16 signed boundary/action trials match the Euler-dynamics prediction: equality with zero velocity remains nonterminal for that step; crossing ends the episode and still earns 1.
- Forced failure at a one-step time limit sets both end flags and earns one reward of 1.
- The original collector records controlled episodes of lengths 1, 1 and 3 with no post-end transitions. Their independent returns are in `mixed_length_trajectories.csv`; batched versus mean separate-episode gradient error is 1.863e-9.
- One-step positive/negative advantages move the isolated selected-action probability from 0.5 to approximately 0.505/0.495.

`results.json` contains all numerical records. No additional recovery experiments are part of this validator.
