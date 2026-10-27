# CartPole dip: policy change versus episode noise

This is the basic-policy experiment described in the report. Exact replay of seed 1668 matched all 8,000 rewards and final actor weights. Checkpoints after 5,120 and 6,400 episodes were frozen and evaluated on 500 fresh paired reset/action seeds. Their means were 453.108 and 83.114; the during-minus-before change was -369.994 with paired 95% bootstrap interval [-377.430, -362.038].

A further 2,000 fresh episodes at the earlier frozen policy gave trailing 100-episode means 430.11–472.29; continued training reached 74.64. Checkpoint selection was exploratory. Intervals condition on the selected policies, not variation across training seeds; overlapping rolling windows are descriptive.

`results.json`, `plan.json`, two checkpoint actors and per-episode CSVs retain the reported evidence. From the parent folder run `python instability_experiment.py`, then `python plot_instability_experiment.py`. Reproduction writes only this experiment folder and preserves original training files.
