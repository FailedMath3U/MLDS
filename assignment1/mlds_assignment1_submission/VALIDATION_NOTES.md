# Report claims and evidence

| Report claim | Existing or added evidence | How it checks the claim |
|---|---|---|
| Shuttle service and saturation | Existing boundary checks; added `validate_shuttle_and_images.py` | Independently computes every scalar transition/cost and pooled probability mass, across all 201 states, two actions and five arrivals |
| No continuation after decision 499 | Existing `finite_terminal_condition`, `last_period_is_myopic` | Terminal costs zero; final-period values/actions equal analytic immediate-cost minima; no policy exists at time 500 |
| VI and PI agreement | Existing saved arrays/summary | All 201 actions agree; max cost difference 1.964e-9, distinct from the PI residual 9.095e-13 |
| Finite/infinite approximation | Existing discounted-tail check | Observed difference 4.942e-9 below omitted-tail bound 6.838e-8 |
| Validation beyond AI confirmation | Existing zero-holding/zero-discount cases and independently coded exhaustive small tree | Changed problems have predetermined answers; tree equals finite DP exactly |
| CartPole failure, final +1, time limit | Existing original validation and `validate_task_edges.py` | Signed boundaries, strict threshold comparisons, simultaneous flags, terminal reward and actual mixed-length collector |
| Episodic gradient aggregation | Existing `validate_task_edges.py` | Batched gradient equals mean of per-episode gradients; max component error 1.863e-9 |
| Training dips beyond fixed-policy episode noise | Existing `instability_experiment/` | Exact replay; frozen-before/during 500-episode paired tests and 2,000-episode frozen controls; exploratory selection disclosed |
| Pong returns span scored points | Existing preflight | (0,1,0,-1) yields (0.019701,0.0199,-0.99,-1), rather than point-wise return resets |
| Pong ends on termination OR truncation | Added `validate_report_claims.py` | Uses actual collector and real ALE with natural termination and forced external TimeLimit; pool rejects post-end stepping |
| Zero first image difference; signed subtraction | Existing preflight plus new image test | Exact -1/+1 appearing/disappearing pixels; zero stationary block; no cross-game first-frame leakage |
| Restricted actions permit complete play | Existing eight-game preflight | 21–22 point events per game and natural ending using actions 2/3 only; does not establish FIRE redundancy |
| Sticky actions retained | Existing seeded emulator replay | Same seeds/fixed actions reproduce frame/reward hashes with probability 0.25; does not show a small effect or exactly one ignored frame |
| Zero baseline recovers REINFORCE | Added validator, both actors | Identical-data gradients compared directly; maximum difference 0 |
| Baseline can learn zero targets | Added validator | Held-out neural fit; actual EMA zero/decay checks; approximation versus exact cancellation distinguished |
| Baseline 1 cancels constant unit returns | Added validator, both actors | One-step task only; exact baseline yields zero gradients; separately measures approximate fitted predictors |
| Current outcomes do not set their own baseline | Code order; existing preflight; new EMA history assertions | Predictions computed before fitting; current actor uses previously stored baseline; detachment and statistical independence are distinct |
| Baseline variance benefit is limited | Existing paired probes | Initial CartPole fit reduces variance; later actual Pong EMA does not consistently help; unclipped estimator measured |

Numerical records for the two added validators are `validation/report_claims.json` and `validation/shuttle_and_images.json`. Their source/result hashes demonstrate preservation of the original files they inspect. Main performance results and training were not regenerated.
