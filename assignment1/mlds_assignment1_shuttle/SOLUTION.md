# Shuttle dispatch solution

## MDP formulation and assumptions

Let S_t be the number of passengers waiting immediately before the dispatch decision,
with state space {0,...,200}. Action a=0 means wait; a=1 means dispatch one shuttle.
Only one shuttle can be dispatched per decision. A dispatched shuttle removes
min(S_t,15) passengers immediately, and is available again at the next decision.
An empty dispatch is permitted but incurs its full cost and is strictly dominated.
There is no shuttle fleet or travel-time state because none was specified.

Write L(s,a)=max(s-15a,0) for passengers remaining after dispatch. The stage cost is

    c(s,a) = 100a + 2 L(s,a).

Holding costs apply before the next arrivals: the newly arriving passengers first
incur holding cost at the next decision. Independent arrivals A_t are uniform on
{1,2,3,4,5}, independent of all earlier states and actions. The transition is

    S_(t+1) = min(L(S_t,a_t) + A_t, 200).

The cap discards overflow without a separate loss penalty. This is an explicit
modeling assumption, not a claim about an actual station. When several arrivals
map to state 200 their probabilities must be ADDED. The finite problem has 500
decisions t=0,...,499 and zero terminal cost at t=500. All methods use gamma=0.95.
Ties select wait. These timing, terminal, and cap choices should appear in the report.

We minimize expected discounted cost J. If the course defines value by reward
r=-c, then V=-J; the optimal actions are identical. Notebook plots use the cost
value convention and label it explicitly.

## (a) Finite horizon enumeration

With J_500(s)=0, enumerate both actions and five arrival outcomes in each state:

    Q_t(s,a) = c(s,a) + (0.95/5) sum_{k=1}^5 J_(t+1)(min(L(s,a)+k,200))
    J_t(s) = min_a Q_t(s,a)
    pi_t(s) = argmin_a Q_t(s,a).

Process t=499,...,0. This is exhaustive state/action enumeration through backward
induction, with O(T * 201 * 2 * 5) work, rather than exponentially enumerating
complete contingent action histories. The assignment's word "enumeration" is
ambiguous; this solution explicitly adopts finite-horizon backward induction.
The solver also validates it against a literal exhaustive decision/arrival tree
for a four-period, seven-state example.

The requested plot is J_0(s) for every state 0,...,200. The code saves ALL 501 value
vectors and ALL 500 policy vectors. The time-zero policy agrees with the stationary
infinite-horizon threshold; finite-horizon policies are generally time dependent.
In the last period, dispatch is never worthwhile: its cost 100 exceeds the maximum
immediate holding saving 2*15=30 and terminal cost is zero.

## (b) Value iteration

Start J^(0)=0 and repeat

    J^(n+1)(s) = min_a [c(s,a) + (0.95/5) sum_k J^(n)(min(L(s,a)+k,200))].

The Bellman operator is a contraction of modulus 0.95 in the maximum norm. Stop
when the Bellman residual of the returned vector is at most 1e-10, and recover its
greedy policy. Residual/(1-gamma) bounds maximum value error. This run takes 518
updates. `results/figures/b_value_iteration_value.pdf` is the requested value plot.

## (c) Policy iteration

Start with the always-wait policy. For each policy pi, construct its stochastic
transition matrix P_pi, then solve

    (I - 0.95 P_pi) J_pi = c_pi.

Improve by choosing the minimizing action in each state using J_pi. Terminate
when no actions change. This run takes six policy evaluation/improvement rounds,
including the final evaluation confirming stability. Both actions are considered
independently at every state; no threshold form is assumed or hard-coded.

The resulting optimal stationary policy is:

    pi*(s) = 0 (wait)      for s = 0,...,12
             1 (dispatch)  for s = 13,...,200.

The threshold being below capacity is reasonable: the discounted future holding
cost avoided can justify departing with 13 or 14 passengers. A purely immediate
cost comparison would miss this benefit. At high initial queues repeated dispatches
are necessary because only 15 passengers can depart per decision.

| Initial queue | Minimum infinite-horizon discounted cost |
|---|---:|
| 0 | 576.670167766 |
| 15 | 676.670167766 |
| 200 | 3821.586477184 |

The extra 100 at state 15 compared with state 0 follows directly: dispatching 15
leaves no one waiting, and both states have the same distribution of next arrivals.

## Edge cases and validation experiments

Run `python shuttle.py` to execute all 14 checks and recreate the numerical evidence.
Read `results/summary.json` for full precision, expected outcomes and measured errors.
This includes substantive algorithm experiments, beyond array-format checks:

1. **Optimality residual.** Substitute the policy-iteration solution into the
   minimizing Bellman equation. The maximum residual is 9.095e-13, implying a
   maximum cost error bound of approximately 1.819e-11 (up to floating-point error).
   This establishes near-optimality for the stated MDP, not merely policy stability.
2. **Independent algorithms.** VI and PI choose identical actions in all 201
   states. Their maximum value difference is 1.964e-9. The smallest absolute
   action-cost gap is 3.778, far above the numerical error level, so the dispatch
   decisions are numerically unambiguous.
3. **Finite versus infinite horizon.** The maximum difference between J_0 for
   T=500 and J* is 4.942e-9. With maximum stage cost 470, a conservative discounted
   tail bound is 0.95^500 * 470 / 0.05 = 6.838e-8. The measured difference is below
   this bound. Agreement is expected because the late costs are heavily discounted.
4. **Terminal condition.** All terminal costs are zero as specified.
5. **Last-period behavior.** Every last-period action is wait and the value equals
   the minimum immediate cost, confirming that future costs are not charged past T.
6. **Boundary dynamics.** Empty and under-capacity queues never become negative;
   15 passengers clear exactly and 16 leave one. At state 200, waiting maps every
   arrival to 200; dispatch maps the next queue to 186,...,190. This checks the
   physical boundary and saturation interpretation, including probability pooling.
7. **Zero holding cost.** With c_h=0, always wait has cost zero and is optimal;
   the algorithm returns exactly that result for all states.
8. **No service and no arrivals.** Set K=0 and A=0. Dispatch cannot help; analytic
   cost is 2s/(1-0.95). The numerical result agrees exactly in the recorded run.
9. **Free complete service.** Set dispatch cost zero, capacity 200 and A=0. All
   nonempty states dispatch and optimal cost is zero, as the algorithm confirms.
10. **Zero discount.** Set gamma=0. Only immediate costs matter, so waiting is
    optimal everywhere and J(s)=2s. The algorithm returns this expected solution.
11. **Exhaustive small problem.** A separately coded scalar decision/arrival tree
    for seven states and four periods matches vectorized backward induction exactly.
12.–14. **Independent Monte Carlo validation.** From each of states 0, 15 and 200,
    simulate 20,000 trajectories of length 300 with seed 1668. These transitions
    and costs are computed directly rather than reusing the solver's array builder.
    Compare the observed discounted costs with the exact policy-evaluation values.

| Initial state | Simulation mean | Exact cost | Standard error |
|---|---:|---:|---:|
| 0 | 576.671469 | 576.670168 | 0.219213 |
| 15 | 676.814669 | 676.670168 | 0.221520 |
| 200 | 3821.605190 | 3821.586477 | 0.532280 |

All errors fall within four standard errors plus a conservative truncation bound.
The rollout tail bound is about 0.001951. The notebook displays 1.96-standard-error
bars; they are approximate normal confidence intervals for the estimated MEAN,
not intervals for individual trajectory costs. Simulation is a consistency check;
Bellman residuals provide the stronger optimality evidence. The implementations
share an MDP specification, so algorithm agreement alone cannot detect every modeling
mistake; analytic experiments and independent scalar/simulation code address that.

## Additional three months of validation

- Sweep dispatch/holding costs, capacity, discount factor and arrival distributions;
  check policy changes and isolate when threshold policies cease to be appropriate.
- Compare larger caps (250, 500, 1000), capped/lost arrivals versus penalties for
  lost demand, and an uncapped approximation. Quantify sensitivity to the boundary.
- Compare finite horizons and nonzero terminal penalties; examine the full
  time-dependent policy near termination rather than only its time-zero slice.
- Compare with an independently constructed linear-programming solution, and use
  higher precision near parameter settings where action costs are nearly tied.
- Use multiple simulation seeds, more starting queues and explicit error/coverage
  studies. Simulate realistic bursty or correlated demand to assess robustness to
  misspecification rather than assuming uniform independent arrivals.
- Extend the state with shuttle availability/travel times and study whether the
  one-step reusable-shuttle simplification changes operational conclusions.

## Artifacts

`shuttle.py`: all algorithms and validation experiments.
`shuttle_plots.ipynb`: separately executed plotting notebook, with embedded outputs.
`results/state_results.csv`: all states, three cost solutions and optimal actions.
`results/shuttle_results.npz`: complete finite-horizon values/policies and histories.
`results/summary.json`: parameters, numerical results and validation evidence.
`results/figures/`: the three required figures plus two validation figures, each as
PNG and PDF. See `README.md` for environment and execution instructions.
