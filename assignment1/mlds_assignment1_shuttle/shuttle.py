"""Exact dynamic programming for Assignment 1's shuttle dispatch MDP.
Run: python shuttle.py. All outputs are relative to this file.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np


@dataclass(frozen=True)
class Model:
    capacity: int = 15
    dispatch_cost: float = 100.0
    holding_cost: float = 2.0
    max_queue: int = 200
    gamma: float = 0.95
    arrivals: tuple = (1, 2, 3, 4, 5)

    def arrays(self):
        if not (0 <= self.gamma < 1) or self.capacity < 0 or self.max_queue < 0:
            raise ValueError('Require 0 <= gamma < 1 and nonnegative capacity/queue cap.')
        if not self.arrivals or any(a < 0 or int(a) != a for a in self.arrivals):
            raise ValueError('Arrivals must be nonnegative integers with nonempty support.')
        states = np.arange(self.max_queue + 1)
        remaining = np.maximum(states[:, None] - self.capacity * np.arange(2), 0)
        costs = self.dispatch_cost * np.arange(2) + self.holding_cost * remaining
        next_states = np.minimum(remaining[:, :, None] + np.array(self.arrivals), self.max_queue)
        return states, remaining, costs, next_states


def action_costs(model, value):
    _, _, cost, nxt = model.arrays()
    return cost + model.gamma * value[nxt].mean(axis=2)


def finite_horizon(model, horizon=500):
    """Enumerate states/actions by backward induction; terminal cost is zero.

    values[t,s] is minimum expected cost from decision t, discounting relative to t.
    actions[t,s] is a time-dependent action; ties choose wait (action 0).
    """
    values = np.zeros((horizon + 1, model.max_queue + 1))
    actions = np.zeros((horizon, model.max_queue + 1), dtype=np.int64)
    for t in range(horizon - 1, -1, -1):
        q = action_costs(model, values[t + 1])
        actions[t] = q.argmin(axis=1)
        values[t] = q.min(axis=1)
    return values, actions


def value_iteration(model, tolerance=1e-10, max_iterations=20000):
    value = np.zeros(model.max_queue + 1)
    history = []
    for iteration in range(1, max_iterations + 1):
        updated = action_costs(model, value).min(axis=1)
        history.append(float(np.max(np.abs(updated - value))))
        value = updated
        residual = float(np.max(np.abs(action_costs(model, value).min(axis=1) - value)))
        if residual <= tolerance:
            return value, action_costs(model, value).argmin(axis=1), np.array(history), iteration
    raise RuntimeError('Value iteration failed to converge.')


def evaluate_policy(model, policy):
    """Solve (I - gamma P_pi) J_pi = c_pi exactly up to floating-point error."""
    states, _, costs, nxt = model.arrays()
    p = np.zeros((len(states), len(states)))
    for i in range(len(model.arrivals)):
        np.add.at(p, (states, nxt[states, policy, i]), 1 / len(model.arrivals))
    return np.linalg.solve(np.eye(len(states)) - model.gamma * p, costs[states, policy])


def policy_iteration(model, max_iterations=1000):
    policy = np.zeros(model.max_queue + 1, dtype=np.int64)
    changes = []
    for iteration in range(1, max_iterations + 1):
        value = evaluate_policy(model, policy)
        improved = action_costs(model, value).argmin(axis=1)
        changes.append(int(np.count_nonzero(improved != policy)))
        if np.array_equal(improved, policy):
            return value, policy, np.array(changes), iteration
        policy = improved
    raise RuntimeError('Policy iteration failed to converge.')


def validate(model, finite_values, finite_policy, vi_value, vi_policy, pi_value, pi_policy):
    """Algorithm-level validation, analytic special cases, and independent rollouts."""
    results = {}
    def check(name, passed, **evidence):
        results[name] = {'passed': bool(passed), **evidence}
        if not passed:
            raise AssertionError(f'{name}: {evidence}')

    q = action_costs(model, pi_value)
    residual = float(np.max(np.abs(q.min(axis=1) - pi_value)))
    check('bellman_optimality', residual < 1e-9, residual=residual,
          certified_max_value_error=residual / (1 - model.gamma))
    gap = float(np.max(np.abs(vi_value - pi_value)))
    check('vi_pi_agreement', gap < 1e-7 and np.array_equal(vi_policy, pi_policy),
          max_cost_difference=gap, differing_actions=int(np.count_nonzero(vi_policy != pi_policy)))
    max_stage_cost = float(model.arrays()[2].max())
    bound = model.gamma ** 500 * max_stage_cost / (1 - model.gamma)
    difference = float(np.max(np.abs(finite_values[0] - pi_value)))
    check('finite_infinite_horizon_agreement', difference <= bound + 1e-8,
          max_cost_difference=difference, theoretical_tail_bound=bound)
    check('finite_terminal_condition', np.all(finite_values[-1] == 0), terminal_cost=0)
    immediate = model.arrays()[2]
    check('last_period_is_myopic', np.allclose(finite_values[-2], immediate.min(axis=1)) and
          np.array_equal(finite_policy[-1], immediate.argmin(axis=1)),
          last_period_dispatch_states=np.flatnonzero(finite_policy[-1]).tolist())

    _, remaining, _, nxt = model.arrays()
    check('boundary_transitions', remaining[0, 1] == 0 and remaining[15, 1] == 0 and
          remaining[16, 1] == 1 and np.all(nxt[200, 0] == 200) and
          np.array_equal(nxt[200, 1], np.arange(186, 191)),
          empty_dispatch_next=nxt[0, 1].tolist(), full_queue_wait_next=nxt[200, 0].tolist(),
          full_queue_dispatch_next=nxt[200, 1].tolist())

    special = Model(holding_cost=0)
    v, p, _, _ = policy_iteration(special)
    check('zero_holding_cost', np.allclose(v, 0) and np.all(p == 0), max_cost=float(np.abs(v).max()))
    special = Model(capacity=0, arrivals=(0,))
    v, p, _, _ = policy_iteration(special)
    expected = 2 * np.arange(201) / (1 - special.gamma)
    check('no_service_no_arrivals', np.allclose(v, expected) and np.all(p == 0),
          max_analytic_error=float(np.abs(v - expected).max()))
    special = Model(dispatch_cost=0, capacity=200, arrivals=(0,))
    v, p, _, _ = policy_iteration(special)
    check('free_complete_service', np.allclose(v, 0) and np.all(p[1:] == 1), max_cost=float(np.abs(v).max()))
    special = Model(gamma=0)
    v, p, _, _ = value_iteration(special)
    check('zero_discount', np.allclose(v, special.arrays()[2].min(axis=1)) and np.all(p == 0),
          explanation='Saving at most 15*2=30 this period cannot justify a dispatch costing 100.')

    # Exhaustive decision/arrival tree using scalar arithmetic, independent of vectorized arrays.
    tiny = Model(capacity=2, dispatch_cost=3, holding_cost=1, max_queue=6, gamma=0.8, arrivals=(1, 2))
    def tree(state, steps):
        if steps == 0:
            return 0.0
        costs = []
        for action in (0, 1):
            left = max(state - action * tiny.capacity, 0)
            continuation = sum(tree(min(left + a, tiny.max_queue), steps - 1) for a in tiny.arrivals) / 2
            costs.append(action * tiny.dispatch_cost + tiny.holding_cost * left + tiny.gamma * continuation)
        return min(costs)
    tiny_values, _ = finite_horizon(tiny, 4)
    error = max(abs(tree(s, 4) - tiny_values[0, s]) for s in range(7))
    check('exhaustive_small_tree', error < 1e-12, horizon=4, states=7, max_error=float(error))

    # Monte Carlo transitions/costs coded directly rather than using model.arrays().
    rng = np.random.default_rng(1668)
    samples, horizon = 20000, 300
    rollouts = []
    for initial in (0, 15, 200):
        state = np.full(samples, initial, dtype=int)
        total = np.zeros(samples)
        for t in range(horizon):
            action = pi_policy[state]
            left = np.maximum(state - action * model.capacity, 0)
            total += model.gamma ** t * (action * model.dispatch_cost + model.holding_cost * left)
            state = np.minimum(left + rng.choice(model.arrivals, samples), model.max_queue)
        mean = float(total.mean())
        se = float(total.std(ddof=1) / np.sqrt(samples))
        tail = model.gamma ** horizon * max_stage_cost / (1 - model.gamma)
        rollouts.append({'initial_state': initial, 'sample_mean': mean, 'standard_error': se,
                         'exact_cost': float(pi_value[initial]), 'absolute_error': abs(mean-pi_value[initial])})
        check(f'rollout_state_{initial}', abs(mean - pi_value[initial]) <= 4 * se + tail,
              **rollouts[-1], samples=samples, horizon=horizon, acceptance='4 standard errors plus tail bound')
    return results


def main():
    out = Path(__file__).resolve().parent / 'results'
    out.mkdir(exist_ok=True)
    model = Model()
    finite_values, finite_policy = finite_horizon(model)
    vi_value, vi_policy, vi_history, vi_iterations = value_iteration(model)
    pi_value, pi_policy, pi_changes, pi_iterations = policy_iteration(model)
    checks = validate(model, finite_values, finite_policy, vi_value, vi_policy, pi_value, pi_policy)
    np.savez_compressed(out / 'shuttle_results.npz', states=np.arange(201),
        finite_costs=finite_values, finite_policy=finite_policy, vi_cost=vi_value, vi_policy=vi_policy,
        pi_cost=pi_value, pi_policy=pi_policy, vi_changes=vi_history, pi_changes=pi_changes)
    dispatch = np.flatnonzero(pi_policy)
    transitions = np.flatnonzero(np.diff(pi_policy)) + 1
    summary = {'model': asdict(model), 'horizon': 500, 'vi_iterations': vi_iterations,
        'pi_iterations': pi_iterations, 'policy_switch_states': transitions.tolist(),
        'dispatch_states': dispatch.tolist(), 'cost_at_0': float(pi_value[0]),
        'cost_at_15': float(pi_value[15]), 'cost_at_200': float(pi_value[200]),
        'minimum_action_gap': float(np.min(np.abs(np.diff(action_costs(model, pi_value), axis=1)))),
        'validation': checks}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savetxt(out / 'state_results.csv', np.column_stack((np.arange(201), finite_values[0], vi_value, pi_value, pi_policy)),
               delimiter=',', header='queue,finite_cost_t0,vi_cost,pi_cost,optimal_action', comments='',
               fmt=['%d','%.12f','%.12f','%.12f','%d'])
    print(json.dumps({k:v for k,v in summary.items() if k not in ('validation','dispatch_states')}, indent=2))
    print(f'All {len(checks)} validation experiments passed. Outputs: {out}')


if __name__ == '__main__':
    main()
