"""Analyze saved MsPacman data using mathematical bounds; standard library only.

Run from any directory with python3 source/mathematical_checks.py. This writes
only new analysis artifacts, never changes checkpoints or reruns gameplay.
"""
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load(relative):
    return json.loads((ROOT / relative).read_text())


def huber(magnitude):
    return 0.5 * magnitude ** 2 if magnitude <= 1 else magnitude - 0.5


def main():
    config = load('results/seed1668/config.json')
    with (ROOT / 'results/seed1668/training.csv').open() as stream:
        training = list(csv.DictReader(stream))
    trace = load('results/report_checks/diagnostic_trajectory.json')
    batch = config['batch']
    gamma = config['gamma']
    epsilon = config['eval_epsilon']
    actions = config['actions']
    tolerance = 1e-6

    learning = [row for row in training if int(row['updates']) > 0]
    losses = []
    for row in learning:
        mean_error, loss = float(row['td_abs']), float(row['loss'])
        assert math.isfinite(mean_error) and math.isfinite(loss)
        assert mean_error >= 0 and loss >= 0
        lower = huber(mean_error)
        upper = huber(batch * mean_error) / batch
        losses.append(dict(episode=int(row['episode']), mean_absolute_error=mean_error,
                           loss=loss, lower=lower, upper=upper))
    loss_violations = [row for row in losses
                       if not row['lower'] - tolerance <= row['loss'] <= row['upper'] + tolerance]
    q_values = [q for row in trace for q in row['q_values']]
    assert all(math.isfinite(q) for q in q_values)
    implied_values = [sum(row['q_values']) / actions for row in trace]
    centered_advantages = [q - value for row, value in zip(trace, implied_values)
                           for q in row['q_values']]
    q_upper = 1 / (1 - gamma)
    centered_advantage_bound = (actions - 1) * q_upper / actions

    frame_rows = [(int(row['length']), int(row['emulator_frames'])) for row in training]
    collections = [load('results/evaluation/episodes.json'),
                   load('results/seed1668/initial_validation.json')['episodes']]
    collections.extend(item['episodes'] for item in load('results/seed1668/validation_history.json'))
    for rows in collections:
        frame_rows.extend((row['length'], row['emulator_frames']) for row in rows)
    frame_violations = [(length, frames) for length, frames in frame_rows
                        if not 4 * length - 3 <= frames <= 4 * length]

    ties = [sum(q == max(row['q_values']) for q in row['q_values']) for row in trace]
    random_count = sum(row['random_action'] for row in trace)
    non_greedy = sum(row['q_values'][row['action']] < max(row['q_values']) for row in trace)
    deterministic_errors = sum(not row['random_action'] and
                               row['q_values'][row['action']] < max(row['q_values']) for row in trace)

    # These are realized returns on one trajectory, not expected Q-values.
    rewards = [max(-1.0, min(1.0, row['reward'])) for row in trace]
    assert set(rewards) <= {0.0, 1.0}
    returns = [0.0] * len(trace)
    reward_counts = [0] * len(trace)
    return_violations = []
    future_return = 0.0
    future_events = 0
    for t in reversed(range(len(trace))):
        future_return = rewards[t] + gamma * future_return
        future_events += int(rewards[t] > 0)
        returns[t], reward_counts[t] = future_return, future_events
        upper = -math.expm1(future_events * math.log(gamma)) / (1 - gamma)
        lower = gamma ** (len(trace) - t - future_events) * upper
        if not lower - tolerance <= future_return <= upper + tolerance:
            return_violations.append(t)
    initial_upper = -math.expm1(reward_counts[0] * math.log(gamma)) / (1 - gamma)
    errors = [row['q_values'][row['action']] - actual for row, actual in zip(trace, returns)]
    gradient_norms = [float(row['gradient_norm']) for row in learning]

    result = dict(
        data_scope='Saved logs and one diagnostic game; no new training or gameplay.',
        huber_bounds=dict(status='PASS' if not loss_violations else 'FAIL', records=len(losses),
                          batch=batch, formula='h(m) <= mean(h(abs(error))) <= h(batch*m)/batch',
                          tolerance=tolerance, violations=loss_violations,
                          minimum_lower_slack=min(row['loss'] - row['lower'] for row in losses),
                          minimum_upper_slack=min(row['upper'] - row['loss'] for row in losses),
                          example=losses[-1]),
        q_lower_bound=dict(status='PASS' if min(q_values) >= 0 else 'FAIL',
                           assumption='All rewards in the task are nonnegative.',
                           recorded_reward_support=sorted(set(row['reward'] for row in trace)),
                           predictions=len(q_values), minimum=min(q_values), maximum=max(q_values),
                           violations=sum(q < 0 for q in q_values),
                           limitation='Only this diagnostic game has all nine predictions saved at every decision.'),
        dueling_implications=dict(status='PASS' if max(abs(a) for a in centered_advantages) <= centered_advantage_bound + tolerance else 'FAIL',
                                  assumption='True Q values are in [0,100] and the dueling head uses mean centering.',
                                  implied_value_minimum=min(implied_values), implied_value_maximum=max(implied_values),
                                  theoretical_centered_advantage_magnitude_bound=centered_advantage_bound,
                                  observed_centered_advantage_maximum_magnitude=max(abs(a) for a in centered_advantages),
                                  scope='Values and centered advantages are inferred from saved Q vectors. This is a consequence of the Q bounds, not an independent measurement of either output head.'),
        exploration=dict(status='DESCRIPTIVE', decisions=len(trace), epsilon=epsilon,
                         unique_greedy_action_at_every_recorded_state=all(k == 1 for k in ties),
                         conditional_non_greedy_probability_if_unique=epsilon * (actions - 1) / actions,
                         random_branch_count=random_count, non_greedy_count=non_greedy,
                         random_branch_greedy_count=random_count - non_greedy,
                         nonrandom_action_errors=deterministic_errors,
                         observed_greedy_fraction=1 - non_greedy / len(trace),
                         scope='The formula is conditional on each state. Counts fluctuate. Episode length is a stopping time, so no fixed-length binomial p-value is claimed.'),
        frame_bound=dict(status='PASS' if not frame_violations else 'FAIL', records=len(frame_rows),
                         formula='4*decisions - 3 <= emulator_frames <= 4*decisions',
                         violations=frame_violations,
                         observed_offsets=sorted(set(4 * length - frames for length, frames in frame_rows))),
        finite_event_return_bound=dict(status='PASS' if not return_violations else 'FAIL',
                                       scope='Realized discounted clipped returns, not network Q predictions.',
                                       formula='gamma**(H-K)*(1-gamma**K)/(1-gamma) <= G <= (1-gamma**K)/(1-gamma)',
                                       suffixes=len(returns), violations=return_violations,
                                       initial_events=reward_counts[0], initial_upper=initial_upper,
                                       initial_return=returns[0]),
        calibration_observation=dict(status='DESCRIPTIVE', initial_q=trace[0]['q_values'][trace[0]['action']],
                                     initial_realized_discounted_return=returns[0],
                                     mean_absolute_difference=sum(abs(e) for e in errors) / len(errors),
                                     mean_difference=sum(errors) / len(errors),
                                     scope='One stochastic trajectory, generated with epsilon=.01. This is not a test that Q must equal each realized return.'),
        gradient_logging=dict(maximum_logged_norm=max(gradient_norms), records_above_10=sum(n > 10 for n in gradient_norms),
                              meaning='The trainer records the norm returned by clip_grad_norm_, which is the norm before clipping.',
                              limitation='The post-clipping norm is not logged, so its bound cannot be independently checked from these records.'),
        input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
            'source/mspacman_dqn.py', 'results/seed1668/config.json', 'results/seed1668/training.csv',
            'results/report_checks/diagnostic_trajectory.json', 'results/evaluation/episodes.json',
            'results/seed1668/initial_validation.json', 'results/seed1668/validation_history.json']})
    output = ROOT / 'results/mathematical_checks.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    example = result['huber_bounds']['example']
    print(json.dumps({name: value for name, value in result.items() if name != 'input_sha256'}, indent=2))
    if any(isinstance(value, dict) and value.get('status') == 'FAIL' for value in result.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
