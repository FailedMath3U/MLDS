"""REINFORCE and robust persistence. Imported only by the main GPU process."""
from pathlib import Path
import os
import json
import math
import time
import hashlib
import numpy as np
import torch
from torch import nn
from pong_env import ACTIONS

GAMMA = .99


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_save(path, value, previous=False):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    if previous and path.exists():
        os.replace(path, path.with_suffix('.previous.pt'))
    os.replace(tmp, path)


def returns(rewards, gamma=GAMMA):
    out = np.empty(len(rewards), dtype=np.float32)
    continuation = 0.
    for t in range(len(rewards)-1, -1, -1):
        continuation = float(rewards[t]) + gamma * continuation
        out[t] = continuation
    return out


def make_policy(seed, device):
    # Reinitialize to exactly the same actor weights for both methods.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        policy = nn.Sequential(nn.Linear(6400, 200), nn.ReLU(), nn.Linear(200, 2))
        nn.init.kaiming_uniform_(policy[0].weight, nonlinearity='relu')
        nn.init.zeros_(policy[0].bias)
        nn.init.orthogonal_(policy[2].weight, gain=.01)
        nn.init.zeros_(policy[2].bias)
    return policy.to(device)


def model_hash(policy):
    h = hashlib.sha256()
    for key, value in policy.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def collect(pool, policy, count, reset_base, action_base, start_index, device):
    """Full games, no point resets, no FIRE/NOOP actions, no rewards reshaped.

    Each episode has its own action RNG. Evaluation can resume at any episode
    boundary without changing that episode's random actions.
    """
    seeds = [reset_base + start_index + i for i in range(count)]
    xs = pool.reset(seeds)
    randoms = [np.random.default_rng(action_base + start_index + i) for i in range(count)]
    alive = set(range(count))
    episodes = [{'x': [], 'a': [], 'r': [], 'p2': [], 'terminated': False,
                 'truncated': False, 'reset_seed': seeds[i],
                 'action_seed': action_base + start_index + i} for i in range(count)]
    policy.eval()
    started = time.monotonic()
    with torch.no_grad():
        while alive:
            active = sorted(alive)
            batch = np.stack([xs[i] for i in active])
            probabilities = policy(torch.as_tensor(batch, device=device, dtype=torch.float32)).softmax(-1).cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise FloatingPointError('Nonfinite policy probabilities')
            actions = {}
            for row, i in enumerate(active):
                index = int(randoms[i].random() < float(probabilities[row, 1]))
                ep = episodes[i]
                ep['x'].append(xs[i]); ep['a'].append(index)
                ep['p2'].append(float(probabilities[row, 0]))
                actions[i] = ACTIONS[index]
            outputs = pool.step(actions)
            for i, (x, reward, terminated, truncated) in outputs.items():
                ep = episodes[i]
                ep['r'].append(reward)
                xs[i] = x
                if terminated or truncated:
                    ep['terminated'], ep['truncated'] = terminated, truncated
                    alive.remove(i)
                if len(ep['r']) > 27005:
                    raise RuntimeError('Pong exceeded its default 108000 emulator-frame cap.')
    for ep in episodes:
        ep['x'] = np.stack(ep['x'])
        ep['a'] = np.asarray(ep['a'], dtype=np.int64)
        ep['r'] = np.asarray(ep['r'], dtype=np.float32)
        ep['p2'] = np.asarray(ep['p2'], dtype=np.float32)
        ep['g'] = returns(ep['r'])
        ep['length'] = len(ep['r'])
        ep['score'] = float(ep['r'].sum())
        ep['discounted_return'] = float(ep['g'][0])
        if not (set(np.unique(ep['r'])) <= {-1., 0., 1.} and -21 <= ep['score'] <= 21):
            raise AssertionError('Unexpected Pong score/reward semantics')
    elapsed = time.monotonic() - started
    return episodes, {'seconds': elapsed, 'decisions': sum(ep['length'] for ep in episodes)}


def flat_arrays(episodes):
    x = np.concatenate([ep['x'] for ep in episodes])
    a = np.concatenate([ep['a'] for ep in episodes])
    g = np.concatenate([ep['g'] for ep in episodes])
    t = np.concatenate([np.arange(ep['length'], dtype=np.float64) for ep in episodes])
    d = np.power(GAMMA, t).astype(np.float32)
    return x, a, g, d


def actor_gradient(policy, episodes, baseline, device, chunk=512):
    """Accumulate a full on-policy batch gradient; no updates between chunks.

    This is the exact discounted episodic reward-to-go estimator, including
    gamma**t. No point-boundary reset, return centering, standardization,
    entropy bonus, or PPO surrogate is used.
    """
    x, a, g, d = flat_arrays(episodes)
    policy.zero_grad(set_to_none=True)
    loss_total = 0.
    for begin in range(0, len(a), chunk):
        end = begin + chunk
        weights = d[begin:end] * (g[begin:end] - baseline) / len(episodes)
        if not np.any(weights):
            continue
        xx = torch.as_tensor(x[begin:end], device=device, dtype=torch.float32)
        aa = torch.as_tensor(a[begin:end], device=device)
        ww = torch.as_tensor(weights, device=device)
        logp = policy(xx).log_softmax(-1).gather(1, aa[:,None]).squeeze(1)
        loss = -(logp * ww).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite actor loss')
        loss.backward()
        loss_total += float(loss.detach())
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters()):
        raise FloatingPointError('Nonfinite actor gradient')
    return loss_total, float(np.sum(d * g) / max(np.sum(d), 1e-12))


class Agent:
    def __init__(self, mode, config, device):
        self.mode, self.config, self.device = mode, config, device
        self.policy = make_policy(config['seed'], device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['lr'])
        self.baseline = 0.
        self.history = []
        self.updates = 0
        self.best_validation = -float('inf')
        self.best_validation_episodes = 0
        self.initial_hash = model_hash(self.policy)

    def update(self, episodes):
        # Frozen BEFORE rollout; only past completed batches determine baseline.
        b = self.baseline if self.mode == 'baseline' else 0.
        self.policy.train()
        loss, baseline_target = actor_gradient(self.policy, episodes, b, self.device, self.config['chunk'])
        norm = float(nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['clip_norm']))
        self.optimizer.step()
        if not all(torch.isfinite(p).all() for p in self.policy.parameters()):
            raise FloatingPointError('Nonfinite policy parameters after update')
        self.updates += 1
        if self.mode == 'baseline':
            alpha = self.config['baseline_ema_rate']
            self.baseline = (1-alpha)*b + alpha*baseline_target
        for ep in episodes:
            p = np.clip(ep['p2'], 1e-7, 1-1e-7)
            scored = np.flatnonzero(ep['r'])
            self.history.append({'episode': len(self.history)+1, 'update': self.updates,
                'score': ep['score'], 'length': ep['length'],
                'discounted_return': ep['discounted_return'],
                'first_point_step': int(scored[0]) if len(scored) else None,
                'points_won': int(np.sum(ep['r'] == 1)), 'points_lost': int(np.sum(ep['r'] == -1)),
                'terminated': ep['terminated'], 'truncated': ep['truncated'],
                'mean_entropy': float(np.mean(-p*np.log(p)-(1-p)*np.log(1-p))),
                'saturation_fraction': float(np.mean((p < .001) | (p > .999))),
                'actor_loss': loss, 'gradient_norm_before_clip': norm,
                'baseline_used': b, 'baseline_target': baseline_target,
                'baseline_after_update': self.baseline,
                'reset_seed': ep['reset_seed'], 'action_seed': ep['action_seed']})
        return {'loss': loss, 'gradient_norm': norm, 'baseline': b}

    def state(self):
        return {'mode': self.mode, 'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(), 'baseline': self.baseline,
            'history': self.history, 'updates': self.updates,
            'best_validation': self.best_validation,
            'best_validation_episodes': self.best_validation_episodes,
            'initial_hash': self.initial_hash}

    def restore(self, state):
        self.policy.load_state_dict(state['policy'])
        self.optimizer.load_state_dict(state['optimizer'])
        for key in ('baseline','history','updates','best_validation','best_validation_episodes','initial_hash'):
            setattr(self, key, state[key])


def episode_record(ep, index):
    return {'episode': index, 'score': ep['score'], 'length': ep['length'],
            'discounted_return': ep['discounted_return'],
            'terminated': ep['terminated'], 'truncated': ep['truncated'],
            'reset_seed': ep['reset_seed'], 'action_seed': ep['action_seed']}
