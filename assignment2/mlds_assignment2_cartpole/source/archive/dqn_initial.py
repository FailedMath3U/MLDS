"""Assignment 2: reproducible single-GPU DQN for the unmodified CartPole-v1.

Training/evaluation only; report generation belongs on the local computer.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

GAMMA = 0.95


class QNetwork(nn.Module):
    def __init__(self, inputs=4, actions=2, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, actions))

    def forward(self, state):
        return self.net(state)


class ReplayBuffer:
    """Uniform, seeded replay on the single selected GPU; retains terminal obs."""
    def __init__(self, capacity, inputs, device, seed):
        self.capacity, self.device = capacity, device
        self.size, self.position = 0, 0
        self.rng = np.random.default_rng(seed)
        self.states = torch.empty((capacity, inputs), device=device)
        self.next_states = torch.empty_like(self.states)
        self.actions = torch.empty(capacity, device=device, dtype=torch.long)
        self.rewards = torch.empty(capacity, device=device)
        self.terminated = torch.empty(capacity, device=device, dtype=torch.bool)

    def add(self, state, action, reward, next_state, terminated):
        i = self.position
        self.states[i].copy_(torch.as_tensor(state, device=self.device))
        self.next_states[i].copy_(torch.as_tensor(next_state, device=self.device))
        self.actions[i], self.rewards[i] = int(action), float(reward)
        self.terminated[i] = bool(terminated)
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch):
        idx = torch.as_tensor(self.rng.integers(self.size, size=batch), device=self.device)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.terminated[idx])


@torch.no_grad()
def bellman_targets(rewards, terminated, next_q, gamma=GAMMA):
    # Truncation is an external time limit, not physical failure. Bootstrap it.
    return rewards + gamma * (~terminated).to(rewards.dtype) * next_q


def update(online, target, optimizer, replay, batch, tau):
    s, a, r, ns, term = replay.sample(batch)
    q = online(s).gather(1, a[:, None]).squeeze(1)
    with torch.no_grad():
        y = bellman_targets(r, term, target(ns).max(1).values)
    loss = F.smooth_l1_loss(q, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    with torch.no_grad():
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.lerp_(op, tau)
    return float(loss.detach()), float((q.detach() - y).abs().mean())


def setup(seed, require_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if require_cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly ONE GPU with CUDA_VISIBLE_DEVICES=0.')
        torch.cuda.manual_seed_all(seed)
        return torch.device('cuda:0')
    return torch.device('cpu')


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def checkpoint(path, model, config, step, episode, dev):
    torch.save({'model': model.state_dict(), 'config': config, 'step': step,
                'episode': episode, 'dev': dev}, path)


def load_model(path, device):
    data = torch.load(path, map_location=device, weights_only=False)
    model = QNetwork().to(device)
    model.load_state_dict(data['model'])
    model.eval()
    return model, data


@torch.no_grad()
def rollout(model, seeds, device, batch_size=50, states=None, trace_first=False):
    """Independent seeded, greedy episodes. No learning or autoreset transitions.

    Batching inference is just a speed optimization: each environment's physics,
    reset seed, termination flags and reward remain independent and unchanged.
    Optional states are solely for out-of-distribution diagnostic experiments.
    """
    records, trace = [], []
    for begin in range(0, len(seeds), batch_size):
        chunk = seeds[begin:begin + batch_size]
        envs = [gym.make('CartPole-v1') for _ in chunk]
        obs = []
        for j, (env, seed) in enumerate(zip(envs, chunk)):
            s, _ = env.reset(seed=int(seed))
            if states is not None:
                s = np.asarray(states[begin + j], dtype=np.float32)
                env.unwrapped.state = s.astype(np.float64).copy()
            obs.append(s)
        obs = np.asarray(obs, dtype=np.float32)
        n = len(chunk)
        alive = np.ones(n, dtype=bool)
        returns, discounts = np.zeros(n), np.zeros(n)
        lengths = np.zeros(n, dtype=int)
        q_start = model(torch.as_tensor(obs, device=device)).max(1).values.cpu().numpy()
        ended_term, ended_trunc = np.zeros(n, bool), np.zeros(n, bool)
        for t in range(500):
            qs = model(torch.as_tensor(obs, device=device)).cpu().numpy()
            acts = qs.argmax(1)
            for j in np.flatnonzero(alive):
                before = obs[j].copy()
                ns, r, term, trunc, _ = envs[j].step(int(acts[j]))
                returns[j] += r
                discounts[j] += GAMMA ** t * r
                lengths[j] += 1
                if trace_first and begin == 0 and j == 0 and t < 10:
                    trace.append({'t': t, 'state': before.tolist(),
                                  'q_left': float(qs[j, 0]), 'q_right': float(qs[j, 1]),
                                  'action': int(acts[j]), 'reward': r,
                                  'next_state': ns.tolist(), 'terminated': bool(term),
                                  'truncated': bool(trunc)})
                obs[j] = ns
                if term or trunc:
                    alive[j] = False
                    ended_term[j], ended_trunc[j] = term, trunc
            if not alive.any():
                break
        for j, seed in enumerate(chunk):
            records.append({'index': begin + j, 'seed': int(seed),
                            'reward': float(returns[j]), 'length': int(lengths[j]),
                            'discounted_return': float(discounts[j]),
                            'initial_max_q': float(q_start[j]),
                            'terminated': bool(ended_term[j]),
                            'truncated': bool(ended_trunc[j])})
        for env in envs:
            env.close()
    return records, trace


def summarize(records):
    r = np.asarray([x['reward'] for x in records])
    return {'episodes': len(r), 'mean': float(r.mean()), 'std_population': float(r.std()),
            'std_sample': float(r.std(ddof=1)) if len(r) > 1 else 0.0,
            'min': float(r.min()), 'max': float(r.max()),
            'median': float(np.median(r)), 'successes_500': int((r == 500).sum()),
            'success_rate_500': float((r == 500).mean()),
            'physical_failures': sum(x['terminated'] for x in records),
            'time_limits': sum(x['truncated'] for x in records)}


def train(args):
    device = setup(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args))
    config.update(gamma=GAMMA, env='CartPole-v1', width=128, layers=[4,128,128,2],
                  device=str(device), gpu=torch.cuda.get_device_name(0),
                  visible_gpu_count=torch.cuda.device_count(), torch=torch.__version__,
                  numpy=np.__version__, gymnasium=gym.__version__,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dev_seed_start=1_000_000, dev_episodes=20,
                  bootstrap_on_time_limit=True, reward_shaping=False)
    save_json(out/'config.json', config)
    online, target = QNetwork().to(device), QNetwork().to(device)
    target.load_state_dict(online.state_dict())
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.capacity, 4, device, args.seed + 17)
    rng = np.random.default_rng(args.seed)
    env = gym.make('CartPole-v1')
    probe = np.asarray([env.reset(seed=3_000_000+i)[0] for i in range(64)])
    probe = torch.as_tensor(probe, device=device)
    dev_seeds = list(range(1_000_000,1_000_020))
    initial, _ = rollout(online, dev_seeds, device)
    initial_summary = summarize(initial)
    save_json(out/'initial_validation.json', initial_summary)
    checkpoint(out/'initial.pt', online, config, 0, 0, initial_summary)
    best_score = (-float('inf'), -float('inf'))
    dev_history = []
    steps, episodes, updates = 0, 0, 0
    last_loss, last_td = 0.0, 0.0
    next_validation = args.validate_every
    started = time.time()
    fields = ['episode','steps','reward','epsilon','q_max','q_mean','q_probe',
              'loss','td_abs','updates','terminated','truncated','wall_seconds']
    with (out/'training.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        while steps < args.steps:
            state, _ = env.reset(seed=args.seed * 100_000 + episodes)
            score, qs = 0.0, []
            term, trunc = False, False
            while not (term or trunc):
                epsilon = 1.0 - 0.98 * min(steps/args.epsilon_steps, 1.0)
                with torch.no_grad():
                    q = online(torch.as_tensor(state, device=device)).cpu().numpy()
                action = int(rng.integers(2)) if rng.random() < epsilon else int(q.argmax())
                ns, reward, term, trunc, _ = env.step(action)
                replay.add(state, action, reward, ns, term)
                state = ns
                score += reward
                qs.append(float(q.max()))
                steps += 1
                if steps >= args.warmup and steps % args.train_every == 0:
                    last_loss, last_td = update(online, target, optimizer, replay,
                                               args.batch, args.tau)
                    updates += 1
                if steps >= next_validation:
                    records, _ = rollout(online, dev_seeds, device)
                    summary = summarize(records)
                    summary.update(step=steps,episode=episodes+1)
                    dev_history.append(summary)
                    save_json(out/'validation_history.json', dev_history)
                    rank = (summary['mean'], summary['min'])
                    if rank > best_score:
                        best_score = rank
                        checkpoint(out/'best.pt', online, config, steps, episodes+1, summary)
                    print(json.dumps({'validation':summary,'elapsed':round(time.time()-started,1)}),flush=True)
                    next_validation += args.validate_every
            episodes += 1
            with torch.no_grad():
                qprobe = float(online(probe).max(1).values.mean())
            writer.writerow(dict(episode=episodes, steps=steps, reward=score, epsilon=epsilon,
                                 q_max=max(qs), q_mean=float(np.mean(qs)), q_probe=qprobe,
                                 loss=last_loss, td_abs=last_td, updates=updates,
                                 terminated=int(term), truncated=int(trunc),
                                 wall_seconds=time.time()-started))
            f.flush()
            if episodes % 25 == 0:
                print(f'episode={episodes} steps={steps} reward={score:.0f} '
                      f'epsilon={epsilon:.3f} q={qprobe:.3f}', flush=True)
    final, _ = rollout(online, dev_seeds, device)
    final_summary = summarize(final)
    checkpoint(out/'final.pt', online, config, steps, episodes, final_summary)
    if (final_summary['mean'],final_summary['min']) > best_score:
        checkpoint(out/'best.pt', online, config, steps, episodes, final_summary)
    save_json(out/'training_summary.json',dict(steps=steps,episodes=episodes,updates=updates,
                                             seconds=time.time()-started,final_validation=final_summary))
    env.close()
    print('TRAINING COMPLETE', json.dumps(final_summary),flush=True)


def evaluate(args):
    device = setup(1668)
    model, ck = load_model(args.checkpoint, device)
    # Refuse to silently overwrite a previous final evaluation.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    seeds = list(range(args.seed_start,args.seed_start+args.episodes))
    records, trace = rollout(model,seeds,device,trace_first=True)
    summary = summarize(records)
    summary.update(checkpoint=str(args.checkpoint),checkpoint_step=ck['step'],
                   checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                   seed_start=args.seed_start,greedy=True,gamma=GAMMA)
    save_json(out/'episodes.json',records)
    save_json(out/'summary.json',summary)
    save_json(out/'first_ten_steps.json',trace)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train')
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=1668)
    p.add_argument('--steps',type=int,default=200_000)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--capacity',type=int,default=100_000)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--tau',type=float,default=.005)
    p.add_argument('--warmup',type=int,default=2000)
    p.add_argument('--train-every',type=int,default=4)
    p.add_argument('--epsilon-steps',type=int,default=40_000)
    p.add_argument('--validate-every',type=int,default=10_000)
    p=sub.add_parser('evaluate')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--episodes',type=int,default=500)
    p.add_argument('--seed-start',type=int,default=9_000_000)
    args=parser.parse_args()
    if args.command=='train': train(args)
    else: evaluate(args)


if __name__=='__main__':
    main()
