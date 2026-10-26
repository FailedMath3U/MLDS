"""PyTorch REINFORCE for CartPole-v1, with optional learned state/time baseline.
CPU only: CUDA_VISIBLE_DEVICES is cleared before importing torch.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import argparse
import csv
import json
import time
import random
import numpy as np
import torch
from torch import nn
import gymnasium as gym

ROOT = Path(__file__).resolve().parent
GAMMA = 0.95


def network(inputs=4, outputs=2):
    return nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, outputs))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def returns(rewards, gamma=GAMMA):
    out = np.empty(len(rewards), dtype=np.float32)
    total = 0.0
    for t in range(len(rewards)-1, -1, -1):
        total = rewards[t] + gamma * total
        out[t] = total
    return out


def collect(policy, count, seed_start):
    """Independent real Gymnasium environments; batched NN inference, no gradients.

    Each environment contributes one complete episode. End on either physical
    termination or the required 500-step time limit. No bootstrapping is used.
    """
    envs = [gym.make('CartPole-v1') for _ in range(count)]
    states = np.stack([e.reset(seed=seed_start+i)[0] for i,e in enumerate(envs)])
    active = np.ones(count, dtype=bool)
    episodes = [{'states': [], 'actions': [], 'rewards': [], 'terminated': False,
                 'truncated': False} for _ in range(count)]
    with torch.no_grad():
        while active.any():
            logits = policy(torch.from_numpy(states))
            actions = torch.distributions.Categorical(logits=logits).sample().numpy()
            for i in np.flatnonzero(active):
                ep = episodes[i]
                ep['states'].append(states[i].copy())
                ep['actions'].append(int(actions[i]))
                nxt, reward, terminated, truncated, _ = envs[i].step(int(actions[i]))
                ep['rewards'].append(float(reward))
                states[i] = nxt
                if terminated or truncated:
                    active[i] = False
                    ep['terminated'], ep['truncated'] = bool(terminated), bool(truncated)
    for e in envs:
        e.close()
    for ep in episodes:
        ep['states'] = np.asarray(ep['states'], dtype=np.float32)
        ep['actions'] = np.asarray(ep['actions'], dtype=np.int64)
        ep['returns'] = returns(ep['rewards'])
    return episodes


def packed(episodes):
    lengths = [len(ep['rewards']) for ep in episodes]
    states = torch.from_numpy(np.concatenate([ep['states'] for ep in episodes]))
    actions = torch.from_numpy(np.concatenate([ep['actions'] for ep in episodes]))
    targets = torch.from_numpy(np.concatenate([ep['returns'] for ep in episodes]))
    times = torch.from_numpy(np.concatenate([np.arange(n,dtype=np.float32) for n in lengths]))
    critic_inputs = torch.cat([states, times[:,None]/500], dim=1)
    discount = GAMMA ** times
    return states, actions, targets, times, critic_inputs, discount


def train(mode, seed, episodes=4000, batch_size=32, lr=0.01, directory=None):
    set_seed(seed)
    directory = Path(directory or ROOT/'runs'/f'{mode}_seed{seed}')
    directory.mkdir(parents=True,exist_ok=True)
    policy = network()
    initial = {k:v.clone() for k,v in policy.state_dict().items()}
    # Build baseline AFTER the policy, preserving identical initial actors across modes.
    baseline = network(5,1) if mode == 'baseline' else None
    # Reset sampling stream: baseline initialization must not alter actor action draws.
    torch.manual_seed(seed+100)
    optimizer = torch.optim.Adam(policy.parameters(),lr=lr)
    critic_optimizer = torch.optim.Adam(baseline.parameters(),lr=0.01) if baseline is not None else None
    rows=[]
    start=time.monotonic()
    torch.save({'policy':initial,'seed':seed},directory/'initial.pt')
    for offset in range(0,episodes,batch_size):
        batch=collect(policy,min(batch_size,episodes-offset),seed*100000+offset)
        states,actions,targets,times,critic_inputs,discount=packed(batch)
        with torch.no_grad():
            b=baseline(critic_inputs).squeeze(1) if baseline is not None else torch.zeros_like(targets)
        log_prob=torch.distributions.Categorical(logits=policy(states)).log_prob(actions)
        # Exact discounted episodic objective: outer gamma**t is retained.
        # Fixed scale 1-gamma affects magnitude only; no return/advantage normalization.
        loss=-(1-GAMMA)*(discount*log_prob*(targets-b)).sum()/len(batch)
        optimizer.zero_grad();loss.backward()
        grad_norm=float(torch.nn.utils.clip_grad_norm_(policy.parameters(),10.0))
        optimizer.step()
        critic_loss=None
        if baseline is not None:
            for _ in range(5):
                critic_loss=((baseline(critic_inputs).squeeze(1)-targets)**2).mean()
                critic_optimizer.zero_grad();critic_loss.backward();critic_optimizer.step()
        for j,ep in enumerate(batch):
            rows.append({'episode':offset+j+1,'reward':sum(ep['rewards']),
                         'discounted_return':float(ep['returns'][0]),
                         'terminated':ep['terminated'],'truncated':ep['truncated'],
                         'actor_loss':float(loss.detach()),'gradient_norm_before_clip':grad_norm,
                         'baseline_mse':float(critic_loss.detach()) if critic_loss is not None else ''})
        if (offset//batch_size)%10==0 or offset+len(batch)==episodes:
            recent=np.mean([r['reward'] for r in rows[-100:]])
            print(f'{mode} seed={seed} episode={len(rows)} reward100={recent:.1f} seconds={time.monotonic()-start:.1f}',flush=True)
        if len(rows)%512==0 or len(rows)==episodes:
            torch.save({'policy':policy.state_dict(),'baseline':baseline.state_dict() if baseline is not None else None,
                        'mode':mode,'seed':seed,'episodes':len(rows),'gamma':GAMMA,'lr':lr},directory/'last.pt')
    with (directory/'training.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    config={'environment':'CartPole-v1','library':'gymnasium','gymnasium':gym.__version__,
            'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','gpus_used':0,
            'mode':mode,'seed':seed,'episodes':episodes,'batch_size':batch_size,'lr':lr,'gamma':GAMMA,
            'architecture':'4 -> 64 tanh -> 2 logits','baseline_architecture':'(state, time/500) -> 64 tanh -> 1' if baseline else None,
            'baseline_lr':0.01 if baseline else None,'baseline_epochs_per_batch':5 if baseline else 0,
            'actor_scale':1-GAMMA,'gradient_clip_norm':10.0,'outer_discount':True,
            'return_normalization':False,'checkpoint_selection':'final episode; no evaluation-based selection',
            'seconds':time.monotonic()-start}
    (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    return policy


def evaluate(checkpoint, count=500, seed=9000000):
    checkpoint=Path(checkpoint)
    obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
    policy=network();policy.load_state_dict(obj['policy']);policy.eval()
    set_seed(seed)
    episodes=[]
    for offset in range(0,count,50):
        episodes.extend(collect(policy,min(50,count-offset),seed+offset))
    scores=np.array([sum(ep['rewards']) for ep in episodes])
    info={'episodes':count,'seed_start':seed,'stochastic_actions':True,'mean':float(scores.mean()),
          'population_std':float(scores.std(ddof=0)),'sample_std':float(scores.std(ddof=1)),
          'median':float(np.median(scores)),'min':float(scores.min()),'max':float(scores.max()),
          'fraction_500':float(np.mean(scores==500)),
          'physical_terminations':sum(ep['terminated'] for ep in episodes),
          'time_limit_truncations':sum(ep['truncated'] for ep in episodes),
          'mean_discounted_return':float(np.mean([ep['returns'][0] for ep in episodes]))}
    prefix='untrained_evaluation' if checkpoint.name=='initial.pt' else 'evaluation'
    np.savetxt(checkpoint.parent/f'{prefix}_rewards.csv',scores,delimiter=',',header='reward',comments='',fmt='%d')
    (checkpoint.parent/f'{prefix}.json').write_text(json.dumps(info,indent=2)+'\n')
    print(checkpoint,info,flush=True)
    return info


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['train','evaluate','suite'])
    parser.add_argument('--mode',choices=['no_baseline','baseline'],default='no_baseline')
    parser.add_argument('--seed',type=int,default=1668)
    parser.add_argument('--episodes',type=int,default=4000)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.command=='train':
        train(args.mode,args.seed,args.episodes,args.batch_size,args.lr)
    elif args.command=='evaluate':
        evaluate(args.checkpoint)
    else:
        for seed in (1668,1669,1670):
            for mode in ('no_baseline','baseline'):
                directory=ROOT/'runs'/f'{mode}_seed{seed}'
                train(mode,seed,args.episodes,args.batch_size,args.lr,directory)
                evaluate(directory/'last.pt')
        evaluate(ROOT/'runs/no_baseline_seed1668/initial.pt')


if __name__=='__main__':
    main()
