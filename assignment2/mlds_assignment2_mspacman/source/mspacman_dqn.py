"""Single-GPU image-based Double/Dueling DQN for assignment 2 MsPacman-v5.

Computational output only: numerical logs/checkpoints, no reports or figures.
ALE's default 4-frame skip and 0.25 sticky-action probability are preserved.
"""
import argparse
from collections import deque
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import time

import ale_py
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ENV_ID='ALE/MsPacman-v5'
GAMMA=.99
HEIGHT,WIDTH,STACK,ACTIONS=88,80,4,9
gym.register_envs(ale_py)


def write_json(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def preprocess(rgb):
    """Assignment's int8 preprocessing, stored losslessly with +128 uint8 offset.

    The network converts storage bytes to the assigned signed values / 128.
    Cropping/downsampling does NOT add another temporal frame skip.
    """
    gray=rgb[1:176:2,::2].sum(axis=2)
    gray[gray==210+164+74]=0
    signed=(gray//3-128).astype(np.int8)
    return (signed.astype(np.int16)+128).astype(np.uint8)


class Game:
    def __init__(self):
        self.env=gym.make(ENV_ID)
        if self.env.action_space.n!=ACTIONS:
            raise RuntimeError('MsPacman must expose exactly nine actions.')
        self.frames=deque(maxlen=STACK)

    def reset(self,seed):
        rgb,info=self.env.reset(seed=int(seed))
        frame=preprocess(rgb)
        self.frames.clear()
        self.frames.extend([frame]*STACK)
        return np.stack(self.frames),info

    def step(self,action):
        rgb,r,terminated,truncated,info=self.env.step(int(action))
        self.frames.append(preprocess(rgb))
        return np.stack(self.frames),float(r),bool(terminated),bool(truncated),info

    def close(self): self.env.close()


class Replay:
    """Two single-frame rings, reconstructing four-frame states without resets leaking.

    Each transition retains its actual next frame, even at terminal/truncated steps.
    The oldest three transitions are excluded once the ring wraps, because their
    earlier frame history may have been overwritten. Episode starts pad with the
    initial frame rather than using frames from the preceding game.
    """
    def __init__(self,capacity,seed):
        if capacity<8: raise ValueError('Replay capacity must be at least 8.')
        self.capacity=capacity
        self.total=0
        self.episode_start=0
        self.rng=np.random.default_rng(seed)
        self.frames=np.empty((capacity,HEIGHT,WIDTH),np.uint8)
        self.next_frames=np.empty_like(self.frames)
        self.actions=np.empty(capacity,np.int64)
        self.rewards=np.empty(capacity,np.float32)
        self.terminated=np.empty(capacity,bool)
        self.starts=np.empty(capacity,np.int64)

    def begin_episode(self): self.episode_start=self.total

    def add(self,state,action,reward,next_state,terminated):
        i=self.total%self.capacity
        self.frames[i]=state[-1]
        self.next_frames[i]=next_state[-1]
        self.actions[i]=action
        self.rewards[i]=reward
        self.terminated[i]=terminated
        self.starts[i]=self.episode_start
        self.total+=1

    @property
    def size(self): return min(self.total,self.capacity)

    @property
    def lower(self): return max(0,self.total-self.capacity+STACK-1)

    def get(self,times):
        times=np.asarray(times,dtype=np.int64)
        if np.any(times<self.lower) or np.any(times>=self.total):
            raise ValueError('Requested replay history is not retained.')
        i=times%self.capacity
        history=np.maximum(times[:,None]+np.arange(-STACK+1,1),self.starts[i,None])
        states=self.frames[history%self.capacity]
        next_states=np.concatenate((states[:,1:],self.next_frames[i,None]),axis=1)
        return states,self.actions[i],self.rewards[i],next_states,self.terminated[i]

    def sample(self,batch,device):
        times=self.rng.integers(self.lower,self.total,size=batch)
        return tuple(torch.as_tensor(x,device=device) for x in self.get(times))


class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(STACK,32,8,4),nn.ReLU(),
                                    nn.Conv2d(32,64,4,2),nn.ReLU(),
                                    nn.Conv2d(64,64,3,1),nn.ReLU(),nn.Flatten(),
                                    nn.Linear(64*7*6,512),nn.ReLU())
        self.value=nn.Linear(512,1)
        self.advantage=nn.Linear(512,ACTIONS)

    def forward(self,frames):
        x=self.features(frames.float()/128.-1.)
        advantage=self.advantage(x)
        return self.value(x)+advantage-advantage.mean(dim=-1,keepdim=True)


@torch.no_grad()
def targets(reward,terminated,next_value):
    return reward+GAMMA*(~terminated).to(reward.dtype)*next_value


class Learner:
    def __init__(self,device,lr):
        self.device=device
        self.online=DuelingDQN().to(device)
        self.target=DuelingDQN().to(device)
        self.sync_target()
        self.target.requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=lr,eps=1e-5)

    def sync_target(self): self.target.load_state_dict(self.online.state_dict())

    def update(self,batch):
        s,a,r,ns,term=batch
        q=self.online(s).gather(1,a[:,None]).squeeze(1)
        with torch.no_grad():
            best=self.online(ns).argmax(1,keepdim=True)
            nq=self.target(ns).gather(1,best).squeeze(1)
            y=targets(r,term,nq)
        loss=F.smooth_l1_loss(q,y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm=nn.utils.clip_grad_norm_(self.online.parameters(),10.)
        if not torch.isfinite(norm): raise RuntimeError('Non-finite DQN gradient.')
        self.optimizer.step()
        return dict(loss=float(loss.detach()),td_abs=float((q.detach()-y).abs().mean()),
                    gradient_norm=float(norm))


def setup(seed):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one GPU: CUDA_VISIBLE_DEVICES=0.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=True
    return torch.device('cuda:0')


def save_checkpoint(path,learner,config,episode,steps,updates,validation):
    path=Path(path)
    tmp=path.with_suffix('.tmp')
    torch.save(dict(model=learner.online.state_dict(),target=learner.target.state_dict(),
                    optimizer=learner.optimizer.state_dict(),config=config,episode=episode,
                    steps=steps,updates=updates,validation=validation,
                    python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp)
    os.replace(tmp,path)


def load_model(path,device):
    ck=torch.load(path,map_location=device,weights_only=False)
    model=DuelingDQN().to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model,ck


def summarize(rows):
    values=np.asarray([r['reward'] for r in rows],float)
    return dict(episodes=len(rows),mean=float(values.mean()),std_population=float(values.std()),
                min=float(values.min()),max=float(values.max()),median=float(np.median(values)))


@torch.no_grad()
def evaluate_games(model,device,seeds,epsilon=.01):
    game=Game()
    rows=[]
    first_ten=[]
    for j,seed in enumerate(seeds):
        rng=np.random.default_rng(int(seed)+123456)
        state,info=game.reset(seed)
        total=0.
        t=0
        while True:
            q=model(torch.as_tensor(state[None],device=device)).cpu().numpy()[0]
            action=int(rng.integers(ACTIONS)) if rng.random()<epsilon else int(q.argmax())
            ns,r,term,trunc,info=game.step(action)
            if j==0 and t<10:
                first_ten.append(dict(t=t,action=action,q_values=q.tolist(),reward=r,
                                      lives=int(info['lives']),terminated=term,truncated=trunc,
                                      state_sha256=hashlib.sha256(state.tobytes()).hexdigest()))
            total+=r
            t+=1
            state=ns
            if term or trunc: break
        rows.append(dict(index=j,seed=int(seed),reward=total,length=t,
                         emulator_frames=int(info['episode_frame_number']),
                         terminated=term,truncated=trunc,lives=int(info['lives'])))
    game.close()
    return rows,first_ten


def train(args):
    device=setup(args.seed)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    learner=Learner(device,args.lr)
    game=Game()
    config=dict(vars(args),env=ENV_ID,gamma=GAMMA,stack=STACK,input_shape=[STACK,HEIGHT,WIDTH],
                actions=ACTIONS,frameskip=4,repeat_action_probability=.25,
                max_emulator_frames_per_episode=108000,training_reward='clip(raw_reward,-1,1)',
                reported_reward='unaltered full-game score',life_loss_is_terminal=False,
                target='Double DQN; hard copy every target_every agent decisions',
                architecture='Nature CNN + dueling value/advantage heads',
                parameter_count=sum(p.numel() for p in learner.online.parameters()),
                replay_bytes=2*args.capacity*HEIGHT*WIDTH,
                gpu=torch.cuda.get_device_name(0),visible_gpu_count=torch.cuda.device_count(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),host=socket.gethostname(),
                torch=torch.__version__,gymnasium=gym.__version__,ale_py=ale_py.__version__,
                numpy=np.__version__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                validation_seed_start=1_000_000,evaluation_epsilon=args.eval_epsilon)
    write_json(out/'config.json',config)
    stop_requested=False
    def stop_handler(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,stop_handler)
    signal.signal(signal.SIGINT,stop_handler)
    started=time.time()
    steps=updates=episode=0
    scores=[]
    history=[]
    stats=dict(loss=0.,td_abs=0.,gradient_norm=0.)
    replay=Replay(args.capacity,args.seed+17)
    action_rng=np.random.default_rng(args.seed)
    dev_seeds=list(range(1_000_000,1_000_000+args.dev_episodes))
    def status(phase,**extra):
        value=dict(phase=phase,episode=episode,episodes_requested=args.episodes,steps=steps,
                   updates=updates,replay_size=replay.size,seconds=time.time()-started,
                   steps_per_second=steps/max(time.time()-started,1.),pid=os.getpid(),
                   gpu=config['gpu'],visible_gpu_count=config['visible_gpu_count'],
                   last_100_mean=float(np.mean(scores[-100:])) if scores else None,
                   **stats,**extra)
        write_json(out/'status.json',value)
    status('initial_validation')
    records,_=evaluate_games(learner.online,device,dev_seeds,args.eval_epsilon)
    initial=summarize(records)
    write_json(out/'initial_validation.json',dict(summary=initial,episodes=records))
    save_checkpoint(out/'initial.pt',learner,config,episode,steps,updates,initial)
    save_checkpoint(out/'best.pt',learner,config,episode,steps,updates,initial)
    best=(initial['mean'],initial['min'])
    print('INITIAL VALIDATION',json.dumps(initial),flush=True)
    fields=['episode','steps','length','emulator_frames','reward','moving_average_100',
            'epsilon','lr','q_max','q_mean','q_at_reset','loss','td_abs','gradient_norm',
            'updates','replay_size','positive_reward_events','life_losses','lives_remaining',
            'terminated','truncated','action_counts','wall_seconds']
    try:
        with (out/'training.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            status('training')
            while episode<args.episodes and not stop_requested:
                state,info=game.reset(args.seed*100_000+episode)
                replay.begin_episode()
                lives=int(info['lives'])
                total=q_sum=0.
                q_max=-float('inf')
                q_start=None
                length=positive=lost=0
                action_counts=np.zeros(ACTIONS,dtype=int)
                progress=episode/max(args.episodes-1,1)
                lr=args.min_lr+(args.lr-args.min_lr)*.5*(1+math.cos(math.pi*progress))
                for group in learner.optimizer.param_groups: group['lr']=lr
                while True:
                    epsilon=1.-(1.-args.epsilon_final)*min(steps/args.epsilon_steps,1.)
                    with torch.no_grad():
                        q=learner.online(torch.as_tensor(state[None],device=device)).cpu().numpy()[0]
                    if q_start is None: q_start=float(q.max())
                    action=int(action_rng.integers(ACTIONS)) if action_rng.random()<epsilon else int(q.argmax())
                    ns,r,term,trunc,info=game.step(action)
                    replay.add(state,action,float(np.clip(r,-1.,1.)),ns,term)
                    state=ns
                    steps+=1
                    length+=1
                    total+=r
                    positive+=int(r>0)
                    lost+=max(0,lives-int(info['lives']))
                    lives=int(info['lives'])
                    action_counts[action]+=1
                    q_max=max(q_max,float(q.max()))
                    q_sum+=float(q.max())
                    if steps>=args.warmup and steps%args.train_every==0:
                        stats=learner.update(replay.sample(args.batch,device))
                        updates+=1
                    if steps%args.target_every==0: learner.sync_target()
                    if steps%1000==0:
                        status('training',epsilon=epsilon,current_episode_reward=total,lr=lr)
                        print(f'STEPS {steps} episodes={episode} updates={updates} '
                              f'epsilon={epsilon:.4f} reward_now={total:.0f} loss={stats["loss"]:.5f}',flush=True)
                    if term or trunc: break
                episode+=1
                scores.append(total)
                mean100=float(np.mean(scores[-100:]))
                writer.writerow(dict(episode=episode,steps=steps,length=length,
                                     emulator_frames=int(info['episode_frame_number']),reward=total,
                                     moving_average_100=mean100,epsilon=epsilon,lr=lr,q_max=q_max,
                                     q_mean=q_sum/length,q_at_reset=q_start,**stats,updates=updates,
                                     replay_size=replay.size,positive_reward_events=positive,
                                     life_losses=lost,lives_remaining=lives,terminated=int(term),
                                     truncated=int(trunc),action_counts=json.dumps(action_counts.tolist()),
                                     wall_seconds=time.time()-started))
                f.flush()
                status('training',epsilon=epsilon,last_episode_reward=total,lr=lr)
                print(f'EPISODE {episode}/{args.episodes} steps={steps} score={total:.0f} '
                      f'mean100={mean100:.1f} epsilon={epsilon:.4f}',flush=True)
                if episode%args.save_every==0 or stop_requested:
                    save_checkpoint(out/'last.pt',learner,config,episode,steps,updates,None)
                if episode%args.validate_every==0 or episode==args.episodes:
                    status('validation')
                    records,_=evaluate_games(learner.online,device,dev_seeds,args.eval_epsilon)
                    summary=summarize(records)
                    summary.update(episode=episode,steps=steps)
                    history.append(dict(summary=summary,episodes=records))
                    write_json(out/'validation_history.json',history)
                    rank=(summary['mean'],summary['min'])
                    if rank>best:
                        best=rank
                        save_checkpoint(out/'best.pt',learner,config,episode,steps,updates,summary)
                    print('VALIDATION',json.dumps(summary),flush=True)
            save_checkpoint(out/'final.pt',learner,config,episode,steps,updates,None)
            save_checkpoint(out/'last.pt',learner,config,episode,steps,updates,None)
            completed=episode==args.episodes
            write_json(out/'training_summary.json',dict(completed=completed,episodes=episode,
                       steps=steps,updates=updates,seconds=time.time()-started,
                       last_100_mean=float(np.mean(scores[-100:])) if scores else None,
                       best_validation_mean=best[0]))
            status('training_complete' if completed else 'stopped')
            if not completed: raise SystemExit(2)
    except Exception as error:
        status('failed',error=repr(error))
        raise
    finally:
        game.close()
    print('TRAINING COMPLETE',flush=True)


def evaluate(args):
    device=setup(1668)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    model,ck=load_model(args.checkpoint,device)
    rows,trace=evaluate_games(model,device,list(range(args.seed_start,args.seed_start+args.episodes)),args.epsilon)
    summary=summarize(rows)
    summary.update(checkpoint=str(args.checkpoint),checkpoint_episode=ck['episode'],
                   checkpoint_steps=ck['steps'],epsilon=args.epsilon,gamma=GAMMA,
                   seed_start=args.seed_start,env=ENV_ID,
                   checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
    write_json(out/'episodes.json',rows)
    write_json(out/'first_ten_steps.json',trace)
    write_json(out/'summary.json',summary)
    print('FINAL EVALUATION',json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    t.add_argument('--out',required=True)
    t.add_argument('--episodes',type=int,default=5000)
    t.add_argument('--seed',type=int,default=1668)
    t.add_argument('--capacity',type=int,default=200_000)
    t.add_argument('--warmup',type=int,default=20_000)
    t.add_argument('--batch',type=int,default=32)
    t.add_argument('--train-every',type=int,default=4)
    t.add_argument('--target-every',type=int,default=10_000)
    t.add_argument('--epsilon-steps',type=int,default=1_000_000)
    t.add_argument('--epsilon-final',type=float,default=.05)
    t.add_argument('--lr',type=float,default=1e-4)
    t.add_argument('--min-lr',type=float,default=2.5e-5)
    t.add_argument('--save-every',type=int,default=100)
    t.add_argument('--validate-every',type=int,default=250)
    t.add_argument('--dev-episodes',type=int,default=10)
    t.add_argument('--eval-epsilon',type=float,default=.01)
    e=sub.add_parser('evaluate')
    e.add_argument('--checkpoint',required=True)
    e.add_argument('--out',required=True)
    e.add_argument('--episodes',type=int,default=500)
    e.add_argument('--seed-start',type=int,default=9_000_000)
    e.add_argument('--epsilon',type=float,default=.01)
    args=p.parse_args()
    if args.command=='train': train(args)
    else: evaluate(args)


if __name__=='__main__': main()
