"""Short report-support computations; original held-out evaluation is not rerun.

Runs a matched random-policy reference and one separately seeded diagnostic game.
Writes numerical data only. Render plots and reports on the Mac.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

from mspacman_dqn import (ENV_ID,ACTIONS,Game,load_model,setup,
                         summarize,write_json)


def parameter_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def random_reference(out):
    env=gym.make(ENV_ID)
    rows=[]
    for i in range(500):
        seed=9_000_000+i
        rng=np.random.default_rng(seed+123456)
        _,info=env.reset(seed=seed)
        score=0.
        length=0
        while True:
            action=int(rng.integers(ACTIONS))
            _,r,term,trunc,info=env.step(action)
            score+=r
            length+=1
            if term or trunc: break
        rows.append(dict(index=i,seed=seed,reward=score,length=length,
                         terminated=bool(term),truncated=bool(trunc),lives=int(info['lives'])))
        if (i+1)%50==0: print('RANDOM BASELINE',i+1,'/ 500',flush=True)
    env.close()
    write_json(out/'random_episodes.json',rows)
    write_json(out/'random_summary.json',summarize(rows))
    return summarize(rows)


@torch.no_grad()
def diagnostic_game(model,device,out):
    seed=6_000_000
    rng=np.random.default_rng(seed+123456)
    game=Game()
    state,info=game.reset(seed)
    rows=[]
    total=0.
    lives=int(info['lives'])
    t=0
    while True:
        q=model(torch.as_tensor(state[None],device=device)).cpu().numpy()[0]
        random_action=bool(rng.random()<.01)
        action=int(rng.integers(ACTIONS)) if random_action else int(q.argmax())
        ns,r,term,trunc,info=game.step(action)
        total+=r
        new_lives=int(info['lives'])
        rows.append(dict(t=t,action=action,random_action=random_action,q_values=q.tolist(),
                         reward=r,cumulative_score=total,lives_before=lives,lives_after=new_lives,
                         terminated=term,truncated=trunc))
        t+=1
        state=ns
        lives=new_lives
        if term or trunc: break
    game.close()
    assert total==sum(r['reward'] for r in rows)
    assert all(r['random_action'] or r['action']==int(np.argmax(r['q_values'])) for r in rows)
    positive=[r for r in rows if r['reward']>0]
    life_events=[r for r in rows if r['lives_after']<r['lives_before']]
    summary=dict(seed=seed,epsilon=.01,steps=t,score=total,positive_reward_events=len(positive),
                 sum_recorded_rewards=sum(r['reward'] for r in rows),
                 life_loss_events=[dict(step=r['t']+1,from_lives=r['lives_before'],
                                       to_lives=r['lives_after'],terminated=r['terminated'],
                                       truncated=r['truncated']) for r in life_events],
                 first_rewards=[dict(step=r['t']+1,reward=r['reward'],cumulative_score=r['cumulative_score'])
                                for r in positive[:10]],
                 greedy_action_checks=sum(not r['random_action'] for r in rows),
                 random_action_count=sum(r['random_action'] for r in rows),
                 )
    write_json(out/'diagnostic_trajectory.json',rows)
    write_json(out/'diagnostic_summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    device=setup(1668)
    start=time.time()
    baseline=random_reference(out)
    model,_=load_model(args.checkpoint,device)
    before=parameter_hash(model)
    diagnostic=diagnostic_game(model,device,out)
    after=parameter_hash(model)
    assert before==after
    summary=dict(status='PASS',random=baseline,diagnostic=diagnostic,
                 model_parameters_unchanged=before==after,model_parameter_sha256=after,
                 checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                 seconds=time.time()-start,gpu=torch.cuda.get_device_name(0),
                 visible_gpu_count=torch.cuda.device_count(),
                 note='Original 500 trained-policy games remain untouched; no tuning or retraining.')
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
