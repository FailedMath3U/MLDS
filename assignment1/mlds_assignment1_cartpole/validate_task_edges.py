"""Supplementary CartPole task checks; originals are read-only.
Run: python validate_task_edges.py. Writes task_validation/ only.
"""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import torch
import gymnasium as gym
import cartpole as cp

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'task_validation'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gradient(policy, episodes, baseline=0.):
    states, actions, targets, times, inputs, discount=cp.packed(episodes)
    logp=torch.distributions.Categorical(logits=policy(states)).log_prob(actions)
    loss=-(1-cp.GAMMA)*(discount*logp*(targets-baseline)).sum()/len(episodes)
    return torch.cat([g.flatten() for g in torch.autograd.grad(loss,policy.parameters())])


def boundary_checks():
    records=[]
    for kind in ('cart','pole'):
        for sign in (-1,1):
            for location in ('exact','crossing'):
                for action in (0,1):
                    env=gym.make('CartPole-v1');env.reset(seed=1)
                    state=np.zeros(4,dtype=np.float64)
                    if kind=='cart':
                        state[0]=sign*(env.unwrapped.x_threshold if location=='exact' else 2.39)
                        state[1]=0 if location=='exact' else sign*1.
                    else:
                        state[2]=sign*(env.unwrapped.theta_threshold_radians if location=='exact' else .20)
                        state[3]=0 if location=='exact' else sign*1.
                    env.unwrapped.state=state
                    nxt,reward,term,trunc,_=env.step(action)
                    expected=location=='crossing'
                    assert bool(term)==expected and not trunc and reward==1
                    records.append({'kind':kind,'sign':sign,'location':location,'action':action,
                                    'initial_state':state.tolist(),'next_state':nxt.tolist(),
                                    'terminated':bool(term),'truncated':bool(trunc),'reward':reward})
                    env.close()
    env=gym.make('CartPole-v1',max_episode_steps=1);env.reset(seed=1)
    env.unwrapped.state=np.array([2.39,1.,0.,0.])
    _,reward,term,trunc,_=env.step(0);env.close()
    assert term and trunc and reward==1
    return {'boundary_trials':records,'trials_passed':len(records),
            'simultaneous_failure_and_limit':{'terminated':bool(term),'truncated':bool(trunc),'reward':reward}}


class ControlledStart(gym.Wrapper):
    def __init__(self,env,state):
        super().__init__(env);self.start=np.array(state,dtype=np.float64)
    def reset(self,**kwargs):
        _,info=self.env.reset(**kwargs)
        self.env.unwrapped.state=self.start.copy()
        return self.start.astype(np.float32),info


def collector_checks(policy):
    # Exercise the actual original collector, with controlled environments.
    original_make=cp.gym.make
    specifications=iter([(500,[2.39,1,0,0]),(1,[0,0,0,0]),(3,[0,0,0,0])])
    def controlled_make(_):
        horizon,state=next(specifications)
        return ControlledStart(original_make('CartPole-v1',max_episode_steps=horizon),state)
    cp.gym.make=controlled_make
    try: episodes=cp.collect(policy,3,9900000)
    finally: cp.gym.make=original_make
    lengths=[len(e['rewards']) for e in episodes]
    assert lengths==[1,1,3]
    assert episodes[0]['terminated'] and not episodes[0]['truncated']
    assert episodes[1]['truncated'] and not episodes[1]['terminated']
    assert episodes[2]['truncated'] and not episodes[2]['terminated']
    assert np.array_equal(episodes[0]['returns'],np.array([1],dtype=np.float32))
    all_gradient=gradient(policy,episodes)
    separate=torch.stack([gradient(policy,[ep]) for ep in episodes]).mean(0)
    error=float((all_gradient-separate).abs().max())
    assert torch.allclose(all_gradient,separate,atol=2e-6,rtol=2e-5)
    with (OUT/'mixed_length_trajectories.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['episode','time','reward','return','terminated','truncated'])
        writer.writeheader()
        for i,ep in enumerate(episodes):
            for t,r in enumerate(ep['rewards']):writer.writerow({'episode':i,'time':t,'reward':r,
                'return':float(ep['returns'][t]),'terminated':ep['terminated'] and t==len(ep['rewards'])-1,
                'truncated':ep['truncated'] and t==len(ep['rewards'])-1})
    return {'lengths':lengths,'first_episode_return':float(episodes[0]['returns'][0]),
            'batch_vs_mean_episode_gradient_max_absolute_error':error,
            'prediction':'Batch gradient equals mean of individual episode gradients; no post-end steps or cross-episode returns.'}


def advantage_sign_checks():
    records=[]
    for baseline in (0.,2.):
        logits=torch.nn.Parameter(torch.zeros(2))
        optimizer=torch.optim.Adam([logits],lr=.01)
        dist=torch.distributions.Categorical(logits=logits)
        before=float(dist.probs[1].detach())
        # One transition: target 1, action 1; gradient is the exact actor loss.
        loss=-(1-cp.GAMMA)*dist.log_prob(torch.tensor(1))*(1-baseline)
        optimizer.zero_grad();loss.backward();optimizer.step()
        after=float(torch.softmax(logits.detach(),0)[1])
        assert (after>before) if baseline==0 else (after<before)
        records.append({'target':1.,'baseline':baseline,'advantage':1-baseline,
                        'before':before,'after':after})
    return records


def main():
    OUT.mkdir(exist_ok=True);cp.set_seed(8877)
    protected=[ROOT/'cartpole.py',ROOT/'validate.py']+[ROOT/f'runs/{mode}_seed1668/last.pt' for mode in ('no_baseline','baseline')]
    before={str(p.relative_to(ROOT)):digest(p) for p in protected}
    report={'device':'cpu','gpus_used':0,'boundary_checks':boundary_checks(),
            'collector_checks':collector_checks(cp.network()),'advantage_sign_checks':advantage_sign_checks()}
    after={str(p.relative_to(ROOT)):digest(p) for p in protected}
    assert before==after
    report['original_hashes_unchanged']=True;report['original_sha256']=before
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
