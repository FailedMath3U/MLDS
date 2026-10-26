"""Algorithm experiments for CartPole. Run after cartpole.py suite."""
import csv
import json
from pathlib import Path
import itertools
import math
import numpy as np
import torch
from torch import nn
import gymnasium as gym
from cartpole import ROOT, GAMMA, network, set_seed, returns, collect, packed, evaluate


def main():
    torch.set_num_threads(1)
    out=ROOT/'validation';out.mkdir(exist_ok=True)
    results={}
    def record(name,passed=None,**evidence):
        results[name]={'passed':bool(passed) if passed is not None else None,**evidence}
        if passed is False:
            raise AssertionError(name)
    got=returns([1,1,1])
    record('analytic_discounted_returns',bool(np.allclose(got,[2.8525,1.95,1],atol=1e-6)),
           rewards=[1,1,1],expected=[2.8525,1.95,1],observed=got.tolist())
    record('zero_discount_is_immediate_reward',bool(np.array_equal(returns([2,3,4],0),[2,3,4])))

    # Exact enumeration of a two-step MDP with reward equal to its binary action.
    # Both decisions share a single Bernoulli logit, so the exact objective is
    # (1+gamma)*sigmoid(logit). Test the SAME outer-discount convention as training.
    theta=0.37;prob=1/(1+math.exp(-theta))
    exact=(1+GAMMA)*prob*(1-prob)
    estimates=[]
    for baseline in (0.,1.7):
        estimate=0.
        for a0,a1 in itertools.product((0,1),repeat=2):
            path_prob=(prob if a0 else 1-prob)*(prob if a1 else 1-prob)
            rtg=returns([a0,a1])
            score=(a0-prob)*(float(rtg[0])-baseline)+GAMMA*(a1-prob)*(float(rtg[1])-baseline)
            estimate+=path_prob*score
        estimates.append(estimate)
    record('enumerated_policy_gradient_and_baseline_invariance',bool(max(abs(x-exact) for x in estimates)<1e-7),
           analytic_gradient=exact,no_baseline_gradient=estimates[0],constant_baseline_gradient=estimates[1],
           explanation='Enumerate all four paths in a two-step MDP; subtracting an action-independent baseline leaves the expected gradient unchanged.')
    theta=torch.tensor(0.,requires_grad=True)
    opt=torch.optim.Adam([theta],lr=.01)
    before=float(torch.sigmoid(theta.detach()))
    loss=-torch.distributions.Bernoulli(logits=theta).log_prob(torch.tensor(1.))*1.
    opt.zero_grad();loss.backward();opt.step()
    after=float(torch.sigmoid(theta.detach()))
    record('rewarded_action_probability_increases',after>before,before=before,after=after)

    # Repeated real environment rollouts with matching reset seeds and fixed actions.
    trajectories=[]
    for _ in range(2):
        env=gym.make('CartPole-v1');state,_=env.reset(seed=500001)
        trace=[state.tolist()]
        for action in [0,1]*5:
            state,_,terminated,truncated,_=env.step(action);trace.append(state.tolist())
            if terminated or truncated:break
        env.close();trajectories.append(trace)
    record('seeded_environment_replay',trajectories[0]==trajectories[1],steps=len(trajectories[0])-1)
    env=gym.make('CartPole-v1');env.reset(seed=1)
    env.unwrapped.state=np.array([2.41,0.,0.,0.])
    _,r,term,trunc,_=env.step(1);env.close()
    record('cart_position_boundary',bool(term and not trunc and r==1),terminated=bool(term),truncated=bool(trunc),reward=float(r))
    env=gym.make('CartPole-v1');env.reset(seed=1)
    env.unwrapped.state=np.array([0.,0.,.22,0.])
    _,r,term,trunc,_=env.step(0);env.close()
    record('pole_angle_boundary',bool(term and not trunc and r==1),terminated=bool(term),truncated=bool(trunc),reward=float(r))
    # A one-decision TimeLimit isolates truncation from physical termination.
    env=gym.make('CartPole-v1',max_episode_steps=1);env.reset(seed=500001)
    _,_,term,trunc,_=env.step(0);env.close()
    record('time_limit_is_truncation',bool(trunc and not term),terminated=bool(term),truncated=bool(trunc),
           explanation='Training and evaluation end on either flag; reward-to-go stops at the specified episodic horizon without bootstrapping.')

    # Frozen-policy variance study: train critic on INDEPENDENT trajectories, then
    # measure both gradient estimators on the SAME held-out trajectories.
    set_seed(1668);policy=network();critic=network(5,1)
    torch.manual_seed(7711)
    fitting=collect(policy,512,6000000)
    states,actions,targets,times,inputs,discount=packed(fitting)
    optim=torch.optim.Adam(critic.parameters(),lr=.01)
    for _ in range(150):
        loss=((critic(inputs).squeeze(1)-targets)**2).mean()
        optim.zero_grad();loss.backward();optim.step()
    heldout=collect(policy,512,7000000)
    gradients=[[],[]]
    for ep in heldout:
        st,ac,rt,tm,ci,di=packed([ep])
        with torch.no_grad():b=critic(ci).squeeze(1)
        for which,base in enumerate((torch.zeros_like(rt),b)):
            logp=torch.distributions.Categorical(logits=policy(st)).log_prob(ac)
            objective=(1-GAMMA)*(di*logp*(rt-base)).sum()
            grad=torch.autograd.grad(objective,policy.parameters())
            gradients[which].append(torch.cat([g.flatten() for g in grad]).numpy())
    g0,g1=map(np.array,gradients)
    v0=float(g0.var(axis=0,ddof=1).sum());v1=float(g1.var(axis=0,ddof=1).sum())
    ratio=v1/v0
    mean_diff=float(np.linalg.norm(g1.mean(axis=0)-g0.mean(axis=0)))
    diff_se=float(np.sqrt((g1-g0).var(axis=0,ddof=1).sum()/len(g0)))
    record('frozen_policy_gradient_variance',None,trajectories=512,fitting_trajectories=512,
           variance_no_baseline=v0,variance_baseline=v1,variance_ratio=ratio,
           variance_reduction_fraction=1-ratio,mean_gradient_difference_norm=mean_diff,
           rms_standard_error_of_mean_difference=diff_se,baseline_mse=float(loss.detach()),
           explanation='Trace of sample gradient covariance (sum of per-parameter variances). A fixed policy and paired held-out paths isolate baseline effects from policy learning. Measured reduction is evidence, not a universal guarantee.')
    np.savez_compressed(out/'gradient_variance.npz',no_baseline=g0,baseline=g1)

    # Uniform random reference, separate from an untrained random-weight neural net.
    uniform=network()
    for p in uniform.parameters():p.data.zero_()
    torch.manual_seed(4455)
    random_episodes=[]
    for offset in range(0,500,50):random_episodes.extend(collect(uniform,50,8000000+offset))
    scores=np.array([sum(ep['rewards']) for ep in random_episodes])
    record('uniform_random_reference',None,episodes=500,mean=float(scores.mean()),std=float(scores.std()),
           mean_discounted_return=float(np.mean([ep['returns'][0] for ep in random_episodes])))
    np.savetxt(out/'random_rewards.csv',scores,fmt='%d',header='reward',comments='')

    # Save ten transparent decisions from the final no-baseline model.
    obj=torch.load(ROOT/'runs/no_baseline_seed1668/last.pt',map_location='cpu',weights_only=True)
    policy.load_state_dict(obj['policy']);torch.manual_seed(123456)
    env=gym.make('CartPole-v1');state,_=env.reset(seed=9100000)
    trace=[]
    for t in range(10):
        with torch.no_grad():
            dist=torch.distributions.Categorical(logits=policy(torch.from_numpy(state)))
            action=int(dist.sample());p_right=float(dist.probs[1])
        nxt,reward,term,trunc,_=env.step(action)
        trace.append({'t':t,'position':float(state[0]),'velocity':float(state[1]),
                      'angle':float(state[2]),'angular_velocity':float(state[3]),'p_right':p_right,
                      'action':action,'reward':float(reward),'terminated':bool(term),'truncated':bool(trunc)})
        state=nxt
        if term or trunc:break
    env.close()
    with (out/'first_ten_steps.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(trace[0]));writer.writeheader();writer.writerows(trace)
    results['trace']={'file':'first_ten_steps.csv','rows':len(trace),'reset_seed':9100000,'torch_seed':123456}
    (out/'validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
