"""Additional numerical checks claimed in the user-written report.
CPU only; original algorithms, checkpoints and result files are read-only.
Run from this directory: python validate_report_claims.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys, json, hashlib
import numpy as np
import torch
import gymnasium as gym

ROOT=Path(__file__).resolve().parent
PROJECTS=ROOT.parent
sys.path.insert(0,str(PROJECTS/'mlds_assignment1_cartpole'))
sys.path.insert(0,str(PROJECTS/'mlds_assignment1_pong'))
import cartpole as cp
import pong_core as pc
from pong_env import preprocess, make_env
OUT=ROOT/'validation'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def cart_gradient(policy,episodes,baseline=None):
    states,actions,targets,times,inputs,discount=cp.packed(episodes)
    with torch.no_grad():
        b=torch.zeros_like(targets) if baseline is None else baseline(inputs).squeeze(1)
    logp=torch.distributions.Categorical(logits=policy(states)).log_prob(actions)
    loss=-(1-cp.GAMMA)*(discount*logp*(targets-b)).sum()/len(episodes)
    grads=torch.autograd.grad(loss,policy.parameters())
    return torch.cat([x.flatten() for x in grads])


def pong_gradient(policy,episodes,baseline):
    pc.actor_gradient(policy,episodes,baseline,torch.device('cpu'))
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten() for p in policy.parameters()])


class RewardOverride(gym.RewardWrapper):
    def __init__(self,env,reward):super().__init__(env);self.fixed=reward
    def reward(self,reward):return float(self.fixed)


def cart_episodes(policy,count,seed,reward=None,horizon=None):
    original_make=cp.gym.make
    def make(_):
        env=original_make('CartPole-v1',**({'max_episode_steps':horizon} if horizon else {}))
        return RewardOverride(env,reward) if reward is not None else env
    cp.gym.make=make
    try:return cp.collect(policy,count,seed)
    finally:cp.gym.make=original_make


def fit_cart_baseline(episodes,heldout,target_value,seed):
    torch.manual_seed(seed);critic=cp.network(5,1)
    training=cp.packed(episodes);testing=cp.packed(heldout)
    x=training[4];y=training[2];xt=testing[4];yt=testing[2]
    assert torch.all(y==target_value) and torch.all(yt==target_value)
    with torch.no_grad():before=float(((critic(xt).squeeze(1)-yt)**2).mean())
    opt=torch.optim.Adam(critic.parameters(),lr=.01)
    for _ in range(400):
        loss=((critic(x).squeeze(1)-y)**2).mean()
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        prediction=critic(xt).squeeze(1)
        mse=float(((prediction-yt)**2).mean());mean=float(prediction.mean())
    assert mse<1e-4 and mse<before/100
    return {'fitting_episodes':len(episodes),'heldout_episodes':len(heldout),'target':target_value,
            'adam_steps':400,'adam_lr':.01,'heldout_mse_before':before,'heldout_mse_after':mse,
            'heldout_prediction_mean':mean}


class OnePongPool:
    """A real single ALE environment with an optional external time limit."""
    def __init__(self,horizon=None):
        self.env=make_env()
        if horizon is not None:self.env=gym.wrappers.TimeLimit(self.env,max_episode_steps=horizon)
        self.finished=False;self.steps=0
    def reset(self,seeds):
        assert len(seeds)==1
        raw,_=self.env.reset(seed=int(seeds[0]));self.previous=preprocess(raw)
        self.finished=False;self.steps=0
        return {0:np.zeros(6400,dtype=np.int8)}
    def step(self,actions):
        assert not self.finished,'Collector stepped after episode end'
        assert set(actions)=={0} and actions[0] in (2,3)
        raw,reward,term,trunc,_=self.env.step(actions[0]);self.steps+=1
        current=preprocess(raw);difference=(current-self.previous).ravel();self.previous=current
        self.finished=bool(term or trunc)
        return {0:(difference,float(reward),bool(term),bool(trunc))}
    def close(self):self.env.close()


def synthetic_pong(rewards):
    r=np.array(rewards,dtype=np.float32);n=len(r)
    return {'x':np.zeros((n,6400),dtype=np.int8),'a':np.arange(n,dtype=np.int64)%2,
            'r':r,'g':pc.returns(r),'p2':np.full(n,.5,dtype=np.float32),
            'length':n,'score':float(r.sum()),'discounted_return':float(pc.returns(r)[0]),
            'terminated':False,'truncated':True,'reset_seed':0,'action_seed':0}


def ema_probe(reward,start,steps):
    config={'seed':1668,'lr':.001,'chunk':512,'clip_norm':5.,'baseline_ema_rate':.05}
    agent=pc.Agent('baseline',config,torch.device('cpu'));agent.baseline=start
    episode=synthetic_pong([reward])
    for i in range(steps):
        old=agent.baseline;agent.update([episode])
        assert agent.history[-1]['baseline_used']==old
    expected=reward+(start-reward)*(.95**steps)
    assert abs(agent.baseline-expected)<1e-12
    return {'one_step_reward':reward,'initial_baseline':start,'updates':steps,
            'observed_baseline':agent.baseline,'predicted_baseline':expected,
            'note':'Repeated fixed synthetic data tests EMA mechanics, not on-policy learning performance.'}


def main():
    OUT.mkdir(exist_ok=True);cp.set_seed(44201)
    protected=[PROJECTS/'mlds_assignment1_cartpole/cartpole.py',PROJECTS/'mlds_assignment1_cartpole/validate.py',
               PROJECTS/'mlds_assignment1_pong/pong_core.py',PROJECTS/'mlds_assignment1_pong/pong_env.py',
               PROJECTS/'mlds_assignment1_pong/run_pong.py']
    before={str(p.relative_to(PROJECTS)):sha(p) for p in protected}
    evidence={};policy=cp.network()
    episodes=cart_episodes(policy,16,31000000)
    zero=cp.network(5,1)
    for p in zero.parameters():p.data.zero_()
    a=cart_gradient(policy,episodes);b=cart_gradient(policy,episodes,zero)
    error=float((a-b).abs().max());assert error==0
    evidence['cart_zero_baseline_recovers_reinforce']={'max_gradient_difference':error,'episodes':16}
    fit=cart_episodes(policy,64,31100000,reward=0.)
    test=cart_episodes(policy,64,31200000,reward=0.)
    evidence['cart_zero_reward_baseline_fit']=fit_cart_baseline(fit,test,0.,701)
    norm=float(cart_gradient(policy,test).norm());assert norm==0
    evidence['cart_zero_returns_zero_baseline_gradient_norm']=norm
    fit=cart_episodes(policy,64,31300000,reward=1.,horizon=1)
    test=cart_episodes(policy,64,31400000,reward=1.,horizon=1)
    evidence['cart_one_step_unit_return_baseline_fit']=fit_cart_baseline(fit,test,1.,702)
    one=cp.network(5,1)
    for p in one.parameters():p.data.zero_()
    one[-1].bias.data.fill_(1.)
    norm=float(cart_gradient(policy,test,one).norm());assert norm==0
    evidence['cart_exact_unit_baseline_gradient_norm']=norm
    # Exact action enumeration in a real one-step CartPole state: both rewards 1.
    state=torch.tensor([[.01,-.02,.03,-.01]],dtype=torch.float32)
    dist=torch.distributions.Categorical(logits=policy(state))
    expected=torch.zeros(sum(p.numel() for p in policy.parameters()))
    sample_norms=[]
    for action in (0,1):
        logp=torch.distributions.Categorical(logits=policy(state)).log_prob(torch.tensor([action])).sum()
        g=torch.cat([v.flatten() for v in torch.autograd.grad(logp,policy.parameters())])
        expected+=dist.probs[0,action].detach()*g;sample_norms.append(float(g.norm()))
    residual=float(expected.norm());assert residual<1e-6
    evidence['cart_one_step_action_independent_objective']={'expected_score_gradient_norm':residual,
        'individual_score_gradient_norms':sample_norms,'objective':1.,'exact_baseline_gradient_norm':0.}
    pong=pc.make_policy(1668,torch.device('cpu'))
    ep=synthetic_pong([0,1,0,-1])
    # Independent autograd reference has no baseline subtraction.
    x,a,g,d=pc.flat_arrays([ep]);xx=torch.tensor(x,dtype=torch.float32);aa=torch.tensor(a)
    logp=pong(xx).log_softmax(-1).gather(1,aa[:,None]).squeeze(1)
    loss=-(logp*torch.tensor(d*g)).sum()
    reference=torch.cat([v.flatten() for v in torch.autograd.grad(loss,pong.parameters())])
    actual=pong_gradient(pong,[ep],0.);error=float((reference-actual).abs().max());assert error<1e-7
    evidence['pong_zero_baseline_recovers_reinforce']={'max_gradient_difference':error}
    norm=float(pong_gradient(pong,[synthetic_pong([0,0,0])],0.).norm());assert norm==0
    evidence['pong_zero_returns_zero_baseline_gradient_norm']=norm
    norm=float(pong_gradient(pong,[synthetic_pong([1])],1.).norm());assert norm==0
    evidence['pong_exact_unit_baseline_gradient_norm']=norm
    evidence['pong_zero_reward_ema_from_zero']=ema_probe(0.,0.,1)
    evidence['pong_zero_reward_ema_from_nonzero']=ema_probe(0.,2.,160)
    evidence['pong_unit_one_step_return_ema']=ema_probe(1.,0.,160)
    real_episodes=[]
    for horizon,name in [(None,'normal_termination'),(1,'external_time_limit')]:
        pool=OnePongPool(horizon)
        try:
            batch,_=pc.collect(pool,pong,1,31500000,31600000,0,torch.device('cpu'))
            episode=batch[0]
            assert pool.steps==episode['length']
            if horizon is None:assert episode['terminated'] and not episode['truncated']
            else:assert episode['truncated'] and not episode['terminated'] and episode['length']==1
            assert episode['g'][-1]==episode['r'][-1]
            evidence['pong_'+name]={'length':episode['length'],'score':episode['score'],
                'terminated':episode['terminated'],'truncated':episode['truncated'],
                'final_reward':float(episode['r'][-1]),'final_return':float(episode['g'][-1]),
                'no_steps_after_end':True,'first_input_all_zero':bool(np.all(episode['x'][0]==0))}
        finally:pool.close()
    after={str(p.relative_to(PROJECTS)):sha(p) for p in protected};assert before==after
    result={'passed':True,'device':'cpu','gpus_used':0,'evidence':evidence,'original_sources_unchanged':True,'source_sha256':before}
    (OUT/'report_claims.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
