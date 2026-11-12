"""Preliminary MsPacman checks: task behavior and the real GPU training update."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from mspacman_dqn import (ACTIONS,HEIGHT,WIDTH,Game,Learner,Replay,preprocess,
                         targets,setup,write_json,save_checkpoint,load_model,summarize)


def replay_experiment():
    replay=Replay(11,123)
    expected={}
    # Short games, including one-step episodes, followed by repeated ring wrap.
    for episode,length in enumerate([2,1,7,2,8,3]):
        replay.begin_episode()
        initial=episode*30
        stack=np.full((4,HEIGHT,WIDTH),initial,np.uint8)
        for t in range(length):
            frame=np.full((HEIGHT,WIDTH),initial+t+1,np.uint8)
            ns=np.concatenate([stack[1:],frame[None]],axis=0)
            expected[replay.total]=(stack.copy(),t%9,float(t),ns.copy(),t==length-1)
            replay.add(stack,t%9,float(t),ns,t==length-1)
            stack=ns
        for index in range(replay.lower,replay.total):
            got=replay.get([index])
            for actual,prediction in zip(got,expected[index]):
                np.testing.assert_array_equal(actual[0],prediction)
    return dict(passed=True,transitions=replay.total,capacity=replay.capacity,
                episode_lengths=[2,1,7,2,8,3],
                checks='Four-frame history equals explicit stacks before/after resets and multiple ring wraps; terminal next frame retained.')


def pixel_experiment():
    rgb=np.zeros((210,160,3),np.uint8)
    rgb[1,0]=[255,255,255]
    rgb[1,2]=[210,164,74]
    rgb[1,4]=[30,60,90]
    gray=preprocess(rgb)
    signed=gray.astype(np.int16)-128
    observed=signed[0,:4].tolist()
    assert observed==[127,-128,-68,-128],observed
    return dict(expected_signed_pixels=[127,-128,-68,-128],observed_signed_pixels=observed,
                shape=list(gray.shape),passed=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    args=p.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    start=time.time()
    device=setup(1668)
    result=dict(preprocessing=pixel_experiment(),replay=replay_experiment())
    reward=torch.tensor([1.,1.,1.],device=device)
    terminal=torch.tensor([False,True,False],device=device)
    nq=torch.tensor([10.,10.,5.],device=device)
    y=targets(reward,terminal,nq).cpu().numpy()
    np.testing.assert_allclose(y,[10.9,1.,5.95],atol=1e-6)
    result['bellman']=dict(expected=[10.9,1.,5.95],observed=y.tolist(),
                           cases=['continue','physical game over','external time limit'])

    game=Game()
    state,info=game.reset(7_000_000)
    result['environment']=dict(id=game.env.spec.id,actions=int(game.env.action_space.n),
                                action_meanings=game.env.unwrapped.get_action_meanings(),
                                stack_shape=list(state.shape),initial_lives=int(info['lives']),
                                frameskip=game.env.unwrapped._frameskip,
                                sticky_action_probability=game.env.unwrapped.ale.getFloat('repeat_action_probability'))
    replay=Replay(2048,223)
    replay.begin_episode()
    rng=np.random.default_rng(991)
    baselines=[]
    score=0.
    decisions=changes=life_losses=nonterminal_life_losses=0
    lives=int(info['lives'])
    for step in range(4000):
        action=int(rng.integers(ACTIONS))
        ns,r,term,trunc,info=game.step(action)
        replay.add(state,action,float(np.clip(r,-1.,1.)),ns,term)
        changes+=int(np.any(ns[-1]!=state[-1]))
        if int(info['lives'])<lives:
            life_losses+=lives-int(info['lives'])
            nonterminal_life_losses+=int(not term and not trunc)
        lives=int(info['lives'])
        score+=r
        decisions+=1
        state=ns
        if term or trunc:
            baselines.append(dict(seed=7_000_000+len(baselines),reward=score,length=decisions,
                                  lives=int(info['lives']),terminated=term,truncated=trunc))
            if len(baselines)>=3: break
            state,info=game.reset(7_000_000+len(baselines))
            replay.begin_episode()
            lives=int(info['lives'])
            score=0.
            decisions=0
    assert len(baselines)==3 and changes>100 and nonterminal_life_losses>0
    assert all(x['lives']==0 and x['terminated'] for x in baselines)
    result['random_games']=dict(summary=summarize(baselines),episodes=baselines,
                                changing_frames=changes,life_losses=life_losses,
                                nonterminal_life_losses=nonterminal_life_losses,
                                interpretation='One episode is a full game, not one life. Nonzero scores and changing images verify actual gameplay.')
    game.close()

    # A fixed action sequence must reproduce exactly when seeded, even with stickiness.
    game=Game()
    traces=[]
    actions=np.random.default_rng(812).integers(9,size=100)
    for repeat in range(2):
        state,_=game.reset(314159)
        digest=hashlib.sha256(state.tobytes())
        for action in actions:
            state,r,term,trunc,info=game.step(int(action))
            digest.update(state.tobytes())
            digest.update(str((r,term,trunc,info['lives'])).encode())
            if term or trunc: break
        traces.append(digest.hexdigest())
    assert traces[0]==traces[1]
    result['repeatability']=dict(action_count=100,seed=314159,identical=traces[0]==traces[1],sha256=traces[0])
    game.close()

    learner=Learner(device,1e-4)
    batch=replay.sample(32,device)
    before_parameters=[p.detach().clone() for p in learner.online.parameters()]
    before_target=[p.detach().clone() for p in learner.target.parameters()]
    # Predictable learning intervention: every sampled transition is made terminal
    # with target +1, so its sampled action-value should move toward +1.
    s,a,_,ns,_=batch
    controlled=(s,a,torch.ones(32,device=device),ns,torch.ones(32,dtype=torch.bool,device=device))
    with torch.no_grad():
        initial_loss=float(F.smooth_l1_loss(learner.online(s).gather(1,a[:,None]).squeeze(1),controlled[2]))
    for _ in range(40): metrics=learner.update(controlled)
    with torch.no_grad():
        final_loss=float(F.smooth_l1_loss(learner.online(s).gather(1,a[:,None]).squeeze(1),controlled[2]))
    assert final_loss<initial_loss*.5,(initial_loss,final_loss)
    assert any(not torch.equal(p,b) for p,b in zip(learner.online.parameters(),before_parameters))
    assert all(torch.equal(p,b) for p,b in zip(learner.target.parameters(),before_target))
    # Then exercise the actual replay and nonterminal Double DQN target repeatedly.
    for _ in range(100): metrics=learner.update(replay.sample(32,device))
    assert all(torch.isfinite(p).all() for p in learner.online.parameters())
    assert all(torch.equal(p,b) for p,b in zip(learner.target.parameters(),before_target))
    learner.sync_target()
    assert all(torch.equal(p,q) for p,q in zip(learner.online.parameters(),learner.target.parameters()))
    config=dict(gamma=.99,env='ALE/MsPacman-v5',purpose='preflight only; not a trained result')
    checkpoint=out/'checkpoint_roundtrip.pt'
    save_checkpoint(checkpoint,learner,config,3,replay.total,140,None)
    model,ck=load_model(checkpoint,device)
    with torch.no_grad():
        error=float((model(s)-learner.online(s)).abs().max())
    assert error==0
    result['gpu_learning']=dict(initial_controlled_loss=initial_loss,final_controlled_loss=final_loss,
                                production_updates=140,last_real_replay_update=metrics,
                                online_parameters_changed=True,target_frozen_until_explicit_sync=True,
                                synchronized_target_matches=True,checkpoint_max_q_difference=error,
                                gpu=torch.cuda.get_device_name(0),visible_gpu_count=torch.cuda.device_count(),
                                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    result['seconds']=time.time()-start
    result['status']='PASS'
    write_json(out/'preflight.json',result)
    print('PREFLIGHT PASS',flush=True)
    import json
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
