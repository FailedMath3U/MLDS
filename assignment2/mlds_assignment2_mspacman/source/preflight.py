"""GPU checks used in the MsPacman section: terminal targets, full games, learning.

Writes numerical evidence only; generate plots and reports on the local Mac.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from mspacman_dqn import ACTIONS,Game,Learner,Replay,targets,setup,write_json,summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    start=time.time()
    device=setup(1668)
    result={}
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
    for _ in range(40): learner.update(controlled)
    with torch.no_grad():
        final_loss=float(F.smooth_l1_loss(learner.online(s).gather(1,a[:,None]).squeeze(1),controlled[2]))
    assert final_loss<initial_loss*.5,(initial_loss,final_loss)
    assert any(not torch.equal(p,b) for p,b in zip(learner.online.parameters(),before_parameters))
    assert all(torch.equal(p,b) for p,b in zip(learner.target.parameters(),before_target))

    learner.sync_target()
    assert all(torch.equal(p,q) for p,q in zip(learner.online.parameters(),learner.target.parameters()))
    result['gpu_learning']=dict(initial_controlled_loss=initial_loss,final_controlled_loss=final_loss,
                                controlled_updates=40,online_parameters_changed=True,
                                target_frozen_until_explicit_sync=True,synchronized_target_matches=True,
                                gpu=torch.cuda.get_device_name(0),visible_gpu_count=torch.cuda.device_count())
    result['seconds']=time.time()-start
    result['status']='PASS'
    write_json(out/'preflight.json',result)
    print('PREFLIGHT PASS',flush=True)


if __name__=='__main__':main()
