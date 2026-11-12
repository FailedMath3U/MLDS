"""Numerical diagnostics used in the final CartPole report.

Produces the action maps, one successful game trace, and the initial-state
stress grid. Generate all plots and the PDF locally on the Mac.
"""
import argparse
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from dqn import load_model,rollout,save_json,setup,summarize

@torch.no_grad()
def phase_maps(model,device):
    angles=np.linspace(-.21,.21,101)
    velocities=np.linspace(-2.,2.,101)
    theta,omega=np.meshgrid(angles,velocities)
    maps=[]
    for x in [-1.5,0.,1.5]:
        s=np.column_stack([np.full(theta.size,x),np.zeros(theta.size),theta.ravel(),omega.ravel()]).astype(np.float32)
        q=model(torch.as_tensor(s,device=device)).cpu().numpy()
        maps.append({'x':x,'right_minus_left':(q[:,1]-q[:,0]).reshape(theta.shape).tolist(),
                     'max_q':q.max(1).reshape(theta.shape).tolist()})
    return {'angles':angles.tolist(),'angular_velocities':velocities.tolist(),'maps':maps}

@torch.no_grad()
def policy_trace(model,device):
    env=gym.make('CartPole-v1')
    state,_=env.reset(seed=4_000_000)
    rows=[]
    for t in range(500):
        q=model(torch.as_tensor(state,device=device)).cpu().numpy()
        action=int(q.argmax())
        ns,r,term,trunc,_=env.step(action)
        rows.append({'t':t,'state':state.tolist(),'q_left':float(q[0]),'q_right':float(q[1]),
                     'action':action,'reward':r,'next_state':ns.tolist(),
                     'terminated':bool(term),'truncated':bool(trunc)})
        state=ns
        if term or trunc: break
    env.close()
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    device=setup(1668)
    model,_=load_model(args.checkpoint,device)
    save_json(out/'policy_trajectory.json',policy_trace(model,device))
    save_json(out/'phase_maps.json',phase_maps(model,device))
    xs=np.linspace(-1.8,1.8,9)
    thetas=np.linspace(-.18,.18,9)
    states=[[float(x),0.,float(theta),0.] for theta in thetas for x in xs]
    rows,_=rollout(model,list(range(5_000_000,5_000_000+len(states))),device,states=states)
    for row,state in zip(rows,states):row['initial_state']=state
    save_json(out/'robustness_grid.json',{'xs':xs.tolist(),'thetas':thetas.tolist(),
                                        'episodes':rows,'summary':summarize(rows)})
    print('Saved the phase map, successful game trace, and initial-state grid.')


if __name__=='__main__': main()
