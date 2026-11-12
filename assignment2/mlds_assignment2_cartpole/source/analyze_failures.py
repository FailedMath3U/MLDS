"""Reproduce the recorded DQN failures for the final report's failing-game plot.

Uses the original evaluation batch arrangement. The original held-out results
remain unchanged; this is a post-hoc diagnostic, with no retraining.
"""
import argparse
import hashlib
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from dqn import GAMMA,load_model,save_json,setup

SEEDS=[9_000_412,9_000_458,9_000_459]
EXPECTED={9_000_412:462,9_000_458:205,9_000_459:421}

@torch.no_grad()
def reproduce_dqn(model,device,out):
    results={}
    # Preserve the original evaluation batch size and seed ordering. A different
    # matrix-multiplication batch shape can change near-tied floating predictions.
    for start in [9_000_400,9_000_450]:
        envs=[gym.make('CartPole-v1') for _ in range(50)]
        obs=np.array([env.reset(seed=start+i)[0] for i,env in enumerate(envs)],np.float32)
        initial=obs.copy()
        alive=np.ones(50,bool)
        traces={seed:[] for seed in SEEDS if start<=seed<start+50}
        model(torch.as_tensor(obs,device=device)).max(1).values.cpu().numpy()
        for t in range(500):
            q=model(torch.as_tensor(obs,device=device)).cpu().numpy()
            actions=q.argmax(1)
            for i in np.flatnonzero(alive):
                seed=start+int(i)
                before=obs[i].copy()
                ns,r,term,trunc,_=envs[i].step(int(actions[i]))
                if seed in traces:
                    traces[seed].append(dict(t=t,state=before.tolist(),action=int(actions[i]),
                                             q_left=float(q[i,0]),q_right=float(q[i,1]),reward=r,
                                             next_state=ns.tolist(),terminated=bool(term),truncated=bool(trunc)))
                obs[i]=ns
                if term or trunc: alive[i]=False
            if not alive.any(): break
        for seed,trace in traces.items():
            i=seed-start
            final=trace[-1]['next_state']
            failure=[]
            if abs(final[0])>2.4: failure.append('cart position')
            if abs(final[2])>float(envs[i].unwrapped.theta_threshold_radians): failure.append('pole angle')
            results[seed]=dict(seed=seed,initial_state=initial[i].tolist(),
                               initial_angle_degrees=float(np.degrees(initial[i,2])),
                               original_evaluation_score=EXPECTED[seed],reproduced_score=len(trace),
                               matches_original=len(trace)==EXPECTED[seed],final_state=final,
                               final_angle_degrees=float(np.degrees(final[2])),
                               failure_boundary=failure,terminated=trace[-1]['terminated'],
                               initial_discounted_return_if_failure_at_original_length=(1-GAMMA**EXPECTED[seed])/(1-GAMMA))
            save_json(out/f'dqn_seed{seed}.json',trace)
        for env in envs: env.close()
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    device=setup(1668)
    model,checkpoint=load_model(args.checkpoint,device)
    cases=reproduce_dqn(model,device,out)
    summary=dict(checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                 checkpoint_step=checkpoint['step'],cases=list(cases.values()),
                 all_dqn_failures_reproduced=all(row['matches_original'] for row in cases.values()),
                 original_evaluation_unchanged=True)
    save_json(out/'summary.json',summary)
    print('Recorded failure lengths:',[row['reproduced_score'] for row in cases.values()])


if __name__=='__main__': main()
