"""Separate sampling noise from changes in CartPole policy quality.
Exploratory checkpoint selection uses existing training curves; new evaluation
seeds are reserved before inspecting their outcomes. Original files unchanged.
CPU-only exact replay, followed by independently coded sequential evaluation.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv,json,hashlib,time
import numpy as np
import torch
from torch import nn
import gymnasium as gym
from cartpole import set_seed,network,collect,packed,GAMMA

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'instability_experiment'
CHECKPOINTS={'no_baseline':(5120,6400,6912,8000),'baseline':(3072,6144,6912,8000)}

def writecsv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def state_hash(state):
 h=hashlib.sha256()
 for key,value in state.items():h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def replay(mode):
 seed=1668
 config=json.loads((ROOT/f'runs/{mode}_seed{seed}/config.json').read_text())
 assert config['episodes']==8000 and config['batch_size']==32 and config['lr']==.01
 set_seed(seed)
 policy=network()
 initial=torch.load(ROOT/f'runs/{mode}_seed{seed}/initial.pt',map_location='cpu',weights_only=True)['policy']
 assert state_hash(policy.state_dict())==state_hash(initial)
 critic=network(5,1) if mode=='baseline' else None
 torch.manual_seed(seed+100)
 optimizer=torch.optim.Adam(policy.parameters(),lr=.01)
 critic_optimizer=torch.optim.Adam(critic.parameters(),lr=.01) if critic is not None else None
 original=list(csv.DictReader((ROOT/f'runs/{mode}_seed{seed}/training.csv').open()))
 original_scores=np.array([float(r['reward']) for r in original])
 rewards=[];snapshots={};start=time.monotonic()
 for offset in range(0,8000,32):
  batch=collect(policy,32,seed*100000+offset)
  states,actions,targets,times,critic_inputs,discount=packed(batch)
  with torch.no_grad():
   b=critic(critic_inputs).squeeze(1) if critic is not None else torch.zeros_like(targets)
  log_prob=torch.distributions.Categorical(logits=policy(states)).log_prob(actions)
  loss=-(1-GAMMA)*(discount*log_prob*(targets-b)).sum()/len(batch)
  optimizer.zero_grad();loss.backward()
  torch.nn.utils.clip_grad_norm_(policy.parameters(),10.0)
  optimizer.step()
  if critic is not None:
   for _ in range(5):
    critic_loss=((critic(critic_inputs).squeeze(1)-targets)**2).mean()
    critic_optimizer.zero_grad();critic_loss.backward();critic_optimizer.step()
  rewards.extend(sum(ep['rewards']) for ep in batch)
  n=len(rewards)
  if n in CHECKPOINTS[mode]:
   snapshots[n]={k:v.detach().clone() for k,v in policy.state_dict().items()}
   torch.save({'policy':snapshots[n],'training_episodes':n,'mode':mode},OUT/f'{mode}_{n}.pt')
   print('SNAPSHOT',mode,n,'replay-last100',np.mean(rewards[-100:]),flush=True)
 replayed=np.asarray(rewards)
 final_original=torch.load(ROOT/f'runs/{mode}_seed{seed}/last.pt',map_location='cpu',weights_only=True)['policy']
 record={'all_8000_rewards_identical':bool(np.array_equal(replayed,original_scores)),
         'number_different_rewards':int(np.sum(replayed!=original_scores)),
         'final_weights_identical':state_hash(policy.state_dict())==state_hash(final_original),
         'final_weight_hash':state_hash(policy.state_dict()),
         'original_weight_hash':state_hash(final_original),'seconds':time.monotonic()-start}
 (OUT/f'{mode}_replay.json').write_text(json.dumps(record,indent=2)+'\n')
 print('REPLAY',mode,record,flush=True)
 if not (record['all_8000_rewards_identical'] and record['final_weights_identical']):
  raise RuntimeError('Replay is not identical; do not attribute its intermediate models to the original trajectory.')
 return snapshots,record

def evaluate(state,count,reset_base,action_base):
 # Separate evaluator: never calls training collector and never uses torch RNG.
 policy=network();policy.load_state_dict(state);policy.eval()
 before=state_hash(policy.state_dict())
 env=gym.make('CartPole-v1');assert env.spec.max_episode_steps==500
 rows=[]
 with torch.inference_mode():
  for i in range(count):
   obs,_=env.reset(seed=reset_base+i)
   rng=np.random.default_rng(action_base+i)
   total=0.;steps=0;terminated=truncated=False
   while not (terminated or truncated):
    p=policy(torch.as_tensor(obs,dtype=torch.float32)).softmax(-1).numpy()
    action=int(rng.random()<p[1])
    obs,reward,terminated,truncated,_=env.step(action)
    total+=float(reward);steps+=1
    assert steps<=500
   assert total==steps
   rows.append({'episode':i+1,'reset_seed':reset_base+i,'action_seed':action_base+i,
                'reward':total,'terminated':terminated,'truncated':truncated})
 env.close()
 assert before==state_hash(policy.state_dict())
 return np.array([r['reward'] for r in rows]),rows

def bootstrap_mean(values,seed):
 rng=np.random.default_rng(seed);means=[]
 for _ in range(5):
  samples=rng.integers(0,len(values),size=(1000,len(values)))
  means.extend(values[samples].mean(axis=1))
 return [float(v) for v in np.quantile(means,[.025,.975])]

def main():
 OUT.mkdir(exist_ok=True)
 plan={'training_seed':1668,'checkpoints':CHECKPOINTS,'evaluation_episodes_per_checkpoint':500,
       'frozen_control_episodes':2000,'evaluation_reset_base':9400000,'evaluation_action_base':34000000,
       'control_reset_base':9500000,'control_action_base':35000000,'bootstrap_replicates':5000,
       'selection':'Exploratory: checkpoints chosen from the already observed training logs, before any new evaluation outcomes.',
       'hypothesis':'If the dip is only fixed-policy episode noise, held-out means at before/during checkpoints should not show a large change. A frozen high-performing policy supplies a sampling-only comparison.',
       'original_files_modified':False,'device':'cpu','gpus_used':0}
 (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 summary={'plan':plan,'methods':{}}
 # Complete all training replays BEFORE constructing evaluation networks so
 # evaluation cannot alter either training RNG stream.
 replays={}
 for mode in CHECKPOINTS:replays[mode]=replay(mode)
 for mode,(snapshots,replay_record) in replays.items():
  results={};scores={}
  for n,state in snapshots.items():
   values,rows=evaluate(state,500,9400000,34000000)
   scores[n]=values
   writecsv(OUT/f'{mode}_{n}_evaluation.csv',rows)
   results[str(n)]={'mean':float(values.mean()),'sd':float(values.std()),'mean_bootstrap_95ci':bootstrap_mean(values,101+n),
                    'min':float(values.min()),'max':float(values.max()),'count':500,
                    'fraction_at_500':float(np.mean(values==500))}
   print('EVALUATION',mode,n,results[str(n)],flush=True)
  before,during,recovered,final=CHECKPOINTS[mode]
  paired={}
  for key,a,b in [('during_minus_before',during,before),('final_minus_during',final,during)]:
   difference=scores[a]-scores[b]
   paired[key]={'mean_difference':float(difference.mean()),'paired_standard_error':float(difference.std(ddof=1)/np.sqrt(500)),
                'paired_bootstrap_95ci':bootstrap_mean(difference,202+a),'pairs':500}
  values,rows=evaluate(snapshots[before],2000,9500000,35000000)
  writecsv(OUT/f'{mode}_frozen_control.csv',rows)
  frozen_ma=np.convolve(values,np.ones(100)/100,mode='valid')
  original=list(csv.DictReader((ROOT/f'runs/{mode}_seed1668/training.csv').open()))
  continued=np.array([float(r['reward']) for r in original[before:before+2000]])
  active_ma=np.convolve(continued,np.ones(100)/100,mode='valid')
  control={'snapshot_training_episodes':before,'episodes':2000,'mean':float(values.mean()),'sd':float(values.std()),
           'frozen_100episode_mean_min':float(frozen_ma.min()),'frozen_100episode_mean_max':float(frozen_ma.max()),
           'continued_training_100episode_mean_min':float(active_ma.min()),'continued_training_100episode_mean_max':float(active_ma.max()),
           'comparison':'Same policy frozen throughout control; fresh independent seeds. Continued-training series uses original subsequent 2000 episodes. Rolling windows overlap; their range is descriptive, not a p-value.'}
  summary['methods'][mode]={'replay':replay_record,'checkpoints':results,'paired_comparisons':paired,'frozen_control':control}
  (OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
  print('CONTROL',mode,control,flush=True)
 print('EXPERIMENT COMPLETE',flush=True)
if __name__=='__main__':main()
