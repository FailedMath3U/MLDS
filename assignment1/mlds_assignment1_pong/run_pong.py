"""Single-GPU, resumable Pong suite. GPU binding is imposed by launch.sh.
CPU worker imports do not initialize CUDA. No report generation is performed.
"""
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parent
MODES = ('no_baseline', 'baseline')
DEFAULT = {'environment':'ALE/Pong-v5', 'gamma':.99, 'seed':1668,
    'workers':8, 'batch':8, 'chunk':512, 'lr':.001, 'clip_norm':5.,
    'baseline_ema_rate':.05, 'episodes_per_mode':20000, 'max_train_hours':16.,
    'validation_every':1024, 'validation_episodes':32, 'evaluation_episodes':500,
    'checkpoint_every_updates':5, 'diagnostic_every':4096,
    'gpu_uuid':'GPU-cef65388-88b0-f39a-6bac-c5316eb78540',
    'policy':'6400 -> 200 ReLU -> 2 logits; binary frame differences',
    'objective':'full-game discounted episodic return; outer gamma**t retained',
    'no_point_return_reset':True, 'return_normalization':False,
    'baseline':'constant EMA of discount-weighted reward-to-go from PREVIOUS batches',
    'lr_rationale':'0.001 instead of assignment default 0.01 for stability of image-policy Adam updates',
    'checkpoint_selection':'best 32-game validation mean; 500-game test sets never used for selection'}
STOP = False


def source_hash():
    h=hashlib.sha256()
    for path in [ROOT/'pong_env.py',ROOT/'pong_core.py',ROOT/'run_pong.py']:
        h.update(path.name.encode());h.update(path.read_bytes())
    return h.hexdigest()


def request_stop(signum, frame):
    global STOP
    STOP=True
    print(f'Signal {signum}: stop requested; finishing current complete-game batch before checkpoint.',flush=True)


def csv_write(path, rows):
    if not rows:
        return
    path=Path(path);temp=path.with_suffix('.csv.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def save_trace(ep, directory, name):
    directory.mkdir(parents=True,exist_ok=True)
    points=np.flatnonzero(ep['r'])
    indices=set(range(min(10,ep['length'])))|set(range(max(0,ep['length']-10),ep['length']))
    if len(points):
        indices.update(range(max(0,int(points[0])-5),min(ep['length'],int(points[0])+6)))
    indices=sorted(indices)
    rows=[{'step':i,'p_action2_right':float(ep['p2'][i]),'network_action_index':int(ep['a'][i]),
        'ale_action':(2,3)[ep['a'][i]],'reward':float(ep['r'][i]),
        'reward_to_go':float(ep['g'][i]),'outer_discount':float(.99**i),
        'nonzero_input_pixels':int(np.count_nonzero(ep['x'][i]))} for i in indices]
    csv_write(directory/f'{name}.csv',rows)
    np.savez_compressed(directory/f'{name}_frames.npz',steps=indices,frame_differences=ep['x'][indices])


def evaluate(pool, policy, config, device, directory, count, reset_base, action_base, name):
    """Atomic per-batch progress. Per-episode RNG makes resumed evaluation reproducible."""
    from pong_core import collect, episode_record, atomic_json, model_hash
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{name}.json'
    fingerprint=model_hash(policy)
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['model_hash'] != fingerprint:
            raise RuntimeError(f'Evaluation output belongs to a different checkpoint: {path}')
        rows=saved['episodes']
    else:
        rows=[]
        selected=directory/'test_selected.json'
        if name=='test_final' and selected.exists():
            cached=json.loads(selected.read_text())
            if cached['complete'] and cached['model_hash']==fingerprint and cached['requested_episodes']==count:
                atomic_json(path,cached)
                csv_write(directory/f'{name}_rewards.csv',cached['episodes'])
                print('Reused identical selected/final policy evaluation',flush=True)
                return cached
    while len(rows)<count:
        n=min(config['batch'],count-len(rows))
        episodes, timing=collect(pool,policy,n,reset_base,action_base,len(rows),device)
        for ep in episodes:
            rows.append(episode_record(ep,len(rows)+1))
        scores=np.array([row['score'] for row in rows])
        saved={'model_hash':fingerprint,'requested_episodes':count,'completed_episodes':len(rows),
            'complete':len(rows)==count,'mean':float(scores.mean()),'population_std':float(scores.std()),
            'sample_std':float(scores.std(ddof=1)) if len(scores)>1 else None,
            'win_fraction':float(np.mean(scores>0)), 'mean_length':float(np.mean([r['length'] for r in rows])),
            'stochastic_actions':True,'reset_base':reset_base,'action_base':action_base,'episodes':rows}
        atomic_json(path,saved)
        if len(rows)==n:
            save_trace(episodes[0],directory,f'{name}_trace')
        if STOP and name.startswith('test'):
            return saved
        if name.startswith('test'):
            print(f'{directory.parent.name}/{directory.name} {name}: {len(rows)}/{count}, mean={saved["mean"]:.2f}',flush=True)
    csv_write(directory/f'{name}_rewards.csv',rows)
    return saved


def gradient_probe(pool, agent, config, device, path, index, fitted_batch=None):
    """Welford variance across full trajectory gradients at ONE fixed actor.
    Does not update actor, optimizer or training EMA. Uses independent reset/action seeds.
    """
    import torch
    from pong_core import collect, actor_gradient, flat_arrays, atomic_json, model_hash
    if path.exists():
        return json.loads(path.read_text())
    before=model_hash(agent.policy)
    if agent.mode=='baseline' and agent.updates:
        baseline=agent.baseline
        fit_count=0
    else:
        fitting=fitted_batch
        if fitting is None:
            fitting,_=collect(pool,agent.policy,config['batch'],40000000+index*10000,41000000+index*10000,0,device)
        _,_,g,d=flat_arrays(fitting)
        baseline=float(np.sum(d*g)/np.sum(d));fit_count=len(fitting)
    total=0;means=[None,None];m2=[None,None];diff_mean=None;diff_m2=None
    for offset in range(0,16,config['batch']):
        episodes,_=collect(pool,agent.policy,min(config['batch'],16-offset),42000000+index*10000,43000000+index*10000,offset,device)
        for ep in episodes:
            vectors=[]
            for which,b in enumerate((0.,baseline)):
                actor_gradient(agent.policy,[ep],b,device,config['chunk'])
                vec=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in agent.policy.parameters()]).detach().cpu().numpy().astype(np.float64)
                vectors.append(vec)
                if means[which] is None:
                    means[which]=np.zeros_like(vec);m2[which]=np.zeros_like(vec)
                delta=vec-means[which];means[which]+=delta/(total+1);m2[which]+=delta*(vec-means[which])
            diff=vectors[1]-vectors[0]
            if diff_mean is None:diff_mean=np.zeros_like(diff);diff_m2=np.zeros_like(diff)
            delta=diff-diff_mean;diff_mean+=delta/(total+1);diff_m2+=delta*(diff-diff_mean)
            total+=1
        if offset==0:save_trace(episodes[0],path.parent,path.stem+'_trace')
    v0=float(np.sum(m2[0])/(total-1));v1=float(np.sum(m2[1])/(total-1))
    result={'training_episodes':len(agent.history),'model_hash':before,'test_trajectories':total,
        'baseline_fitting_trajectories':fit_count,'baseline_used':baseline,
        'gradient_variance_no_baseline':v0,'gradient_variance_with_baseline':v1,
        'variance_ratio':v1/v0 if v0 else None,
        'mean_gradient_difference_norm':float(np.linalg.norm(diff_mean)),
        'rms_standard_error_of_paired_mean_difference':float(np.sqrt(np.sum(diff_m2)/(total-1)/total)),
        'actor_unchanged':model_hash(agent.policy)==before,
        'interpretation':'Same frozen actor and paired held-out trajectories; measured outcome, not a forced improvement criterion.'}
    agent.policy.zero_grad(set_to_none=True)
    if not result['actor_unchanged']:raise AssertionError('Probe changed actor')
    atomic_json(path,result)
    return result


def preflight(config, device):
    import torch
    from pong_env import Pool, preprocess, make_env, metadata
    from pong_core import Agent, returns, actor_gradient, atomic_json, atomic_save, model_hash
    out=ROOT/'preflight';out.mkdir(exist_ok=True)
    checks={}
    def check(name, passed, **evidence):
        checks[name]={'passed':bool(passed),**evidence}
        print(f'PREFLIGHT {name}: {passed}',flush=True)
        if not passed:raise AssertionError((name,evidence))
    start=time.monotonic()
    image=np.zeros((210,160,3),dtype=np.uint8)
    image[35,0,0]=144;image[37,2,0]=109;image[39,4,0]=255;image[0,0,0]=255
    original=image.copy();x=preprocess(image)
    check('preprocessing_crop_background_and_no_input_mutation',x.shape==(80,80) and x[0,0]==0 and x[1,1]==0 and x[2,2]==1 and x.sum()==1 and np.array_equal(image,original))
    check('discounted_returns_across_point_boundaries',np.allclose(returns([0,1,0,-1]),[.019701,.0199,-.99,-1],atol=1e-6),
          observed=returns([0,1,0,-1]).tolist(),explanation='No artificial return reset at a nonzero point reward.')
    check('zero_discount',np.array_equal(returns([0,1,0,-1],0),[0,1,0,-1]))
    import itertools
    theta=.37;p=1/(1+np.exp(-theta));exact=1.99*p*(1-p);est=[]
    for baseline in (0.,1.7):
        total=0.
        for a,b in itertools.product((0,1),repeat=2):
            probability=(p if a else 1-p)*(p if b else 1-p)
            g=returns([a,b]);total+=probability*((a-p)*(g[0]-baseline)+.99*(b-p)*(g[1]-baseline))
        est.append(float(total))
    check('exact_expected_gradient_and_baseline_invariance',max(abs(v-exact) for v in est)<1e-7,
          analytic_gradient=float(exact),enumerated_gradients=est)
    traces=[]
    for _ in range(2):
        env=make_env();raw,_=env.reset(seed=700001);h=hashlib.sha256(raw.tobytes())
        total=0.
        for a in (2,3)*50:
            raw,reward,terminated,truncated,_=env.step(a)
            h.update(raw.tobytes());h.update(np.float64(reward).tobytes());total+=reward
            if terminated or truncated:break
        traces.append(h.hexdigest());meta=metadata(env);env.close()
    check('seeded_emulator_replay_with_sticky_actions',traces[0]==traces[1],hash=traces[0],environment=meta)
    check('single_gpu_and_cuda_kernels',torch.cuda.device_count()==1 and device.type=='cuda',
          visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_name=torch.cuda.get_device_name(0),torch=torch.__version__)
    agents={m:Agent(m,config,device) for m in MODES}
    check('matched_initial_policy',agents['no_baseline'].initial_hash==agents['baseline'].initial_hash,
          initial_hash=agents['no_baseline'].initial_hash)
    zero={'x':np.zeros((2,6400),dtype=np.int8),'a':np.array([0,1]),'g':np.zeros(2,dtype=np.float32),'length':2}
    actor_gradient(agents['no_baseline'].policy,[zero],0,device,config['chunk'])
    zero_norm=sum(float(p.grad.abs().sum()) for p in agents['no_baseline'].policy.parameters() if p.grad is not None)
    check('zero_reward_produces_zero_gradient',zero_norm==0.,gradient_l1=zero_norm)
    positive={'x':np.zeros((1,6400),dtype=np.int8),'a':np.array([0]),'g':np.ones(1,dtype=np.float32),'length':1}
    test=Agent('no_baseline',config,device)
    with torch.no_grad():before=float(test.policy(torch.zeros((1,6400),device=device)).softmax(-1)[0,0])
    actor_gradient(test.policy,[positive],0.,device,config['chunk']);test.optimizer.step()
    with torch.no_grad():after=float(test.policy(torch.zeros((1,6400),device=device)).softmax(-1)[0,0])
    check('positive_return_increases_chosen_action_probability',after>before,before=before,after=after)
    pool=Pool(config['workers'])
    try:
        episodes,timing=__import__('pong_core').collect(pool,agents['no_baseline'].policy,config['batch'],500000,600000,0,device)
        check('complete_games_without_fire_or_noop',all(ep['terminated'] or ep['truncated'] for ep in episodes) and all(np.count_nonzero(ep['r'])>0 for ep in episodes),
              scores=[ep['score'] for ep in episodes],lengths=[ep['length'] for ep in episodes],
              points=[int(np.count_nonzero(ep['r'])) for ep in episodes],allowed_actions=[2,3])
        check('frame_difference_reset_and_range',all(np.all(ep['x'][0]==0) and set(np.unique(ep['x'])) <= {-1,0,1} for ep in episodes))
        save_trace(episodes[0],out,'initial_game_trace')
        # Same on-policy data, independent one-update learning-rate probes.
        probe=np.concatenate([ep['x'][:128] for ep in episodes])
        probe_tensor=torch.as_tensor(probe,device=device,dtype=torch.float32)
        with torch.no_grad():old_prob=agents['no_baseline'].policy(probe_tensor).softmax(-1)
        lr_evidence={}
        for rate in (.01,config['lr']):
            candidate=Agent('no_baseline',{**config,'lr':rate},device)
            candidate.update(episodes)
            with torch.no_grad():
                new_prob=candidate.policy(probe_tensor).softmax(-1).clamp_min(1e-12)
                kl=float((old_prob*(old_prob.clamp_min(1e-12).log()-new_prob.log())).sum(-1).mean())
                entropy=float(-(new_prob*new_prob.log()).sum(-1).mean())
            lr_evidence[str(rate)]={'mean_forward_kl_after_one_update':kl,'mean_entropy':entropy}
            del candidate
        check('learning_rate_step_size_probe',all(np.isfinite(v['mean_forward_kl_after_one_update']) for v in lr_evidence.values()),
              learning_rates=lr_evidence,chosen_lr=config['lr'],
              interpretation='Same initial actor and same complete-game data. This checks update aggressiveness, not final performance.')
        del probe_tensor
        agent=agents['no_baseline'];agent.update(episodes)
        atomic_save(out/'resume_test.pt',agent.state())
        restored=Agent('no_baseline',config,device)
        restored.restore(torch.load(out/'resume_test.pt',map_location=device,weights_only=False))
        # Repeat the SAME fixed training-data update in two copies. This tests
        # optimizer restoration, not an additional on-policy training experiment.
        agent.update(episodes);restored.update(episodes)
        check('checkpoint_restores_optimizer_and_next_update',model_hash(agent.policy)==model_hash(restored.policy),
              uninterrupted_hash=model_hash(agent.policy),restored_hash=model_hash(restored.policy),
              completed_updates=restored.updates)
        b_agent=agents['baseline'];b_before=b_agent.baseline;b_agent.update(episodes)
        check('baseline_uses_only_previous_batches',b_before==0 and all(row['baseline_used']==b_before for row in b_agent.history),
              used=b_before,updated=b_agent.baseline)
        # Variance is measured, not asserted to improve for every possible policy.
        fresh=Agent('no_baseline',config,device)
        variance=gradient_probe(pool,fresh,config,device,out/'initial_gradient_variance.json',0,fitted_batch=episodes)
        checks['initial_gradient_variance_measurement']={'passed':variance['actor_unchanged'],**variance}
        runtime={'seconds':timing['seconds'],'episodes':len(episodes),'decisions':timing['decisions'],
            'decisions_per_second':timing['decisions']/timing['seconds'],
            'initial_seconds_per_game':timing['seconds']/len(episodes),
            'initial_game_budget_hours':2*config['episodes_per_mode']*timing['seconds']/len(episodes)/3600,
            'warning':'Later games can be much longer. Validation, gradient probes and final testing add time.'}
        print('BENCHMARK',json.dumps(runtime),flush=True)
    finally:
        pool.close()
    import gymnasium,ale_py,platform
    result={'passed':all(value['passed'] for value in checks.values()),'source_hash':source_hash(),
        'config':config,'checks':checks,'benchmark':runtime,'environment':meta,
        'software':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
                    'gymnasium':gymnasium.__version__,'ale_py':ale_py.__version__},
        'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
        'preflight_seconds':time.monotonic()-start}
    atomic_json(out/'results.json',result)
    print('PREFLIGHT COMPLETE',flush=True)


def train(config, device):
    import torch
    from pong_env import Pool
    from pong_core import Agent, atomic_save, atomic_json, collect, model_hash
    proof_path=ROOT/'preflight/results.json'
    if not proof_path.exists():raise RuntimeError('Run preflight first')
    proof=json.loads(proof_path.read_text())
    if not proof['passed'] or proof['source_hash']!=source_hash() or proof['config']!=config:
        raise RuntimeError('Preflight is stale or failed: source/config changed. Re-run it before training.')
    agents={mode:Agent(mode,config,device) for mode in MODES}
    state={'schema':1,'source_hash':source_hash(),'config':config,'elapsed_seconds':0.,
           'training_complete':False,'last_validation':{mode:0 for mode in MODES},
           'last_diagnostic':{mode:0 for mode in MODES}}
    checkpoint=ROOT/'checkpoints/latest.pt';checkpoint.parent.mkdir(exist_ok=True)
    candidates=[checkpoint,checkpoint.with_suffix('.previous.pt')]
    restored=False
    for candidate in candidates:
        if not candidate.exists():continue
        try:
            loaded=torch.load(candidate,map_location=device,weights_only=False)
            if loaded['config']!=config or loaded['source_hash']!=source_hash():
                raise RuntimeError('Checkpoint configuration/source mismatch')
            for mode in MODES:agents[mode].restore(loaded['agents'][mode])
            state={k:v for k,v in loaded.items() if k not in ('agents','torch_rng','cuda_rng')}
            torch.set_rng_state(loaded['torch_rng'].cpu())
            torch.cuda.set_rng_state(loaded['cuda_rng'].cpu())
            restored=True
            print(f'RESUMED {candidate}; episodes {[len(a.history) for a in agents.values()]}',flush=True)
            break
        except Exception:
            print(f'Could not load {candidate}:\n{traceback.format_exc()}',flush=True)
    if any(c.exists() for c in candidates) and not restored:
        raise RuntimeError('Existing checkpoints could not be recovered; refusing to silently start over.')
    started=time.monotonic();previous_elapsed=state['elapsed_seconds']
    def persist(phase):
        state['elapsed_seconds']=previous_elapsed+time.monotonic()-started
        payload={**state,'agents':{m:a.state() for m,a in agents.items()},
                 'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
        atomic_save(checkpoint,payload,previous=True)
        for m,a in agents.items():
            directory=ROOT/'runs'/m;directory.mkdir(parents=True,exist_ok=True)
            csv_write(directory/'training.csv',a.history)
        status(phase)
    def status(phase):
        elapsed=previous_elapsed+time.monotonic()-started
        atomic_json(ROOT/'status.json',{'phase':phase,'pid':os.getpid(),'gpu_uuid':config['gpu_uuid'],
            'elapsed_hours':elapsed/3600,'max_training_hours':config['max_train_hours'],
            'updated_unix':time.time(),'models':{m:{'episodes':len(a.history),'updates':a.updates,
                'recent100_mean':float(np.mean([r['score'] for r in a.history[-100:]])) if a.history else None,
                'baseline':a.baseline,'best_validation_mean':a.best_validation if np.isfinite(a.best_validation) else None,
                'best_validation_episodes':a.best_validation_episodes} for m,a in agents.items()},
            'gpu_memory_allocated_bytes':torch.cuda.memory_allocated(),
            'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated()})
    persist('starting')
    pool=Pool(config['workers'])
    try:
        if not restored:
            initial=evaluate(pool,agents['no_baseline'].policy,config,device,ROOT/'references',32,3000000,3100000,'untrained')
            for mode in MODES:
                atomic_save(ROOT/'runs'/mode/'initial.pt',{'policy':agents[mode].policy.state_dict(),'config':config,'episodes':0})
        while not state['training_complete']:
            counts=[len(a.history) for a in agents.values()]
            elapsed=previous_elapsed+time.monotonic()-started
            if min(counts)>=config['episodes_per_mode'] or (len(set(counts))==1 and elapsed>=config['max_train_hours']*3600):
                state['training_complete']=True;persist('training_finished');break
            target=min(min(counts)+config['batch'],config['episodes_per_mode'])
            for mode in sorted(MODES,key=lambda m:len(agents[m].history)):
                agent=agents[mode];n=len(agent.history)
                if n>=target:continue
                if STOP:persist('paused');return
                frozen_b=agent.baseline
                batch,timing=collect(pool,agent.policy,min(config['batch'],config['episodes_per_mode']-n),1000000,2000000,n,device)
                if agent.baseline!=frozen_b:raise AssertionError('Baseline changed during rollout')
                metrics=agent.update(batch);n=len(agent.history)
                print(f'{mode}: episode={n}, score100={np.mean([r["score"] for r in agent.history[-100:]]):.2f}, '
                      f'grad={metrics["gradient_norm"]:.4g}, baseline={metrics["baseline"]:.4g}, '
                      f'fps={timing["decisions"]/timing["seconds"]:.0f}',flush=True)
                status('training')
                if agent.updates%config['checkpoint_every_updates']==0:persist('training')
                if n-state['last_validation'][mode]>=config['validation_every']:
                    result=evaluate(pool,agent.policy,config,device,ROOT/'runs'/mode/'validation',
                        config['validation_episodes'],10000000,11000000,f'episode_{n:06d}')
                    state['last_validation'][mode]=n
                    atomic_save(ROOT/'checkpoints'/f'{mode}_episode_{n:06d}.pt',
                        {'policy':agent.policy.state_dict(),'config':config,'episodes':n,'validation_mean':result['mean']})
                    if result['mean']>agent.best_validation:
                        agent.best_validation=result['mean'];agent.best_validation_episodes=n
                        atomic_save(ROOT/'runs'/mode/'best_validation.pt',
                            {'policy':agent.policy.state_dict(),'config':config,'episodes':n,'validation_mean':result['mean']})
                    print(f'VALIDATION {mode} n={n}: mean={result["mean"]:.2f}; best={agent.best_validation:.2f}',flush=True)
                    persist('training')
                if n-state['last_diagnostic'][mode]>=config['diagnostic_every']:
                    result=gradient_probe(pool,agent,config,device,ROOT/'diagnostics'/f'{mode}_{n:06d}.json',
                        10+2*(n//config['diagnostic_every'])+MODES.index(mode))
                    state['last_diagnostic'][mode]=n;persist('training')
                if STOP:persist('paused');return
        # Always compare the final policy with earlier validations before final test.
        for mode,agent in agents.items():
            n=len(agent.history)
            result=evaluate(pool,agent.policy,config,device,ROOT/'runs'/mode/'validation',
                config['validation_episodes'],10000000,11000000,f'episode_{n:06d}')
            if result['mean']>agent.best_validation or not (ROOT/'runs'/mode/'best_validation.pt').exists():
                agent.best_validation=result['mean'];agent.best_validation_episodes=n
                atomic_save(ROOT/'runs'/mode/'best_validation.pt',{'policy':agent.policy.state_dict(),
                    'config':config,'episodes':n,'validation_mean':result['mean']})
            atomic_save(ROOT/'runs'/mode/'final.pt',{'policy':agent.policy.state_dict(),'config':config,'episodes':n})
        persist('evaluating')
        if STOP:status('paused');return
        for mode,agent in agents.items():
            for selection,filename in (('selected','best_validation.pt'),('final','final.pt')):
                saved=torch.load(ROOT/'runs'/mode/filename,map_location=device,weights_only=False)
                policy=__import__('pong_core').make_policy(config['seed'],device)
                policy.load_state_dict(saved['policy'])
                result=evaluate(pool,policy,config,device,ROOT/'runs'/mode/'evaluation',config['evaluation_episodes'],
                    20000000,21000000,f'test_{selection}')
                atomic_json(ROOT/'runs'/mode/'evaluation'/f'{selection}_provenance.json',
                    {'checkpoint':filename,'training_episodes':saved['episodes'],'selection_used_test_scores':False,
                     'evaluation_mean':result['mean'],'population_std':result['population_std']})
                del policy
                if STOP:status('paused');return
        persist('complete')
        summary={mode:{key:json.loads((ROOT/'runs'/mode/'evaluation'/f'test_{key}.json').read_text())
                 for key in ('selected','final')} for mode in MODES}
        atomic_json(ROOT/'summary.json',summary)
        print('ALL TRAINING AND 500-EPISODE TEST EVALUATIONS COMPLETE',flush=True)
    finally:
        pool.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['preflight','train','status'])
    args=parser.parse_args()
    if args.phase=='status':
        path=ROOT/'status.json'
        print(path.read_text() if path.exists() else 'No training status yet.');return
    ROOT.mkdir(exist_ok=True)
    lock=(ROOT/'job.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A Pong preflight/training process is already running.')
    (ROOT/'trainer.pid').write_text(str(os.getpid())+'\n')
    config_path=ROOT/'config.json'
    from pong_core import atomic_json
    if not config_path.exists():atomic_json(config_path,DEFAULT)
    config=json.loads(config_path.read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=config['gpu_uuid']:
        raise RuntimeError('Bind EXACTLY the configured GPU UUID before running.')
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one CUDA device must be available; no silent CPU fallback.')
    device=torch.device('cuda:0')
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    try:
        if args.phase=='preflight':preflight(config,device)
        else:train(config,device)
    except BaseException:
        atomic_json(ROOT/'failure.json',{'time_unix':time.time(),'pid':os.getpid(),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
