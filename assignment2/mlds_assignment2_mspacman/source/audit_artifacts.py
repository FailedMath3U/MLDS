"""Mac-only consistency audit; never retrains or reruns the evaluation."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent.parent
RES=ROOT/'results'


def load(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg=load(RES/'seed1668/config.json')
    training=load(RES/'seed1668/training_summary.json')
    evaluation=load(RES/'evaluation/episodes.json')
    summary=load(RES/'evaluation/summary.json')
    pre=load(RES/'preflight.json')
    assert training['completed'] and training['episodes']==5000
    assert cfg['env']=='ALE/MsPacman-v5' and cfg['gamma']==.99
    assert cfg['actions']==9 and cfg['visible_gpu_count']==1 and cfg['stack']==4
    assert cfg['frameskip']==4 and cfg['repeat_action_probability']==.25
    assert cfg['source_sha256']==sha(ROOT/'source/mspacman_dqn.py')
    digest=sha(RES/'seed1668/best.pt')
    assert summary['checkpoint_sha256']==digest
    assert summary['epsilon']==.01 and len(evaluation)==500
    assert [r['seed'] for r in evaluation]==list(range(9_000_000,9_000_500))
    values=[r['reward'] for r in evaluation]
    assert math.isclose(statistics.mean(values),summary['mean'],abs_tol=1e-10)
    assert math.isclose(statistics.pstdev(values),summary['std_population'],abs_tol=1e-10)
    for row in evaluation:
        assert row['terminated'] or row['truncated']
        assert 4*row['length']-3<=row['emulator_frames']<=4*row['length']
    with (RES/'seed1668/training.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==5000 and sum(int(r['length']) for r in rows)==training['steps']
    assert int(rows[-1]['updates'])==training['updates']
    assert int(rows[-1]['steps'])==training['steps']
    assert math.isclose(statistics.mean(float(r['reward']) for r in rows[-100:]),training['last_100_mean'])
    for row in rows:
        assert sum(json.loads(row['action_counts']))==int(row['length'])
        assert int(row['positive_reward_events'])<=int(row['length'])
    history=load(RES/'seed1668/validation_history.json')
    assert len(history)==20 and all(len(h['episodes'])==10 for h in history)
    best=max(history,key=lambda h:(h['summary']['mean'],h['summary']['min']))
    assert best['summary']['episode']==summary['checkpoint_episode']
    assert best['summary']['steps']==summary['checkpoint_steps']
    initial=load(RES/'seed1668/initial_validation.json')
    dev_seeds=[r['seed'] for r in initial['episodes']]
    assert dev_seeds==list(range(1_000_000,1_000_010))
    assert all([r['seed'] for r in h['episodes']]==dev_seeds for h in history)
    assert not set(dev_seeds).intersection(r['seed'] for r in evaluation)
    random=load(RES/'report_checks/random_episodes.json')
    random_summary=load(RES/'report_checks/random_summary.json')
    assert len(random)==500 and [r['seed'] for r in random]==[r['seed'] for r in evaluation]
    assert math.isclose(statistics.mean(r['reward'] for r in random),random_summary['mean'])
    checks=load(RES/'report_checks/summary.json')
    assert checks['status']=='PASS' and checks['checkpoint_sha256']==digest
    assert checks['model_parameters_unchanged'] and checks['visible_gpu_count']==1
    trajectory=load(RES/'report_checks/diagnostic_trajectory.json')
    diagnostic=load(RES/'report_checks/diagnostic_summary.json')
    assert len(trajectory)==diagnostic['steps']
    assert sum(r['reward'] for r in trajectory)==diagnostic['score']==diagnostic['sum_recorded_rewards']
    for r in trajectory:
        assert r['random_action'] or r['action']==max(range(9),key=lambda a:r['q_values'][a])
    assert pre['status']=='PASS' and pre['repeatability']['identical']
    assert pre['gpu_learning']['final_controlled_loss']<pre['gpu_learning']['initial_controlled_loss']
    names=['training_rewards','validation_progress','maximum_q_values','evaluation_histogram',
           'random_comparison','survival_and_scoring_rate','one_game_score_and_lives','image_preprocessing']
    for name in names:
        assert (ROOT/'figures'/(name+'.png')).read_bytes().startswith(b'\x89PNG')
        assert (ROOT/'figures'/(name+'.pdf')).read_bytes().startswith(b'%PDF')
    result=dict(status='PASS',training_games=len(rows),training_steps=training['steps'],
                final_evaluation_games=len(evaluation),random_reference_games=len(random),
                recomputed_mean=statistics.mean(values),recomputed_population_sd=statistics.pstdev(values),
                selected_checkpoint_episode=best['summary']['episode'],
                selection_uses_validation_only=True,source_hash_verified=True,
                checkpoint_sha256=digest,reported_score_matches_reward_ledger=True,
                figures_verified=len(names))
    (RES/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
