"""Generate MsPacman plots and numeric metrics locally from recorded data."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
RES=ROOT/'results'
FIG=ROOT/'figures'

def load(path): return json.loads(Path(path).read_text())

def training_rows():
    with (RES/'seed1668/training.csv').open() as f: return list(csv.DictReader(f))

def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def score_rates(rows):
    n=len(rows)
    score=sum(float(r['reward']) for r in rows)
    length=sum(int(r['length']) for r in rows)
    return dict(games=n,mean_score=score/n,mean_length=length/n,
                points_per_100_decisions=100*score/length,
                mean_positive_events=sum(int(r['positive_reward_events']) for r in rows)/n)

def compute_metrics():
    rows=training_rows()
    evaluation=load(RES/'evaluation/episodes.json')
    random=load(RES/'report_checks/random_episodes.json')
    assert [r['seed'] for r in evaluation]==[r['seed'] for r in random]
    trained=np.array([r['reward'] for r in evaluation])
    baseline=np.array([r['reward'] for r in random])
    differences=trained-baseline
    rng=np.random.default_rng(1668)
    indices=rng.integers(0,len(trained),size=(10000,len(trained)))
    mean_ci=np.quantile(trained[indices].mean(1),[.025,.975])
    gain_ci=np.quantile(differences[indices].mean(1),[.025,.975])
    metrics=dict(first_100=score_rates(rows[:100]),last_100=score_rates(rows[-100:]),
                 mean_score_bootstrap95=mean_ci.tolist(),
                 paired_mean_gain=float(differences.mean()),paired_gain_bootstrap95=gain_ci.tolist(),
                 paired_wins=int((differences>0).sum()),paired_ties=int((differences==0).sum()),
                 paired_losses=int((differences<0).sum()),
                 trained_to_random_mean_ratio=float(trained.mean()/baseline.mean()),
                 bootstrap_seed=1668,bootstrap_resamples=10000,
                 interpretation='Game-level uncertainty for one fixed trained model, not across training seeds.')
    (RES/'report_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    return rows,metrics

def figures(rows):
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                        'figure.facecolor':'white'})
    episodes=np.array([int(r['episode']) for r in rows])
    reward=np.array([float(r['reward']) for r in rows])
    fig,ax=plt.subplots(figsize=(8.4,3.0))
    ax.plot(episodes,reward,lw=.4,alpha=.45,color='#8baac9',label='One training game')
    ax.plot(episodes[99:],np.convolve(reward,np.ones(100)/100,'valid'),color='#154e79',lw=2,
            label='Average of the last 100 games')
    ax.set(xlabel='Completed training games',ylabel='Original game score',title='Training rewards improve over the run')
    ax.legend(fontsize=8,loc='upper left')
    save(fig,'training_rewards')

    h=load(RES/'seed1668/validation_history.json')
    initial=load(RES/'seed1668/initial_validation.json')['summary']
    ev=load(RES/'evaluation/summary.json')
    x=[0]+[r['summary']['episode'] for r in h]
    y=[initial['mean']]+[r['summary']['mean'] for r in h]
    fig,ax=plt.subplots(figsize=(8.4,2.8))
    ax.plot(x,y,marker='o',ms=4,color='#267b67',label='Same 10 validation seeds; 1% random actions')
    ax.axvline(ev['checkpoint_episode'],color='#b46c13',ls='--',label='Selected checkpoint')
    ax.set(xlabel='Completed training games',ylabel='Mean validation score',title='Fixed-protocol validation shows learning, with setbacks')
    ax.legend(fontsize=8,loc='upper left')
    save(fig,'validation_progress')

    fig,ax=plt.subplots(figsize=(8.4,3.0))
    ax.plot(episodes,[float(r['q_max']) for r in rows],lw=.6,alpha=.65,label='Maximum across visited states and all 9 actions')
    ax.plot(episodes,[float(r['q_at_reset']) for r in rows],lw=1,color='#b46c13',label='Max-Q at the start of each game')
    ax.set(xlabel='Completed training games',ylabel='Predicted discounted, clipped reward',title='Maximum Q-values during training')
    ax.legend(fontsize=8,loc='upper left')
    save(fig,'maximum_q_values')

    scores=np.array([r['reward'] for r in load(RES/'evaluation/episodes.json')])
    random_scores=np.array([r['reward'] for r in load(RES/'report_checks/random_episodes.json')])
    bins=np.arange(0,np.ceil(max(scores.max(),random_scores.max())/100)*100+101,100)
    fig,ax=plt.subplots(figsize=(8.4,3.1))
    ax.hist(random_scores,bins=bins,color='#9ba8b6',alpha=.65,label='Uniform random actions, 500 games')
    ax.hist(scores,bins=bins,color='#267b67',alpha=.8,label='Trained model, 500 games')
    ax.axvline(scores.mean(),color='#154e79',ls='--',lw=1)
    ax.set(xlabel='Original full-game score',ylabel='Number of games',title='Final 500-game score histogram with a matched random reference')
    ax.legend(fontsize=8,loc='upper right')
    save(fig,'evaluation_histogram')

    random_summary=load(RES/'report_checks/random_summary.json')
    fig,ax=plt.subplots(figsize=(7.6,2.7))
    bars=ax.bar(['Random actions','Trained network'],[random_summary['mean'],ev['mean']],
                color=['#9ba8b6','#267b67'],width=.5)
    for bar,value in zip(bars,[random_summary['mean'],ev['mean']]):
        ax.text(bar.get_x()+bar.get_width()/2,value+35,f'{value:,.2f}',ha='center')
    ax.set(ylabel='Mean original score',ylim=(0,ev['mean']*1.2),title='Easy check: does the trained network beat random play?')
    save(fig,'random_comparison')

    blocks=[rows[i:i+100] for i in range(0,len(rows),100)]
    rates=[score_rates(b) for b in blocks]
    x=[int(b[-1]['episode']) for b in blocks]
    fig,axes=plt.subplots(2,1,figsize=(8.4,3.8),sharex=True,layout='constrained')
    axes[0].plot(x,[r['mean_length'] for r in rates],marker='.',color='#154e79')
    axes[1].plot(x,[r['points_per_100_decisions'] for r in rates],marker='.',color='#267b67')
    axes[0].set(ylabel='Decisions per game',title='Non-standard diagnostic: longer games and faster scoring')
    axes[1].set(ylabel='Points per 100 decisions',xlabel='Training game (non-overlapping 100-game groups)')
    save(fig,'survival_and_scoring_rate')

    trajectory=load(RES/'report_checks/diagnostic_trajectory.json')
    summary=load(RES/'report_checks/diagnostic_summary.json')
    t=np.arange(len(trajectory)+1)
    score=[0.]+[r['cumulative_score'] for r in trajectory]
    lives=[trajectory[0]['lives_before']]+[r['lives_after'] for r in trajectory]
    fig,axes=plt.subplots(2,1,figsize=(8.4,3.6),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1]})
    axes[0].step(t,score,where='post',color='#267b67')
    axes[1].step(t,lives,where='post',color='#154e79')
    for event in summary['life_loss_events']:
        for ax in axes: ax.axvline(event['step'],color='#b14c43',ls='--',lw=.8)
    axes[0].set(ylabel='Cumulative original score',title='One diagnostic game: score continues after the first life losses')
    axes[1].set(ylabel='Lives',xlabel='Agent decision',yticks=sorted(set(lives)))
    save(fig,'one_game_score_and_lives')

    images=np.load(RES/'report_checks/diagnostic_images.npz')
    chosen=int(np.argmin(abs(images['steps']-100)))
    fig,axes=plt.subplots(1,5,figsize=(10.5,3.0),layout='constrained')
    axes[0].imshow(images['rgb'][chosen]); axes[0].set_title('Original RGB\n210 × 160')
    for j,ax in enumerate(axes[1:]):
        ax.imshow(images['stacks'][chosen,j],cmap='gray',vmin=0,vmax=255)
        ax.set_title(f'Input frame {j+1}\n88 × 80')
    for ax in axes: ax.axis('off')
    fig.suptitle(f"What the network sees near decision {int(images['steps'][chosen])}")
    save(fig,'image_preprocessing')


def main():
    rows,_=compute_metrics()
    figures(rows)
    print('Generated eight MsPacman figures and numerical metrics.')


if __name__=='__main__': main()
