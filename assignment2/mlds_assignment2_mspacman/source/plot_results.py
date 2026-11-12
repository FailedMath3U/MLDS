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

    blocks=[rows[i:i+100] for i in range(0,len(rows),100)]
    rates=[score_rates(b) for b in blocks]
    x=[int(b[-1]['episode']) for b in blocks]
    fig,axes=plt.subplots(2,1,figsize=(8.4,3.8),sharex=True,layout='constrained')
    axes[0].plot(x,[r['mean_length'] for r in rates],marker='.',color='#154e79')
    axes[1].plot(x,[r['points_per_100_decisions'] for r in rates],marker='.',color='#267b67')
    axes[0].set(ylabel='Decisions per game',title='Non-standard diagnostic: longer games and faster scoring')
    axes[1].set(ylabel='Points per 100 decisions',xlabel='Training game (non-overlapping 100-game groups)')
    save(fig,'survival_and_scoring_rate')


def main():
    rows=training_rows()
    figures(rows)
    print('Generated the five MsPacman figures used in the report.')


if __name__=='__main__': main()
