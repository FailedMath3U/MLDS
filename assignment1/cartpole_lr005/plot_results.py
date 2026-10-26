"""Plot all six completed runs; original artifacts are not modified."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'results/figures'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    summary=[]
    for mode in ('no_baseline','baseline'):
        fig,axes=plt.subplots(3,2,figsize=(11,9),layout='constrained')
        for row,seed in enumerate((1668,1669,1670)):
            folder=ROOT/'runs'/f'{mode}_seed{seed}'
            config=json.loads((folder/'config.json').read_text())
            assert config['lr']==.005 and config['episodes']==8000
            assert config['gamma']==.95 and config['outer_discount']
            scores=np.genfromtxt(folder/'training.csv',delimiter=',',names=True)['reward']
            evaluation=json.loads((folder/'evaluation.json').read_text())
            test=np.atleast_1d(np.loadtxt(folder/'evaluation_rewards.csv',skiprows=1))
            assert len(test)==500
            axes[row,0].scatter(np.arange(1,len(scores)+1),scores,s=2,alpha=.15)
            axes[row,0].plot(np.arange(100,len(scores)+1),np.convolve(scores,np.ones(100)/100,mode='valid'),color='navy')
            axes[row,0].set(title=f'Seed {seed}: training',xlabel='Training episode',ylabel='Episode reward',ylim=(0,510))
            axes[row,1].hist(test,bins=np.linspace(0,500,26),edgecolor='white')
            axes[row,1].set(title=f"Final policy: {evaluation['mean']:.3f} ± {evaluation['population_std']:.3f}",xlabel='Episode reward',ylabel='Count (500 evaluations)')
            summary.append({'mode':mode,'seed':seed,'lr':.005,'mean':evaluation['mean'],
                            'population_std':evaluation['population_std'],'fraction_500':evaluation['fraction_500']})
        fig.suptitle(f'CartPole-v1, {mode}; constant actor LR=0.005; final checkpoints')
        for ext in ('png','pdf'):fig.savefig(DEST/f'{mode}.{ext}',dpi=170)
        plt.close(fig)
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (ROOT/'results/summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
