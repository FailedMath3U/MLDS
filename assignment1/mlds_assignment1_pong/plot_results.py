"""Plot saved Pong results on CPU; safe to run while training is active."""
from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
out=ROOT/'plots';out.mkdir(exist_ok=True)
summary={}
for mode in ('no_baseline','baseline'):
    directory=ROOT/'runs'/mode
    training=directory/'training.csv'
    if training.exists():
        with training.open() as f:rows=list(csv.DictReader(f))
        if rows:
            x=np.array([int(r['episode']) for r in rows]);y=np.array([float(r['score']) for r in rows])
            fig,ax=plt.subplots(figsize=(8,4))
            ax.plot(x,y,alpha=.2,lw=.6,label='Full-game score')
            if len(y)>=100:ax.plot(x[99:],np.convolve(y,np.ones(100)/100,'valid'),lw=2,label='100-game mean')
            ax.set(xlabel='Training game',ylabel='Undiscounted Pong score',title=mode.replace('_',' '),ylim=(-22,22))
            ax.legend();fig.tight_layout()
            for ext in ('png','pdf'):fig.savefig(out/f'{mode}_training.{ext}',dpi=170)
            plt.close(fig)
            fig,axes=plt.subplots(2,1,figsize=(8,5),sharex=True)
            axes[0].plot(x,[float(r['mean_entropy']) for r in rows]);axes[0].set(ylabel='Action entropy')
            axes[1].plot(x,[float(r['gradient_norm_before_clip']) for r in rows]);axes[1].set(ylabel='Gradient norm',xlabel='Training game')
            fig.tight_layout();fig.savefig(out/f'{mode}_diagnostics.png',dpi=170);plt.close(fig)
    summary[mode]={}
    for selection in ('selected','final'):
        path=directory/'evaluation'/f'test_{selection}.json'
        if not path.exists():continue
        data=json.loads(path.read_text());scores=[ep['score'] for ep in data['episodes']]
        summary[mode][selection]={k:v for k,v in data.items() if k!='episodes'}
        fig,ax=plt.subplots(figsize=(8,4))
        ax.hist(scores,bins=np.arange(-21.5,22,1),edgecolor='white')
        ax.set(xlabel='Full-game Pong score',ylabel='Evaluation games',title=f'{mode} ({selection}): {len(scores)}/500 evaluations')
        ax.text(.03,.95,f"Mean {data['mean']:.3f}; SD {data['population_std']:.3f}",transform=ax.transAxes,va='top')
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(out/f'{mode}_{selection}_histogram.{ext}',dpi=170)
        plt.close(fig)
(out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
print(f'Plots saved to {out}')
