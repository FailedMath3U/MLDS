"""Plot only the final-policy results reported in the submission."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
out=ROOT/'plots';out.mkdir(exist_ok=True)
for mode in ('no_baseline','baseline'):
    folder=ROOT/'runs'/mode
    with (folder/'training.csv').open() as f:rows=list(csv.DictReader(f))
    x=np.array([int(r['episode']) for r in rows]);y=np.array([float(r['score']) for r in rows])
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(x,y,alpha=.2,lw=.6,label='Episode reward')
    ax.plot(x[99:],np.convolve(y,np.ones(100)/100,'valid'),lw=2,label='100-game mean')
    ax.set(xlabel='Training game',ylabel='Undiscounted Pong score',title=mode.replace('_',' '),ylim=(-22,22))
    ax.legend();fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'{mode}_training.{ext}',dpi=170)
    plt.close(fig)
    data=json.loads((folder/'evaluation/test_final.json').read_text());scores=[ep['score'] for ep in data['episodes']]
    fig,ax=plt.subplots(figsize=(8,4));ax.hist(scores,bins=np.arange(-21.5,22,1),edgecolor='white')
    ax.set(xlabel='Full-game Pong score',ylabel='Evaluation games',title=f"{mode}: mean {data['mean']:.3f}, SD {data['population_std']:.3f}")
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'{mode}_final_histogram.{ext}',dpi=170)
    plt.close(fig)
print(f'Final-policy plots saved to {out}')
