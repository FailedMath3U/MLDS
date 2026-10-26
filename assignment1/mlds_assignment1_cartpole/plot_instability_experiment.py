"""Plot the completed controlled experiment from saved CSV/JSON, CPU only."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'instability_experiment'
data=json.loads((OUT/'results.json').read_text())
plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
for row,mode in enumerate(('no_baseline','baseline')):
 with (ROOT/f'runs/{mode}_seed1668/training.csv').open() as f:original=list(csv.DictReader(f))
 y=np.array([float(r['reward']) for r in original]);x=np.arange(1,len(y)+1)
 result=data['methods'][mode]
 ax=axes[row,0]
 ax.plot(x[99:],np.convolve(y,np.ones(100)/100,'valid'),label='Original training: 100-episode mean',color='#2563eb')
 positions=np.array([int(n) for n in result['checkpoints']]);results=list(result['checkpoints'].values())
 means=np.array([r['mean'] for r in results]);cis=np.array([r['mean_bootstrap_95ci'] for r in results])
 ax.errorbar(positions,means,yerr=[means-cis[:,0],cis[:,1]-means],fmt='D',color='#c2410c',capsize=5,ms=5,label='Frozen checkpoint: 500 fresh episodes, 95% CI',zorder=4)
 ax.set(xlabel='Training episodes completed',ylabel='Episode score / mean score',ylim=(0,515),title='Basic REINFORCE' if mode=='no_baseline' else 'REINFORCE with learned baseline')
 ax.legend(fontsize=8,loc='lower left')
 ax=axes[row,1]
 with (OUT/f'{mode}_frozen_control.csv').open() as f:control=list(csv.DictReader(f))
 frozen=np.array([float(r['reward']) for r in control]);n=result['frozen_control']['snapshot_training_episodes']
 continued=y[n:n+len(frozen)];idx=np.arange(100,len(frozen)+1)
 ax.plot(idx,np.convolve(frozen,np.ones(100)/100,'valid'),label=f'Frozen model after {n} training episodes',color='#0f766e',lw=1.6)
 ax.plot(idx,np.convolve(continued,np.ones(100)/100,'valid'),label='Original model continued learning',color='#c2410c',lw=1.6)
 ax.set(xlabel='Subsequent episode index (different samples)',ylabel='100-episode mean score',title='Fixed-policy sampling noise vs continued learning')
 ax.legend(fontsize=8,loc='lower left')
fig.suptitle('CartPole: sampling noise alone does not explain these training dips',fontsize=15)
fig.tight_layout(rect=(0,.02,1,.95))
fig.text(.5,.012,'Exact replay matched both original runs. Evaluations use fresh seeds; checkpoint selection was exploratory. Intervals are conditional on fixed policies.',ha='center',fontsize=9)
for ext in ('png','pdf'):fig.savefig(OUT/f'controlled_comparison.{ext}',dpi=180,bbox_inches='tight')
print('Saved controlled_comparison.png and .pdf')
