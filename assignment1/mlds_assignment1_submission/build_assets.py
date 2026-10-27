"""Rebuild only the final report's figures and statistics from saved raw results.
Does not train or depend on another report directory.
"""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PROJECTS=ROOT.parent
CART=PROJECTS/'mlds_assignment1_cartpole'
PONG=PROJECTS/'mlds_assignment1_pong'
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42})

def readcsv(path):
    with path.open() as f:return list(csv.DictReader(f))

def save(fig,name,png=False):
    fig.tight_layout()
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    if png:fig.savefig(FIG/f'{name}.png',dpi=170,bbox_inches='tight')
    plt.close(fig)

metrics=[]
for game in ('cartpole','pong'):
    for mode,prefix in [('no_baseline','Basic'),('baseline','Baseline')]:
        color='#2563eb' if mode=='no_baseline' else '#0f766e'
        if game=='cartpole':
            directory=CART/f'runs/{mode}_seed1668'
            rows=readcsv(directory/'training.csv');key='reward'
            scores=np.loadtxt(directory/'evaluation_rewards.csv',skiprows=1)
            record=json.loads((directory/'evaluation.json').read_text());expected=8000
        else:
            directory=PONG/f'runs/{mode}'
            rows=readcsv(directory/'training.csv');key='score'
            record=json.loads((directory/'evaluation/test_final.json').read_text())
            scores=np.array([ep['score'] for ep in record['episodes']]);expected=20000
        assert len(rows)==expected and len(scores)==500
        assert np.isclose(scores.mean(),record['mean'],atol=1e-12)
        assert np.isclose(scores.std(),record['population_std'],atol=1e-12)
        label='Cart' if game=='cartpole' else 'Pong'
        metrics.extend([f'\\newcommand{{\\{label}{prefix}Mean}}{{{scores.mean():.3f}}}',
                        f'\\newcommand{{\\{label}{prefix}SD}}{{{scores.std():.3f}}}'])
        x=np.array([int(r['episode']) for r in rows]);y=np.array([float(r[key]) for r in rows])
        fig,ax=plt.subplots(figsize=(6.8,3.05))
        ax.plot(x,y,lw=.6,alpha=.23,color=color,label='Episode reward')
        ax.plot(x[99:],np.convolve(y,np.ones(100)/100,'valid'),lw=1.9,color=color,label='100-episode moving average')
        ax.set(xlabel='Training episode (complete game)',ylabel='Undiscounted episode reward',xlim=(1,len(x)),
               ylim=(0,510) if game=='cartpole' else (-22,22))
        ax.legend(loc='best',fontsize=9);save(fig,f'{game}_{mode}_training')
        fig,ax=plt.subplots(figsize=(6.8,3.05))
        bins=np.linspace(0,500,26) if game=='cartpole' else np.arange(-21.5,22,1)
        ax.hist(scores,bins=bins,color=color,edgecolor='white',linewidth=.4)
        ax.axvline(scores.mean(),c='black',ls='--',lw=1,label=f'Mean {scores.mean():.3f}; SD {scores.std():.3f}')
        ax.set(xlabel='Undiscounted episode reward',ylabel='Number of evaluation episodes',
               xlim=(0,510) if game=='cartpole' else (-22,22))
        ax.legend(fontsize=9);save(fig,f'{game}_{mode}_hist')
(ROOT/'metrics_macros.tex').write_text('\n'.join(metrics)+'\n')

shuttle=PROJECTS/'mlds_assignment1_shuttle'
d=np.load(shuttle/'results/shuttle_results.npz')
s=json.loads((shuttle/'results/summary.json').read_text())
assert all(v['passed'] for v in s['validation'].values())
plt.rcParams.update({'font.size':11})
for name,cost,title in [('finite',d['finite_costs'][0],'500 decisions, time zero'),
                         ('infinite',d['vi_cost'],'Infinite horizon')]:
    fig,axes=plt.subplots(2,1,figsize=(6.8,4.5),sharex=True,layout='constrained')
    axes[0].plot(d['states'],cost,color='#2563eb',lw=1.8)
    axes[0].set(ylabel='Optimal discounted cost',title=f'{title}: cost')
    axes[1].plot(d['states'],-cost,color='#b45309',lw=1.8)
    axes[1].set(ylabel='Optimal discounted value',xlabel='Passengers waiting before dispatch',title='Value = negative of cost',xlim=(0,200))
    for ext in ('pdf','png'):fig.savefig(FIG/f'shuttle_{name}_cost_value.{ext}',dpi=170)
    plt.close(fig)
fig,ax=plt.subplots(figsize=(6.6,2.05),layout='constrained')
ax.step(d['states'],d['pi_policy'],where='post',color='#7c3aed',lw=1.8)
ax.axvline(13,color='gray',ls='--',lw=1)
ax.annotate('First dispatch state: 13',xy=(13,.7),xytext=(45,.45),arrowprops={'arrowstyle':'->','color':'gray'})
ax.set(xlabel='Passengers waiting before dispatch',ylabel='Optimal action',xlim=(0,200),ylim=(-.15,1.15))
ax.set_yticks([0,1],['Wait','Dispatch']);fig.savefig(FIG/'shuttle_policy.pdf');plt.close(fig)

claims=json.loads((ROOT/'validation/report_claims.json').read_text())
assert claims['passed'] and claims['original_sources_unchanged']
e=claims['evidence'];macros=[]
def macro(name,value):macros.append('\\newcommand{\\'+name+'}{'+str(value)+'}')
def scientific(x):
    mantissa,exponent=f'{x:.2e}'.split('e')
    return mantissa+r'\times10^{'+str(int(exponent))+'}'
macro('ZeroFitMSE',scientific(e['cart_zero_reward_baseline_fit']['heldout_mse_after']))
macro('OneFitMean',f'{e["cart_one_step_unit_return_baseline_fit"]["heldout_prediction_mean"]:.6f}')
macro('OneFitMSE',scientific(e['cart_one_step_unit_return_baseline_fit']['heldout_mse_after']))
macro('ZeroEMA',f'{e["pong_zero_reward_ema_from_nonzero"]["observed_baseline"]:.6f}')
macro('OneEMA',f'{e["pong_unit_one_step_return_ema"]["observed_baseline"]:.6f}')
macro('PongTerminationLength',e['pong_normal_termination']['length'])
(ROOT/'validation_macros.tex').write_text('\n'.join(macros)+'\n')
print('Audited raw results and regenerated the final report figures and numerical macros.')
