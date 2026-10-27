"""Copy the recorded plots and add shuttle reward-value plots; no training."""
from pathlib import Path
import json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PROJECTS=ROOT.parent
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
pg=PROJECTS/'mlds_assignment1_policy_gradient_report'
for source in (pg/'figures').glob('*.pdf'):shutil.copy2(source,FIG/source.name)
shuttle=PROJECTS/'mlds_assignment1_shuttle'
shutil.copy2(shuttle/'results/report_figures/c_policy_iteration_policy.pdf',FIG/'shuttle_policy.pdf')
d=np.load(shuttle/'results/shuttle_results.npz');s=json.loads((shuttle/'results/summary.json').read_text())
assert all(v['passed'] for v in s['validation'].values())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42})
for name,cost,title in [('finite',d['finite_costs'][0],'500 decisions, time zero'),
                         ('infinite',d['vi_cost'],'Infinite horizon')]:
    value=-cost
    fig,axes=plt.subplots(2,1,figsize=(6.8,4.5),sharex=True,layout='constrained')
    axes[0].plot(d['states'],cost,color='#2563eb',lw=1.8)
    axes[0].set(ylabel='Optimal discounted cost',title=f'{title}: cost')
    axes[1].plot(d['states'],value,color='#b45309',lw=1.8)
    axes[1].set(ylabel='Optimal discounted value',xlabel='Passengers waiting before dispatch',title='Value = negative of cost',xlim=(0,200))
    for ext in ('pdf','png'):fig.savefig(FIG/f'shuttle_{name}_cost_value.{ext}',dpi=170)
    plt.close(fig)
    for label,y,color in [('cost',cost,'#2563eb'),('value',value,'#b45309')]:
        fig,ax=plt.subplots(figsize=(6.8,2.4),layout='constrained')
        ax.plot(d['states'],y,color=color,lw=1.8)
        ax.set(xlabel='Passengers waiting before dispatch',ylabel=f'Optimal discounted {label}',title=title,xlim=(0,200))
        for ext in ('pdf','png'):fig.savefig(FIG/f'shuttle_{name}_{label}.{ext}',dpi=170)
        plt.close(fig)
claims=json.loads((ROOT/'validation/report_claims.json').read_text())
assert claims['passed'] and claims['original_sources_unchanged']
e=claims['evidence']
macros=[]
def macro(name,value):macros.append('\\newcommand{\\'+name+'}{'+str(value)+'}')
for source in ['metrics_macros.tex']:
    shutil.copy2(pg/source,ROOT/source)
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
print('Copied eight policy-gradient plots; generated cost and negative-cost plots for both shuttle horizons; audited new validation evidence.')
