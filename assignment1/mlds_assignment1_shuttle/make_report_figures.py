"""Compact vector figures with readable labels for the LaTeX report."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
out=root/'results/report_figures'
out.mkdir(exist_ok=True)
d=np.load(root/'results/shuttle_results.npz')
s=json.loads((root/'results/summary.json').read_text())
plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.18,'pdf.fonttype':42})
for field,name,color in [('finite_costs','a_finite_horizon_value','#2563eb'),('vi_cost','b_value_iteration_value','#0f766e')]:
 fig,ax=plt.subplots(figsize=(6.6,2.05),layout='constrained')
 y=d[field][0] if field=='finite_costs' else d[field]
 ax.plot(d['states'],y,color=color,lw=1.8)
 ax.set(xlabel='Passengers waiting before dispatch',ylabel='Optimal discounted cost',xlim=(0,200))
 fig.savefig(out/(name+'.pdf')); plt.close(fig)
fig,ax=plt.subplots(figsize=(6.6,2.05),layout='constrained')
ax.step(d['states'],d['pi_policy'],where='post',color='#7c3aed',lw=1.8)
ax.axvline(13,color='gray',ls='--',lw=1)
ax.annotate('First dispatch state: 13',xy=(13,0.7),xytext=(45,0.45),arrowprops={'arrowstyle':'->','color':'gray'})
ax.set(xlabel='Passengers waiting before dispatch',ylabel='Optimal action',xlim=(0,200),ylim=(-.15,1.15))
ax.set_yticks([0,1],['Wait','Dispatch'])
fig.savefig(out/'c_policy_iteration_policy.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(6.6,2.05),layout='constrained')
r=[s['validation'][f'rollout_state_{i}'] for i in (0,15,200)]
ax.errorbar(range(3),[v['sample_mean'] for v in r],yerr=[1.96*v['standard_error'] for v in r],fmt='o',capsize=5,label='Monte Carlo mean ± 1.96 SE')
ax.scatter(range(3),[v['exact_cost'] for v in r],marker='x',c='black',label='Exact policy value',zorder=4)
ax.set_xticks(range(3),['0','15','200'])
ax.set(xlabel='Initial passengers waiting',ylabel='Discounted cost',xlim=(-.3,2.3))
ax.legend(fontsize=9,loc='upper left')
fig.savefig(out/'validation_rollouts.pdf');plt.close(fig)
print('Generated four report figures.')
p=root/'SHUTTLE_REPORT.tex'
t=p.read_text().replace('\\graphicspath{{results/figures/}}','\\graphicspath{{results/report_figures/}}')
for w in ('0.77','0.88','0.82'):
 t=t.replace('width='+w+'\\linewidth','width=0.90\\linewidth')
p.write_text(t)
