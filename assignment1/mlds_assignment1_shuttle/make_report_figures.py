"""Compact vector figures with readable labels for the LaTeX report."""
from pathlib import Path
import json
import numpy as np
from shuttle import Model, action_costs, evaluate_policy
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
# Audit saved solutions and derive report numbers without rewriting solver outputs.
model=Model()
assert s['model']=={'capacity':15,'dispatch_cost':100.0,'holding_cost':2.0,
                   'max_queue':200,'gamma':.95,'arrivals':[1,2,3,4,5]}
assert d['finite_costs'].shape==(501,201)
assert d['finite_policy'].shape==(500,201)
assert np.array_equal(d['pi_policy'],(d['states']>=13).astype(int))
assert np.array_equal(d['vi_policy'],d['pi_policy'])
assert all(v['passed'] for v in s['validation'].values())
q=action_costs(model,d['pi_cost'])
full_policy=(d['states']>=15).astype(int)
full_value=evaluate_policy(model,full_policy)
metrics={'threshold_action_costs':{str(i):{'wait':float(q[i,0]),'dispatch':float(q[i,1])}
                                  for i in (12,13)},
         'cost_at_zero_optimal':float(d['pi_cost'][0]),
         'cost_at_zero_wait_until_full':float(full_value[0]),
         'wait_until_full_extra_cost':float(full_value[0]-d['pi_cost'][0]),
         'vi_residual':float(np.max(abs(action_costs(model,d['vi_cost']).min(axis=1)-d['vi_cost']))),
         'policy_iteration_residual':s['validation']['bellman_optimality']['residual'],
         'validation_count':len(s['validation'])}
(root/'results/report_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
macros=[]
for name,value in [('FullShuttleCost',full_value[0]),('OptimalEmptyCost',d['pi_cost'][0]),
                   ('WaitCostTwelve',q[12,0]),('DispatchCostTwelve',q[12,1]),
                   ('WaitCostThirteen',q[13,0]),('DispatchCostThirteen',q[13,1])]:
 macros.append('\\newcommand{\\'+name+'}{'+f'{value:.3f}'+'}')
(root/'report_metrics.tex').write_text('\n'.join(macros)+'\n')
print('Generated four report figures and audited report metrics; original solver outputs preserved.')
