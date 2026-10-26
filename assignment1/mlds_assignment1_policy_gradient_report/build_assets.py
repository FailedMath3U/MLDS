"""Audit saved data and generate every reported table/plot; does not retrain."""
from pathlib import Path
import csv,json,math,re,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
CART=ROOT.parent/'mlds_assignment1_cartpole'
PONG=ROOT.parent/'mlds_assignment1_pong'
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
SEEDS=(1668,1669,1670)
MODES=('no_baseline','baseline')
COLORS={'no_baseline':'#2563eb','baseline':'#0f766e'}
plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42})

def readcsv(p):
 with p.open() as f:return list(csv.DictReader(f))
def readjson(p):return json.loads(p.read_text())
def save(fig,name):
 fig.tight_layout()
 for ext in ('pdf','png'):fig.savefig(FIG/f'{name}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
def learning(rows,key,name,color):
 x=np.array([int(r['episode']) for r in rows]);y=np.array([float(r[key]) for r in rows])
 fig,ax=plt.subplots(figsize=(6.8,3.05))
 ax.plot(x,y,lw=.6,alpha=.23,color=color,label='Episode reward')
 ax.plot(x[99:],np.convolve(y,np.ones(100)/100,'valid'),lw=1.9,color=color,label='100-episode moving average')
 ax.set(xlabel='Training episode (complete game)',ylabel='Undiscounted episode reward',xlim=(1,len(x)))
 ax.set_ylim((0,510) if key=='reward' else (-22,22));ax.legend(loc='best',fontsize=9)
 save(fig,name)
def histogram(scores,name,color,is_pong):
 fig,ax=plt.subplots(figsize=(6.8,3.05))
 bins=np.arange(-21.5,22,1) if is_pong else np.linspace(0,500,26)
 ax.hist(scores,bins=bins,color=color,edgecolor='white',linewidth=.4)
 ax.axvline(scores.mean(),c='black',ls='--',lw=1,label=f'Mean {scores.mean():.3f}; SD {scores.std():.3f}')
 ax.set(xlabel='Undiscounted episode reward',ylabel='Number of evaluation episodes',xlim=(-22,22) if is_pong else (0,510))
 ax.legend(fontsize=9);save(fig,name)

metrics={'cartpole':{},'pong':{},'audits':[]}
cart_table=[]
for mode in MODES:
 for seed in SEEDS:
  directory=CART/'runs'/f'{mode}_seed{seed}'
  c=readjson(directory/'config.json');e=readjson(directory/'evaluation.json')
  rows=readcsv(directory/'training.csv');scores=np.loadtxt(directory/'evaluation_rewards.csv',skiprows=1)
  assert len(rows)==8000 and len(scores)==500
  assert c['gamma']==.95 and c['lr']==.01 and c['environment']=='CartPole-v1'
  assert np.isclose(scores.mean(),e['mean'],atol=1e-12) and np.isclose(scores.std(),e['population_std'],atol=1e-12)
  u=rows[::32]
  stats={'mean':float(scores.mean()),'sd':float(scores.std()),'evaluation_episodes':len(scores),
    'min':float(scores.min()),'max':float(scores.max()),'at500':float(np.mean(scores==500)),
    'updates':len(u),'clipped_updates':sum(float(r['gradient_norm_before_clip'])>10 for r in u),
    'final100_training_mean':float(np.mean([float(r['reward']) for r in rows[-100:]])),
    'mean_discounted_return':e['mean_discounted_return']}
  metrics['cartpole'][f'{mode}_{seed}']=stats
  label='No baseline' if mode=='no_baseline' else 'Learned baseline'
  cart_table.append(f'{label} & {seed} & {stats["mean"]:.3f} & {stats["sd"]:.3f} & {stats["at500"]:.3f} \\\\')
  if seed==1668:
   learning(rows,'reward',f'cartpole_{mode}_training',COLORS[mode])
   histogram(scores,f'cartpole_{mode}_hist',COLORS[mode],False)
for mode in MODES:
 directory=PONG/'runs'/mode
 c=readjson(PONG/'config.json');rows=readcsv(directory/'training.csv')
 assert len(rows)==20000 and c['gamma']==.99 and c['lr']==.001
 u=rows[::8]
 e=readjson(directory/'evaluation/test_final.json');scores=np.array([v['score'] for v in e['episodes']])
 csv_scores=np.array([float(v['score']) for v in readcsv(directory/'evaluation/test_final_rewards.csv')])
 assert len(scores)==500 and e['complete'] and np.array_equal(scores,csv_scores)
 assert np.isclose(scores.mean(),e['mean'],atol=1e-12) and np.isclose(scores.std(),e['population_std'],atol=1e-12)
 selected=readjson(directory/'evaluation/test_selected.json');provenance=readjson(directory/'evaluation/selected_provenance.json')
 stats={'mean':float(scores.mean()),'sd':float(scores.std()),'evaluation_episodes':len(scores),
    'min':float(scores.min()),'max':float(scores.max()),'win_fraction':float(np.mean(scores>0)),
    'updates':len(u),'clipped_updates':sum(float(v['gradient_norm_before_clip'])>5 for v in u),
    'clip_fraction':float(np.mean([float(v['gradient_norm_before_clip'])>5 for v in u])),
    'final100_training_mean':float(np.mean([float(v['score']) for v in rows[-100:]])),
    'mean_discounted_return':float(np.mean([v['discounted_return'] for v in e['episodes']])),
    'mean_length':float(np.mean([v['length'] for v in e['episodes']])),
    'terminations':sum(v['terminated'] for v in e['episodes']),
    'truncations':sum(v['truncated'] for v in e['episodes']),
    'selected_mean':selected['mean'],'selected_sd':selected['population_std'],
    'selected_training_episodes':provenance['training_episodes']}
 assert not provenance['selection_used_test_scores']
 metrics['pong'][mode]=stats
 learning(rows,'score',f'pong_{mode}_training',COLORS[mode])
 histogram(scores,f'pong_{mode}_hist',COLORS[mode],True)

cv=readjson(CART/'validation/validation.json')
pp=readjson(PONG/'preflight/results.json')
assert pp['passed'] and all(v.get('passed') is not False for v in cv.values())
metrics['cartpole_validation']=cv
metrics['pong_preflight']=pp
metrics['pong_diagnostics']={p.stem:readjson(p) for p in sorted((PONG/'diagnostics').glob('*.json'))}
metrics['cartpole_random_reference']=cv['uniform_random_reference']
metrics['cartpole_untrained_reference']=readjson(CART/'runs/no_baseline_seed1668/untrained_evaluation.json')
pr=readjson(PONG/'references/untrained.json')
metrics['pong_untrained_reference']={k:v for k,v in pr.items() if k!='episodes'}
metrics['pong_untrained_reference']['mean_discounted_return']=float(np.mean([e['discounted_return'] for e in pr['episodes']]))
metrics['audits']=['All six CartPole runs contain 8000 training episodes and 500 evaluations.',
 'Both Pong runs contain 20000 training games and 500 complete final-policy evaluations.',
 'All primary means and population SDs independently recomputed from individual rewards.',
 'Pong final CSV reward arrays agree with their JSON episode records.',
 'Required discounts, learning rates, episode counts and environments verified.',
 'All eight required plots generated from complete episode logs and final-policy rewards.',
 'CartPole validation assertions and Pong preflight assertions passed.',
 'Pong checkpoint selection provenance confirms test scores were not used for selection.']

# Compact supplementary trace tables, using actual saved decisions.
trace=readcsv(CART/'validation/first_ten_steps.csv')
cart_trace=[]
for row in trace:
 cart_trace.append(f"{row['t']} & {float(row['position']):.5f} & {float(row['velocity']):.5f} & {float(row['angle']):.5f} & {float(row['angular_velocity']):.5f} & {float(row['p_right']):.4f} & {row['action']} \\\\")
pong_trace=readcsv(PONG/'runs/no_baseline/evaluation/test_final_trace.csv')
first_point=[r for r in pong_trace if float(r['reward'])!=0][0]
metrics['pong_trace']={'first_decision':pong_trace[0],'first_point_row':first_point}
varrows=[]
for key,d in metrics['pong_diagnostics'].items():
 mode='Basic actor' if key.startswith('no_baseline') else 'EMA-baseline actor'
 varrows.append(f'{mode} & {d["training_episodes"]} & {d["gradient_variance_no_baseline"]:.2f} & {d["gradient_variance_with_baseline"]:.2f} & {d["variance_ratio"]:.3f} \\\\')

tex=[]
def macro(name,value):tex.append('\\newcommand{\\' + name + '}{' + str(value) + '}')
for mode,prefix in [('no_baseline','Basic'),('baseline','Baseline')]:
 for game,data in [('Cart',metrics['cartpole'][f'{mode}_1668']),('Pong',metrics['pong'][mode])]:
  macro(game+prefix+'Mean',f'{data["mean"]:.3f}');macro(game+prefix+'SD',f'{data["sd"]:.3f}')
  macro(game+prefix+'Clipped',data['clipped_updates'])
macro('CartVarianceReduction',f'{100*cv["frozen_policy_gradient_variance"]["variance_reduction_fraction"]:.2f}')
macro('PongInitialVarianceReduction',f'{100*(1-pp["checks"]["initial_gradient_variance_measurement"]["variance_ratio"]):.2f}')
macro('PongReferenceDiscounted',f'{metrics["pong_untrained_reference"]["mean_discounted_return"]:.6f}')
(ROOT/'metrics_macros.tex').write_text('\n'.join(tex)+'\n')
(ROOT/'cartpole_seed_rows.tex').write_text('\n'.join(cart_table)+'\n')
(ROOT/'cartpole_trace_rows.tex').write_text('\n'.join(cart_trace)+'\n')
(ROOT/'pong_variance_rows.tex').write_text('\n'.join(varrows)+'\n')
prows=[]
chosen=pong_trace[:10]+[first_point,pong_trace[-1]]
for row in chosen:
 prows.append(f"{int(row['step'])} & {float(row['p_action2_right']):.4f} & {int(row['ale_action'])} & {float(row['reward']):.0f} & {float(row['reward_to_go']):.5f} & {float(row['outer_discount']):.3g} " + chr(92)*2)
(ROOT/'pong_trace_rows.tex').write_text('\n'.join(prows)+'\n')
defs=[]
for command,file in [('CartSeedRows','cartpole_seed_rows.tex'),('CartTraceRows','cartpole_trace_rows.tex'),('PongVarianceRows','pong_variance_rows.tex'),('PongTraceRows','pong_trace_rows.tex')]:
 defs.append('\\newcommand{\\'+command+'}{%\n'+(ROOT/file).read_text().rstrip()+'%\n}')
(ROOT/'table_defs.tex').write_text('\n'.join(defs)+'\n')
(ROOT/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')

submission=ROOT/'submission.json'
if not submission.exists():submission.write_text(json.dumps({'github_username':'FailedMath3U','repository_name':'','repository_url':''},indent=2)+'\n')
s=json.loads(submission.read_text())
def esc(x):
 return str(x).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('&',r'\&').replace('%',r'\%').replace('#',r'\#')
meta=[r'\newcommand{\GithubUsername}{'+esc(s['github_username'])+'}']
if s['repository_url']:
 meta+=[r'\newcommand{\RepositoryName}{'+esc(s['repository_name'])+'}',r'\newcommand{\RepositoryLink}{\url{'+s['repository_url']+'}}',r'\newcommand{\SubmissionNotice}{}']
else:
 meta+=[r'\newcommand{\RepositoryName}{\textcolor{red}{Not provided}}',r'\newcommand{\RepositoryLink}{\textcolor{red}{Repository URL required before submission}}',r'\newcommand{\SubmissionNotice}{\textbf{Submission metadata incomplete:} add the actual repository name and URL before submitting this PDF.}']
(ROOT/'submission.tex').write_text('\n'.join(meta)+'\n')
print('Audited six CartPole and two Pong runs; generated eight required plots and numerical tables.')
print('Repository metadata complete:',bool(s['repository_url']))
