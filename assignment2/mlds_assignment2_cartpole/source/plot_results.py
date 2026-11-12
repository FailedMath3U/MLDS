"""Generate CartPole figures locally from recorded data, without gameplay.

Includes training rewards, maximum Q-values, final reward histogram, validation,
initial-state stress, action maps, a separate green trajectory, and pole angles.
"""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
FIG=ROOT/'figures'
RES=ROOT/'results'

def load(path): return json.loads(Path(path).read_text())

def csv_data(path):
    with Path(path).open() as f:
        rows=list(csv.DictReader(f))
    return {key:np.array([float(row[key]) for row in rows]) for key in rows[0]}

def smooth(x,n=100):
    return np.convolve(x,np.ones(n)/n,mode='valid')

def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def figures(run,ev,diag):
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,
                        'axes.spines.right':False,'figure.facecolor':'white'})
    train=csv_data(run/'training.csv')
    fig,ax=plt.subplots(figsize=(8.4,3.0))
    ax.plot(train['episode'],train['q_max'],lw=.65,alpha=.7,label='Maximum over states/actions visited in episode')
    ax.plot(train['episode'],train['q_probe'],lw=1.4,label='Mean max-Q over 64 fixed reset states')
    ax.axhline(20.,color='#444444',ls='--',lw=1,label='1 / (1 - 0.95) = 20')
    ax.set(xlabel='Training episode',ylabel='Predicted action value',title='DQN maximum Q-values during training')
    ax.legend(fontsize=8,loc='lower right')
    save(fig,'maximum_q_values')

    fig,ax=plt.subplots(figsize=(8.4,3.0))
    ax.plot(train['episode'],train['reward'],lw=.5,color='#8baac9',alpha=.65,label='Exploratory episode reward')
    if len(train['reward'])>=100:
        ax.plot(train['episode'][99:],smooth(train['reward']),color='#154e79',lw=2,label='Trailing 100-episode mean')
    ax.axhline(500,color='#555555',ls='--',lw=.8)
    ax.set(xlabel='Training episode',ylabel='Undiscounted episode reward',ylim=(0,520),title='Training rewards (epsilon-greedy behavior)')
    ax.legend(fontsize=8,loc='lower right')
    save(fig,'training_rewards')

    rewards=np.array([x['reward'] for x in load(ev/'episodes.json')])
    failures=sorted(int(x['reward']) for x in load(ev/'episodes.json') if x['terminated'])
    failure_label=', '.join(map(str,failures)) if failures else 'none'
    summary=load(ev/'summary.json')
    fig,ax=plt.subplots(figsize=(8.4,2.8))
    histogram_bins=np.arange(-5,506,10)
    counts,_,bars=ax.hist(rewards,bins=histogram_bins,color='#267b67',edgecolor='none',linewidth=0)
    ax.set_yscale('log')
    ax.set(xlabel='Episode reward',ylabel='Episodes (logarithmic scale)',xlim=(0,507),
           ylim=(.5,800),title='All 500 evaluation episodes: logarithmic count scale')
    ax.set_yticks([1,10,100,500],labels=['1','10','100','500'])
    for count,bar in zip(counts,bars):
        if count>0:
            ax.text(bar.get_x()+bar.get_width()/2,count*1.12,str(int(count)),
                    ha='center',va='bottom',fontsize=8)
    ax.text(.02,.95,f"Mean {summary['mean']:.3f}\nPopulation SD {summary['std_population']:.3f}\n500-step episodes: {summary['successes_500']}/500\nEarly failures: {failure_label}",
            va='top',transform=ax.transAxes)
    save(fig,'evaluation_histogram')

    fig,ax=plt.subplots(figsize=(8.4,3.0))
    for directory in sorted((RES/'runs').iterdir()):
        hist=directory/'validation_history.json'
        if not hist.exists(): continue
        h=load(hist)
        label='Ordinary DQN' if directory.name=='seed1668' else 'Symmetric Double DQN'
        ax.plot([x['step'] for x in h],[x['mean'] for x in h],marker='.',label=label)
        ts=load(directory/'training_summary.json')
        ax.scatter([ts['steps']],[ts['final_validation']['mean']],marker='x',s=40,zorder=5)
    ax.axvline(summary['checkpoint_step'],ls=':',color='#267b67',label='Chosen checkpoint step')
    ax.set(xlabel='Training environment steps',ylabel='Mean reward over 20 validation episodes',ylim=(0,520),title='Validation used for model selection; final 500 seeds kept separate')
    ax.legend(fontsize=7,loc='lower right')
    save(fig,'validation_comparison')

    # Dedicated validation view for the selected run, without mixing it with
    # exploratory training rewards or the separate 500-game final evaluation.
    history=load(run/'validation_history.json')
    initial=load(run/'initial_validation.json')
    final=load(run/'training_summary.json')
    points=[dict(initial,step=0)]+history
    steps=np.array([p['step'] for p in points])
    means=np.array([p['mean'] for p in points])
    deviations=np.array([p['std_population'] for p in points])
    successes=np.array([p['successes_500'] for p in points])
    selected=next(p for p in history if p['step']==summary['checkpoint_step'])
    fig,axes=plt.subplots(2,1,figsize=(8.4,4.6),sharex=True,layout='constrained',
                          gridspec_kw={'height_ratios':[2,1]})
    axes[0].fill_between(steps,np.maximum(0,means-deviations),np.minimum(500,means+deviations),
                         color='#267b67',alpha=.15,label='Mean ± 1 SD across validation games')
    axes[0].plot(steps,means,'o-',ms=3.5,lw=1.4,color='#267b67',label='Mean of 20 greedy validation games')
    axes[0].scatter([selected['step']],[selected['mean']],marker='*',s=135,
                     color='#d69b1f',edgecolor='#714b00',zorder=6,label=f"Selected checkpoint: {selected['step']:,} steps")
    axes[0].scatter([final['steps']],[final['final_validation']['mean']],marker='X',s=65,
                     color='#b94b43',zorder=6,label=f"End of training: {final['steps']:,} steps")
    axes[0].axhline(500,color='#777777',ls='--',lw=.8)
    axes[0].set(ylabel='Mean episode reward',ylim=(0,530),
                title='CartPole validation performance: same 20 reset seeds, no exploration')
    fig.legend(loc='outside upper center',ncol=2,fontsize=7,frameon=False)
    axes[1].plot(steps,successes,'o-',ms=3.5,lw=1.4,color='#154e79')
    axes[1].scatter([selected['step']],[selected['successes_500']],marker='*',s=135,
                     color='#d69b1f',edgecolor='#714b00',zorder=6)
    axes[1].scatter([final['steps']],[final['final_validation']['successes_500']],marker='X',s=65,
                     color='#b94b43',zorder=6)
    for ax in axes:
        ax.axvline(selected['step'],color='#b58319',ls=':',lw=1)
        ax.grid(axis='y',alpha=.15)
    axes[1].set(ylabel='Full 500-step games\n(out of 20)',xlabel='Training environment steps',
                yticks=[0,5,10,15,20],ylim=(-.7,21.5),xticks=[0,50000,100000,150000,200000])
    axes[1].ticklabel_format(axis='x',style='plain')
    save(fig,'validation_performance')

    grid=load(diag/'robustness_grid.json')
    matrix=np.array([x['reward'] for x in grid['episodes']]).reshape(len(grid['thetas']),len(grid['xs']))
    fig,ax=plt.subplots(figsize=(7.6,4.3))
    im=ax.imshow(matrix,origin='lower',aspect='auto',vmin=0,vmax=500,cmap='viridis',
                 extent=[min(grid['xs'])-.225,max(grid['xs'])+.225,min(grid['thetas'])-.0225,max(grid['thetas'])+.0225])
    for i,theta in enumerate(grid['thetas']):
        for j,x in enumerate(grid['xs']):
            ax.text(x,theta,str(int(matrix[i,j])),ha='center',va='center',fontsize=7,color='white' if matrix[i,j]<300 else '#102338')
    ax.set(xlabel='Initial cart position (meters)',ylabel='Initial pole angle (radians)',
           title='Stress test: 81 initial states beyond the training reset distribution')
    fig.colorbar(im,ax=ax,label='Steps survived / episode reward')
    save(fig,'robustness_initial_states')



def phase_figures():
    maps=json.loads((RES/'validation/phase_maps.json').read_text())
    trajectory=json.loads((RES/'validation/policy_trajectory.json').read_text())
    states=np.asarray([row['state'] for row in trajectory])
    bound=np.deg2rad(12)
    fig,axes=plt.subplots(1,3,figsize=(10.6,3.6),sharex=True,sharey=True,layout='constrained')
    maxgap=max(float(np.quantile(np.abs(m['right_minus_left']),.98)) for m in maps['maps'])
    norm=TwoSlopeNorm(vmin=-max(maxgap,.01),vcenter=0,vmax=max(maxgap,.01))
    for ax,m in zip(axes,maps['maps']):
        gap=np.asarray(m['right_minus_left'])
        im=ax.pcolormesh(maps['angles'],maps['angular_velocities'],gap,
                         cmap='coolwarm',norm=norm,shading='auto',rasterized=True)
        if gap.min()<0<gap.max():
            ax.contour(maps['angles'],maps['angular_velocities'],gap,
                       levels=[0],colors='black',linewidths=.8)
        ax.axvline(-bound,color='black',lw=.7,ls='--')
        ax.axvline(bound,color='black',lw=.7,ls='--')
        ax.set(xlabel='Pole angle (radians)',
               title=f"Cart position x={m['x']:+.1f} m\nCart velocity = 0")
    axes[0].set_ylabel('Pole angular velocity (radians/s)')
    fig.colorbar(im,ax=axes,label='Q(right) - Q(left); red favors right',shrink=.8)
    fig.suptitle('Non-standard diagnostic: action margins across physical state slices')
    fig.savefig(FIG/'action_margin_phase_maps.pdf',bbox_inches='tight')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.4,4.0),layout='constrained')
    ax.plot(states[:,2],states[:,3],color='#20a368',lw=1.1)
    ax.set(xlabel='Pole angle (radians)',ylabel='Pole angular velocity (radians/s)',
           xlim=(min(maps['angles']),max(maps['angles'])),
           title='Green trajectory: states throughout one 500-step game')
    ax.margins(y=.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(FIG/'policy_trajectory.pdf',bbox_inches='tight')
    fig.savefig(FIG/'policy_trajectory.png',bbox_inches='tight',dpi=180)
    plt.close(fig)


def pole_angle_figure():
    passing=json.loads((RES/'validation/policy_trajectory.json').read_text())
    failing=json.loads((RES/'failure_analysis/dqn_seed9000458.json').read_text())
    fig,axes=plt.subplots(2,1,figsize=(8.4,4.0),sharex=True,sharey=True,layout='constrained')
    for ax,rows,title in zip(axes,[passing,failing],
                             ['Winning game: 500 steps (seed 4000000)',
                              'Failing game: 205 steps (seed 9000458)']):
        states=np.asarray([rows[0]['state']]+[r['next_state'] for r in rows])
        ax.plot(np.arange(len(states)),np.degrees(states[:,2]),color='#154e79',lw=1.2)
        ax.axhline(12,color='#c2413b',ls='--',lw=.9)
        ax.axhline(-12,color='#c2413b',ls='--',lw=.9)
        ax.set(title=title,ylabel='Pole angle (degrees)',ylim=(-13.5,13.5),xlim=(0,500))
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    last_angle=float(np.degrees(failing[-1]['next_state'][2]))
    axes[1].plot([len(failing)],[last_angle],marker='x',color='#c2413b',ms=7)
    axes[1].annotate(f'{last_angle:.3f} degrees',xy=(len(failing),last_angle),xytext=(285,7),
                     arrowprops=dict(arrowstyle='->',color='#c2413b'),fontsize=9)
    axes[1].set_xlabel('Environment step')
    fig.savefig(FIG/'pole_angle_comparison.pdf',bbox_inches='tight')
    fig.savefig(FIG/'pole_angle_comparison.png',bbox_inches='tight',dpi=180)
    plt.close(fig)



def main():
    selected=load(RES/'selected_model.json')
    figures(RES/'runs'/selected['run'],RES/'evaluation',RES/'validation')
    with plt.rc_context(matplotlib.rcParamsDefault):
        phase_figures()
        pole_angle_figure()
    print('Generated nine CartPole figures from recorded data.')


if __name__=='__main__': main()
