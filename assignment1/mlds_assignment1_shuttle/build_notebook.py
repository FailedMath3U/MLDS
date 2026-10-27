"""Create and execute only the shuttle plots used in the final submission."""
from pathlib import Path
import os,sys,json
import nbformat as nb
from nbclient import NotebookClient
root=Path(__file__).resolve().parent
cells=[nb.v4.new_markdown_cell('''# Shuttle: cost, value and policy

Loads the reported solver results without training. Cost is minimized; reward value is its negative. See `../mlds_assignment1_submission/ASSIGNMENT1_REPORT.pdf` for the model, assumptions and validation.''')]
cells.append(nb.v4.new_code_cell('''from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
ROOT=Path.cwd()
if not (ROOT/'shuttle.py').exists(): ROOT=ROOT/'mlds_assignment1_shuttle'
data=np.load(ROOT/'results/shuttle_results.npz')
FIG=ROOT/'results/figures';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
def curve(values,label,title,name):
    fig,ax=plt.subplots(figsize=(8,3.5))
    ax.plot(data['states'],values)
    ax.set(xlabel='Passengers waiting before dispatch',ylabel=label,title=title,xlim=(0,200))
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(FIG/f'{name}.{ext}',dpi=170)
    plt.show()
'''))
for name,expression,label,title in [
 ('finite_cost',"data['finite_costs'][0]",'Optimal discounted cost','Enumeration: 500 decisions, time zero'),
 ('finite_value',"-data['finite_costs'][0]",'Optimal discounted reward value','Enumeration: value = negative cost'),
 ('infinite_cost',"data['vi_cost']",'Optimal discounted cost','Infinite-horizon value iteration'),
 ('infinite_value',"-data['vi_cost']",'Optimal discounted reward value','Value iteration: value = negative cost')]:
    cells.append(nb.v4.new_code_cell(f'curve({expression},{label!r},{title!r},{name!r})'))
cells.append(nb.v4.new_code_cell('''fig,ax=plt.subplots(figsize=(8,3.5))
ax.step(data['states'],data['pi_policy'],where='post')
ax.set(xlabel='Passengers waiting before dispatch',ylabel='Optimal action',title='Policy iteration',xlim=(0,200),ylim=(-.15,1.15))
ax.set_yticks([0,1],['Wait','Dispatch'])
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(FIG/f'policy.{ext}',dpi=170)
plt.show()
'''))
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path=root/'shuttle_plots.ipynb';nb.write(book,path)
k=Path('/tmp/mlds-shuttle-final-jupyter/kernels/python3');k.mkdir(parents=True,exist_ok=True)
(k/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Python 3','language':'python'}))
os.environ['JUPYTER_PATH']='/tmp/mlds-shuttle-final-jupyter'
os.environ['IPYTHONDIR']='/tmp/mlds-shuttle-final-ipython'
os.environ['JUPYTER_RUNTIME_DIR']='/tmp/mlds-shuttle-final-runtime'
NotebookClient(book,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
nb.write(book,path);print(f'Executed {path}')
