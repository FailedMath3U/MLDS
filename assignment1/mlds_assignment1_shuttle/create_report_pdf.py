"""Build the shuttle report PDF from saved solver results and figures.
Requires reportlab, matplotlib and numpy. Run from any working directory.
"""
from pathlib import Path
from io import BytesIO
import json
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT/'results/summary.json').read_text())
checks=summary['validation']
fontroot=Path('/usr/share/fonts/truetype/dejavu')
for name, file in [('Body','DejaVuSerif.ttf'),('BodyBold','DejaVuSerif-Bold.ttf'),('Head','DejaVuSans.ttf'),('HeadBold','DejaVuSans-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(fontroot/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Text',fontName='Body',fontSize=9.2,leading=12.7,spaceAfter=6))
styles.add(ParagraphStyle(name='Section',fontName='HeadBold',fontSize=12,leading=15,textColor=colors.HexColor('#16324f'),spaceAfter=9,spaceBefore=7,keepWithNext=True))
styles.add(ParagraphStyle(name='TitleCustom',fontName='HeadBold',fontSize=21,leading=26,textColor=colors.HexColor('#16324f'),spaceAfter=12))
styles.add(ParagraphStyle(name='CaptionCustom',fontName='Head',fontSize=7.8,leading=10,spaceAfter=7,textColor=colors.HexColor('#334155')))
styles.add(ParagraphStyle(name='CellCustom',fontName='Head',fontSize=8.2,leading=11))
styles.add(ParagraphStyle(name='CodeCustom',fontName='Courier',fontSize=8.3,leading=12,spaceAfter=8))
story=[]
def p(s): story.append(Paragraph(s,styles['Text']))
def h(s): story.append(Paragraph(s,styles['Section']))
def eq(s):
    buf=BytesIO()
    fig=plt.figure(figsize=(7,0.5))
    fig.text(0.5,0.5,'$'+s+'$',ha='center',va='center',fontsize=11)
    fig.savefig(buf,format='png',dpi=200,bbox_inches='tight',pad_inches=0.08)
    plt.close(fig); buf.seek(0)
    w,hh=ImageReader(buf).getSize(); scale=min(460/w,1)
    story.append(Image(buf,width=w*scale,height=hh*scale))
    story.append(Spacer(1,7))
def plot(file,caption,width=460):
    path=ROOT/'results/figures'/file
    w,hh=ImageReader(str(path)).getSize()
    image=Image(str(path),width=width,height=hh*width/w)
    story.append(KeepTogether([image,Paragraph(caption,styles['CaptionCustom'])]))
def table(headers,rows,widths):
    data=[[Paragraph(escape(str(s)),styles['CellCustom']) for s in row] for row in [headers]+rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2e8f0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),0.7,colors.HexColor('#64748b')),('LINEBELOW',(0,-1),(-1,-1),0.5,colors.HexColor('#cbd5e1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafc')]),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.extend([t,Spacer(1,10)])
def page(): story.append(PageBreak())

story.append(Paragraph('Shuttle Dispatch',styles['TitleCustom']))
p('Assignment 1 · Value and Policy Iteration')
h('1. Markov decision process')
p('The state S<sub>t</sub> is the number of passengers waiting immediately before the dispatch decision, with state space {0, …, 200}. Action a = 0 means wait and a = 1 means dispatch one shuttle. A dispatched shuttle immediately serves up to 15 passengers. The remaining queue and stage cost are')
eq(r'L(s,a)=\max\{s-15a,0\},\qquad c(s,a)=100a+2L(s,a).')
p('Independent arrivals are uniform on {1, 2, 3, 4, 5}. They occur after dispatch and the holding-cost assessment, so the transition is')
eq(r'S_{t+1}=\min\{L(S_t,a_t)+A_t,200\},\qquad \gamma=0.95.')
p('The transition probability for a next state is the number of arrival outcomes producing that state, divided by five. In particular, probabilities are combined when several arrival outcomes reach the queue cap.')
p('<b>Assumptions.</b> At most one shuttle is dispatched per decision, with no separate fleet-availability or travel-time constraint. New arrivals first incur holding cost at the next decision. Overflow above 200 is discarded without an extra penalty. An empty dispatch is allowed but strictly dominated. The finite problem has 500 decisions, t = 0, …, 499, and zero terminal cost at t = 500. Ties select wait.')
p('<b>Value convention.</b> J denotes minimum expected discounted <i>cost</i>, so lower is better. If rewards are defined as r = −c, the reward value is V = −J and the optimal policy is unchanged. All plots label the cost convention explicitly.')
h('2. Finite-horizon enumeration')
p('Enumeration was interpreted as exhaustive state/action enumeration by backward induction. At every time and state, both actions and all five arrival outcomes are evaluated. This reuses continuation values instead of enumerating exponentially many complete contingent histories.')
eq(r'J_{500}(s)=0,\qquad J_t(s)=\min_{a\in\{0,1\}}Q_t(s,a),')
eq(r'Q_t(s,a)=c(s,a)+\frac{0.95}{5}\sum_{k=1}^{5}J_{t+1}(\min\{L(s,a)+k,200\}).')
p('The recursion runs from t = 499 to 0. All 501 value vectors and 500 time-dependent policy vectors are saved. The required time-zero value plot appears in Figure 1.')
page()
h('3. Finite- and infinite-horizon values')
plot('a_finite_horizon_value.png','Figure 1. Minimum expected discounted cost at time zero for T = 500 and zero terminal cost.',380)
p('The finite-horizon policy depends on the remaining time. In the final period, waiting is optimal everywhere: dispatch costs 100 but saves at most 2 × 15 = 30 in immediate holding cost. With zero terminal cost, no subsequent benefit can justify dispatch.')
h('Value iteration')
p('Starting with J⁽⁰⁾ = 0, value iteration repeatedly applies the same minimizing Bellman update using the previous value vector as the continuation value. Iteration stops when the residual of the returned vector is at most 10<super>−10</super>. This took 518 updates.')
eq(r'\|J-J^*\|_\infty\leq\frac{\|\mathcal{T}J-J\|_\infty}{1-0.95}.')
plot('b_value_iteration_value.png','Figure 2. Infinite-horizon optimal discounted cost obtained by value iteration.',380)
page()
h('4. Policy iteration and dispatch rule')
p('Policy iteration starts from the always-wait policy. Given its transition matrix P<sub>π</sub> and cost vector c<sub>π</sub>, each policy is evaluated by solving')
eq(r'(I-0.95P_\pi)J_\pi=c_\pi.')
p('Policy improvement chooses the minimizing action in every state using this evaluated cost. Termination occurs when the policy stops changing. The method required six evaluation/improvement rounds, including the final stability check. Both actions were optimized independently at every state; no threshold structure was imposed.')
plot('c_policy_iteration_policy.png','Figure 3. Optimal stationary policy: wait with 0–12 passengers and dispatch with 13–200 passengers.',380)
p('<b>The optimal dispatch threshold is 13 passengers.</b> Dispatching before the shuttle is full can be worthwhile because it avoids future discounted holding costs. Comparing only immediate costs would miss that benefit. When starting from a large queue, repeated dispatches are needed because one shuttle can serve only 15 passengers per decision.')
table(['Initial queue','Optimal infinite-horizon discounted cost'],[[s,f'{summary[f"cost_at_{s}"]:.9f}'] for s in (0,15,200)],[120,340])
p('The difference J*(15) − J*(0) = 100 is consistent with the transition model. Dispatching at 15 leaves an empty queue, while waiting at zero also leaves an empty queue. Both have the same next-arrival distribution, so their values differ only by the dispatch cost.')
page()
h('5. Edge cases and validation')
p('All 14 validation checks passed. The experiments assess algorithm behavior and optimality, beyond data-format or basic implementation checks.')
p('<b>Bellman optimality and agreement.</b> Value iteration and policy iteration selected identical actions in all 201 states. Their maximum cost difference was 1.964 × 10<super>−9</super>. The policy-iteration solution had Bellman residual 9.095 × 10<super>−13</super>, yielding a contraction-based maximum-error bound of approximately 1.819 × 10<super>−11</super>, subject to floating-point rounding. The smallest absolute action-cost gap was 3.778, much larger than the numerical error.')
p('<b>Finite versus infinite horizon.</b> The largest difference between the 500-period time-zero cost and the infinite-horizon cost was 4.942 × 10<super>−9</super>. Since the largest stage cost is 470, the omitted tail is bounded by 470(0.95)<super>500</super> / (1 − 0.95) = 6.838 × 10<super>−8</super>. The observed difference is below this bound. Both the zero terminal vector and the analytically expected last-period waiting policy were verified.')
p('<b>Queue boundaries.</b> Service never produces a negative queue. A queue of 15 clears exactly and a queue of 16 leaves one passenger. At state 200, waiting gives next state 200 for every arrival; dispatch gives next states 186 through 190. An empty dispatch has the same next-state distribution as waiting but costs 100 more.')
p('<b>Predictable special cases.</b> Model parameters were changed to produce analytically known behavior, then the algorithms were rerun:')
table(['Experiment','Expected behavior','Observed'],[
['Zero holding cost','Always wait; optimal cost zero','Matched all states'],
['Zero capacity, no arrivals','Always wait; J(s) = 2s / (1 − 0.95)','Zero analytic error'],
['Free dispatch, capacity 200, no arrivals','Dispatch at nonempty states; cost zero','Matched all states'],
['Zero discount factor','Always wait; J(s) = 2s','Matched all states']], [135,235,90])
p('<b>Exhaustive small problem.</b> A separately coded scalar decision/arrival tree for a seven-state, four-period problem matched vectorized backward induction exactly. Complete tree enumeration is feasible at this scale and checks that the optimized recursion handles the action and arrival branches correctly.')
p('<b>Independent simulation.</b> For each initial queue of 0, 15 and 200, 20,000 trajectories of 300 decisions were simulated with seed 1668. Transitions and costs were computed directly rather than reusing the solver’s transition-array builder. Results are shown below and in Figure 4.')
rows=[]
for s in (0,15,200):
    r=checks[f'rollout_state_{s}']
    rows.append([s,f'{r["sample_mean"]:.6f}',f'{r["exact_cost"]:.6f}',f'{r["standard_error"]:.6f}'])
table(['Initial queue','Simulation mean','Exact cost','Standard error'],rows,[85,125,125,125])
p('Every observed error was within four standard errors plus the conservative 300-period truncation bound of approximately 0.001951. Monte Carlo agreement checks policy values; the Bellman residual provides the stronger optimality evidence.')
page()
h('6. Simulation evidence and reproducibility')
plot('validation_rollouts.png','Figure 4. Monte Carlo means with ±1.96 standard-error bars and exact policy values. Bars describe uncertainty in the estimated mean, not the spread of individual trajectory costs.',380)
p('Agreement between solution methods alone cannot detect every modeling error because they share the MDP specification. The analytic variants, independently coded exhaustive tree, and separately implemented simulator provide additional checks. Results remain conditional on the stated arrival, timing, terminal-cost and overflow assumptions.')
p('From the assignment folder, install dependencies and run the solver, followed by the separate plotting notebook:')
for s in ['python -m pip install -r requirements.txt','python shuttle.py','jupyter lab shuttle_plots.ipynb']:
    story.append(Paragraph(escape(s),styles['CodeCustom']))
p('The solver runs all three methods and every validation experiment; a failed check raises an error. Full evidence is saved in results/summary.json, per-state values in results/state_results.csv, and the full finite-horizon values and policies in results/shuttle_results.npz. Running all notebook cells regenerates the PNG and PDF plots in results/figures/.')
h('7. Additional three months of validation')
p('An extended study would first sweep dispatch costs, holding costs, capacity, discount factors and arrival distributions. This would measure changes in the optimal dispatch threshold and identify settings where threshold behavior fails. Larger queue caps and alternative overflow penalties would test sensitivity to the artificial boundary.')
p('A second stage would compare the results against an independently formulated linear program and use higher precision near action ties. Different horizons and terminal penalties would reveal how the finite-horizon policy changes near termination. Additional simulation seeds and starting queues would support error and confidence-interval coverage studies.')
p('Finally, bursty or correlated arrivals and explicit shuttle travel times and availability would test robustness to model misspecification. These extensions address whether the simplifying assumptions change operational conclusions, beyond verifying the numerical solution under independent uniform demand.')

def footer(canvas,doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#cbd5e1'))
    canvas.line(54,40,558,40)
    canvas.setFont('Head',8)
    canvas.setFillColor(colors.HexColor('#64748b'))
    canvas.drawString(54,27,'Assignment 1 | Shuttle dispatch')
    canvas.drawRightString(558,27,str(doc.page))
    canvas.restoreState()
output=ROOT/'SHUTTLE_REPORT.pdf'
doc=SimpleDocTemplate(str(output),pagesize=(612,792),rightMargin=54,leftMargin=54,topMargin=42,bottomMargin=54,title='Assignment 1: Shuttle Dispatch',author='',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(f'Saved {output} ({output.stat().st_size:,} bytes)')
