"""Apply only factual corrections and required additions to the supplied draft."""
from pathlib import Path
import difflib
ROOT=Path(__file__).resolve().parent
s=(ROOT/'USER_DRAFT.tex').read_text()

def change(old,new):
    global s
    if old not in s:raise ValueError('Missing original text: '+old[:100])
    s=s.replace(old,new,1)

change(r'\usepackage{amsmath}',r'''\usepackage{amsmath}
\usepackage{amssymb,booktabs}
\usepackage[margin=1in]{geometry}
\usepackage[hidelinks]{hyperref}
\usepackage[section]{placeins}
\setlength{\emergencystretch}{2em}
\renewcommand{\topfraction}{.95}
\renewcommand{\bottomfraction}{.95}
\renewcommand{\textfraction}{.05}
\renewcommand{\floatpagefraction}{.9}
\graphicspath{{figures/}}
\input{metrics_macros.tex}
\input{validation_macros.tex}
\newcommand{\shuttleplots}[2]{%
\begin{figure}[!htbp]\centering
\includegraphics[width=.92\linewidth]{#1}
\caption{#2}
\end{figure}\FloatBarrier}
\newcommand{\pgplots}[4]{%
\begin{figure}[!htbp]\centering
\includegraphics[width=.88\linewidth]{#1_training.pdf}\par\smallskip
\includegraphics[width=.88\linewidth]{#1_hist.pdf}
\caption{#2. Top: every training episode reward and its trailing 100-episode average. Bottom: 500 stochastic final-policy evaluation rewards; mean #3, population standard deviation #4.}
\end{figure}}''')
change('\\section{Introduction}\n',r'''\section{Introduction}
GitHub username: \texttt{FailedMath3U}. Repository name: \texttt{MLDS}. Link: \url{https://github.com/FailedMath3U/MLDS}. The code, tests and saved numerical results are included there; run instructions are at the end of this report.

For both games, the main plots use training seed 1668 and the final fixed-budget model. Evaluation uses 500 fresh episodes with sampled actions and no learning. Plots and evaluation statistics use undiscounted episode rewards, while training uses the prescribed discounted objective. Standard deviations below are population standard deviations. CartPole used CPU only; Pong used one GPU.
''')
change("We will \\textbf{assume} that we will only incur the cost of passengers waiting who will not have been picked up by a shuttle by the next time step, and that we don't incur the cost of the ones who'll arrive by the next timestep. Given a state $t$ and action $a$ let $c(s,a)$ denote our cost.","We will \\textbf{assume} that holding cost is charged for passengers remaining immediately after dispatch, before new arrivals. Given a state $s$ and action $a$ let $c(s,a)$ denote our cost.")
change('For the sake of convenience we will not mention rewards for the rest of the section.','For the sake of convenience we will mostly use costs for the rest of the section. The plots also show the reward value, which is the negative of the cost.')
change('Thus we have completely modeled our problem as a formal Markov decision process.','We also assume overflow above 200 is discarded without an extra penalty, and that there is no separate shuttle-availability or travel-time constraint. Thus we have completely modeled our problem as a formal Markov decision process.')
change("Note that the optimal value we're computing is only an approximation of the optimal value of the MDP above; not only because of the finite horizon but because our implicit policy will change depending on the time.","This computes the optimal cost of the finite-horizon problem, up to numerical error. It approximates the infinite-horizon cost, and its policy may change depending on the time.")
change('We start by having our final values be 0.','We start by having our final values be $J_{500}(s)=0$, assuming no terminal penalty.')
change(r'$$J_t(s) = \min_a  \left[ c(s,a) + E(J_{t+1}(s_{t+1}(s_t,a))\right], $$',r'\[J_t(s) = \min_a \left[c(s,a)+\gamma\mathbb{E}\!\left[J_{t+1}(s_{t+1}(s,a))\right]\right],\]')
change(r'$$J_t(s) = \min_{pi} E_{\pi} \left[ \sum_{u=t}^{499} \gamma^{u-t} c(s_u, a_u) \ | \  s_t = s \right] $$',r'''\[J_t(s) = \min_{\pi}\mathbb{E}_{\pi}\!\left[\sum_{u=t}^{499}\gamma^{u-t}c(s_u,a_u)\,\middle|\,s_t=s\right].\]
Here $J_t(s)$ is the minimum expected remaining discounted cost; $\pi$ is the decision rule and $\mathbb E_\pi$ averages the random arrivals. Define the reward value by $V_t(s)=-J_t(s)$.
\shuttleplots{shuttle_finite_cost_value.pdf}{Enumeration, 500 decisions: optimal time-zero cost $J_0(s)$ (top) and reward value $V_0(s)=-J_0(s)$ (bottom).}''')
change('so total cost will be quadratic (amount times duration stuck there).','a larger queue means more people waiting and more time needed to clear it, although discounting means the cost need not be exactly quadratic.')
change(r'\subsection{(b) Policy Iteration (Infinite Horizon)}',r'\subsection{(b) Value Iteration (Infinite Horizon)}')
change(r'\[ J^{n+1}(s) = \min_a \left[ c(s,a) + \gamma E(J^n (\boldsymbol{s}(s,a)) \right] \]',r'\[J^{n+1}(s)=\min_a\left[c(s,a)+\gamma\mathbb E\!\left[J^n(\boldsymbol{s}(s,a))\right]\right].\]')
change('It is mathematically guaranteed to converge sufficiently (change is less than $10^{-10}$ for all states) in a bounded amount of time (a function of $\\gamma$). Then we terminate.',r'Because the Bellman update is a contraction with factor $\gamma<1$, it converges. We terminate when another Bellman update would change the current vector by at most $10^{-10}$ in every state (the Bellman residual).')
change('This consistency validates our independent numerical experiments.','This consistency supports the numerical results, although the methods share the same model.\n'+r'\shuttleplots{shuttle_infinite_cost_value.pdf}{Value iteration: optimal infinite-horizon cost $J^*(s)$ (top) and reward value $V^*(s)=-J^*(s)$ (bottom).}')
change(r'\subsection{Policy Iteration (Infinite Horizon)}',r'\subsection{(c) Policy Iteration (Infinite Horizon)}')
a=s.index('Policy iteration is an alternating process.');b=s.index('Then we see if we can greedily improve',a)
s=s[:a]+r'''Policy iteration is an alternating process. Given a policy $\pi$ (in this case a function from states to actions), its cost satisfies
\[J_\pi(s)=c(s,\pi(s))+\gamma\mathbb E\!\left[J_\pi(\boldsymbol{s}(s,\pi(s)))\right].\]
It's actually implemented using the $J$ vector (for all states) and the transition probability matrix $P_\pi$. We solve
\[(I-\gamma P_\pi)J_\pi=c_\pi,\]
where $I$ is the identity matrix, $P_\pi(s,s')=P(s'\mid s,\pi(s))$ and $c_\pi(s)=c(s,\pi(s))$. Thus our code uses a linear solve for policy evaluation, rather than an inner iteration with a tolerance.
'''+s[b:]
change(r'\[ \pi(s) = \arg \min_a   \left[ c(s,a) + \gamma E(J^*_\pi (\boldsymbol{s}(s,a)))\right]\]',r'\[\pi_{\mathrm{new}}(s)=\arg\min_a\left[c(s,a)+\gamma\mathbb E\!\left[J_\pi(\boldsymbol{s}(s,a))\right]\right].\]')
change('It is guaranteed that this policy is at least as low cost as $\\pi$.','It is guaranteed that this new policy is at least as low cost as $\\pi$.')
change('We can then apply the previous iterative procedure to compute the cost. Then alternate back into policy iteration.','We can then evaluate the new policy with the same linear solve, and alternate back into policy improvement.')
change("We can thus plot the optimal policy by plotting whether it'll dispatch or not at a given number of customers waiting.","We can thus plot the optimal policy by plotting whether it'll dispatch or not at a given number of customers waiting. This took six evaluation/improvement rounds, starting from always waiting, including the final stability check. Ties are resolved by waiting.\n"+r'\shuttleplots{shuttle_policy.pdf}{Policy iteration: wait at queues 0--12 and dispatch at queues 13--200.}')
a=s.index('It is mathematically evident that the optimal policy');b=s.index('\\subsection{Edge Cases}',a)
s=s[:a]+r'''The computed optimal policy dispatches whenever we're above a certain threshold and otherwise waits. No threshold was imposed on the search. If the cost of waiting times the number of waiters is sufficiently high relative to $\gamma$, it might even make sense to order a shuttle when you have less than 15. Thus the threshold has face validity. More quantitatively, at queue 12 the costs of waiting and dispatching, including optimal continuation, are 672.892 and 676.670; at 13 they are 680.770 and 676.670. These comparisons explain the threshold of 13.

'''+s[b:]
change('We made sure that for the 501st time step $J_{500}=0,$ $\\pi_{500}=0$  and passengers top being added, preventing ghost continuation.',r'We made sure that $J_{500}=0$. There is no decision at $t=500$, preventing ghost continuation costs. At $t=499$, waiting is optimal in every state: dispatch costs 100 but can save at most $2\cdot15=30$ immediately, with no future benefit or terminal penalty. We checked this prediction against the saved final-period policy.')
a=s.index('Value iteration and policy iteration selected');b=s.index('We can also compare the results',a)
s=s[:a]+r'''Value iteration and policy iteration selected the same actions for all 201 states. The differences of the values were at most $1.964\times10^{-9}$. The policy-iteration Bellman residual was $9.095\times10^{-13}$, bounding value error by residual$/ (1-\gamma)=1.819\times10^{-11}$, subject to rounding. The minimum absolute difference between the two action costs across states was 3.778, meaning that numerical errors are unlikely to change the optimal action. Solver agreement is useful, but their shared model could still contain a common error.

'''+s[b:]
change("We've checked that at zero passengers, our policy prefers to wait rather than dispatch.",r'''We've checked that at zero passengers, our policy prefers to wait rather than dispatch. An exhaustive audit of all 2,010 state/action/arrival combinations also matched scalar transitions and costs, with every transition-probability row summing to 1.

For independent predictions, setting holding cost to zero gave zero optimal cost and always waiting; setting $\gamma=0$ gave $J(s)=2s$ and always waiting. A separately coded exhaustive decision/arrival tree with seven states and four decisions matched backward induction exactly.''')
change('\\subsection{Validation With an Additional 3 months}\n',r'''\subsection{Validation With an Additional 3 months}
We would vary arrival distributions, dispatch and holding costs, and the queue cap to see how the threshold changes. We would also compare a separately implemented linear-programming solution, and test overflow penalties and terminal costs. Adding correlated arrivals or shuttle travel times would test whether the original policy remains useful when its modeling assumptions change.
''')
change('{REINFORCE}',r'\textbf{REINFORCE}')
change('an easy to compute unbiased estimator of the return under $\\pi_\\theta$','an easy to compute unbiased estimator of the gradient of expected discounted return under $\\pi_\\theta$')
change(r"Then we use $G_t \nabla_\theta \log(\pi_\theta(S_t, A_t)$ as our gradient, where intuitively log corrects for the fact that we want to nudge rarely seen $(S_t,A_t)$ more since we'll be nudging them less often (there is a mathematical derivation).",r'Then we use terms $\gamma^tG_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)$ in our gradient. The log comes from the identity $\nabla_\theta\pi_\theta=\pi_\theta\nabla_\theta\log\pi_\theta$.')
change(r'Hence given a stepsize $\alpha > 0$ use the gradient as input into the Adam optimizer, for each $(S_t,A_t)$ in the sequence used to compute $G_t$.',r'''Hence given a stepsize $\alpha>0$, we sum the discounted gradient terms within each episode, average over the batch of episodes, and use the result for one Adam update. The objective is $\mathbb E[\sum_{t=0}^{T-1}\gamma^t r_t]$, where $T$ is the episode length and $r_t$ its reward after decision $t$. Our code uses a fixed gradient scale of 0.05 for CartPole and 1 for Pong; it does not standardize returns. Gradient-norm caps are 10 and 5 respectively, so the clipped estimator need not remain unbiased.''')
change('plus an absorbing state representing a horizontal pole.','with failure when the angle exceeds $12^\\circ$ or the cart leaves $[-2.4,2.4]$. Episodes are also capped at 500 steps.')
change("The transition rules are entirely deterministic, a simple application of Newton's laws. The reward is +1 for every timestep it stays out of the terminal absorbing state. The discount factor is $\\gamma = 0.95$ which is appropriate since this game doesn't involve any long term consequences.","The transition rules are entirely deterministic given state and action, a numerical application of Newton's laws. The initial four components are sampled uniformly from $(-0.05,0.05)$, with a distinct recorded seed per episode. The reward is +1 for every transition, including the failure transition. The discount factor is the specified $\\gamma=0.95$.")
change('when the angle reaches a certain value or the cart leaves its allowed range','when the angle exceeds its allowed range or the cart leaves its allowed range')
change('There is one hidden layer with tanh activation functions.','There is one hidden layer with 64 units and tanh activation functions.')
change('there is a huge amount of variance','there can be variance')
change('slightly changing policies per epsiode','changing policies per update')
change('So I plotted the 100 average of episode returns, and although it still wobbled around the policy value, it was sufficiently flat relative to the significant changes in our REINFORCE plot. Thus we\'ve validated that our training improvement is not just sampling noise.',r'''So I plotted the 100 average of episode returns with an earlier policy frozen for 2,000 fresh episodes. Its averages ranged from 430.11 to 472.29, while continued training fell to 74.64. Frozen checkpoints before and during the dip scored 453.108 and 83.114 over 500 fresh paired episodes, with a paired 95\% bootstrap interval of $[-377.430,-362.038]$ for the change. Thus the selected dip reflects a policy change beyond fixed-policy episode noise. The checkpoints were selected after looking at the curves; this does not identify the cause of the bad updates.''')
change("We've experimented with smaller step sizes but then although the growth is much more stable, it is also much more slow. We will see that REINFORCE with a good baseline gives much steadier training gains with the same speed. This is because it's also an unbiased estimator of the expected return, but with lower variance.",r'''We've experimented with smaller step sizes, but the final performance was not consistently better. Across three seeds, the basic method's mean final evaluation scores were 418.570, 163.178 and 247.102 for rates 0.01, 0.005 and 0.002 respectively. A good baseline can reduce gradient variance, but does not guarantee steadier training gains.

We trained for 8,000 episodes with Adam rate 0.01 and batches of 32 complete episodes. The primary final policy scored $500.000\pm0.000$ over 500 evaluations; all reached the limit without physical failure.
\pgplots{cartpole_no_baseline}{CartPole REINFORCE, seed 1668}{\CartBasicMean}{\CartBasicSD}''')
change("The timelimit is 500 and we terminated the episode there if the it hadn't already failed. We tested that the episode actually terminated and that our return reset.",r'The timelimit is 500 and we stopped the episode there if it had not already failed. We checked time-limit truncation separately from physical termination. Either flag ends our capped episode, with zero continuation. A failure on the last allowed step can set both flags; we tested that case too.')
change('We weigh all gradients for all choices equally. If we were to instead weigh episodes inversely proportional to their length (so in some sense equally), then we would be not have an unbiased estimator of the policy gradient.',r'Within each episode, decision terms are weighted by $\gamma^tG_t$ (and the fixed scale), summed, and then averaged over 32 episodes. Dividing each episode contribution by its own length would change this estimator. Dividing the whole batch by its total state count would instead rescale that batch, with a scale depending on episode lengths.')
change('\\subsection{CartPole Validation and Correctness Checks}\n',r'''\subsection{CartPole Validation and Correctness Checks}
An untrained neural policy scored $20.926\pm11.467$ and uniform random actions scored $22.782\pm12.058$, each over 500 episodes. These controls support improvement beyond initialization. For an analytic check, a two-step Bernoulli-action task with reward equal to its action has objective $(1+\gamma)p$, where $p$ is the probability of action 1. Enumerating all four paths gave gradient 0.471188756 versus analytic derivative 0.471188748. Returns $(1,1,1)$ also matched $(2.8525,1.95,1)$.

Controlled episodes of lengths 1, 1 and 3 exercised the actual batched collector. The one-step return was 1 and the combined gradient matched the average of separate episode gradients to $1.863\times10^{-9}$. A saved ten-step trace records states, probabilities, actions and rewards; at step 3, left was sampled even though right had probability 0.7711, as can happen under a stochastic policy.
''')
change('\\subsection{CartPole 3 Additional Months}\n',r'''\subsection{CartPole 3 Additional Months}
We would test more training seeds and initial positions, angles and velocities outside the narrow reset region. We would vary batch size and baseline-fitting frequency alongside learning rate, and measure critic error and gradient variance throughout training to investigate the dips rather than only extending training.
''')
change('So our state space implicitly contains the direction the ball is moving.','So our observation implicitly contains cues to the direction the ball is moving.')
change("But the policy can just learn not to do that, so it shouldn't hinder our model.","This can hide its position and may hinder our model; we have not established that the policy learns to avoid the problem.")
change('The two possible actions are to move your paddle 8 downsampled pixels in either directions.','The two possible actions are ALE actions 2 and 3 (RIGHT and LEFT), which move the paddle vertically in opposite directions. They are directional commands, not a verified fixed pixel displacement.')
change('We do not directly use or compute any probabilities.','We do not directly use or compute any transition probabilities; the policy still computes action probabilities.')
change("We use a discount factor of $\\gamma = 0.99$'\npresumably because we need to think more ahead than for the CartPole problem.","We use the specified discount factor of $\\gamma=0.99$.")
change('We can think of this as an absorbing state.','We stop return accumulation at either episode-ending flag.')
change('As you can see, we got significant improvements, but have still not achieved the maximum theoretical performance.',r'''We chose this lower rate because, on identical game data, one update changed the action distribution less than rate 0.01 (mean KL divergence $1.996\times10^{-6}$ versus $3.423\times10^{-4}$). This is a step-size check, not proof of better long-run stability. We trained for 20,000 full games, updating after batches of 8. The final score was $-18.610\pm1.681$ over 500 games, versus $-20.250\pm0.935$ for a 32-game untrained reference. All 500 final games were losses and none was truncated, so the policy did not learn to win.
\pgplots{pong_no_baseline}{Pong REINFORCE, seed 1668}{\PongBasicMean}{\PongBasicSD}''')
change('$(0.0197, 0.0199, -0.99, -1)$','$(0.019701, 0.0199, -0.99, -1)$')
change("We tested that we don't continue to compute returns past the game end in either situation.",r"We tested that we don't continue to compute returns past the game end in either situation. A real complete-game test ended after \PongTerminationLength{} decisions with final reward and return both $-1$; a forced one-step limit truncated with final reward and return both 0. Neither collector stepped the environment after ending.")
change("It doesn't really matter anyway.","This means the first action has no motion information; later differences use only frames from the same game.")
change('Normally players serve by using the FIRE command. But here our bot couldn\'t play fire. This was not a problem as we ran tests and all the games terminated by scoring events. The game specs don\'t mention what happens if we use FIRE.',r"The environment includes a FIRE command, but here our bot couldn't play fire. This did not prevent play: eight preflight games using only actions 2 and 3 each produced 21--22 scoring events and ended naturally. This does not show that FIRE has no effect on unrestricted gameplay.")
change("However the policy will get negative rewards if it tries to move into the boundary and then doesn't get out by moving in the opposite direction.","Boundary commands are not directly penalized; a negative reward occurs only when the opponent scores. Thus we cannot assume rewards automatically solve the missing-position problem.")
change('We had no choice but to retain them. Because we actually move 4 Pong-v5 actions per model action, this results in at least one of our 4 actions being ignored but the rest executed. This should have a fairly small effect because in the next model timestep, the model can see where it ended up and act using that.',r'We retained this behavior to match v5. The repeat probability is 0.25 per emulator frame, so within four frames there can be zero, one or more repetitions; it is not guaranteed that exactly one requested action is ignored. Identical reset seeds and fixed action sequences reproduced frame/reward hashes. This tests reproducibility, not that stickiness has a small effect.')
change('\\subsection{}',r'''\subsection{Pong 3 Additional Months}
We would compare frame differences with frame stacks and a small convolutional policy, retaining v5 and actions 2/3. We would measure how much of the gradient comes from late-game rewards, since $0.99^{1000}\approx4.32\times10^{-5}$, and compare learning rates, batch sizes and clipping on separate validation seeds. Multiple training seeds would be needed to distinguish a robust improvement from a favorable run.''')
change('A baseline is just an amount we subtract off of $G_t$ independent of the rest of the episode.','A baseline is just an amount we subtract off of $G_t$ that does not depend on the current action, given the information before that action.')
change("This doesn't change the expectation of the gradient but changes it's variance.","This doesn't change the expectation of the unclipped gradient estimator but changes its variance.")
change('predict an baseline based on the action and state','predict a baseline based on the state and time')
change('We use a new new neural network with input thh state and the time (dived by 500).','We use a new neural network with input the state and the time (divided by 500).')
change('We train it with 5 Adam steps with learning rate $0.01$ to minimize MSE (since we want to avoid huge updates to our original NN). We update our actor model after completing our batch of 32 episodes.',r'We update our actor model after completing our batch of 32 episodes, using detached baseline predictions from its existing weights. Afterward we train the separate baseline with 5 Adam steps at learning rate $0.01$ to minimize MSE against the observed $G_t$s. Thus current outcomes are used to fit predictions for later batches.')
change('The plot of undiscounted training reward is smoother now but it has roughly the same performance. This is what we should expect from the Theory.',r'''The plot still has large changes, and the primary final model scored $260.308\pm57.013$, worse than the basic model. Four of the 500 evaluations reached 500; 496 failed physically. Theory does not guarantee a better final policy. Across three seeds, the average final scores were 323.588 with the baseline and 418.570 without it.
\pgplots{cartpole_baseline}{CartPole with a learned baseline, seed 1668}{\CartBaselineMean}{\CartBaselineSD}''')
change("Here we use an exponential moving average of past baselines with a newly calculated terms using only this batch's returns. For a given state that we visited we compute this new term using the discounted sum of the future $G_t$s in that episode, divided by the sum of the discount factors.",r'''Here we use a scalar exponential moving average. Batch $k$ uses the existing $b_k$, starting at $b_0=0$. After the actor update, we use this batch's returns to compute
\[\bar G_k=\frac{\sum_{i,t}\gamma^tG_t^{(i)}}{\sum_{i,t}\gamma^t},\qquad b_{k+1}=0.95b_k+0.05\bar G_k.\]
The index $i$ runs over all eight games, and $t$ is the decision within each game. Thus it is a discount-weighted average of reward-to-go over the batch, not a second return calculation or a value fitted separately for each state.''')
change('The plot is fairly similar to the one without a baseline, which is what we would expect since the original had almost no variance.',r'''The plot is fairly similar to the one without a baseline. The final result was $-18.446\pm1.851$ over 500 games, only 0.164 above the basic method; all were losses. This does not establish a benefit across training seeds or lower gradient variance.
\pgplots{pong_baseline}{Pong with a constant EMA baseline, seed 1668}{\PongBaselineMean}{\PongBaselineSD}''')
change('If the rewards are all 0, then the our chosen baselines should be 0. We tested this in both the CartPole and Pong case.',r'''If the rewards are all 0, then the target returns are 0 and a fitted baseline should approach 0. We tested this in both cases. Fitting the CartPole baseline for 400 Adam steps on 64 modified episodes gave held-out MSE $\ZeroFitMSE$ on 64 separate episodes. Pong's EMA stayed exactly 0 when initialized at 0; from 2 it decayed to \ZeroEMA{} after 160 zero-return updates, matching $2(0.95)^{160}$. The latter is a fixed-data baseline-mechanics experiment, not a new on-policy performance run.''')
change('Setting all the baselines to 0 recovers the REINFORCE.',r'Setting all the baselines to 0 recovers REINFORCE. On identical data, the maximum gradient difference was exactly 0 in both implementations. Zero returns with zero baseline also gave gradient norm 0.')
change('We made sure to update our baseline after we updated our policy, otherwise our baseline would depend on the samples episodes.',r'We made sure to fit our baseline after updating the policy, so its predictions for the current actor update use weights from earlier data. Fitting on the current outcomes first can introduce dependence on the current actions, even if actions are not explicit network inputs.')
change('If the rewards for all actions are 1 and the games terminate, then the baselines should converge to 1, and our gradients should be 0. This is unlike the case without a baseline.',r'''If an episode lasts only one decision and every action receives reward 1, then all returns are 1. With an exact baseline of 1, both actor implementations gave gradient norm 0. Fitting the CartPole predictor on one-step episodes gave held-out mean \OneFitMean{} and MSE $\OneFitMSE$; Pong's EMA approached \OneEMA{} after 160 updates from 0. Finite fitting is approximate, so exact cancellation is checked separately with the baseline fixed to 1. Without a baseline, individual sampled gradients can be nonzero, although their expectation is still zero because every action has the same payoff. For longer episodes with reward 1 at every decision, returns exceed 1, so the target baseline need not be 1.

Finally, we checked whether baselines actually reduced gradient variance. At a frozen initial CartPole actor, fitting a separate baseline on 512 episodes and testing on 512 different paired episodes reduced summed gradient-component variance from 5.338262 to 0.842421 (84.22\%). This used 150 fitting steps, rather than the five per training batch. Pong's independently fitted initial constant reduced variance from 369.320 to 20.763 on 16 held-out games, but the actual EMA's later variance ratios were 0.873, 1.017, 0.924 and 1.011. Thus variance reduction did not persist at every checkpoint; these probes measure unclipped gradients, and 16 games gives noisy estimates.''')
change('\\subsection{3 More Months}\n',r'''\subsection{3 More Months}
We would compare constant, state-dependent and time-dependent baselines on identical frozen policies, using separate fitting data and more held-out episodes. We would track baseline error and gradient variance throughout training, with uncertainty intervals, and separate the effect of clipping from baseline subtraction. This would test when lower gradient variance actually produces better control.

\section{Running the Code}
Paths below are relative to \texttt{assignment1/} in the repository. Use a fresh copy for training to preserve saved runs, and install each project's \texttt{requirements.txt}. Shuttle runs with \texttt{python shuttle.py}; its separate \texttt{shuttle\_plots.ipynb} regenerates plots. For CartPole:
\begin{verbatim}
cd mlds_assignment1_cartpole
python cartpole.py suite --episodes 8000 --lr 0.01
python validate.py
python validate_task_edges.py
python instability_experiment.py
python plot_instability_experiment.py
\end{verbatim}
These run the six training/evaluation runs, numerical and environment checks, and exact replay/frozen-policy comparison. Evidence is saved in \texttt{runs/}, \texttt{validation/}, \texttt{task\_validation/} and \texttt{instability\_experiment/}. The lower-rate folders provide \texttt{run.sh}.

For Pong, activate the deepdish Pong environment and work in a fresh copy of \texttt{mlds\_assignment1\_pong}. Set a single GPU UUID consistently in \texttt{config.json}, \texttt{supervise.sh} and \texttt{CUDA\_VISIBLE\_DEVICES}; the supervisor also specifies the Python executable. Then run:
\begin{verbatim}
python run_pong.py preflight
nohup setsid bash supervise.sh > nohup.log 2>&1 < /dev/null &
\end{verbatim}
Preflight checks returns, analytic gradients, image processing, restricted actions, seeded replay and checkpoint recovery. Training adds variance probes. Results are in \texttt{preflight/}, \texttt{runs/} and \texttt{diagnostics/}. Existing checkpoints resume instead of starting over.

The additional tests claimed in this draft are in \texttt{mlds\_assignment1\_submission/}. With PyTorch, Gymnasium and ALE installed, run:
\begin{verbatim}
python validate_report_claims.py
python validate_shuttle_and_images.py
python build_assets.py
tectonic ASSIGNMENT1_REPORT.tex
\end{verbatim}
The first tests zero/unit-return baseline fitting, gradient cancellation, zero-baseline equivalence, and real Pong termination/truncation. The second checks every shuttle transition and signed/stationary image differences. Both save numerical evidence under \texttt{validation/} without modifying the original training files. The asset builder copies saved game plots and generates both cost and negative-cost shuttle plots.
''')
# Keep short discussions together when a figure is placed nearby.
start=s.index(r'\subsection{Pong 3 Additional Months}')
end=s.index(r'\section{Policy Gradient with Baseline}',start)
s=s[:start]+r'\begin{samepage}'+'\n'+s[start:end]+r'\end{samepage}'+'\n\n'+s[end:]
start=s.index(r'\subsection{Edge Cases, Correctness Checks and Validation}')
end=s.index('\n\nSetting all the baselines',start)
s=s[:start]+r'\begin{samepage}'+'\n'+s[start:end]+'\n'+r'\end{samepage}'+s[end:]
s=s.replace(r'\end{samepage}',r'\par\end{samepage}')
(ROOT/'ASSIGNMENT1_REPORT.tex').write_text(s)
diff=''.join(difflib.unified_diff((ROOT/'USER_DRAFT.tex').read_text().splitlines(keepends=True),s.splitlines(keepends=True),fromfile='USER_DRAFT.tex',tofile='ASSIGNMENT1_REPORT.tex'))
(ROOT/'DRAFT_CHANGES.diff').write_text(diff)
print('Revised draft and exact change diff written.')
