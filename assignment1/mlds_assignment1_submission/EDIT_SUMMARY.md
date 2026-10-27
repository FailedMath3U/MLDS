# What changed

The original text is preserved in `USER_DRAFT.tex`; `DRAFT_CHANGES.diff` shows every edit. `ASSIGNMENT1_REPORT.tex` remains the editable main file.

- Added the required GitHub identity, execution instructions, eight game plots and their 500-episode mean/SD results, and the three shuttle plots. The two shuttle cost plots additionally show their negative-cost reward values, as requested.
- Filled the empty validation and additional-three-months sections using the same direct, explanatory style.
- Corrected the finite-horizon Bellman discount, exact versus approximate objectives, the mislabeled value-iteration heading, and the policy-evaluation linear solve.
- Replaced the invalid threshold argument and AI confirmation with the measured state-12/state-13 action costs; corrected solver error and minimum-action-gap numbers.
- Corrected the CartPole stopping/reward description and episodic gradient weighting, and removed claims that the smaller rates or baseline necessarily produced steadier/better learning.
- Replaced the unverified fixed Pong displacement with the implemented action commands; corrected FIRE, sticky-action and boundary-reward claims. Pong's failure to win remains explicit.
- Corrected baseline action independence, actor/critic update order, the EMA equation, and the distinction between constant rewards and constant returns.
- Implemented and ran the missing numerical validations. Original training algorithms/results and earlier report files were preserved; the new report is stored separately.

No attempt was made to polish unusual but accurate wording. Main game results and training were not rerun or selected to favor the new report.
