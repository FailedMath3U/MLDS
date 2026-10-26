# Pong-v5: one-GPU long-run protocol

This is training/validation code, not a report. The report will be written after
results are available. Local source is under Downloads/mlds_assignment1_pong;
the live job and outputs are under ~/mlds_assignment1_pong on deepdish5.

## Resource and algorithm choices

- Exactly one physical GPU is visible: GPU 0, UUID
  GPU-cef65388-88b0-f39a-6bac-c5316eb78540 (RTX 5060 Ti, 16 GB).
- A single main process holds both policies on that GPU. Eight CPU workers emulate
  independent games; they do not import PyTorch or use GPUs. Methods alternate
  complete-game batches so both progress in one job.
- Environment: `ALE/Pong-v5` (the maintained namespaced Pong-v5), Gymnasium 1.2.2,
  ALE 0.11.2. Defaults retained: frameskip 4, sticky-action probability 0.25,
  minimal action space, 108000 emulator-frame limit. No double frame skipping,
  no life-loss episodes, no reward clipping/shaping, no FIRE or NOOP startup.
- The policy samples ONLY ALE actions 2 (RIGHT) and 3 (LEFT), as assigned.
- Frames use the assignment's crop/downsample/background removal to 80x80 binary
  images. The network receives successive-frame differences (zero on reset),
  providing motion information. Stored inputs are int8, evaluated as float32.
- Policy: 6400 -> 200 ReLU -> 2 categorical logits. Gamma=0.99. Adam lr=0.001
  (lower than the assignment's default 0.01); a same-data one-update preflight
  compares both rates' KL/entropy changes. Gradient norm cap=5; actual norms saved.
- Full-game Monte Carlo reward-to-go; no resetting returns at point boundaries.
  The actor loss includes the outer gamma**t for the discounted episodic objective.
  NO centering, return standardization, entropy bonus, PPO or TD bootstrapping.
- Part 1 uses baseline 0. Part 2 uses a scalar constant baseline, an EMA with rate
  0.05 of past batches' discount-weighted mean reward-to-go. It is frozen before
  collecting each batch, independent of current sampled actions, and updated only
  afterward. Both methods start from identical weights and reset/action seed schedules.

## Budget and checkpoint selection

The configured maximum is 20,000 games per method, with a 16-hour wall budget for
training plus intermediate validation/diagnostics. Time is checked at balanced
batch boundaries; final 500-game evaluations run afterward and can take additional
hours. This is a ceiling, not a forecast of training speed or guaranteed success.
Budget and all hyperparameters are preserved in config.json.

Each method validates every 1024 training games on 32 reserved games. Validation
reset/action seeds are separate from training and final tests. Every validation
checkpoint is retained, and the best validation mean selects a policy BEFORE
viewing test rewards. Both selected and final policies automatically receive
500 stochastic test games each. If they are the identical model, the same saved
500-game evaluation is reused. Final tests use reset seeds 20000000–20000499.
No model selection uses those test scores.

## Protection against interruption

- A lock prevents duplicate main processes. Source/config hashes must match the
  passed preflight; the long job refuses to start on untested edited code.
- Full actor/Adam states, EMA, logs, counters, selection metadata and RNG states
  are atomically checkpointed every five updates and at validation/phase changes.
  The preceding checkpoint is retained as a fallback.
- Complete-game collection and per-episode environment/action seeds allow
  deterministic restart at a batch checkpoint, including after a mid-cycle exit.
- Initial and milestone policies, first-point timing, entropy, saturation, score,
  lengths, per-point outcomes, gradients and baseline values are saved for analysis.
- The nohup supervisor retries failures twice from the saved checkpoint. It does
  not silently restart training from zero if checkpoints cannot be read.
- Evaluation saves atomic progress every batch and resumes from completed games.
  Each episode has an independent action RNG, so resume does not change the
  stochastic evaluation for its reset/action seed.
- SIGTERM/SIGINT asks the trainer to finish a complete batch, checkpoint, and pause.
  A paused job exits normally; the supervisor does not automatically undo the pause.

## Validation built in before and during training

`python run_pong.py preflight` must pass before launch. It verifies:

1. Exact crop, background removal, binary output, and no mutation of the raw frame.
2. Hand-calculated discounted returns, including a reward sequence crossing point
   boundaries; zero discount; no unintended point-wise return reset.
3. An exactly enumerated two-step MDP: analytic gradient equals expected sampled
   gradient, with/without an action-independent baseline.
4. A positive-return policy update increases its chosen-action probability; zero
   returns without a baseline produce zero gradient.
5. Seeded emulator replay with sticky actions, action meanings, frame-difference
   reset/range, complete-game scoring using only actions 2 and 3.
6. Exactly one visible CUDA device and real GPU forward/backward compatibility.
7. Identical initial actors across methods and baseline use restricted to earlier batches.
8. Save/load of the Adam optimizer plus identical next parameter update on the same
   fixed real-game data (an optimizer recovery test, not extra on-policy learning).
9. Learning-rate step-size experiment on the same initial actor and complete-game
   data, recording policy KL/entropy at 0.01 versus 0.001.
10. Frozen-policy, paired-trajectory gradient-variance measurement. A baseline is
    fitted on separate trajectories, then both estimators are measured on the
    same held-out games. Full per-parameter Welford variances are summed; mean
    difference and its standard error are recorded. Improvement is measured,
    never imposed as an assertion.

Additional frozen-policy variance probes run every 4096 training games per method.
They do NOT update policy weights, optimizers, training EMA or training RNG streams.
Saved trace CSVs expose action probabilities, ALE actions, rewards, reward-to-go,
outer discounts and nonzero image pixels. Paired compressed frame differences
allow later visual inspection. Traces include the beginning, first point and end
of a game rather than silently treating points as episodes.

These tests establish reproducibility and mathematical/implementation checks;
they do not guarantee successful learning. Score curves and final results will
show actual performance, including failure or seed limitations.

## Monitoring and control (on deepdish5)

```bash
cd ~/mlds_assignment1_pong
cat status.json
tail -n 20 nohup.log
nvidia-smi -i 0
```

Main process ID: trainer.pid; supervisor ID: supervisor.pid.
To pause gracefully:

```bash
kill -TERM "$(cat trainer.pid)"
```

After a deliberate pause, resume with the same unchanged source/config using:

```bash
nohup setsid bash supervise.sh > nohup_resume.log 2>&1 < /dev/null &
echo $! > supervisor.pid
```

Do not start a second job while the first is running. Source changes intentionally
invalidate preflight. The three source files affecting the job hash are
pong_env.py, pong_core.py and run_pong.py; plotting/report work does not affect it.

## Output map

- `preflight/results.json`: assertions, versions, environment defaults and benchmark.
- `preflight/initial_gradient_variance.json`: initial fixed-policy variance study.
- `config.json`: exact run configuration and rationale.
- `checkpoints/latest.pt`, `latest.previous.pt`: full recoverable suite states.
- `checkpoints/*_episode_*.pt`: independent milestone policies.
- `runs/{no_baseline,baseline}/training.csv`: training metrics including every game.
- `runs/*/best_validation.pt`, `final.pt`: selected/final models.
- `runs/*/validation/`: checkpoint selection data on reserved validation games.
- `runs/*/evaluation/`: 500-game test progress, rewards, SDs, and selection provenance.
- `diagnostics/`: gradient-variance measurements and trace/frame data over training.
- `status.json`: live phase, progress, elapsed time, GPU memory and selection summaries.
- `summary.json`: final selected/final metrics once all tests finish.
- `failure.json`: last exception if any; development preflight failures can predate
  successful launch, so compare timestamps with status and current logs.
- `plots/`: score curves with 100-game mean, entropy/gradient diagnostics and final
  histograms; generated by plot_results.py at normal exit or manually at any time.

No report PDF or LaTeX source is created on deepdish5 by this job.
