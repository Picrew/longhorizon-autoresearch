# Method levers for long-horizon agentic SFT

This is **not** a hyperparameter grid. The research question is: *what changes to
the training method make a small model better at choosing good actions across a
long agent trajectory?* The loop is adaptive — each experiment is chosen from the
evidence so far (see RESEARCH_LOG.md). This file is the menu of **method-level**
levers to draw from, why each should help long-horizon, and how it's wired.

The metric is fixed (see run_experiment.py header): held-out causal-LM loss on the
agent's **decision tokens** (`<|assistant|>` spans), lower is better. Method
changes are judged against that fixed yardstick.

## Implemented (config knobs in run_experiment.py)

- **`loss_on`: `"all"` → `"assistant"`** — mask the loss on system/user/**tool-output**
  tokens; train only on the agent's own tokens. *Long traces are mostly
  environment observations (API JSON, file dumps) that are noise to memorise;
  supervising only decisions concentrates capacity on the actual skill.* This is
  the single most important lever and the reason the metric was redefined.
- **`truncation`: `"right"` / `"tail"` / `"head_tail"`** — when a trace exceeds
  `max_length`, standard `"right"` keeps the head and **drops the ending** (where
  the goal is achieved). `"tail"` keeps the end; `"head_tail"` keeps goal+recent
  context and drops the middle ("lost in the middle"). *Directly about not
  throwing away the long-horizon payoff.*
- **`max_length`** — train-time context window (the eval window is fixed, so this
  trades sequence coverage against steps-per-budget).
- LoRA placement/capacity (`lora_r/alpha/dropout/target_modules`), `lr`,
  schedule/warmup, `weight_decay`, `neftune_noise_alpha`, effective batch,
  `attn_implementation` — supporting knobs, used only when evidence motivates.

## To implement when evidence motivates (build lazily, one at a time)

- **Late-token / position loss weighting** — upweight decision tokens deeper in
  the trajectory (quality decay and goal drift happen late). Needs a weighted
  cross-entropy in a custom `compute_loss`.
- **Decision-point upweighting** — upweight the first assistant tokens right after
  each tool observation (the moments that actually require a choice).
- **Length-balanced / curriculum sampling** — reweight or order examples by length
  so a few 30k-token traces don't dominate the gradient; optionally ramp short→long.
- **Packing with cross-trace attention masking** — pack multiple traces to fill the
  window for throughput, but block attention across trace boundaries (no leakage).
- **Goal re-anchoring augmentation** — periodically re-inject the original goal
  into very long traces so the model is trained to hold the objective (ties to the
  SPT "goal stability" thesis).
- **RoPE / context-length scaling** — extend usable context for the longest traces.
- **Self-prediction auxiliary (SPT-lite)** — insert self-prediction targets
  (goal recall / trajectory forecast / "should I stop?") whose labels come from the
  trace's own future. The project's flagship idea; highest-novelty, build last.

## Phase 3+ bold directions (long-agent step-change, experiments → 80)
After discovering the metric noise is ~0.004 (so sub-0.005 single-run deltas are unreliable),
the search shifted to **big levers + rigour**, focused on *training to improve long agents*:

- **Data scale-up (long-heavy):** the model was trained on only 2,730 examples / ~1,045 long.
  Train on a 16k+ slice weighted 60% long (`big_long_v1`), eval on the fixed pilot val
  (leakage-filtered). More long agentic data on an undertrained model is the most reliable
  step-change.
- **Train to convergence:** the full-budget win showed undertraining; with more data, push
  train_seconds far higher (longer flagship runs are OK now) and watch the scaling curve.
- **SPT self-prediction auxiliary (the project's thesis):** at checkpoints in long traces,
  add self-prediction targets whose labels are the trace's own future — goal recall,
  trajectory-position/“will this succeed”, quality self-rating. L = L_task + λ·L_selfpred.
  The on-thesis long-horizon method; build incrementally.
- **Decision-point upweighting / goal re-anchoring:** test at scale, not at the noise floor.
- **Efficiency for more effective training:** sequence packing + flash-attention to fit more
  real tokens per unit time (longer effective training within a budget).
- **Higher-capacity PEFT:** rsLoRA / DoRA / higher rank, now that data isn't the bottleneck.

## Rules
1. Prefer a **method change** over a hyperparameter tweak when both are plausible.
2. One change per experiment, so attribution is clean (karpathy discipline).
3. Every experiment records *why it was chosen given prior results* in RESEARCH_LOG.md.
4. Keep only if it beats the running-best decision loss **by more than the noise margin
   (~0.004)**; for important claims, confirm with a 2-seed re-run.
5. Chase **step-changes**, not noise-level drift.
