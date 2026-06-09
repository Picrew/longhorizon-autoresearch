# Research log

The narrative of the search: for each experiment, **why it was chosen given the
results so far**, what happened, and what it implies for the next one. This is the
heart of "adaptive" — the loop is not a pre-written plan, it is decisions made from
evidence. (The machine-readable record is `experiments/ledger.jsonl`; the recipe is
`method.md`; the chart is `progress.png`.)

Metric: held-out **decision-token** loss (loss on `<|assistant|>` tokens only),
lower is better. Fixed budget ~20 min train + eval per experiment on one RTX 4090.

---

### exp 0001 — full-sequence baseline (pre-metric-upgrade)
**Chosen because:** establish a top-of-curve reference with the naive recipe
(QLoRA r16 attn-only, lr 1e-4, no warmup, train on all tokens, full-sequence loss).
**Status:** ran as the very first experiment; scored on the *old* full-sequence
metric. Kept in the ledger as a reference but **excluded from the decision-loss
curve** — the curve re-anchors at the first decision-loss experiment.
**Lesson that triggered the redesign:** full-sequence loss rewards predicting
tool-output/observation tokens, which is not the agentic skill and can't even see
the most important method change (decision-only training). → redefined the metric
to decision-token loss; the search now targets *methods*, not knobs (see METHODS.md).

### exp 0001 result (full-seq metric)
Full-sequence val loss **0.93794** in 64 steps / ~20 min train. Per bucket (full-seq):
long 0.939, medium 0.917, short 1.119. Validated the harness end-to-end. Recorded
as the reference; not on the decision-loss curve.

---

### exp 0002 — re-anchor on the decision-token metric
**Chosen because:** the metric was redefined to decision-token loss, so the curve
needs a clean anchor measured the new way. Same recipe as 0001 (r16 attn-only,
lr 1e-4, `loss_on=all`), `reanchor=true`. This is the real top-of-curve.
**Note from the smoke run:** decision tokens are only ~30% of all tokens, so under
`loss_on=all` ~70% of the training signal is spent predicting observations.

### exp 0003 — loss_on=assistant (first method change)
**Chosen because:** directly acts on the 30/70 split above — mask the loss on
system/user/**tool-output** tokens and supervise only the agent's own decisions.
*Hypothesis:* concentrating capacity on decision tokens lowers held-out decision
loss, most on the long bucket (where observation tokens dominate the trace).
Single change vs the 0002 anchor, so attribution is clean. If it wins, the next
question is *where* the gain lands (per-bucket) → likely points at truncation or
late-token weighting next.

### exp 0002 & 0003 results
- **0002** (decision-loss anchor): decision loss **0.80773**. Per bucket: long 0.809,
  medium 0.783, short 0.992. (Note: the smoke showed only ~30% of tokens are agent
  decisions.)
- **0003** (loss_on=assistant): decision loss **0.769**, **kept (−0.039)**. Per bucket:
  long 0.774 (−0.034), medium 0.743 (−0.040), short 0.927 (−0.065).
- **Reading:** masking the loss to the agent's own tokens helps everywhere, but
  **least on the long bucket** — exactly where long-horizon capability lives and
  where the most headroom remains. In long traces the decision tokens are sparse
  and sit late in a context dominated by observations, and the ~64-step budget
  barely touches them.

### exp 0004 — oversample long traces ×2
**Chosen because:** directly acts on "long helped least" — give the fixed step
budget more exposure to long traces by duplicating long_horizon examples ×2 (on
top of assistant-masking). Clean single change vs the 0003 recipe.

### exp 0005 — late-token loss weighting (1→2 ramp)
**Chosen because:** the long-horizon thesis is that quality decays and goals drift
*late* in a trajectory. Upweight decision tokens deeper in the context so the
model is pushed to stay sharp there. Weights are scale-matched to mean-1 (only the
relative emphasis changes; effective LR unchanged — verified the train-loss scale
after fixing a grad-accum normalisation bug in the custom weighted loss).

### exp 0004 & 0005 results — both discarded, and that's the signal
- **0004** (oversample long×2): **0.7776**, discarded. Every bucket got *worse*,
  including long (0.779 vs 0.774). In a ~64-step budget, duplicating long traces
  trades diversity for repetition → worse generalisation. Also ran fewer steps (60).
- **0005** (late-token weight 1→2): **0.77317**, discarded. Also worse across buckets
  (long 0.778). Upweighting late tokens de-emphasises the early/mid decisions the
  fixed head-eval rewards.
- **Reading:** the two intuitive "spend more on long/late" levers both hurt. So the
  long-bucket headroom is **not** an exposure or late-emphasis problem within the
  8192 window — the likely bottleneck is simply **undertraining** (64 steps is tiny)
  and/or **LoRA capacity**. Next: diagnose that fork before touching the eval window.

### exp 0006 — lr 1e-4 → 2e-4
**Chosen because:** in a step-limited budget the dominant question is how much the
model learns per step. grad-norms were healthy (~0.7–0.9) at 1e-4, so there's room.
If 2e-4 lowers loss broadly, undertraining is the bottleneck → pursue schedule /
throughput (more steps per budget) next.

### exp 0007 — LoRA r32 + all-linear targets
**Chosen because:** r16 on attn-only may be too little capacity to fit decision
behaviour. Add MLP targets (gate/up/down) and r32/α64. If this helps, capacity is
the bottleneck → tune rank/placement; if not, capacity is fine and it's an
optimisation/throughput problem.

### exp 0006 & 0007 results — undertraining confirmed
- **0006** (lr 1e-4 → 2e-4): **0.73336**, **kept (−0.0356)** — the biggest gain after
  assistant-masking. Helped every bucket: long 0.744 (−0.030), medium 0.706 (−0.037),
  short 0.863 (−0.064). **The bottleneck was undertraining**, exactly as the 0004/0005
  failures implied: with only ~64 steps, learning faster beats spending the budget on
  long/late tokens.
- **0007** (r32 + all-linear targets): **OOM / FAILED** — r32 with MLP targets at 8192
  ctx + 4-bit + grad-ckpt exceeded 24 GB. Capacity question unanswered; keep capacity
  tests memory-safe (attn-only r32, or lower ctx, or expandable_segments).
- **Direction:** push the undertraining lever. (1) more LR, (2) a proper schedule, (3)
  more steps per budget (throughput). New best to stack on = 0006 (lr 2e-4).

### exp 0008 — lr 2e-4 → 3e-4
Continue the LR search to find the peak before it overshoots.

### exp 0009 — train ctx 8192 → 6144 (throughput)
More optimizer steps per fixed budget. The eval stays fixed @8192, so this is an
honest test of the steps-vs-coverage trade-off for long traces (does more training
beat full per-trace context?). Watch the long_horizon bucket specifically.

### exp 0010 — cosine cooldown (lr 2e-4, planned 64 steps, 3-step warmup)
A real LR schedule: warm up briefly then decay over the run. Standard cooldown often
buys a little once the LR magnitude is right.

### exp 0008 & 0009 results — both kept; throughput is real
- **0008** (lr 3e-4): **0.71926**, kept (−0.014). LR still climbing (long 0.732, med 0.691,
  short 0.837). Peak not yet found → try 4e-4.
- **0009** (train ctx 8192→6144): **0.71048**, kept (−0.009), and ran **78 steps** vs 65.
  Crucially the long bucket *improved* (0.7324→0.7248) even though it's evaluated at the
  fixed 8192. **More steps beat more per-trace coverage** here — and it did NOT hurt
  long-horizon, so this isn't the "short seqs are useless" failure mode; it's honest
  throughput. → push ctx lower (5120) and watch where the long bucket turns.
- 0010 (cosine cooldown) was still running at this check-in; result next time.
- Stacked best = ...assistant + lr3e-4 + ctx6144 = 0.710. Curve: 0.808→0.769→0.733→0.719→0.710.

### exp 0011 — lr 3e-4 → 4e-4
Keep climbing the LR ridge until it regresses (then we've bracketed the peak).

### exp 0012 — train ctx 6144 → 5120 (more throughput)
Push the steps-vs-coverage trade further. Watch long_horizon: if it finally regresses,
we've found the coverage floor for long traces; if not, steps keep winning.

### exp 0013 — LoRA r16 → r32 (attn-only, memory-safe)
Re-ask the capacity question 0007 couldn't (it OOM'd with MLP targets at 8192). r32
attn-only is cheap; tests whether more rank helps now that LR/steps are tuned.

### exp 0010/0011/0012 results
- **0010** (cosine cooldown): **0.758**, discarded — but FLAWED: planned_steps=64 capped
  training to 64 steps (vs 78 at the wall-clock budget), so this conflated decay with
  *less training*. Cosine not ruled out; a fair test needs planned_steps ≈ achievable
  steps at that ctx. Lesson: don't set planned_steps below the budget.
- **0011** (lr 4e-4): **0.70361**, kept (−0.007). LR still climbing → try 5e-4 (+warmup).
- **0012** (ctx 5120): **0.69836**, kept (−0.005), 90 steps. Long bucket still improving
  (0.720→0.715) — more steps keep beating coverage; floor not yet hit → try ctx 4096.
- Best = assistant + lr4e-4 + ctx5120 = **0.698**. Curve now 0.808→...→0.698 (6 kept).

### exp 0014 — lr 4e-4 → 5e-4
Keep climbing; high LR may want warmup (see 0016).

### exp 0015 — train ctx 5120 → 4096 (coverage-floor probe)
Aggressive throughput. Eval stays @8192 (long-horizon). If the long bucket finally
regresses, lock the floor (this is where the user's "full context matters" intuition
would kick in); if not, more steps still win.

### exp 0016 — warmup 5 steps
A brief warmup often lets a higher LR train stably from step 1 (constant_with_warmup,
so horizon-independent — unlike the flawed cosine 0010).

**Next major move (planned):** once lr/ctx/warmup peak, upgrade the eval to score long
traces beyond 8192 via chunked cross-entropy (reanchor) so tail/head_tail truncation
and other genuinely long-horizon method levers become measurable.

### exp 0013 & 0014 results
- **0013** (LoRA r16→r32 attn-only): **0.68925**, kept (−0.009). Capacity is a real,
  fresh axis (long 0.708, med 0.658, short 0.787). → try r64.
- **0014** (lr 5e-4): **0.68933**, discarded — tied with 4e-4 (+0.0001). **LR has peaked
  at 4e-4; lock it.** No more LR experiments.
- Best = assistant + lr4e-4 + ctx5120 + r32 = **0.689**. Curve 0.808→…→0.698→0.689 (7 kept).
- (0015 ctx4096 and 0016 warmup5 still pending.)

### exp 0017 — LoRA r32 → r64 (alpha 128)
Capacity still helping; push rank once more before it plateaus.

### exp 0018 — NEFTune noise alpha 5
A training-method lever (embedding-noise regularisation) from METHODS.md; sometimes
helps generalisation. Quick to test, low OOM risk.

**Decision:** knob ridge is nearly done (LR locked, capacity ~1 step left). NEXT check-in,
once 0015/0016/0017/0018 are read, execute the EVAL-WINDOW UPGRADE (chunked CE beyond
8192, reanchor) and pivot the search to long-horizon method levers (truncation, late
weighting re-tested on the faithful metric, decision-point upweighting).

### exp 0015/0016/0017/0018 results — knob ridge done
- **0015** (ctx 4096): 0.690, discard. 110 steps but the **long bucket regressed**
  (0.710 vs 0.708) — over-truncation. **Coverage floor = 5120**; the user's
  full-context intuition kicks in below it.
- **0016** (warmup 5): 0.692, discard. Constant LR from step 0 is fine for LoRA.
- **0017** (r64): 0.6886, kept but only −0.0007 → capacity plateauing.
- **0018** (NEFTune α=5): **0.68717**, kept (−0.0014). Small regularisation win.
- v1 best recipe = assistant + lr4e-4 + ctx5120 + r64 + neftune5 = **0.687**.
  v1 curve: 0.808→…→0.687 (9 kept, ~15% relative). Knobs exhausted.

### EVAL-WINDOW UPGRADE (v2) — the long-horizon-faithfulness fix
The v1 metric only scored the first 8192 tokens, so late-trajectory decisions
(the actual long-horizon test) were invisible, and truncation levers were
unmeasurable. v2: `score_heldout` now scores decision tokens up to **16384** via
**chunked cross-entropy** (2048-position chunks) to avoid the [seq,vocab] float32
OOM. Smoke on the 6 longest val traces (29k–32k tokens, capped at 16k): no OOM
(peak 20.4 GB), eval 22.7 s/6 rows; those long traces are only ~12.5% decision
tokens. metric_kind='decision_loss_v2'; train trimmed to 1020 s to keep total
≤30 min with the heavier eval. The curve re-anchors here.

### exp 0019 — v2 reanchor (current best recipe, scored on v2)
Establishes the long-horizon curve's top from the v1-best recipe.

### exp 0020/0021/0022 — long-horizon levers, now measurable
- **0020 truncation=tail**: train on the END of long traces (late decisions) — the
  eval can finally see them.
- **0021 truncation=head_tail**: keep goal + recent context, drop the middle.
- **0022 late-token weighting (1→2)**: re-test 0005 under v2; upweighting late
  decisions should now pay off when the metric rewards late-trajectory quality.

### v2 results begin
- **0019** (v2 anchor, best recipe scored to 16384): **0.67903** (long 0.681, med 0.662,
  short 0.786). Total wall 1462 s (train 1021 + eval 425) — comfortably ≤30 min.
- **0020** (truncation=tail): **0.67753**, kept (−0.0015), but the **long bucket was flat**
  (0.681→0.6808); the gain came from medium. At ctx5120, tail-training just swaps head
  decisions for tail ones.
- **Key realisation the v2 metric exposes:** the model trains on ≤5120 tokens of context
  but v2 evaluates decisions made with up to 16384 tokens. That train/eval context
  mismatch is the long-horizon gap. Under v1 (head@8192 eval) shorter ctx won because
  more-steps dominated and the eval never tested long context. Under v2 it should pay to
  train on more context → re-test ctx upward.

### exp 0023 / 0024 — re-test training context length under v2
0023 ctx 8192, 0024 ctx 6144 (vs the current 5120). If the long bucket now improves with
longer ctx, v2 rewards true long-context training (the user's full-context intuition,
finally measured); if 5120 still wins, more-steps still dominates even on the long eval.

### v2 long-horizon levers — head_tail wins; the context/steps tradeoff is real
- **0021** (truncation=head_tail): **0.67423**, kept (−0.0033, biggest v2 gain). Helped the
  long bucket too (0.6808→0.6783). Keeping goal+recent and dropping the middle is the right
  way to fit long traces into a fixed window.
- **0022** (late-weight v2): 0.6778, discard — late-token upweighting fails again, even when
  the eval rewards late decisions. Coverage (head_tail) beats reweighting.
- **0023** (ctx 8192): 0.6757 overall, discarded — BUT the **long bucket hit 0.6754, the best
  long number yet**. Only 55 steps (vs 76), so medium/short undertrained and dragged the
  aggregate down. **Finding: long traces want more training context; the fixed step budget
  makes it cost shorter-trace quality.** The user's full-context intuition, quantified.
- Best = …+head_tail = **0.6742**.

### exp 0025 / 0026 — head_tail's optimal context
Since long traces like more context and head_tail is the winner, test head_tail at ctx 6144
(0025) and 7168 (0026) vs 5120: does giving head_tail more coverage capture the long-bucket
gain without losing too many steps?

<!-- next entries appended at each steering check-in -->
