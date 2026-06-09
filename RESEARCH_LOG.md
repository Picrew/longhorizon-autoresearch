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

<!-- next entries appended at each steering check-in -->
