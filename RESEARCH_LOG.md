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

<!-- next entries appended at each steering check-in -->
