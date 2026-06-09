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

<!-- next entries appended at each steering check-in -->
