# longhorizon-autoresearch

An autonomous, karpathy/autoresearch-style self-improving training loop for
**agentic long-horizon SFT**, run end-to-end on a single RTX 4090. An agent
(Claude) proposes one change at a time, trains under a fixed budget, scores a
held-out metric, keeps or discards, logs the reasoning, and repeats — ~44
experiments across three phases, chasing a real step-change in how well a small
model models the *decisions an agent makes deep into a long trajectory*.

![progress](progress.png)

*The curve above is the long-horizon-faithful phase (v2 metric). The two big
drops are the headline result: scaling **data + compute together**.*

## Setup
- **Base model:** `Qwen3-4B-Instruct-2507`, 4-bit QLoRA, one RTX 4090.
- **Data:** quality-filtered multi-step agent×environment trajectories
  (long-horizon weighted). Two slices: a 2.7k pilot and a **9.1k long-heavy** set
  (60% traces ≥12k tokens).
- **Metric (lower=better):** held-out causal-LM loss on the agent's **decision
  tokens** only (the `<|assistant|>` spans — its reasoning + tool-calls), *not*
  on tool-output/observation tokens (those are environment noise, not the skill).
  The eval is a **fixed yardstick** so method changes never move the goalposts.
- **Budget:** fixed wall-clock per experiment (a method that trains faster wins by
  fitting more steps — the karpathy framing). Late phase used longer budgets to
  probe the compute frontier.

Everything is reproducible from files: `experiments/ledger.jsonl` (every run),
`method.md` (the running-best recipe), `RESEARCH_LOG.md` (the *why* behind each
step), `EXPERIMENTS.md` (the table), `autoresearch/` (the harness).

---

## The three phases

### Phase 1 — knob tuning on a coarse metric (eval @ 8192, head only)
Started from a deliberately naive baseline and let the loop tune. **0.808 → 0.687**
(9 kept improvements). The one genuine *method* win here was the rest were
hyperparameters/throughput:
- **`loss_on=assistant`** — supervise only the agent's own tokens, masking the ~70%
  of tokens that are observations. The single biggest lever.
- LR → 4e-4, context 8192→5120 (throughput: more steps/budget), LoRA r16→r64.

### Phase 2 — the eval-window upgrade (the long-horizon-faithfulness fix)
Phase 1's metric only scored the first 8192 tokens, so *late-trajectory* decisions —
the actual long-horizon test — were invisible, and truncation methods were
unmeasurable. We rebuilt the eval to score decision tokens **up to 16384 tokens**
via **chunked cross-entropy** (`decision_loss_v2`), and re-anchored the curve.

We also discovered (via seed re-runs) that the **run-to-run noise is ~0.004**, so we
added a noise-margin guard and **2-seed-confirm every claimed win**. This retro­
actively showed several Phase-1/early-v2 "wins" were noise — a discipline that
shaped everything after.

### Phase 3 — the data + compute step-change ⭐
On the faithful metric, the recipe stalled around **0.674** with knobs. The real
lever turned out to be **scale** — but only when *data and compute move together*:

| training data | steps | decision loss |
|---|---|---|
| 2.7k pilot | 95 | 0.6657 |
| **9.1k long-heavy** | 91 | 0.6688 *(more data, same compute → wash)* |
| **9.1k long-heavy** | 258 | 0.6512 |
| **9.1k long-heavy** | 388 | **0.6401** ⭐ best (2-seed confirmed) |
| 9.1k long-heavy | 516 | 0.6431 *(plateau; long bucket regresses)* |

**Data alone was a wash and compute alone had saturated; together they gave a
−0.026 step-change (0.6657 → 0.6401), confirmed across two seeds.** The model was
*both* data- and compute-limited — invisible while the knob phase held both fixed.
It plateaus at ~388 steps (before even one epoch), and **more LoRA capacity (r128)
made it worse**, so ~0.640 is the QLoRA/proxy ceiling for this setup.

The long bucket — the thing we actually care about — moved most: **0.681 → 0.6495**.

---

## Final recipe (exp 0039)
```
base:        Qwen3-4B-Instruct-2507, 4-bit nf4 QLoRA
data:        9.1k long-heavy agentic traces (60% long), leakage-filtered vs eval
loss_on:     assistant   (mask system / user / tool-output tokens)
truncation:  head_tail   (keep the goal + recent context, drop the middle)
max_length:  5120
LoRA:        r64, alpha128, dropout0.05, attn-only (q,k,v,o)
optim:       lr 4e-4, constant-with-warmup, paged_adamw_8bit, bs1 × grad_accum8
budget:      ~5400 s train (~388 optimizer steps)
```
**Held-out decision-token loss 0.6401** (long-horizon metric, scored to 16384).
v2-phase reduction **0.679 → 0.640 (−5.7%)**; the data+compute step-change alone
**0.6657 → 0.6401 (−0.026)**.

## Honest negatives (these did *not* help, and that's a result)
- **Reweighting** (late-token / position weighting) — failed twice, even on the
  long-horizon metric. Coverage (`head_tail`) beats reweighting.
- **Cosine decay, NEFTune, oversampling long traces, r96/r128 capacity** — all
  within or worse than the ~0.004 noise floor.
- **SPT self-prediction (goal-recall)** — *no improvement on this proxy* (0.6428 vs
  0.6401). And rightly so: the val set has no self-check spans, so a next-token loss
  **cannot** measure goal-stability — it can only see that the auxiliary slightly
  dilutes the decision-token budget. **SPT's real value needs a capability eval
  (Hexagon-Bench), not this proxy.** This is the sharpest methodological takeaway:
  *pick the metric that can see the thing you're trying to improve.*

## What actually moves long-agent SFT (the summary)
1. **Supervise the agent's decisions, not the environment's tokens** (`loss_on=assistant`).
2. **Don't drop the trajectory's payoff** — `head_tail` truncation keeps goal + ending.
3. **Scale data and compute together** — the step-change; neither alone works.
4. Everything else here was noise — measure honestly (fixed metric, 2-seed, noise margin).

## Reproduce
```bash
# on the GPU box (data + model alongside the harness)
python autoresearch/run_experiment.py \
  --config autoresearch/configs/baseline.json --exp-id repro \
  --data-root <root> --model <root>/models/Qwen3-4B-Instruct-2507 --out-dir runs/repro
# or run the full autonomous loop:
python autoresearch/loop.py --data-root <root> --model <model> \
  --max-experiments 80 --keep-margin 0.004 --per-exp-timeout 20000
```
See [RESEARCH_LOG.md](RESEARCH_LOG.md) for the full decision narrative,
[METHODS.md](METHODS.md) for the lever menu, [EXPERIMENTS.md](EXPERIMENTS.md) for
the table, and `experiments/<id>/result.json` for per-experiment details.
