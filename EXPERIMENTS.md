# Experiment log

Autonomous, adaptive autoresearch over agentic long-horizon SFT (Qwen3-4B QLoRA,
one RTX 4090). Each experiment is chosen from the evidence so far (the reasoning
lives in [RESEARCH_LOG.md](RESEARCH_LOG.md)); the search targets **training methods**
for long traces, not a hyperparameter grid (see [METHODS.md](METHODS.md)).

Primary metric = held-out **decision-token** loss (loss on `<|assistant|>` tokens
only), lower is better, fixed ~20-min wall-clock training budget per experiment.
Exp 0001 used the old full-sequence metric and is shown for reference (not on the
decision-loss curve).

| # | slug | hypothesis | metric | kind | kept | steps | train s |
|---|------|------------|--------|------|------|-------|---------|

| 0001 | baseline | naive QLoRA: r16 attn-only, lr1e-4, no warmup, no packing | 0.93794 | full_seq | ✅ | 64 | 1215.5 |
| 0002 | baseline-decision-loss | Re-anchor the curve on the decision-token metric: same recipe as 0001 (r16 attn-only, l... | 0.80773 | decision_loss | ✅ | 64 | 1201.2 |
| 0003 | loss-on-assistant | First method change: supervise only the agent's own tokens (mask system/user/tool-outpu... | 0.76900 | decision_loss | ✅ | 64 | 1200.2 |

_3 experiments logged; 2 on the decision-loss curve, 2 kept improvements._
