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
| 0004 | oversample-long2 | Long bucket improved least under loss_on=assistant. Spend more of the fixed 64-step bud... | 0.77760 | decision_loss |  | 60 | 1217.7 |
| 0005 | late-token-weight-2 | Quality decay / goal drift happen late in a trajectory. Upweight later-in-context decis... | 0.77317 | decision_loss |  | 64 | 1215.5 |
| 0006 | lr-2e-4 | Two long-horizon levers failed -> likely undertraining in the 64-step budget. Test lear... | 0.73336 | decision_loss | ✅ | 65 | 1217.6 |
| 0007 | lora-r32-all-linear | Test if capacity is the bottleneck: LoRA r16 attn-only -> r32 alpha64 with MLP targets ... | failed | full_seq |  |  |  |
| 0008 | lr-3e-4 | Continue LR search: 2e-4 -> 3e-4 (undertraining confirmed by 0006). | 0.71926 | decision_loss | ✅ | 65 | 1217.4 |
| 0009 | ctx-6144-throughput | Throughput: train ctx 8192->6144 to fit more optimizer steps in the fixed budget. Eval ... | 0.71048 | decision_loss | ✅ | 78 | 1205.8 |
| 0010 | cosine-cooldown-64 | Proper LR schedule on the lr2e-4 best: cosine decay over planned 64 steps with 3-step w... | 0.75848 | decision_loss |  | 64 | 994.9 |
| 0011 | lr-4e-4 | Keep climbing LR ridge: 3e-4 -> 4e-4 until it regresses. | 0.70361 | decision_loss | ✅ | 78 | 1206.0 |
| 0012 | ctx-5120 | More throughput: train ctx ->5120 for more steps. Watch long bucket for the coverage fl... | 0.69836 | decision_loss | ✅ | 90 | 1206.9 |
| 0013 | lora-r32-attn | Memory-safe capacity test (0007 OOM'd with MLP): LoRA r16->r32 alpha64, attn-only. | 0.68925 | decision_loss | ✅ | 90 | 1205.5 |
| 0014 | lr-5e-4 | Keep climbing LR ridge: 4e-4 -> 5e-4. | 0.68933 | decision_loss |  | 90 | 1205.9 |
| 0015 | ctx-4096 | Coverage-floor probe: train ctx ->4096 for more steps. Eval stays @8192. Watch long buc... | 0.68985 | decision_loss |  | 110 | 1201.4 |
| 0016 | warmup-5 | Brief warmup (5 steps) to stabilize the now-high LR from step 1. | 0.69172 | decision_loss |  | 90 | 1206.1 |
| 0017 | lora-r64 | Capacity still helping (r32 gave -0.009); push rank r32->r64, alpha->128. | 0.68860 | decision_loss | ✅ | 90 | 1207.8 |
| 0018 | neftune-5 | Method lever: NEFTune embedding-noise regularisation (alpha 5) on the best recipe. | 0.68717 | decision_loss | ✅ | 90 | 1207.7 |
| 0019 | baseline-v2-longeval | Reanchor on decision_loss_v2 (eval scores decision tokens up to 16384, chunked CE). Re-... | 0.67903 | decision_loss_v2 | ✅ | 76 | 1020.9 |
| 0020 | truncation-tail | Now that eval sees beyond 8192: train on the END of long traces (truncation=tail keeps ... | 0.67753 | decision_loss_v2 | ✅ | 76 | 1020.8 |
| 0021 | truncation-head-tail | Keep goal+recent context, drop the middle (head_tail) -- trains on both the goal setup ... | 0.67423 | decision_loss_v2 | ✅ | 76 | 1021.4 |
| 0022 | late-weight-v2 | Re-test late-token loss weighting (1->2 ramp) under v2: upweighting late decisions shou... | 0.67784 | decision_loss_v2 |  | 75 | 1020.4 |
| 0023 | ctx-8192-v2 | Re-test long context under v2: train ctx ->8192 (more context coverage, fewer steps). D... | 0.67568 | decision_loss_v2 |  | 55 | 1037.6 |
| 0024 | ctx-6144-v2 | Intermediate context under v2: ctx ->6144 to bracket the v2-optimal training context (5... | 0.67488 | decision_loss_v2 |  | 66 | 1028.4 |
| 0025 | headtail-ctx6144 | head_tail wins + long traces want more context: test head_tail at ctx 6144 (vs 5120). C... | 0.67590 | decision_loss_v2 |  | 66 | 1027.5 |

_25 experiments logged; 16 on the v1 (head@8192) curve, 7 on the v2 (long-horizon, to 16384) curve, 13 kept improvements total._
