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
| 0026 | headtail-ctx7168 | head_tail at ctx 7168 -- more coverage still. Brackets head_tail's optimal training con... | 0.67839 | decision_loss_v2 |  | 59 | 1022.6 |
| 0027 | train-1260 | Use more of the 30-min budget: train_seconds 1020->1260 (eval is only ~7min). Undertrai... | 0.66566 | decision_loss_v2 | ✅ | 95 | 1270.8 |
| 0028 | train-1320 | Push training to the full budget: train_seconds 1320 (total ~30min). | 0.66626 | decision_loss_v2 |  | 99 | 1323.5 |
| 0029 | cosine-95 | Cosine decay done right at full budget: planned_steps=95 (matches the ~95-step run), wa... | 0.67553 | decision_loss_v2 |  | 95 | 1270.8 |
| 0030 | lora-r96 | More capacity now that training is longer: r64->r96, alpha->192. | 0.66628 | decision_loss_v2 |  | 94 | 1266.4 |
| 0031 | ctx8192-fullbudget | Re-test ctx8192 at train1260 (~63 steps vs 55 before): does the long bucket's appetite ... | 0.67169 | decision_loss_v2 |  | 67 | 1263.2 |
| 0032 | best-seed43 | Robustness/noise re-run of the BEST config with seed 43 (quantify the ~0.001 noise on t... | 0.67004 | decision_loss_v2 |  | 95 | 1268.9 |
| 0033 | neftune-10 | Cheap genuine test: more embedding noise, NEFTune alpha 5->10. | 0.66716 | decision_loss_v2 |  | 95 | 1271.3 |
| 0034 | neftune-off | Ablation: 0018's NEFTune win (-0.0014) was noise-level. Remove it -- if the loss is unc... | 0.66618 | decision_loss_v2 |  | 95 | 1271.8 |
| 0035 | bigdata-t1260 | SCALE-UP: train on the 9.1k long-heavy slice (4.7k long, vs pilot 2730/1045) at train_s... | 0.66878 | decision_loss_v2 |  | 91 | 1267.6 |
| 0036 | bigdata-t2400 | Scale compute on big data: train_seconds 2400 (~40min). More steps to actually traverse... | timeout | full_seq |  |  |  |
| 0037 | bigdata-t3600 | Flagship scale run: big data + train_seconds 3600 (~1hr, ~270 steps). Find the data+com... | 0.65118 | decision_loss_v2 | ✅ | 258 | 3600.4 |
| 0038 | bigdata-seed43 | 2-seed confirm of the 0037 step-change (big-data@3600=0.6512, -0.0145): same config, se... | 0.64813 | decision_loss_v2 |  | 260 | 3609.9 |
| 0039 | bigdata-t5400 | Push compute scaling further: train_seconds 5400 (~390 steps, ~40% of an epoch on the 9... | 0.64014 | decision_loss_v2 | ✅ | 388 | 5406.0 |
| 0040 | spt-goalrecall | SPT self-prediction (thesis method): spt_goal_recall=true on the big-data best -- train... | 0.64277 | decision_loss_v2 |  | 385 | 5405.7 |
| 0041 | bigdata-t7200 | Push compute frontier: train_seconds 7200 (~520 steps, ~46% epoch). Scaling not plateau... | 0.64312 | decision_loss_v2 |  | 516 | 7213.4 |
| 0042 | bigdata-t7200-s43 | 2-seed confirm of the train7200 frontier (seed 43). Keeps GPU busy within the 9000s tim... | 0.63921 | decision_loss_v2 |  | 518 | 7211.0 |
| 0043 | r128-t5400 | Capacity break-attempt: r64->r128 (alpha256) at the best compute (train5400/388 steps).... | 0.65508 | decision_loss_v2 |  | 385 | 5402.2 |
| 0044 | r128-t7200 | Capacity + compute: r128 @ train7200 (516 steps). If r64 plateaued at 388 due to capaci... | 0.65449 | decision_loss_v2 |  | 512 | 7206.4 |

_44 experiments logged; 16 on the v1 (head@8192) curve, 25 on the v2 (long-horizon, to 16384) curve, 16 kept improvements total._
