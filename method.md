# Current best method

- best experiment: **0027**
- held-out val loss (lower=better): **0.66566**

## Config
```json
{
  "train_file": "data/agentic_experiments/pilot_4090_longhorizon_v1/train.sft.jsonl",
  "val_file": "data/agentic_experiments/pilot_4090_longhorizon_v1/val.sft.jsonl",
  "loss_on": "assistant",
  "truncation": "head_tail",
  "max_length": 5120,
  "lora_r": 64,
  "lora_alpha": 128,
  "lora_dropout": 0.05,
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
  ],
  "lr": 0.0004,
  "lr_scheduler_type": "constant_with_warmup",
  "warmup_steps": 0,
  "warmup_ratio": 0.0,
  "planned_steps": null,
  "weight_decay": 0.0,
  "optim": "paged_adamw_8bit",
  "batch_size": 1,
  "grad_accum": 8,
  "max_grad_norm": 1.0,
  "neftune_noise_alpha": 5.0,
  "attn_implementation": "sdpa",
  "train_seconds": 1260,
  "max_steps_cap": 100000,
  "seed": 42,
  "eval_limit": 0
}
```

## Accepted improvements (in order)

- `0001` baseline: naive QLoRA: r16 attn-only, lr1e-4, no warmup, no packing -> loss 0.93794
- `0002` baseline-decision-loss: Re-anchor the curve on the decision-token metric: same recipe as 0001 (r16 attn-only, lr1e-4, loss_on=all) but scored on decision loss. This is the true top-of-curve. -> loss 0.80773
- `0003` loss-on-assistant: First method change: supervise only the agent's own tokens (mask system/user/tool-output). Smoke showed ~70% of tokens are observations -- noise to memorise; concentrating capacity on the ~30% decision tokens should lower decision loss. -> loss 0.769
- `0006` lr-2e-4: Two long-horizon levers failed -> likely undertraining in the 64-step budget. Test learning faster: lr 1e-4 -> 2e-4 on the loss_on=assistant best. -> loss 0.73336
- `0008` lr-3e-4: Continue LR search: 2e-4 -> 3e-4 (undertraining confirmed by 0006). -> loss 0.71926
- `0009` ctx-6144-throughput: Throughput: train ctx 8192->6144 to fit more optimizer steps in the fixed budget. Eval stays fixed @8192; honest steps-vs-coverage test. Watch long bucket. -> loss 0.71048
- `0011` lr-4e-4: Keep climbing LR ridge: 3e-4 -> 4e-4 until it regresses. -> loss 0.70361
- `0012` ctx-5120: More throughput: train ctx ->5120 for more steps. Watch long bucket for the coverage floor. -> loss 0.69836
- `0013` lora-r32-attn: Memory-safe capacity test (0007 OOM'd with MLP): LoRA r16->r32 alpha64, attn-only. -> loss 0.68925
- `0017` lora-r64: Capacity still helping (r32 gave -0.009); push rank r32->r64, alpha->128. -> loss 0.6886
- `0018` neftune-5: Method lever: NEFTune embedding-noise regularisation (alpha 5) on the best recipe. -> loss 0.68717
- `0019` baseline-v2-longeval: Reanchor on decision_loss_v2 (eval scores decision tokens up to 16384, chunked CE). Re-score the current best recipe (assistant+lr4e-4+ctx5120+r64+neftune5) on the long-horizon-faithful metric; train_seconds 1020 to keep total <=30min with the heavier eval. -> loss 0.67903
- `0020` truncation-tail: Now that eval sees beyond 8192: train on the END of long traces (truncation=tail keeps the trajectory's late decisions). Previously unmeasurable under head@8192 eval. -> loss 0.67753
- `0021` truncation-head-tail: Keep goal+recent context, drop the middle (head_tail) -- trains on both the goal setup and the late payoff of long traces. -> loss 0.67423
- `0027` train-1260: Use more of the 30-min budget: train_seconds 1020->1260 (eval is only ~7min). Undertrained model should drop above the ~0.001 noise floor. -> loss 0.66566
