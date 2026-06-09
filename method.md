# Current best method

- best experiment: **0006**
- held-out val loss (lower=better): **0.73336**

## Config
```json
{
  "train_file": "data/agentic_experiments/pilot_4090_longhorizon_v1/train.sft.jsonl",
  "val_file": "data/agentic_experiments/pilot_4090_longhorizon_v1/val.sft.jsonl",
  "loss_on": "assistant",
  "truncation": "right",
  "max_length": 8192,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
  ],
  "lr": 0.0002,
  "lr_scheduler_type": "constant_with_warmup",
  "warmup_steps": 0,
  "warmup_ratio": 0.0,
  "planned_steps": null,
  "weight_decay": 0.0,
  "optim": "paged_adamw_8bit",
  "batch_size": 1,
  "grad_accum": 8,
  "max_grad_norm": 1.0,
  "neftune_noise_alpha": null,
  "attn_implementation": "sdpa",
  "train_seconds": 1200,
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
