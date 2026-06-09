"""Run ONE autoresearch experiment: QLoRA SFT under a fixed wall-clock budget,
then score held-out validation loss. Emits a single result.json.

This is the unit the autoresearch loop optimizes. The metric is held-out
token-weighted causal-LM loss (lower is better) -- the same objective SFT
minimizes, evaluated on data never trained on. Because loss is token-weighted
and long traces dominate the val token count, the headline number is naturally
long-horizon-weighted.

A method that trains *faster* (packing, flash-attn, shorter ctx) legitimately
wins by fitting more optimizer steps into the same wall-clock budget -- this is
the karpathy/autoresearch framing: fixed compute, best metric.

Config is a JSON dict; see autoresearch/configs/baseline.json for all knobs.
Everything is seeded so two identical configs give the same number.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


# ----------------------------- config ---------------------------------------

DEFAULTS: dict = {
    # data (relative to --data-root)
    "train_file": "data/agentic_experiments/pilot_4090_longhorizon_v1/train.sft.jsonl",
    "val_file": "data/agentic_experiments/pilot_4090_longhorizon_v1/val.sft.jsonl",
    # sequence / throughput
    "max_length": 8192,
    "packing": False,
    # lora
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # optimisation
    "lr": 1.0e-4,
    # constant_with_warmup is horizon-independent, so it stays correct under a
    # wall-clock stop. To test cosine/linear decay, set "planned_steps" too.
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 0,        # absolute warmup steps (horizon-independent)
    "warmup_ratio": 0.0,      # only used when planned_steps is set
    "planned_steps": None,    # scheduler horizon; None => wall-clock is the only stop
    "weight_decay": 0.0,
    "optim": "paged_adamw_8bit",
    "batch_size": 1,
    "grad_accum": 8,
    "max_grad_norm": 1.0,
    "neftune_noise_alpha": None,
    "attn_implementation": "sdpa",  # or "flash_attention_2"
    # budget
    "train_seconds": 1300,   # wall-clock training cap (~21.7 min)
    "max_steps_cap": 100000, # safety cap when no planned_steps
    "seed": 42,
}


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(json.loads(Path(path).read_text()))
    return cfg


# ------------------------- wall-clock stopper --------------------------------

class WallClockStopper(TrainerCallback):
    """Stop training once `limit_seconds` of wall-clock have elapsed. This is the
    fixed-compute budget: every experiment trains for the same number of seconds,
    so faster methods simply complete more steps."""

    def __init__(self, limit_seconds: float):
        self.limit_seconds = limit_seconds
        self.t0: float | None = None

    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.monotonic()

    def on_step_end(self, args, state, control, **kw):
        if self.t0 is not None and (time.monotonic() - self.t0) >= self.limit_seconds:
            control.should_training_stop = True
        return control


# ------------------------------ eval -----------------------------------------

@torch.no_grad()
def score_heldout(model, tokenizer, val_file: Path, max_length: int) -> dict:
    """Token-weighted next-token loss / ppl / token-acc, overall + per bucket."""
    eos = tokenizer.eos_token_id
    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device

    def new_acc():
        return {"rows": 0, "nll": 0.0, "correct": 0, "n": 0}

    overall = new_acc()
    by_bucket: dict[str, dict] = {}

    with open(val_file) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    for row in rows:
        text = row.get("text", "")
        bucket = row.get("length_bucket", "unknown")
        ids = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
        if eos is not None and (not ids or ids[-1] != eos):
            ids.append(eos)
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids], device=device)
        out = model(input_ids=input_ids, labels=input_ids)
        n = input_ids.shape[1] - 1
        nll_sum = float(out.loss.item()) * n
        pred = out.logits[:, :-1, :].argmax(dim=-1)
        correct = int((pred == input_ids[:, 1:]).sum().item())
        for acc in (overall, by_bucket.setdefault(bucket, new_acc())):
            acc["rows"] += 1
            acc["nll"] += nll_sum
            acc["correct"] += correct
            acc["n"] += n
        del out, pred, input_ids

    def agg(b):
        if b["n"] == 0:
            return {"rows": b["rows"], "tokens": 0, "loss": None, "ppl": None, "token_acc": None}
        loss = b["nll"] / b["n"]
        return {
            "rows": b["rows"],
            "tokens": b["n"],
            "loss": round(loss, 5),
            "ppl": round(math.exp(min(loss, 20.0)), 3),
            "token_acc": round(b["correct"] / b["n"], 5),
        }

    return {"overall": agg(overall), "by_bucket": {k: agg(v) for k, v in sorted(by_bucket.items())}}


# ------------------------------ main -----------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="path to experiment config json")
    p.add_argument("--exp-id", default="manual")
    p.add_argument("--data-root", default=".")
    p.add_argument("--model", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--out-dir", required=True, help="where to write result.json (+ adapter)")
    p.add_argument("--save-adapter", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)

    set_seed(int(cfg["seed"]))
    model_path = Path(args.model)
    train_file = root / cfg["train_file"]
    val_file = root / cfg["val_file"]
    for pth in (model_path, train_file, val_file):
        if not pth.exists():
            raise FileNotFoundError(pth)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=cfg["attn_implementation"],
    )
    model.config.use_cache = False

    dataset = load_dataset("json", data_files={"train": str(train_file), "validation": str(val_file)})

    peft_config = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg["target_modules"]),
    )

    # Scheduler horizon: a planned_steps shapes cosine/linear; otherwise the
    # wall-clock callback is the only stop and we use a high safety cap.
    planned = cfg.get("planned_steps")
    max_steps = int(planned) if planned else int(cfg["max_steps_cap"])
    warmup_steps = int(cfg.get("warmup_steps", 0) or 0)
    warmup_ratio = 0.0 if warmup_steps else float(cfg.get("warmup_ratio", 0.0) or 0.0)

    sft_args = SFTConfig(
        output_dir=str(out_dir / "trainer"),
        dataset_text_field="text",
        max_length=int(cfg["max_length"]),
        packing=bool(cfg["packing"]),
        num_train_epochs=1000,                 # never the binding constraint; wall-clock is
        max_steps=max_steps,
        per_device_train_batch_size=int(cfg["batch_size"]),
        gradient_accumulation_steps=int(cfg["grad_accum"]),
        learning_rate=float(cfg["lr"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=float(cfg["weight_decay"]),
        max_grad_norm=float(cfg["max_grad_norm"]),
        optim=cfg["optim"],
        neftune_noise_alpha=cfg["neftune_noise_alpha"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=2,
        save_strategy="no",
        report_to=[],
        seed=int(cfg["seed"]),
        data_seed=int(cfg["seed"]),
        dataset_num_proc=4,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[WallClockStopper(float(cfg["train_seconds"]))],
    )

    t_train0 = time.time()
    train_out = trainer.train()
    train_seconds = round(time.time() - t_train0, 1)
    steps = int(trainer.state.global_step)
    train_loss = float(train_out.training_loss) if train_out and train_out.training_loss is not None else None

    if args.save_adapter:
        trainer.save_model(str(out_dir / "adapter"))
        tokenizer.save_pretrained(str(out_dir / "adapter"))

    t_eval0 = time.time()
    eval_res = score_heldout(trainer.model, tokenizer, val_file, int(cfg["max_length"]))
    eval_seconds = round(time.time() - t_eval0, 1)

    result = {
        "exp_id": args.exp_id,
        "config_path": args.config,
        "config": cfg,
        "primary_metric": eval_res["overall"]["loss"],   # lower is better
        "val_overall": eval_res["overall"],
        "val_by_bucket": eval_res["by_bucket"],
        "train_loss": train_loss,
        "steps": steps,
        "tokens_seen_approx": steps * int(cfg["batch_size"]) * int(cfg["grad_accum"]) * int(cfg["max_length"]),
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: result[k] for k in ("exp_id", "primary_metric", "steps", "train_seconds", "eval_seconds")}, ensure_ascii=False))
    print(f"wrote {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
