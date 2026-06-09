"""Run ONE autoresearch experiment: QLoRA fine-tune under a fixed wall-clock
budget, then score a FIXED held-out metric. Emits a single result.json.

Why this exists (read this before changing the metric):
  The capability we care about is **an agent choosing good next actions over a
  long trajectory** -- not reproducing environment/tool-output tokens (API JSON,
  file dumps, search results), which are partly unlearnable noise. So the metric
  is the held-out causal-LM loss computed ONLY on the agent's own decision tokens
  (the `<|assistant|>` spans, which include the natural-language reasoning and the
  `<tool_call>` it emits). Full-sequence loss is reported too, as a secondary.

  The metric is a FIXED yardstick: always eval at eval_max_length with standard
  head-truncation, regardless of how the experiment trained. That decouples the
  metric from the method, so changing training max_length / truncation / masking
  never moves the goalposts -- a fair autoresearch comparison.

Budget: fixed wall-clock training (train_seconds). A faster method legitimately
wins by fitting more optimizer steps into the same budget (the karpathy framing).

Config is a JSON dict; see autoresearch/configs/baseline.json. The interesting
knobs are *method* knobs (loss_on, truncation, ...), documented in METHODS.md.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)


# The eval metric is FIXED. Do not make these per-experiment knobs.
EVAL_MAX_LENGTH = 8192
ROLE_RE = re.compile(r"<\|(system|user|assistant|tool)\|>")
DECISION_ROLES = {"assistant"}  # tokens the agent itself produces


DEFAULTS: dict = {
    # data (relative to --data-root)
    "train_file": "data/agentic_experiments/pilot_4090_longhorizon_v1/train.sft.jsonl",
    "val_file": "data/agentic_experiments/pilot_4090_longhorizon_v1/val.sft.jsonl",
    # --- method knobs (the point of this project; see METHODS.md) ---
    "loss_on": "all",          # "all" | "assistant"  (mask non-agent tokens)
    "truncation": "right",     # "right" (keep head) | "tail" (keep end) | "head_tail"
    "max_length": 8192,        # TRAIN sequence cap (a method knob; eval is fixed)
    # --- lora ---
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # --- optimisation ---
    "lr": 1.0e-4,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 0,
    "warmup_ratio": 0.0,
    "planned_steps": None,
    "weight_decay": 0.0,
    "optim": "paged_adamw_8bit",
    "batch_size": 1,
    "grad_accum": 8,
    "max_grad_norm": 1.0,
    "neftune_noise_alpha": None,
    "attn_implementation": "sdpa",
    # --- budget ---
    "train_seconds": 1200,
    "max_steps_cap": 100000,
    "seed": 42,
    "eval_limit": 0,           # 0 = full val set (smoke only)
}


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(json.loads(Path(path).read_text()))
    return cfg


# ------------------------ role-aware tokenisation ----------------------------

def role_at(starts: list[int], roles: list[str], char_pos: int) -> str | None:
    idx = bisect.bisect_right(starts, char_pos) - 1
    return roles[idx] if idx >= 0 else None


def label_token_ids(text: str, tokenizer, *, loss_on: str):
    """Tokenise `text` (no truncation) and return (ids, labels, roles_per_tok).

    labels[i] == ids[i] if token i is supervised under `loss_on`, else -100.
    roles_per_tok is kept so callers can apply a truncation strategy and still
    know which surviving tokens are decision tokens (for the metric)."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    markers = [(m.start(), m.group(1)) for m in ROLE_RE.finditer(text)]
    starts = [s for s, _ in markers]
    roles = [r for _, r in markers]
    keep = DECISION_ROLES if loss_on == "assistant" else {"system", "user", "assistant", "tool"}
    roles_per_tok = [role_at(starts, roles, a) for (a, _b) in offsets]
    labels = [tid if (r in keep) else -100 for tid, r in zip(ids, roles_per_tok)]
    return ids, labels, roles_per_tok


def apply_truncation(ids, labels, max_length: int, strategy: str, eos_id):
    """Truncate to <= max_length under the chosen strategy, then ensure EOS."""
    if len(ids) > max_length:
        if strategy == "tail":
            ids, labels = ids[-max_length:], labels[-max_length:]
        elif strategy == "head_tail":
            h = max_length // 2
            t = max_length - h
            ids = ids[:h] + ids[-t:]
            labels = labels[:h] + labels[-t:]
        else:  # "right": keep the head (standard HF behaviour)
            ids, labels = ids[:max_length], labels[:max_length]
    if eos_id is not None and (not ids or ids[-1] != eos_id):
        ids = ids + [eos_id]
        labels = labels + [eos_id]  # supervise the stop token
    return ids, labels


# ------------------------- wall-clock stopper --------------------------------

class WallClockStopper(TrainerCallback):
    """Fixed-compute budget: stop after limit_seconds of wall-clock training."""

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
def score_heldout(model, tokenizer, val_file: Path, eval_limit: int = 0) -> dict:
    """FIXED metric. Per-token nll computed once per row, aggregated two ways:
      - decision: loss over <|assistant|> tokens only  (PRIMARY, lower better)
      - full:     loss over all tokens                 (secondary)
    plus a per-length-bucket breakdown of the decision loss."""
    eos = tokenizer.eos_token_id
    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device

    def acc():
        return {"rows": 0, "dec_nll": 0.0, "dec_n": 0, "dec_correct": 0,
                "full_nll": 0.0, "full_n": 0, "full_correct": 0}

    overall = acc()
    by_bucket: dict[str, dict] = {}

    with open(val_file) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if eval_limit and eval_limit > 0:
        rows = rows[:eval_limit]

    for row in rows:
        text = row.get("text", "")
        bucket = row.get("length_bucket", "unknown")
        # FIXED: head-truncate at EVAL_MAX_LENGTH, decision mask from roles.
        ids, _lab, roles = label_token_ids(text, tokenizer, loss_on="all")
        ids, roles = ids[:EVAL_MAX_LENGTH], roles[:EVAL_MAX_LENGTH]
        if eos is not None and (not ids or ids[-1] != eos):
            ids = ids + [eos]
            roles = roles + ["assistant"]
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids], device=device)
        out = model(input_ids=input_ids)
        logits = out.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        pred = logits.argmax(dim=-1).reshape(-1)
        tgt = targets.reshape(-1)
        correct = (pred == tgt)
        # decision mask aligns with the predicted token i+1 -> role of token i+1
        dec_mask = torch.tensor([r in DECISION_ROLES for r in roles[1:]], device=device)

        a = by_bucket.setdefault(bucket, acc())
        for acc_ in (overall, a):
            acc_["rows"] += 1
            acc_["full_nll"] += float(nll.sum().item())
            acc_["full_n"] += nll.numel()
            acc_["full_correct"] += int(correct.sum().item())
            acc_["dec_nll"] += float(nll[dec_mask].sum().item())
            acc_["dec_n"] += int(dec_mask.sum().item())
            acc_["dec_correct"] += int(correct[dec_mask].sum().item())
        del out, logits, nll, input_ids

    def agg(b):
        out = {"rows": b["rows"], "dec_tokens": b["dec_n"], "full_tokens": b["full_n"]}
        out["decision_loss"] = round(b["dec_nll"] / b["dec_n"], 5) if b["dec_n"] else None
        out["decision_ppl"] = round(math.exp(min(out["decision_loss"], 20.0)), 3) if out["decision_loss"] else None
        out["decision_acc"] = round(b["dec_correct"] / b["dec_n"], 5) if b["dec_n"] else None
        out["full_loss"] = round(b["full_nll"] / b["full_n"], 5) if b["full_n"] else None
        return out

    return {"overall": agg(overall), "by_bucket": {k: agg(v) for k, v in sorted(by_bucket.items())}}


# ------------------------------ main -----------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--exp-id", default="manual")
    p.add_argument("--data-root", default=".")
    p.add_argument("--model", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--out-dir", required=True)
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
    eos_id = tokenizer.eos_token_id

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, quantization_config=quant,
        dtype=torch.bfloat16, device_map="auto", attn_implementation=cfg["attn_implementation"],
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]), lora_dropout=float(cfg["lora_dropout"]),
        bias="none", task_type="CAUSAL_LM", target_modules=list(cfg["target_modules"]),
    ))

    # ---- build train dataset: role-aware labels + truncation strategy ----
    loss_on = cfg["loss_on"]
    truncation = cfg["truncation"]
    max_length = int(cfg["max_length"])

    def encode(row):
        ids, labels, _roles = label_token_ids(row["text"], tokenizer, loss_on=loss_on)
        ids, labels = apply_truncation(ids, labels, max_length, truncation, eos_id)
        # guard: a sequence with no supervised tokens would give nan loss
        if all(x == -100 for x in labels):
            labels = list(ids)
        return {"input_ids": ids, "labels": labels}

    raw = load_dataset("json", data_files={"train": str(train_file)})["train"]
    train_ds = raw.map(encode, remove_columns=raw.column_names, num_proc=4, desc="encode")

    collator = DataCollatorForSeq2Seq(tokenizer, padding="longest", label_pad_token_id=-100)

    planned = cfg.get("planned_steps")
    max_steps = int(planned) if planned else int(cfg["max_steps_cap"])
    warmup_steps = int(cfg.get("warmup_steps", 0) or 0)
    warmup_ratio = 0.0 if warmup_steps else float(cfg.get("warmup_ratio", 0.0) or 0.0)

    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        max_steps=max_steps,
        num_train_epochs=1000,
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
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=collator, callbacks=[WallClockStopper(float(cfg["train_seconds"]))],
    )

    t0 = time.time()
    train_out = trainer.train()
    train_seconds = round(time.time() - t0, 1)
    steps = int(trainer.state.global_step)
    train_loss = float(train_out.training_loss) if train_out and train_out.training_loss is not None else None

    if args.save_adapter:
        trainer.save_model(str(out_dir / "adapter"))
        tokenizer.save_pretrained(str(out_dir / "adapter"))

    t1 = time.time()
    ev = score_heldout(trainer.model, tokenizer, val_file, int(cfg.get("eval_limit", 0) or 0))
    eval_seconds = round(time.time() - t1, 1)

    result = {
        "exp_id": args.exp_id,
        "config": cfg,
        "metric_kind": "decision_loss",
        "primary_metric": ev["overall"]["decision_loss"],   # lower is better
        "val_overall": ev["overall"],
        "val_by_bucket": ev["by_bucket"],
        "train_loss": train_loss,
        "steps": steps,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: result[k] for k in ("exp_id", "primary_metric", "steps", "train_seconds", "eval_seconds")}, ensure_ascii=False))
    print(f"wrote {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
