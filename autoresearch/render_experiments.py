"""Render EXPERIMENTS.md (the human table on main) from experiments/ledger.jsonl.
Run locally at each check-in after pulling results from the box."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "experiments" / "ledger.jsonl"
OUT = REPO / "EXPERIMENTS.md"

HEAD = """# Experiment log

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
"""


def main() -> None:
    rows = []
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    lines = [HEAD]
    for r in rows:
        kind = r.get("metric_kind") or "full_seq"
        m = r.get("metric")
        m = f"{m:.5f}" if isinstance(m, (int, float)) else (r.get("status") or "—")
        hyp = (r.get("hypothesis") or "").replace("|", "\\|")
        if len(hyp) > 90:
            hyp = hyp[:87] + "..."
        kept = "✅" if r.get("kept") else ""
        lines.append(
            f"| {r.get('exp_id')} | {r.get('slug','')} | {hyp} | {m} | {kind} | {kept} | "
            f"{r.get('steps') or ''} | {r.get('train_seconds') or ''} |"
        )
    dec_kinds = {"decision_loss", "decision_loss_v2"}
    n_kept = sum(1 for r in rows if r.get("kept") and r.get("metric_kind") in dec_kinds)
    n_v1 = sum(1 for r in rows if r.get("metric_kind") == "decision_loss")
    n_v2 = sum(1 for r in rows if r.get("metric_kind") == "decision_loss_v2")
    lines.append("")
    lines.append(f"_{len(rows)} experiments logged; {n_v1} on the v1 (head@8192) curve, "
                 f"{n_v2} on the v2 (long-horizon, to 16384) curve, {n_kept} kept improvements total._")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
