"""Render the karpathy/autoresearch-style progress chart from the ledger.

x = experiment #, y = held-out validation loss (lower is better).
Discarded experiments are grey dots, kept ones are green, and a green step line
traces the running best. Kept experiments get their hypothesis annotated.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "experiments" / "ledger.jsonl"
OUT = REPO / "progress.png"


def main() -> None:
    if not LEDGER.exists():
        print("no ledger yet")
        return
    recs = []
    for line in LEDGER.read_text().splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    # Plot the curve for the CURRENT metric version only (the latest metric_kind
    # present). Earlier-metric rows stay in the ledger but off this curve, so each
    # reanchor (metric change) gets its own clean descending curve.
    kinds = [r.get("metric_kind") for r in recs if r.get("metric") is not None and r.get("metric_kind")]
    if not kinds:
        print("no scored experiments yet")
        return
    current_kind = kinds[-1]
    recs = [r for r in recs if r.get("metric") is not None and r.get("metric_kind") == current_kind]
    if not recs:
        print("no scored experiments for current metric yet")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(recs)))
    ys = [r["metric"] for r in recs]
    kept = [r["kept"] for r in recs]

    running = []
    best = float("inf")
    for r in recs:
        if r["kept"]:
            best = min(best, r["metric"])
        running.append(best if best != float("inf") else r["metric"])

    n_kept = sum(1 for k in kept if k)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(f"Autoresearch Progress: {len(recs)} Experiments, {n_kept} Kept Improvements")

    disc_x = [x for x, k in zip(xs, kept) if not k]
    disc_y = [y for y, k in zip(ys, kept) if not k]
    keep_x = [x for x, k in zip(xs, kept) if k]
    keep_y = [y for y, k in zip(ys, kept) if k]

    ax.scatter(disc_x, disc_y, s=18, c="0.7", label="Discarded", zorder=2)
    ax.scatter(keep_x, keep_y, s=42, c="#16a34a", label="Kept", zorder=4)
    ax.step(xs, running, where="post", color="#16a34a", linewidth=1.5, label="Running best", zorder=3)

    for x, r in zip(xs, recs):
        if r["kept"]:
            label = r.get("slug", "")
            ax.annotate(label, (x, r["metric"]), textcoords="offset points",
                        xytext=(4, 6), rotation=30, fontsize=7, color="#15803d")

    ax.set_xlabel("Experiment # (current metric version)")
    ax.set_ylabel("Held-out decision-token loss (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
