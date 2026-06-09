"""The autoresearch driver.

Runs experiments from a queue DIRECTORY (autoresearch/queue/NNNN-slug.json),
one at a time, under a fixed wall-clock budget each. For every experiment it:

  1. builds the effective config = running-best config + this spec's overrides
     (so accepted improvements stack, karpathy-style),
  2. trains + evals via run_experiment.py,
  3. keeps it iff held-out val loss beats the running best (lower is better),
  4. appends to experiments/ledger.jsonl, writes a human line to live.log,
  5. updates method.md to the current best, regenerates progress.png.

The queue is a directory the *researcher* (a human or Claude) drops new numbered
spec files into between iterations -- the loop never rewrites it, so there is no
race. "Done" is determined from the ledger, so the loop is fully resumable: kill
it and restart, it picks up where it left off.

Run it detached so you can `tail -f autoresearch/live.log`:
  nohup python autoresearch/loop.py --data-root <root> --model <model> \
        --max-experiments 40 > autoresearch/loop.out 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def read_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def load_baseline() -> dict:
    return read_json(HERE / "configs" / "baseline.json", {})


def load_state() -> dict:
    return read_json(REPO / "loop_state.json", {
        "best_metric": None, "best_config": None, "best_exp_id": None, "history": [],
    })


def save_state(state: dict) -> None:
    (REPO / "loop_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2))


def done_exp_ids(ledger: Path) -> set[str]:
    if not ledger.exists():
        return set()
    out = set()
    for line in ledger.read_text().splitlines():
        line = line.strip()
        if line:
            out.add(json.loads(line)["exp_id"])
    return out


def log_live(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(HERE / "live.log", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def render_method_md(state: dict) -> None:
    cfg = state.get("best_config") or {}
    lines = [
        "# Current best method",
        "",
        f"- best experiment: **{state.get('best_exp_id')}**",
        f"- held-out val loss (lower=better): **{state.get('best_metric')}**",
        "",
        "## Config",
        "```json",
        json.dumps(cfg, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Accepted improvements (in order)",
        "",
    ]
    for h in state.get("history", []):
        if h.get("kept"):
            lines.append(f"- `{h['exp_id']}` {h['slug']}: {h.get('hypothesis','')} -> loss {h['metric']}")
    (REPO / "method.md").write_text("\n".join(lines) + "\n")


def regen_plot() -> None:
    try:
        subprocess.run([sys.executable, str(HERE / "plot_progress.py")], check=False, timeout=120)
    except Exception as e:  # noqa: BLE001
        log_live(f"WARN plot failed: {e}")


def run_one(spec_path: Path, exp_id: str, state: dict, args) -> None:
    spec = json.loads(spec_path.read_text())
    slug = spec.get("slug", spec_path.stem)
    hypothesis = spec.get("hypothesis", "")
    base = spec.get("base", "best")
    overrides = spec.get("overrides", {})

    if base == "baseline" or state.get("best_config") is None:
        base_cfg = load_baseline()
    else:
        base_cfg = dict(state["best_config"])
    cfg = dict(base_cfg)
    cfg.update(overrides)

    exp_dir = REPO / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = exp_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    is_baseline = state.get("best_metric") is None
    log_live(f"EXP {exp_id} START  slug={slug}  base={base}  hyp={hypothesis}")
    log_live(f"EXP {exp_id} overrides={json.dumps(overrides, ensure_ascii=False)}")

    metric = None
    steps = None
    train_seconds = None
    by_bucket = None
    status = "ok"
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(HERE / "run_experiment.py"),
             "--config", str(cfg_path),
             "--exp-id", exp_id,
             "--data-root", args.data_root,
             "--model", args.model,
             "--out-dir", str(exp_dir)],
            capture_output=True, text=True, timeout=args.per_exp_timeout,
        )
        (exp_dir / "stdout.log").write_text(proc.stdout)
        (exp_dir / "stderr.log").write_text(proc.stderr)
        result = read_json(exp_dir / "result.json", None)
        if proc.returncode != 0 or result is None:
            status = "failed"
            log_live(f"EXP {exp_id} FAILED rc={proc.returncode}; tail: {proc.stderr.strip()[-400:]}")
        else:
            metric = result["primary_metric"]
            steps = result["steps"]
            train_seconds = result["train_seconds"]
            by_bucket = result.get("val_by_bucket")
    except subprocess.TimeoutExpired:
        status = "timeout"
        log_live(f"EXP {exp_id} TIMEOUT after {args.per_exp_timeout}s")
    except Exception as e:  # noqa: BLE001
        status = "error"
        log_live(f"EXP {exp_id} ERROR {e}")

    wall = round(time.time() - t0, 1)

    kept = False
    if metric is not None:
        if is_baseline:
            kept = True
        elif state["best_metric"] is None or metric < state["best_metric"]:
            kept = True

    if kept:
        prev = state.get("best_metric")
        state["best_metric"] = metric
        state["best_config"] = cfg
        state["best_exp_id"] = exp_id
        delta = "" if prev is None else f" (was {prev}, -{round(prev - metric, 5)})"
        log_live(f"EXP {exp_id} KEPT  val_loss={metric}{delta}  steps={steps}  train_s={train_seconds}")
    else:
        rb = state.get("best_metric")
        log_live(f"EXP {exp_id} DISCARD  val_loss={metric}  running_best={rb}  steps={steps}  train_s={train_seconds}")

    rec = {
        "exp_id": exp_id,
        "slug": slug,
        "hypothesis": hypothesis,
        "base": base,
        "overrides": overrides,
        "status": status,
        "metric": metric,
        "kept": kept,
        "best_so_far": state.get("best_metric"),
        "steps": steps,
        "train_seconds": train_seconds,
        "wall_seconds": wall,
        "val_by_bucket": by_bucket,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(REPO / "experiments" / "ledger.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    state.setdefault("history", []).append({
        "exp_id": exp_id, "slug": slug, "hypothesis": hypothesis,
        "metric": metric, "kept": kept,
    })
    save_state(state)
    render_method_md(state)
    regen_plot()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--model", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--max-experiments", type=int, default=40)
    p.add_argument("--per-exp-timeout", type=int, default=2400, help="hard kill a single exp after N s")
    p.add_argument("--poll-seconds", type=int, default=30, help="when queue empty, wait then recheck")
    p.add_argument("--idle-exit-minutes", type=int, default=180, help="exit if queue stays empty this long")
    args = p.parse_args()

    queue_dir = HERE / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    ledger = REPO / "experiments" / "ledger.jsonl"

    log_live(f"LOOP START max_experiments={args.max_experiments} data_root={args.data_root} model={args.model}")
    idle_since = None

    while True:
        state = load_state()
        done = done_exp_ids(ledger)
        if len(done) >= args.max_experiments:
            log_live(f"LOOP DONE reached max_experiments={args.max_experiments}")
            break

        specs = sorted(p for p in queue_dir.glob("*.json") if not p.name.startswith("."))
        next_spec = None
        for sp in specs:
            exp_id = sp.stem.split("-")[0]
            if exp_id not in done:
                next_spec = (sp, exp_id)
                break

        if next_spec is None:
            if idle_since is None:
                idle_since = time.time()
                log_live(f"QUEUE EMPTY ({len(done)} done). Waiting for new specs...")
            if (time.time() - idle_since) > args.idle_exit_minutes * 60:
                log_live(f"LOOP EXIT idle > {args.idle_exit_minutes} min")
                break
            time.sleep(args.poll_seconds)
            continue

        idle_since = None
        sp, exp_id = next_spec
        run_one(sp, exp_id, state, args)

    log_live("LOOP STOPPED")


if __name__ == "__main__":
    main()
