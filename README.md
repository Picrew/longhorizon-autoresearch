# longhorizon-autoresearch

Autonomous, karpathy/autoresearch-style self-improving training loop for
**agentic long-horizon SFT**, on a single RTX 4090.

The idea (after [@karpathy](https://github.com/karpathy/autoresearch)): give an
agent a small but real LLM-training setup and let it experiment on its own. It
proposes a method change, trains for a fixed budget, checks if a held-out metric
improved, keeps or discards, and repeats. You come back to a log of experiments
and (hopefully) a better recipe.

![progress](progress.png)

## This setup

- **Base model:** `Qwen3-4B-Instruct-2507`, 4-bit QLoRA.
- **Task:** SFT on a quality-filtered, long-horizon-weighted slice of multi-step
  agent×environment trajectories (`pilot_4090_longhorizon_v1`: 2730 train / 303 val).
- **Metric (lower is better):** held-out **token-weighted causal-LM loss** on the
  val split — the same objective SFT minimizes, on data never trained on. Because
  loss is token-weighted and long traces hold ~64% of all val tokens, the headline
  number is naturally **long-horizon-dominated**. Reported with ppl, token-acc,
  and a per-length-bucket breakdown.
- **Compute budget:** a **fixed wall-clock** of ~22 min training + ~5 min eval per
  experiment (≤ 30 min). Fixed wall-clock (not fixed steps) is the karpathy
  framing: a method that improves throughput legitimately wins by fitting more
  optimizer steps into the same budget.
- **Search space:** LR & schedule, warmup, LoRA rank/alpha/dropout/targets,
  sequence packing, max_length, weight decay, NEFTune, effective batch, optimizer,
  data mix/quality. (SFT/QLoRA — not RL; rollout-based RL can't fit a stable
  30-min loop on one 4090.)

## How it runs

`autoresearch/loop.py` pops experiment specs from `autoresearch/queue/` (a
race-free directory the researcher drops numbered `NNNN-slug.json` files into),
builds each effective config as **running-best + this spec's overrides** (so
accepted improvements stack), trains+evals via `autoresearch/run_experiment.py`,
keeps it iff it beats the running best, and records to `experiments/ledger.jsonl`,
`autoresearch/live.log`, `method.md`, and `progress.png`. State lives in files,
so the loop is fully resumable.

```bash
# on the GPU box (data + model live next to this repo)
nohup python autoresearch/loop.py \
    --data-root /home/ljj/ssd1/ljj/research/llm-longhorizon-traces \
    --model    /home/ljj/ssd1/ljj/research/llm-longhorizon-traces/models/Qwen3-4B-Instruct-2507 \
    --max-experiments 40 > autoresearch/loop.out 2>&1 &

tail -f autoresearch/live.log     # watch progress
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for the running log and [method.md](method.md)
for the current best recipe.
