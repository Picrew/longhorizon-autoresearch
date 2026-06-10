# Operations runbook (for the agent driving this loop)

This loop spans many hours and many context windows. **All state is on disk** —
resume by reading files, never from memory.

## Topology
- **Local (this machine):** the git repo, GitHub access as `Picrew`. Path:
  `/Volumes/mac/python_code/longhorizon-autoresearch`. GitHub remote:
  `https://github.com/Picrew/longhorizon-autoresearch.git`.
- **GPU box:** `ssh -p 6000 ljj@124.220.35.225`. Repo mirror + data + model live at
  `/home/ljj/ssd1/ljj/research/llm-longhorizon-traces` (the autoresearch harness is
  rsync'd into `.../llm-longhorizon-traces/autoresearch_repo/`). venv: `.venv4090`.
  **The box has no direct GitHub access** (proxy down) → all git happens locally;
  rsync bridges code (local→box) and results (box→local).

## Check status (each wake-up)
```bash
ssh -p 6000 ljj@124.220.35.225 'cd /home/ljj/ssd1/ljj/research/llm-longhorizon-traces/autoresearch_repo && tail -n 20 autoresearch/live.log; echo ---; ls experiments; echo ---; pgrep -af "autoresearch/loop.py" || echo "LOOP NOT RUNNING"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```

## ⚠ rsync pulls are flaky while the box is training
Under GPU+CPU load, `rsync` pulls intermittently fail with "receiver has empty
file list: exiting" (the push direction is usually fine). ssh commands stay
reliable, so pull **text** files via `ssh cat >` and **binaries** via `scp`:
```bash
ssh -p 6000 ljj@124.220.35.225 "cat $SRV/experiments/ledger.jsonl" > experiments/ledger.jsonl
ssh -p 6000 ljj@124.220.35.225 "cat $SRV/method.md" > method.md
scp -P 6000 ljj@124.220.35.225:$SRV/progress.png ./progress.png
```
ALWAYS verify `wc -l experiments/ledger.jsonl` matches the server before committing.
**Pull to a TEMP file first** (`ssh ... "cat $f" > /tmp/x && [ -s /tmp/x ] && mv /tmp/x dest`):
a bare `ssh "cat f" > dest` **truncates dest to empty if the ssh drops** (the box's ssh
is intermittently flaky). If you do clobber a tracked file, `git checkout -- <file>` restores it.

## Pull results back & publish (each wake-up)
```bash
RB=/Volumes/mac/python_code/longhorizon-autoresearch
SRV=/home/ljj/ssd1/ljj/research/llm-longhorizon-traces/autoresearch_repo
rsync -az -e 'ssh -p 6000' --exclude '._*' --exclude 'trainer' --exclude 'adapter' \
  ljj@124.220.35.225:$SRV/experiments/ $RB/experiments/
rsync -az -e 'ssh -p 6000' --exclude '._*' \
  ljj@124.220.35.225:$SRV/autoresearch/live.log $RB/autoresearch/live.log
rsync -az -e 'ssh -p 6000' --exclude '._*' ljj@124.220.35.225:$SRV/loop_state.json $RB/ 2>/dev/null
rsync -az -e 'ssh -p 6000' --exclude '._*' ljj@124.220.35.225:$SRV/method.md $RB/ 2>/dev/null
rsync -az -e 'ssh -p 6000' --exclude '._*' ljj@124.220.35.225:$SRV/progress.png $RB/ 2>/dev/null
# then: regenerate EXPERIMENTS.md rows from ledger, commit each new kept exp to a
# branch exp/NNNN-slug, update main (EXPERIMENTS.md + ledger + method.md + progress.png), push.
```

## Decision protocol (this is the research — do it every check-in)
The loop is **adaptive**: never pre-bake a long queue. At each check-in:
1. Read the evidence: `experiments/ledger.jsonl`, the per-bucket `val_by_bucket`
   in each `result.json`, `method.md` (current best), and `RESEARCH_LOG.md`.
2. Form ONE hypothesis from what the evidence says — **prefer a method change**
   from `METHODS.md` over a hyperparameter tweak. E.g. "decision-masking helped
   most on the long bucket → try late-token weighting next" or "long-bucket loss
   barely moved while medium improved → the long traces are being truncated; try
   `head_tail`." If a needed lever isn't implemented yet, **implement it in
   run_experiment.py now**, smoke-test it (tiny `train_seconds`+`eval_limit`),
   sync, then queue it.
3. Write the reasoning into `RESEARCH_LOG.md` (the *why*, not just the result).
4. Queue at most 1–2 specs ahead (so the GPU doesn't idle, but the search stays
   adaptive). One change per experiment for clean attribution.

## Add experiments to the queue
```bash
# write $RB/autoresearch/queue/00NN-slug.json  {slug,hypothesis,base,overrides[,reanchor]}
rsync -az -e 'ssh -p 6000' --exclude '._*' $RB/autoresearch/queue/ \
  ljj@124.220.35.225:$SRV/autoresearch/queue/
```
The loop reads the queue dir fresh every iteration and skips exp_ids already in
the ledger — no restart needed, no race. A spec with `"reanchor": true` resets the
running best (used once to start the decision-loss curve cleanly).

## Restart the loop if it died
```bash
ssh -p 6000 ljj@124.220.35.225 'cd /home/ljj/ssd1/ljj/research/llm-longhorizon-traces/autoresearch_repo && . /home/ljj/ssd1/ljj/research/llm-longhorizon-traces/.venv4090/bin/activate && nohup python autoresearch/loop.py --data-root /home/ljj/ssd1/ljj/research/llm-longhorizon-traces --model /home/ljj/ssd1/ljj/research/llm-longhorizon-traces/models/Qwen3-4B-Instruct-2507 --max-experiments 40 > autoresearch/loop.out 2>&1 &'
```
It resumes from the ledger automatically.

## Cadence
~28 min/experiment. Wake every ~60 min, process whatever finished, steer the
queue when there is enough signal, push. Stop at 40 experiments → finalize plot +
README story.
