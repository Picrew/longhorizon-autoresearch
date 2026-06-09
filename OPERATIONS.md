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

## Add new experiments (steering)
Drop higher-numbered specs into the queue dir, then sync to the box:
```bash
# write $RB/autoresearch/queue/00NN-slug.json  {slug,hypothesis,base,overrides}
rsync -az -e 'ssh -p 6000' --exclude '._*' $RB/autoresearch/queue/ \
  ljj@124.220.35.225:$SRV/autoresearch/queue/
```
The loop reads the queue dir fresh every iteration and skips exp_ids already in
the ledger — no restart needed, no race.

## Restart the loop if it died
```bash
ssh -p 6000 ljj@124.220.35.225 'cd /home/ljj/ssd1/ljj/research/llm-longhorizon-traces/autoresearch_repo && . /home/ljj/ssd1/ljj/research/llm-longhorizon-traces/.venv4090/bin/activate && nohup python autoresearch/loop.py --data-root /home/ljj/ssd1/ljj/research/llm-longhorizon-traces --model /home/ljj/ssd1/ljj/research/llm-longhorizon-traces/models/Qwen3-4B-Instruct-2507 --max-experiments 40 > autoresearch/loop.out 2>&1 &'
```
It resumes from the ledger automatically.

## Cadence
~28 min/experiment. Wake every ~60 min, process whatever finished, steer the
queue when there is enough signal, push. Stop at 40 experiments → finalize plot +
README story.
