# Operations runbook (for the agent driving this loop)

This loop spans many hours and many context windows. **All state is on disk** —
resume by reading files, never from memory.

> **Connection details are NOT in this repo.** The GPU box host / user / port and the
> server paths are kept in an untracked local file `connection.local.sh` (gitignored).
> Source it before any remote command:
> ```bash
> source connection.local.sh   # sets: SSH_HOST, SSH_USER, SSH_PORT, SRV, RB
> SSHX="ssh -o ConnectTimeout=25 -p $SSH_PORT $SSH_USER@$SSH_HOST"
> ```
> Placeholders below (`<HOST>`, `<USER>`, `<PORT>`, `<SERVER_ROOT>`) are filled by those vars.

## Topology
- **Local (this machine):** the git repo + GitHub access. Path: `RB` (this repo).
  GitHub remote: `https://github.com/Picrew/longhorizon-autoresearch.git`.
- **GPU box:** `ssh -p <PORT> <USER>@<HOST>`. Repo mirror + data + model live at
  `<SERVER_ROOT>` (harness rsync'd into `<SERVER_ROOT>/autoresearch_repo/`). venv:
  `<SERVER_ROOT>/.venv4090`. **The box has no direct GitHub access** → all git is local;
  rsync bridges code (local→box) and results (box→local).

## ⚠ local ephemeral-port exhaustion ("Can't assign requested address")
Too many rapid ssh/rsync retries exhaust LOCAL ephemeral ports → connect fails with
"Can't assign requested address" (a LOCAL error; the box is fine, the loop keeps running).
Fix: back off ~5–10 min; do ONE batched ssh per check (not retry-loops of many).
(Also seen: a stale static host-route to `<HOST>` misrouting traffic — `sudo route -n delete <HOST>` or toggle VPN.)

## ⚠ rsync pulls are flaky while the box is training
Under load, `rsync` pulls intermittently fail with "receiver has empty file list". Pull
**text** via `$SSHX "cat <file>" > /tmp/x` (verify non-empty + `wc -l` vs the box before
`mv`), **binaries** via `scp -P <PORT> <USER>@<HOST>:...`. `git checkout -- <f>` restores a clobber.

## Decision protocol (the research — every check-in)
Adaptive: never pre-bake a long queue. Read the evidence (`experiments/ledger.jsonl`,
per-`result.json` `val_by_bucket`, `method.md`, `RESEARCH_LOG.md`), form ONE hypothesis
(prefer a method change from `METHODS.md`), 2-seed-confirm any new best (noise ~0.004),
write the reasoning into `RESEARCH_LOG.md`, queue 1–3 specs.

## Add experiments
Drop `autoresearch/queue/NNNN-slug.json` (`{slug,hypothesis,base,overrides[,reanchor]}`),
then `rsync -az -e "$SSHX" autoresearch/queue/ $SSH_USER@$SSH_HOST:$SRV/autoresearch_repo/autoresearch/queue/`.
A spec's `model` override may be **relative to the data-root** (e.g. `models/Qwen3-8B`) —
keep absolute machine paths out of committed configs.

## Restart the loop (if it idle-exited / died)
```bash
$SSHX "cd $SRV/autoresearch_repo && . $SRV/.venv4090/bin/activate && \
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  setsid nohup python autoresearch/loop.py --data-root $SRV \
    --model $SRV/models/Qwen3-4B-Instruct-2507 --max-experiments 84 \
    --keep-margin 0.004 --per-exp-timeout 20000 > autoresearch/loop.out 2>&1 < /dev/null &"
```
It resumes from the ledger. Launch with ONE simple command (nested `bash -c` wrappers don't detach reliably).

## Publish (each check-in)
Pull `experiments/ledger.jsonl` (verify == box `wc -l`), regenerate `progress.png` locally
(`python3 autoresearch/plot_progress.py`), render `EXPERIMENTS.md`
(`python3 autoresearch/render_experiments.py`), update `RESEARCH_LOG.md`, commit + push;
create `exp/NNNN-slug` branches for kept experiments.
