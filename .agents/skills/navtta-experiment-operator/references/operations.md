# Safe experiment operations

## 1. Establish the live contract

Resolve placeholders before running commands; do not paste literal examples into a shell.

1. Locate the top-level repository with `git rev-parse --show-toplevel`.
2. Read `AGENTS.md`, the selected task README, the campaign plan, the exact JSON/YAML spec, and the runner source or `--help`.
3. Record the spec path and `sha256sum SPEC`. Extract limits from the current spec and launcher, including `concurrency`, `max_workers*`, `jobs_per_gpu`, aggregate GPU/CPU-memory ceilings, launch stagger, phases/barriers, episode count, and retry policy.
4. Compare the runner's dry-run job matrix with the spec. Treat a disagreement as a blocker, not as permission to choose a convenient value.

Do not infer caps from currently idle hardware. Hardware availability can lower a reviewed cap but cannot raise it.

## 2. Read-only preflight

Run locally and on the selected server, using the intended repository path:

```bash
git rev-parse HEAD
git status --short --untracked-files=no
git diff --quiet
git diff --cached --quiet
df -h .
nvidia-smi
```

Also verify:

- The requested commit exists on the server and is the checkout used by the process.
- Formal-run tracked-worktree requirements pass before and after execution. Do not hide changes with stash/reset.
- The correct task-specific environment is active and `core` is installed editable in that environment when required.
- Dataset, checkpoint, environment, and episode-order manifests exist and their recorded digests match.
- The batch ID is new, or the operation is an authorized resume of exactly that batch.
- Scheduler locks and PID files agree with live process identity. A stale-looking file is not proof that its owner is dead.
- GPU memory, utilization, compute PIDs, host RAM/cgroup memory, and disk headroom fit the spec's limits.

Use the bundled status reader on the server when possible:

```bash
python3 .agents/skills/navtta-experiment-operator/scripts/read_status.py \
  --repo . --batch-root TASK/results/logs/CAMPAIGN/BATCH \
  --spec TASK/experiments/SPEC.json --batch-id BATCH
```

It reads Git, plan/summary/metrics/exit/validation markers, process metadata, and `nvidia-smi`; it writes nothing and never emits full process arguments. Supply `--eta-workers N` only when `N` is the actual scheduler worker count. ETA is a projection, not completion evidence.

## 3. Launch and monitor

Prefer the repository launcher and its documented durable supervisor. Before a state-changing command, show or retain:

- host, repository, environment, commit, spec path/digest, runner, and exact arguments;
- batch ID, seed, phases, expected jobs, GPU list, and effective caps;
- dry-run outcome and evidence locations.

During monitoring, keep these signals separate:

- **Process:** scheduler/worker PID and elapsed time.
- **Resource:** per-GPU utilization/memory and host/cgroup pressure.
- **Progress:** expected, completed, failed, metrics, and validated job counts.
- **ETA:** median completed-job duration projected through a known worker count; report `unknown` before enough comparable jobs finish.
- **Scientific evidence:** manifests and compact artifacts, audited separately.

A quiet GPU can mean a CPU-heavy phase, a blocked process, or completion. Corroborate it with process state and fresh progress markers.

## 4. Resume safely

1. Re-run read-only status and identify the scheduler plus every worker belonging to the batch.
2. If any owner is live, reconnect to or monitor it; do not start a second scheduler.
3. If interrupted, confirm the original processes are gone and inspect the runner's documented lock semantics.
4. Compare the stored batch/spec/plan identity against the requested commit, spec digest, seed, GPU/cap arguments, and other immutable arguments.
5. Invoke the same runner with the same batch ID and `--resume`. Do not manually mark jobs complete.
6. Use `--retry-failed` or an equivalent only when documented. Preserve the failed attempt and require a new attempt/run identity where the launcher does so.

Remove a stale lock only after all owners are proven dead and the task documentation explicitly permits removal. Never delete result directories to make resume pass.

## 5. Retrieve evidence

First inventory remote files and perform an `rsync --dry-run`. Then transfer only reviewed compact evidence, without `--delete`:

- campaign spec/digest, batch metadata, plan/grid, scheduler summary;
- per-job parameters/config, exit and validation markers;
- compact `metrics.csv`/`metrics.json`, summaries, diagnostics, stream fingerprints;
- canonical `manifest.json` files referenced by jobs or registries.

Do not retrieve raw logs by default: inspect short tails remotely when diagnosing, then explicitly authorize any needed excerpt. Never retrieve model checkpoints, datasets, simulator assets, audio/RIR files, videos, TensorBoard directories, caches, SSH material, shell history, or environment dumps.

Preserve task-relative canonical paths. Compare remote and local `sha256sum` values for transferred manifests and compact artifacts, and record missing or mismatched items. A `manifest.path` pointer alone is not a downloaded manifest.

## 6. Handoff record

Report:

- observed time/host, commit, tracked-worktree state, spec digest, batch ID, command, and cap source;
- scheduler/worker identities, GPU state, completed/failed/validated/expected counts, and ETA assumptions;
- whether the action was monitor, launch, resume, retry, or retrieval;
- evidence destination and digest/missing-file results;
- blockers and the safest next action.

Do not describe process survival or a zero exit code as a validated or formal result.
