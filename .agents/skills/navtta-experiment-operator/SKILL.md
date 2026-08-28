---
name: navtta-experiment-operator
description: Safely preflight, launch, monitor, resume, and retrieve evidence for NavTTA experiments on local or remote GPU servers. Use for Git/server readiness checks, campaign dry runs, process and GPU status, ETA estimates, interrupted-batch recovery, or downloading compact result evidence without copying raw assets or credentials.
---

# NavTTA Experiment Operator

Operate an existing reviewed experiment plan without changing its scientific contract.

## Guardrails

- Resolve the repository with `git rev-parse --show-toplevel`, then read its `AGENTS.md`, the selected task README, experiment plan, machine-readable spec, and launcher `--help`. Confirm task activation from the current request and repository policy; do not infer it from old results or status prose.
- Read the spec and launcher at operation time. Derive job count, GPU mapping, concurrency and memory caps, immutable arguments, output paths, and resume semantics from those files; never rely on values remembered by this skill.
- Treat `references/repos/` as read-only. Keep task artifacts under their task directory and shared code only in `core/`.
- Use an existing SSH agent/config or platform session. Never request, print, copy, or store passwords, private keys, access tokens, cookies, `.env` files, or credential-bearing command lines in the repository or evidence bundle.
- Make status collection read-only. A launch, resume, retry, kill, lock removal, checkout, pull, upload, or download is a separate state-changing action and needs to be within the user's request.
- Never call a running process complete. Completion requires exit evidence, expected metrics, validation evidence, and—when formal results are requested—authenticated manifests.

## Workflow

1. Identify the exact task, campaign, spec, runner, batch ID, repository path, environment, host, and evidence destination. Stop if any identity is ambiguous.
2. Run local and server preflight. Confirm the pinned top-level commit, tracked-worktree policy, runner/spec digest, available disk, environment, dataset/checkpoint provenance, live schedulers/workers, locks, and GPU memory/utilization.
3. Inspect a launcher dry run. Compare its expanded plan and effective limits to the current spec. Lower concurrency when allowed; never exceed a registered cap.
4. Launch only through the repository's existing scheduler/launcher, with a unique batch ID and durable supervision. Record the exact command and start-time identities, but no secrets.
5. Monitor scheduler, worker, GPU, exit, metrics, validation, and manifest evidence separately. Use `python3 .agents/skills/navtta-experiment-operator/scripts/read_status.py --batch-root PATH --spec SPEC --batch-id ID` for a read-only snapshot.
6. Resume only after proving the original scheduler and workers are gone. Reuse the same commit, spec digest, batch ID, seed, assets, and immutable arguments. Use the launcher's documented `--resume`; use retry flags only for failed/invalid jobs and only when the runner supports them.
7. Download compact evidence with an allowlist and no deletion: batch spec/plan, summaries, metrics, validation and exit markers, parameters, diagnostics, and formal manifests. Exclude checkpoints, datasets, scene/audio assets, raw logs, videos, TensorBoard files, caches, and credentials. Preserve relative paths and verify sizes/hashes after transfer.
8. Report the command identity, observed state, completed/failed/validated counts, GPU pressure, ETA method, resume decision, and local evidence paths. Label unknowns explicitly.

Read [references/operations.md](references/operations.md) before a launch, resume, lock intervention, or evidence download.

## Failure policy

Fail closed on a dirty formal worktree, commit/spec mismatch, stale live owner, cap violation, missing provenance, incomplete transfer, hash mismatch, or unsupported resume semantics. Do not repair scheduler state by hand or relaunch under a new identity merely to bypass a guard.
