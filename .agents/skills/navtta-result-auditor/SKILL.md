---
name: navtta-result-auditor
description: Authenticate NavTTA run manifests and audit experiment evidence, matched Source comparisons, configuration drift, and TTA no-op behavior. Use when deciding whether runs are merely alive, have metrics, are validated, or qualify as formal results; when checking downloaded evidence or result tables; or when investigating suspicious gains, zero updates, parameter drift, provenance gaps, or protocol mismatches.
---

# NavTTA Result Auditor

Audit evidence independently of the desired conclusion. Never promote a result because its metric is favorable.

## Evidence levels

- **process:** a PID, scheduler record, lock, or exit marker proves only lifecycle state.
- **metrics:** compact metrics are parseable and complete enough to inspect, but validation/provenance may be absent.
- **validated:** the runner's checks establish expected episodes, finite metrics, diagnostics, configuration, and successful exit for the reviewed protocol.
- **formal:** validated evidence plus an authenticated canonical run manifest binding the top-level commit, configuration, checkpoint digest, dataset/version and stream identity, seed, hardware, immutable identity, and hashed result artifacts. Evidence under `results/legacy/` is never formal.

State each level separately; do not collapse “completed,” “validated,” and “formal.”

## Audit workflow

1. Read `AGENTS.md`, the task README, frozen experiment plan/spec, launcher validation rules, and any result registry. Do not activate or refactor another task line while auditing.
2. Inventory manifests with `python3 .agents/skills/navtta-result-auditor/scripts/manifest_inventory.py PATH --verify all`. Use `--path-map REMOTE=LOCAL` for preserved downloads. Run full verification on the server when large referenced assets are intentionally not downloaded.
3. Authenticate every claimed formal run against expectations derived independently from the frozen spec/registry—not values copied back from the manifest. Recompute immutable identity and file size/SHA256, require successful completion and result artifacts, and run `tools/validate_run_manifest.py` with independently sourced expected arguments.
4. Reconcile plan, job markers, aggregate metrics, per-job metrics/diagnostics, manifest pointers, and canonical manifests. Flag duplicates, missing jobs, stale attempts, orphaned artifacts, and aggregate/per-job disagreements.
5. Match each TTA run to Source on task, benchmark, model/checkpoint, source setting, split, seed/order, dataset/content, episode count, action/evaluation protocol, and other frozen controls. Treat any mismatch as an unmatched comparison unless the plan records and justifies it.
6. Compare effective parameters and diagnostics with the frozen spec. Detect zero-update/no-op adaptation, unexpected trainable scope, non-finite or excessive parameter drift, changed config/checkpoint/stream hashes, and retry/attempt drift. Use thresholds from the spec or runner; never invent a pass threshold.
7. Compute deltas only against the matched authenticated Source. Prefer paired episode-level analysis when the same stream allows it; otherwise label aggregate deltas descriptive. Keep unsupervised TTA separate from methods consuming binary episode feedback.
8. Produce a finding-led report with evidence level, manifest authentication, match key, metric delta, no-op/drift checks, discrepancies, and an explicit `accept`, `provisional`, or `reject` decision.

Read [references/audit-protocol.md](references/audit-protocol.md) for exact checks and failure interpretation.

## Fail-closed rules

Reject formal status for a missing/noncanonical/legacy manifest, identity or artifact hash mismatch, incomplete provenance, failed validation, unmatched Source, config drift, silent no-op, or unexplained diagnostics. Preserve the evidence and downgrade its level; never edit artifacts to make an audit pass.
