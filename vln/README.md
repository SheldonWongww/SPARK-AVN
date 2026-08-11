# Vision-Language Navigation

VLN was explicitly activated for environment and asset preparation on
2026-08-10.  It remains isolated from AVN and ObjectNav.  The active model
coverage is:

- Discrete R2R/REVERIE: DUET, HAMT, and GOAT.
- Continuous R2R-CE: ETPNav and BEVBert.
- StreamVLN on continuous R2R-CE v1.3.

Active, Git-tracked source snapshots live under `baselines/`. Pinned,
read-only upstream working trees also live under
`references/repos/navigation/vln/` and remain ignored by the top-level Git
repository. URLs, commits, permission basis, and asset links are recorded in
`manifests/upstream_repositories.json`. Reproduce the upstream reference
checkouts without downloading datasets or checkpoints with:

```bash
python3 vln/scripts/fetch_upstreams.py
```

Research modifications must be made in `baselines/`, not inside the read-only
reference working trees.  Dataset/checkpoint provenance is recorded in this
task's `manifests/` directory; large assets remain ignored by Git.  The active
export uses the workspace LF policy and omits three upstream-tracked Python
bytecode cache files as well as MP3D connectivity metadata bundled by ETPNav
and BEVBert.

The evolving Source/TTA comparison, literature anchors, and small-search
policy are maintained in [`VLN_TTA_REPORT.md`](VLN_TTA_REPORT.md).  Tracked
machine-readable summaries live under `results/`; downloaded raw logs and
leaderboard prediction files remain ignored.

HAMT R2R uses the paper's final `vitbase-finetune-e2e` checkpoint and its
matching `r2r.e2e.ft.22k` visual features.  Run HAMT alone on both supported
benchmarks with:

```bash
TAG="hamt-e2e-source-$(date -u +%Y%m%dT%H%M%SZ)"
vln/scripts/run_hamt_e2e_source_eval.sh --gpu 0 --run-tag "$TAG"
```

## Canonical source evaluation

The reproducible split order is `val_seen`, `val_unseen`, then `test`.  Online
evaluation uses one process, one environment, and batch size one; each split
starts in a fresh process.  Complete episode-order manifests and the rationale
are in `manifests/episode_order/README.md`.

On the AutoDL host, print three representative settings (three splits each)
without running them:

```bash
cd /root/autodl-tmp/code/NavTTA
vln/scripts/run_source_eval.sh duet-r2r all 0 --dry-run
vln/scripts/run_source_eval.sh goat-reverie all 0 --dry-run
vln/scripts/run_source_eval.sh streamvln-r2r-ce all 0 --dry-run
```

The grouped launcher applies the measured single-GPU schedule automatically:
DUET/HAMT/GOAT first (three parallel model workers, each running R2R then
REVERIE), ETPNav/BEVBert second, and StreamVLN last.  Every setting still uses
fresh processes in canonical split order:

```bash
TAG="grouped-source-$(date -u +%Y%m%dT%H%M%SZ)"
vln/scripts/run_grouped_source_eval.sh --gpu 0 --run-tag "$TAG"
```

Use `--dry-run --skip-preflight` to validate the complete schedule without
evaluation.  The default continuous protocol is `v1.3-unified`.  The optional
`--ce-data-version v1.2-native` applies only to ETPNav/BEVBert; StreamVLN stays
on v1.3, so such a mixed-version run is provenance-only and must not be used as
a cross-model comparison.  A failed worker prevents the next resource group
from starting, interruption terminates active child process groups, and each
launcher log is retained under
`/root/autodl-tmp/tmp/navtta-grouped-source/TAG/`.
Every launcher attempt, including a dry-run, requires a fresh tag; formal runs
also reject any tag already present in source results or run manifests.
Grouped and standalone formal launchers share an atomic per-tag lock, so they
cannot write the same result tag concurrently.

For long evaluations, run the grouped scheduler inside one detached GNU screen
session.  The lifecycle manager keeps the resource-group barriers inside the
existing scheduler and provides persistent control logs, validated PID status,
safe termination, and screen reconnection:

```bash
TAG="grouped-source-$(date -u +%Y%m%dT%H%M%SZ)"
vln/scripts/manage_grouped_source_screen.sh start "$TAG" --gpu 0
vln/scripts/manage_grouped_source_screen.sh status "$TAG"
vln/scripts/manage_grouped_source_screen.sh logs "$TAG" --follow
vln/scripts/manage_grouped_source_screen.sh attach "$TAG"
# Detach again with Ctrl-a d.
vln/scripts/manage_grouped_source_screen.sh stop "$TAG"
```

Use `attach "$TAG" --multi` to join without detaching another screen client.
Stopping sends `TERM` to the validated grouped-runner PID and lets its existing
process-group cleanup finish; do not terminate the screen session directly.
Control logs live under `vln/results/logs/grouped_source_screen/TAG/`, while
per-setting logs retain their paths printed by the grouped runner.  Screen
reconnection is not result resumption: a failed or interrupted formal attempt
still requires a new run tag.

Before a run, repeat the lightweight asset and CPU/offline checks:

```bash
vln/scripts/verify_preflight.py --hash small
vln/scripts/verify_runtime_imports.sh
```

Use `verify_preflight.py --hash all` only when a full re-hash of the multi-GB
checkpoints and features is warranted.

To restart an isolated resource group with a fresh run tag, pass
`--only-group 1`, `--only-group 2`, or `--only-group 3` through
`manage_grouped_source_screen.sh start`.  This is intended for recovery after
an earlier group has already produced immutable formal artifacts.

Formal execution requires a GPU; remove `--dry-run` only on the intended
runtime host.  Supported settings are printed by invoking the script without
arguments.  Validation splits produce local metrics.  `test` only produces
leaderboard trajectory files; no local test metric is valid.  The R2R-CE files
are checked by `vln/scripts/validate_r2r_ce_submission.py` before handoff.
DUET and GOAT keep graph-path segments internally; their test artifacts are
atomically normalized after inference to the official R2R/REVERIE
`[viewpoint_id, heading_radians, elevation_radians]` schema before the strict
validator runs.  This export-only conversion does not alter policy actions or
validation metrics.  R2R-CE validation permits repeated positions for in-place
turns and the simulator-recorded displacement caused by slopes, stairs, and
collision sliding; the configured forward action remains 0.25m, but adjacent
recorded 3D positions are not required to be at most 0.25m apart.

All nine settings passed a two-episode canonical-prefix `val_seen` GPU
lifecycle smoke on 2026-08-10.  Successful outputs use these isolated tags:

- `gpu-smoke-20260810T023726Z`: DUET R2R/REVERIE, HAMT REVERIE, ETPNav,
  BEVBert, and StreamVLN.
- `gpu-smoke-20260810T023726Z-fix1`: HAMT-R2R.
- `gpu-smoke-20260810T023726Z-goatfix2`: GOAT R2R/REVERIE.

These smoke metrics are non-formal lifecycle diagnostics.  Repeat the check
after material runtime or checkpoint-loading changes with:

```bash
TAG="gpu-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
for setting in duet-r2r duet-reverie hamt-r2r hamt-reverie \
  goat-r2r goat-reverie etpnav-r2r-ce bevbert-r2r-ce streamvln-r2r-ce; do
  vln/scripts/run_source_eval.sh "$setting" val_seen 0 \
    --run-tag "$TAG" --smoke-episodes 2
done
```

The two-episode validator requires finite metrics for the exact canonical
prefix: R2R `1199_0`, `1199_1`; REVERIE `6806_482_0`, `6806_482_1`; and R2R-CE
`200`, `201` (all in scene `1LXtFkjw3qL`).  Checkpoint-key validation remains
active during these runs.

ETPNav and BEVBert default to the derived `v1.3-unified` annotations so their
episode starts match StreamVLN.  Use `--ce-data-version v1.2-native` only to
reproduce the upstream v1.2 protocol, and never mix those values in one formal
cross-model table.  `--run-tag TAG` gives every attempt an isolated output
tree; if omitted, a UTC tag is generated automatically.

Non-smoke execution is the formal-result path.  The launcher rejects tracked
changes and untracked execution files, then creates one manifest per
setting/split under `vln/results/runs/`.  Each manifest pins the commit, full
command, runtime hardware, primary and auxiliary checkpoint digests, asset
manifest, annotation bytes, episode-order manifest, and seed.  Review and
commit the preparation changes before removing `--dry-run`; smoke mode is
allowed on a dirty review tree and never creates a formal manifest.

Frozen-winner order robustness is the only use of
`run_source_eval.sh --order-seed 0|1|2`.  It requires a complete `val_seen`
`orders`-stage TTA job for one of the eight staged-search settings.  Seed 0
uses the unchanged canonical order; seeds 1/2 use the tracked derived
manifests described in `manifests/episode_order/README.md`.  The launcher
rejects Source, smoke/prefix, StreamVLN, split `all`, and native CE v1.2 uses,
and records the selected seed as both runtime/model seed and formal-manifest
seed.

Published paper numbers are preserved only as provenance-limited references in
`results/legacy/upstream_published_metrics.json`.  They can be used as cited
Source baselines, but cannot be relabelled as canonical-order or TTA reruns.
Any new formal result must also have a run manifest tied to the top-level Git
commit, exact configuration, asset digests, seed, and hardware.

## Post-search zero-update adapter parity audit

After all five methods have immutable `FROZEN_HPARAMETERS.json` files, run the
separate adapter-parity audit before interpreting TTA gains.  It uses the exact
canonical 256-episode `val_seen` prefix and creates exactly 56 jobs: 40 frozen
method adapters, eight argmax Source controls, and eight sampled Source
controls.  FeedTTA is paired with sampled Source; every other method is paired
with argmax Source.

```bash
SEARCH_BATCH=vln-tta-hparam-final-YYYYMMDDTHHMMSSZ
AUDIT_BATCH=vln-tta-adapter-parity-YYYYMMDDTHHMMSSZ

python3 vln/scripts/run_tta_adapter_parity_audit.py plan \
  --batch-id "$AUDIT_BATCH" --search-batch-id "$SEARCH_BATCH" --gpu 0
python3 vln/scripts/run_tta_adapter_parity_audit.py run \
  --batch-id "$AUDIT_BATCH" --max-workers 3 --max-per-model 1
python3 vln/scripts/run_tta_adapter_parity_audit.py validate \
  --batch-id "$AUDIT_BATCH"
```

This is not a zero-learning-rate ablation.  Explicit audit mode executes each
adapter's native loss, backward, replay, gating, binary-feedback, and update
scheduling paths, intercepting every optimizer/direct parameter write at the
last boundary.  Validation requires positive suppressed attempts, `updates=0`,
zero drift, identical before/after full-model state SHA256, method-specific
positive path evidence (including FSTTA's shadow FAST/SLOW audit trajectory),
the exact pinned episode IDs/order, identical action-trajectory SHA256,
identical per-episode output evidence, and exact aggregate metrics versus the
matched Source job.  Every job uses the formal source-tag lock and immutable
run-manifest lifecycle; its manifest pins the generated 256-record prefix,
canonical parent order, audit config, assets, dataset, seed, commit, and
hardware, with clean-tree checks at run start, immediately before model
execution, and at finalization.
Planning also pins the audit commit, frozen-search artifacts, asset/environment
manifests, datasets, and order manifests; execution requires an all-asset
preflight.  Audit configs use their own schema and
`vln/results/audits/adapter_parity/` namespace, so they cannot be passed through
the ordinary hyperparameter-search protocol or confused with scientific TTA
results.
