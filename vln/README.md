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
turns and enforces the official 0.25m maximum forward displacement.

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

Published paper numbers are preserved only as provenance-limited references in
`results/legacy/upstream_published_metrics.json`.  They can be used as cited
Source baselines, but cannot be relabelled as canonical-order or TTA reruns.
Any new formal result must also have a run manifest tied to the top-level Git
commit, exact configuration, asset digests, seed, and hardware.
