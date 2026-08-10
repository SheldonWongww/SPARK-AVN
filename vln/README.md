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

On the AutoDL host, print all nine source-evaluation commands without running
them:

```bash
cd /root/autodl-tmp/code/NavTTA
vln/scripts/run_source_eval.sh duet-r2r all 0 --dry-run
vln/scripts/run_source_eval.sh goat-reverie all 0 --dry-run
vln/scripts/run_source_eval.sh streamvln-r2r-ce all 0 --dry-run
```

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
