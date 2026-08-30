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

The order-seed-0 R2R `val_unseen` table is currently retained only as a
scene-blocked continual stress test.  Its replacement uses three globally
shuffled `val_seen` streams for model-method selection and freezes the result
before three shuffled `val_unseen` evaluations; see
[`R2R_CROSS_SPLIT_ROBUST_REEVALUATION_V1_PLAN.md`](experiments/R2R_CROSS_SPLIT_ROBUST_REEVALUATION_V1_PLAN.md).

HAMT R2R uses the paper's final `vitbase-finetune-e2e` checkpoint and its
matching `r2r.e2e.ft.22k` visual features.  Run HAMT alone on both supported
benchmarks with:

```bash
TAG="hamt-e2e-source-$(date -u +%Y%m%dT%H%M%SZ)"
vln/scripts/run_hamt_e2e_source_eval.sh --gpu 0 --run-tag "$TAG"
```

## Canonical source evaluation

The split order is `val_seen`, `val_unseen`, then `test`.  Online evaluation
uses one process, one environment, and batch size one; each split starts in a
fresh process.  Episode-order JSON files remain runtime inputs because they
define the online TTA stream.

The active server layout is fixed in the launch scripts:

```text
repository  /data1/wxy/code/NavTTA
VLN storage /data1/wxy/exp_data/NavTTA/vln
environments /data1/wxy/exp_data/NavTTA/vln/envs
cache       /data1/wxy/exp_data/NavTTA/vln/cache
temporary   /data1/wxy/exp_data/NavTTA/vln/tmp
```

On that host, print three representative settings without running them:

```bash
cd /data1/wxy/code/NavTTA
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

Use `--dry-run --skip-runtime-check` to print the complete schedule without
running the full import sweep.  The model environments must still exist.  The
default continuous protocol is `v1.3-unified`.  The optional
`--ce-data-version v1.2-native` applies only to ETPNav/BEVBert; StreamVLN stays
on v1.3, so such a mixed-version run is provenance-only and must not be used as
a cross-model comparison.  A failed worker prevents the next resource group
from starting, interruption terminates active child process groups, and each
launcher log is retained under
`/data1/wxy/exp_data/NavTTA/vln/tmp/navtta-grouped-source/TAG/`.
Every launcher attempt, including a dry-run, requires a fresh tag, and a real
run rejects a tag already present in Source results.
Grouped and standalone launchers share an atomic per-tag lock, so they
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
reconnection is not result resumption: a failed or interrupted attempt
still requires a new run tag.

The grouped runner performs the CPU/offline import check automatically.  It
can also be run directly while preparing environments:

```bash
vln/scripts/verify_runtime_imports.sh
```

`MatterSim` is a native Python 3.8 extension and is not installed by pip.  On
a new server checkout, build it once at the fixed runtime path and then repeat
the import check:

```bash
vln/scripts/build_mattersim.sh
vln/scripts/verify_runtime_imports.sh
```

The builder uses the isolated native toolchain at
`/data1/wxy/exp_data/NavTTA/vln/envs/mattersim-native`, pins the simulator and
pybind11 revisions, applies the OpenCV 4 and conda-forge JsonCpp compatibility
changes, embeds the native-library runtime path, and verifies the resulting
module in the DUET, HAMT, and GOAT environments.  It does not depend on shell
activation or exported compiler variables.  Evaluation launchers never build
it implicitly.

HAMT uses its older Transformers runtime with the pinned local
`bert-base-uncased` snapshot already stored under the DUET checkpoints.  The
launch and runtime-check scripts pass that snapshot to HAMT explicitly, so no
legacy URL-hashed cache files or network access are required.

The shared ETPNav/BEVBert evaluation environment uses `tensorboard==1.15.0`,
the minimum version accepted by its PyTorch runtime.  The unused upstream
TensorFlow imports are omitted from the active evaluation snapshots, so the
obsolete `tensorflow==1.13.1` and unavailable `tb-nightly` packages are not
part of this evaluation runtime.

To restart an isolated resource group with a fresh run tag, pass
`--only-group 1`, `--only-group 2`, or `--only-group 3` through
`manage_grouped_source_screen.sh start`.  This is intended for recovery after
an earlier group has already produced results under a different run tag.

Full execution requires a GPU; remove `--dry-run` only on the intended
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

The two-episode StreamVLN smoke does not cross into the second canonical
scene. A damaged `1pXnuDYAj8r_semantic.ply` therefore first appears at episode
22 as `StanfordImporter::mesh(): unsupported face size 64` followed by a native
abort. Validate and restore that scene from the manifest-pinned official
archive before retrying:

```bash
cd /data1/wxy/code/NavTTA
if ! vln/scripts/verify_streamvln_scene_assets.py 1pXnuDYAj8r; then
  vln/scripts/repair_streamvln_scene_assets.py 1pXnuDYAj8r
fi
vln/scripts/verify_streamvln_scene_assets.py 1pXnuDYAj8r

TAG="streamvln-scene-transition-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
vln/scripts/run_source_eval.sh streamvln-r2r-ce val_seen 2 \
  --run-tag "$TAG" --smoke-episodes 22
```

The repair verifies the complete `mp3d_habitat.zip` against the tracked size
and SHA256, extracts the four scene files into same-filesystem staging,
validates the GLB and binary PLY layouts, retains a timestamped backup, and
then atomically installs the restored scene. Stop all jobs using MP3D before
running it. Do not use the AVN scene tree as a repair source: a matching bad
copy may share the same corruption.

The four-GPU launcher can run the same nine-setting smoke, or the complete
`val_seen` and `val_unseen` Source evaluation.  It assigns ETPNav, BEVBert,
StreamVLN, and all six discrete settings to four independent queues.  It does
not run TTA methods or the hidden-label `test` split.  Every setting uses model
seed 0 and the canonical episode order (order seed 0).  Its default selects the
paper-native released weights and dataset versions while retaining NavTTA's
controlled single-rank, canonical-order execution:

| Model | Released evaluation weight | Evaluation data |
|---|---|---|
| DUET | R2R/REVERIE `best_val_unseen` | Original discrete R2R/REVERIE |
| HAMT | R2R `vitbase-finetune-e2e`; REVERIE `best_val_unseen` | Original discrete R2R/REVERIE |
| GOAT | R2R/REVERIE `best_val_unseen.pt` plus released sidecars | Original discrete R2R/REVERIE |
| ETPNav | `ckpt.iter12000.pth` | `R2R_VLNCE_v1-2_preprocessed_BERTidx` |
| BEVBert | `ckpt.iter9600.pth` | `R2R_VLNCE_v1-2_preprocessed_BERTidx` |
| StreamVLN | released `...scalevln_v1_3` checkpoint | public `R2R_VLNCE_v1-3` |

R2R-CE v1.3 is not just a repackaging of v1.2.  The official release corrected
the initial heading of every episode to match discrete R2R and recomputed the
oracle paths; it also removed 109 no-longer-navigable EnvDrop augmentation
episodes.  The validation episode identities and goals remain the same, but
the changed start rotations make v1.2 and v1.3 different evaluation streams.
The `preprocessed` and `BERTidx` suffixes describe preprocessing/tokenization,
not additional benchmark versions.  See the
[official VLN-CE data changelog](https://jacobkrantz.github.io/vlnce/data).

Run it with:

```bash
SMOKE_TAG="four-gpu-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
vln/scripts/run_four_gpu_source_eval.sh \
  --gpus 0,1,2,3 --run-tag "$SMOKE_TAG" --smoke

SOURCE_TAG="four-gpu-source-$(date -u +%Y%m%dT%H%M%SZ)"
vln/scripts/run_four_gpu_source_eval.sh \
  --gpus 0,1,2,3 --run-tag "$SOURCE_TAG" --skip-runtime-check
```

To rerun only BEVBert and StreamVLN after an affected batch, keep BEVBert
explicitly on the paper-native v1.2 annotations. The two queues run in
parallel:

```bash
SOURCE_TAG="bevbert-streamvln-source-$(date -u +%Y%m%dT%H%M%SZ)"
vln/scripts/run_four_gpu_source_eval.sh \
  --gpus 0,1,2,3 --only-queues 1,2 \
  --ce-data-version v1.2-native \
  --run-tag "$SOURCE_TAG" --skip-runtime-check
```

When queue 1 is selected with `v1.2-native`, the launcher checks frozen sizes
and hashes for the released BEVBert checkpoint, CLIP ViT-B/16, waypoint
predictor, depth encoder, both validation annotation files, and both evaluator
ground-truth files before starting GPU work. It also verifies the manifest
episode counts/order and prints that the one `ckpt.iter9600.pth` checkpoint is
shared by `val_seen` and `val_unseen`. The same check can be run independently
with `vln/scripts/verify_bevbert_source_pairing.py`.

The four-GPU launcher's default `v1.2-native` option selects the ETPNav/BEVBert
paper protocol; StreamVLN remains on its released public v1.3 data and is
labelled `v1.3-native`.  Passing `--ce-data-version v1.3-unified` instead uses
the derived BERT-indexed v1.3 annotations for ETPNav/BEVBert and is an
explicitly different cross-model protocol, not paper parity. For a full run,
batch logs, the expanded plan, Git commit, exit codes, and comparisons are
stored under `vln/results/source/TAG/_launcher/`. Smoke-launcher logs remain
under `/data1/wxy/exp_data/NavTTA/vln/tmp/navtta-four-gpu-source/TAG/`.
Per-split model logs and outputs remain under `vln/results/source/` or
`vln/results/smoke/`. `metrics.csv` checks the Source values transcribed from
the five VLN-TTA papers into the current project workbook;
`paper_metrics.csv` separately compares against each navigation model's
original paper.  For R2R and R2R-CE, SR and SPL are compared after decimal
`ROUND_HALF_UP` to integer percentage points; raw values are retained in both
CSVs.  REVERIE uses the precision reported by its source table.  A comparison
mismatch is a scientific finding, not a launcher failure; the command exits
nonzero only for execution, completeness, or parsing failures.

The paper-native run is not claimed to reproduce the papers' unpublished
episode scheduling or process-level RNG bit for bit: NavTTA intentionally uses
one process, one environment, and its recorded canonical order.  Source does
not adapt state across episodes, but implementation-level nondeterminism can
still prevent bitwise parity; the rerun therefore checks the paper-reported
metric precision rather than requiring identical trajectories.

The standalone `run_source_eval.sh` and grouped scheduler retain their
`v1.3-unified` default because the existing NavTTA R2R-CE experiments were
frozen on v1.3 episode starts.  The paper-native four-GPU run is a separate
reproduction: ETPNav/BEVBert v1.2 and StreamVLN v1.3 must be labelled by
version and must not be presented as one same-stream cross-model table.  Any
future TTA/search campaign following the newly selected paper-native protocol
must use new matched Source controls on ETPNav/BEVBert v1.2; it cannot reuse
the existing unified-v1.3 Source or TTA records.
`--run-tag TAG` gives every attempt an isolated output tree; if omitted, a UTC
tag is generated automatically.

The server launchers intentionally do not inspect asset/environment manifests,
require a clean Git tree, or create formal run manifests.  Their outputs are
exploratory experiment results.  Promote a result into a formal table only
through a separate provenance workflow satisfying the workspace rules.

Frozen-winner order robustness is the only use of
`run_source_eval.sh --order-seed 0|1|2`.  It requires a complete `val_seen`
`orders`-stage TTA job for one of the eight staged-search settings.  Seed 0
uses the unchanged canonical order; seeds 1/2 use the tracked derived
order files described in `manifests/episode_order/README.md`.  The launcher
rejects Source, smoke/prefix, StreamVLN, split `all`, and native CE v1.2 uses,
and passes the selected seed to the runtime/model.

Published paper numbers are preserved only as provenance-limited references in
`results/legacy/upstream_published_metrics.json`.  They can be used as cited
Source baselines, but cannot be relabelled as canonical-order or TTA reruns.
Any new formal result must also have a run manifest tied to the top-level Git
commit, exact configuration, asset digests, seed, and hardware.

## Completed historical R2R model-wise Cartesian full-val search

This section documents the earlier formal-manifest workflow for historical
results.  Its launch recipes are not the active manifest-free server path.
The historical R2R-only search is defined by
`experiments/r2r_modelwise_cartesian_hparam_v2.json` and executed by
`scripts/run_r2r_cartesian_hparam_search.py`.  It covers DUET-R2R, HAMT-R2R,
and GOAT-R2R with Tent, FSTTA, EAM, FeedTTA, and ATENA.  Unlike the legacy
multi-benchmark scheduler, it runs every Cartesian point directly over all
1,021 canonical-order `val_seen` episodes: there is no smoke job, prefix,
screening stage, or promotion stage.  Winners are model-specific and selected
by SR, SPL, lower parameter drift, then fewer updates.

The five grids contain 40, 81, 320, 250, and 100 candidates per setting,
respectively.  Across three settings this is 2,373 TTA jobs.  Three standard
argmax Source jobs and three sampled-Source FeedTTA diagnostic controls bring
the exact campaign total to **2,379 jobs**.  Every reported delta and frozen
winner uses standard argmax Source; sampled Source is diagnostic only.

Those FeedTTA jobs predate the paper/protocol audit and retain the historical
forced-sampling implementation. The active corrected follow-up is
`experiments/r2r_fstta_feedtta_postfix_search_v1.json`: it contains only FSTTA
and FeedTTA, restores target-native argmax FeedTTA, fixes evaluator-endpoint
feedback, adds model-aware FeedTTA scopes, and preserves FSTTA variance over
the full test stream. It reuses the pinned argmax Source controls and has no
sampled control jobs.

The subsequent focused batch is defined by
`experiments/r2r_targeted_gap_refinement_v1.json`. It contains only DUET Tent,
GOAT FeedTTA, and GOAT ATENA (29 full-split jobs), reuses Source, and corrects
ATENA to consume submitted-trajectory evaluator success through a lazy query
callback. Results are documented in
`results/analysis/hparam_search/R2R_TARGETED_GAP_REFINEMENT_V1.md`.

Results are benchmark-first:

```text
vln/results/logs/r2r/hparam_search/<batch>/<method>/
vln/results/tuning/r2r/hparam_search/<batch>/<model>/<method>/jobs/<run-tag>/val_seen/
```

The batch-level `WINNERS.json`, `FROZEN_HPARAMETERS.json`, and `TOP5.csv` are
written under `vln/results/logs/r2r/hparam_search/<batch>/` only after all five
TTA methods have complete validated evidence.  Per-method plans and progress
remain in the method subdirectories.  A formal launch requires a clean tracked
worktree.

Run the long campaign in one detached GNU screen session on the AutoDL host:

```bash
cd /data1/wxy/code/NavTTA
BATCH="vln-r2r-modelwise-cartesian-v2-seed0"
SESSION="navtta-r2r-v2"
LAUNCH_DIR="/data1/wxy/code/NavTTA/vln/results/logs/r2r/hparam_search/${BATCH}/_launcher"

mkdir -p "$LAUNCH_DIR"
python3 vln/scripts/run_r2r_cartesian_hparam_search.py all \
  --batch-id "$BATCH" --gpu 0 --plan-only

screen -dmS "$SESSION" env BATCH="$BATCH" LAUNCH_DIR="$LAUNCH_DIR" bash -lc '
  cd /data1/wxy/code/NavTTA || exit 97
  export PYTHONUNBUFFERED=1
  python3 vln/scripts/run_r2r_cartesian_hparam_search.py all \
    --batch-id "$BATCH" --gpu 0 --resume \
    >>"$LAUNCH_DIR/console.log" 2>&1 &
  pid=$!
  printf "%s\n" "$pid" >"$LAUNCH_DIR/scheduler.pid"
  wait "$pid"
  rc=$?
  printf "%s\n" "$rc" >"$LAUNCH_DIR/exitcode"
  exit "$rc"
'
```

Observe the screen session and the runner's durable batch status independently:

```bash
screen -ls
tail -f "$LAUNCH_DIR/console.log"
screen -r "$SESSION"                       # detach again with Ctrl-a d
python3 vln/scripts/run_r2r_cartesian_hparam_search.py all \
  --batch-id "$BATCH" --status
python3 vln/scripts/run_r2r_cartesian_hparam_search.py all \
  --batch-id "$BATCH" --watch
```

After confirming that no original scheduler screen is still running, resume an
interrupted batch in a new detached session with the identical batch ID, Git
commit, and spec:

```bash
RESUME_SESSION="navtta-r2r-v2-r-$(date -u +%Y%m%dT%H%M%SZ)"
screen -dmS "$RESUME_SESSION" env BATCH="$BATCH" LAUNCH_DIR="$LAUNCH_DIR" bash -lc '
  cd /data1/wxy/code/NavTTA || exit 97
  export PYTHONUNBUFFERED=1
  python3 vln/scripts/run_r2r_cartesian_hparam_search.py all \
    --batch-id "$BATCH" --gpu 0 --resume \
    >>"$LAUNCH_DIR/console.log" 2>&1 &
  pid=$!
  printf "%s\n" "$pid" >"$LAUNCH_DIR/scheduler.pid"
  wait "$pid"
  rc=$?
  printf "%s\n" "$rc" >"$LAUNCH_DIR/exitcode"
  exit "$rc"
'
```

Ordinary `--resume` retains completed jobs and reconnects to live worker PIDs.
If persisted attempts have failed or invalid evidence, use the same command
with `--resume --retry-failed`; retries archive the old console, result tree,
worker state, and formal manifest before receiving a new run tag.  Never run
two schedulers for the same batch concurrently. To stop safely, first verify
the recorded command with `ps -fp "$(cat "$LAUNCH_DIR/scheduler.pid")"`, then
send `kill -TERM "$(cat "$LAUNCH_DIR/scheduler.pid")"`; do not kill the screen
session directly while workers are active.

## Post-search zero-update adapter parity audit

For historical pre-fix campaigns, after all five methods have immutable
`FROZEN_HPARAMETERS.json` files, run the
separate adapter-parity audit before interpreting TTA gains.  It uses the exact
canonical 256-episode `val_seen` prefix and creates exactly 56 jobs: 40 frozen
method adapters, eight argmax Source controls, and eight sampled Source
controls.  That v1 audit pairs historical sampled FeedTTA with sampled Source;
it must not be used to certify the corrected target-native-argmax campaign.

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
