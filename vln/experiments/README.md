# VLN experiment specifications

## R2R model-wise direct Cartesian full-val search

`r2r_modelwise_cartesian_hparam_v2.json` is the immutable design input for the
completed DUET/HAMT/GOAT R2R search.  It is intentionally separate from the older
staged search described below.  Every declared Cartesian point is evaluated
directly on the complete 1,021-episode canonical-order R2R `val_seen` stream at
order seed 0.  There is no smoke run, 256-episode screening prefix, staged
promotion, or cross-model winner.  Each model/method pair freezes its own
winner by SR first, SPL second, then lower parameter drift and fewer updates.

The exact grid sizes are:

| Method | Candidates per setting | Three-setting TTA jobs |
|---|---:|---:|
| Tent | 40 | 120 |
| FSTTA | 81 | 243 |
| EAM | 320 | 960 |
| FeedTTA | 250 | 750 |
| ATENA | 100 | 300 |
| **Total TTA** | **791** | **2,373** |

The campaign also runs one standard argmax Source job and one sampled-Source
FeedTTA diagnostic job for each of the three settings.  These six controls make
the complete campaign exactly **2,379 jobs**.  All winner deltas use the three
standard argmax Source jobs; the sampled controls are diagnostic evidence only
and never replace the formal Source baseline.

Administrative evidence and raw model outputs both use a benchmark-first
layout and contain no `stages/` directory:

```text
vln/results/logs/r2r/hparam_search/<batch>/<method>/
  GRID.json  grid.csv  metrics.csv  SUMMARY.json  progress.json
  jobs/<setting>/<run-tag>/
  WINNER.json  TOP5.csv                 # TTA methods after completion

vln/results/logs/r2r/hparam_search/<batch>/
  WINNERS.json  FROZEN_HPARAMETERS.json  TOP5.csv

vln/results/tuning/r2r/hparam_search/<batch>/<model>/<method>/jobs/
  <run-tag>/val_seen/
```

`GRID.json` pins the top-level Git commit, search-spec digest, exact job count,
settings, canonical order seed, and result-layout version.  Jobs are ordered
round-robin across `duet-r2r`, `hamt-r2r`, and `goat-r2r`.  Full-val commands
deliberately omit both `--episode-limit` and `--order-seed`; `episodes=-1` in
job metadata means the complete canonical seed-0 stream, not a prefix.

The checked-in scheduler caps reflect the observed 32GB vGPU envelope:
Source/control `3/1`, Tent `5/2`, FSTTA `8/3`, EAM `6/2`, FeedTTA `5/2`, and
ATENA `4/1` for `max_workers/max_per_model`. Launches are staggered by 15
seconds and pause above 25,000 MiB GPU memory or 75 GiB cgroup memory. These
are launch guards rather than claimed resource requirements; `resource.csv`
records actual campaign usage.

The foreground lifecycle is:

```text
python3 vln/scripts/run_r2r_cartesian_hparam_search.py all --batch-id SEARCH_ID --gpu 0 --plan-only
python3 vln/scripts/run_r2r_cartesian_hparam_search.py all --batch-id SEARCH_ID --gpu 0 --resume
python3 vln/scripts/run_r2r_cartesian_hparam_search.py all --batch-id SEARCH_ID --status
python3 vln/scripts/run_r2r_cartesian_hparam_search.py all --batch-id SEARCH_ID --watch
python3 vln/scripts/run_r2r_cartesian_hparam_search.py all --batch-id SEARCH_ID --resume --gpu 0
python3 vln/scripts/run_r2r_cartesian_hparam_search.py all --batch-id SEARCH_ID --resume --retry-failed --gpu 0
```

`all` runs `source`, `feedtta_control`, Tent, FSTTA, EAM, FeedTTA, and ATENA in
that order.  Resume revalidates the immutable plans, completed metrics, commit,
and spec digest.  `--retry-failed` is only valid with `--resume` and archives
the previous attempt before assigning a `-retryN` run tag.  See `vln/README.md`
for detached GNU screen launch and recovery commands. Planning and execution
both require a clean tracked worktree. `--watch` exits automatically only after
all 2,379 jobs succeed; use one-shot `--status` when investigating failures.

The batch `vln-r2r-modelwise-cartesian-v2-seed0` completed all 2,379 jobs with
zero failures on commit `a258ac5`.  Its compact analysis is tracked in
`vln/results/analysis/hparam_search/R2R_CARTESIAN_V2_AND_LOCAL_REFINEMENT.md`.

## R2R FSTTA/FeedTTA paper-alignment post-fix search

`r2r_fstta_feedtta_postfix_search_v1.json` is the focused 78-job follow-up.
It contains only FSTTA (12 candidates per model) and FeedTTA (14 per model),
uses strict model/method barriers, and reuses the three pinned standard argmax
Source results. It neither reruns Source nor creates sampled Source controls.
The batch completed 78/78 validated jobs on commit `b2e37dd`; its result report
is [`R2R_FSTTA_FEEDTTA_POSTFIX_V1.md`](../results/analysis/hparam_search/R2R_FSTTA_FEEDTTA_POSTFIX_V1.md).

The implementation changes that define this batch are part of its scientific
identity: FSTTA retains its FAST variance EMA for the complete test stream;
FeedTTA executes the target model's native argmax policy, receives success from
the exact evaluator endpoint, and records one of `paper_full`,
`last_crossmodal`, or `action_head` as its parameter scope. The FeedTTA grid
includes the paper's literal R2R SGR setting `p=.05, alpha=+.1`, the prior
negative-SGR winner neighborhoods, a no-SGR control, and smaller-scope points.

The six phases are:

```text
DUET FSTTA -> DUET FeedTTA -> HAMT FSTTA -> HAMT FeedTTA
            -> GOAT FSTTA -> GOAT FeedTTA
```

Run from a clean committed tree:

```bash
SPEC=vln/experiments/r2r_fstta_feedtta_postfix_search_v1.json
BATCH=vln-r2r-fstta-feedtta-postfix-v1-seed0
python3 vln/scripts/run_r2r_local_refinement.py \
  --spec "$SPEC" --batch-id "$BATCH" --plan-only --print-commands
python3 vln/scripts/run_r2r_local_refinement.py \
  --spec "$SPEC" --batch-id "$BATCH" --resume --confirm-reviewed --gpu 0
python3 vln/scripts/run_r2r_local_refinement.py \
  --spec "$SPEC" --batch-id "$BATCH" --watch --watch-interval 10
```

Prior grouped measurements plus the registered conservative GOAT projections
remain the concurrency authority: FSTTA uses DUET/HAMT/GOAT caps `14/11/14`,
and FeedTTA uses `6/5/6`. This batch directly observed GOAT FSTTA at 12 workers
(all available candidates) and GOAT FeedTTA at its six-worker cap; the
14-worker GOAT FSTTA ceiling remains a projection. The 29,000 MiB planning and
30,000 MiB emergency lines remain unchanged.

## R2R targeted gap refinement

`r2r_targeted_gap_refinement_v1.json` is a completed 29-job follow-up for only the three
remaining weak pairs: nine DUET Tent points, eleven GOAT FeedTTA points, and
nine GOAT ATENA points. It reuses all Source evidence and runs no sampled
controls. Tent keeps `update_interval=1` and searches graph-local LayerNorm
scopes; FeedTTA stays on corrected argmax/action-head semantics and interpolates
below its `1e-6` LR boundary. ATENA first fixes its feedback to a lazy callback
over the submitted evaluator trajectory, then reruns one protocol anchor plus
eight local points. See
[`R2R_TARGETED_GAP_REFINEMENT_PLAN.md`](R2R_TARGETED_GAP_REFINEMENT_PLAN.md)
for the exact rationale, budget, and launch command. All 29 jobs completed
without failure on commit `9c006fb`; the result analysis is
[`R2R_TARGETED_GAP_REFINEMENT_V1.md`](../results/analysis/hparam_search/R2R_TARGETED_GAP_REFINEMENT_V1.md).

## R2R four-method low-learning-rate refinement

`r2r_five_method_local_refinement_v1.json` is the next-round design.  It keeps
Tent at 30 candidates, FSTTA at 214, FeedTTA at 167, and GOAT-only ATENA at 55.
The three completed standard Source results are reused through the pinned
`vln/manifests/r2r_reused_source_controls.json`; only three FeedTTA sampled
controls are executed. This makes 469 jobs before conditional confirmation.
EAM is not searched in this round.
The filename and schema retain the name of an earlier uncommitted draft; the
validated contents and enabled-method map are the authoritative four-method
protocol.

The new campaign is model-major and has a strict barrier:

```text
DUET: all methods complete -> HAMT: all methods complete -> GOAT: all methods
```

`vln/scripts/run_r2r_local_refinement.py` expands the model-specific grids and
enforces strict model and method barriers. The spec is launchable only with the
explicit `--confirm-reviewed` acknowledgement; `max_per_model` alone still does
not enforce the barrier.

For any future increase above a default production cap, use a separate
plan-only calibration batch. The helper clones jobs into isolated calibration
paths and starts with one fresh
five-worker group over the canonical 100-episode prefix. A later worker count
may be chosen from the measured per-worker VRAM only when its linear projection
stays at or below 28,500 MiB, and that larger count is then launched as a new,
independent grouped calibration. A process count alone is not loading evidence:
each tested group must raise VRAM by at least 512 MiB per worker over idle and
remain within 2% for a 20-second/three-sample window. Each group has a
300-second load timeout. The helper also
stops adding workers at 29,000 MiB, aborts its own process groups at 30,000 MiB,
and writes `CALIBRATION.json` plus `resource.csv` without changing formal job
evidence:

```bash
CAL_BATCH=vln-r2r-four-method-low-lr-v1-calibration
python3 vln/scripts/run_r2r_local_refinement.py \
  --batch-id "$CAL_BATCH" --plan-only
python3 vln/scripts/calibrate_r2r_local_refinement.py \
  --batch-id "$CAL_BATCH" --phase-id 01-duet-r2r-fstta \
  --target-workers 5 --initial-workers 5 --episode-limit 100 \
  --stagger-seconds 0 --calibration-id duet-fstta-baseline5

PARENT="$PWD/vln/results/logs/r2r/hparam_search/$CAL_BATCH/calibration/\
01-duet-r2r-fstta/duet-fstta-baseline5/CALIBRATION.json"
# Choose N from the phase's declared counts after inspecting the baseline.
N=10
python3 vln/scripts/calibrate_r2r_local_refinement.py \
  --batch-id "$CAL_BATCH" --phase-id 01-duet-r2r-fstta \
  --target-workers "$N" --initial-workers "$N" --episode-limit 100 \
  --sizing-parent "$PARENT" --stagger-seconds 0 \
  --calibration-id "duet-fstta-verify${N}"
```

The grouped stability defaults can be made more conservative with
`--steady-seconds`, `--steady-samples`, `--load-timeout-seconds`,
`--min-loaded-memory-mib`, and `--steady-relative-tolerance`. Do not reduce
them merely to make a slow-loading model advance. `CALIBRATION.json` records
the previous steady VRAM, confirmed current steady VRAM, and load wait for
every level; only levels with `steady_confirmed: true` can contribute to
`recommended_cap`.

The 100-episode prefix is mandatory for resource sizing and never replaces the
complete 1,021-episode formal run. The helper binds it in the plan, cloned
configs, commands, summary, and parent identity checks. A parentless run is
accepted only for exactly five workers. Any larger declared count requires
`--sizing-parent` pointing to that same phase's completely successful fresh
five-worker run. The authorization projection is
`parent_idle + N * max(spec_estimate, 1.05 * measured_peak_increment_per_worker)`;
it must not exceed 28,500 MiB. The helper repeats the projection with current
idle VRAM before launching anything, then independently requires the selected
N-worker group to load and remain steady below 29,000 MiB. Parent summary and
plan SHA256 digests are recorded, and the summary binds the raw resource CSV
digest. Counts observed while a group is launching are transient and never
approved as formal concurrency.

The helper remains the required path for raising a phase above the default
`production_cap`. Do not run the formal scheduler for `CAL_BATCH`; use a
different batch for formal results.

For the current 469-job launch, the approved defaults are encoded directly as
`production_cap`: DUET Tent/FSTTA/FeedTTA = 10/14/6, HAMT = 10/11/5, and GOAT
Tent/FSTTA/FeedTTA/ATENA = 10/14/6/5. Completed grouped measurements support
all of these except GOAT FSTTA and GOAT FeedTTA. Those two are explicit
user-approved estimates: 14-worker FSTTA projects to 28,102 MiB from its
successful five-worker group, and six-worker FeedTTA projects to 26,400 MiB
from the per-job estimate. The formal 15-second launch stagger, per-method
23,500--25,000 MiB pre-launch gates, 29,000 MiB planning line, and 30,000 MiB
emergency line remain active. Because these are defaults, the formal batch
uses the no-override command below; future increases still require completed
grouped evidence.

After the spec and runner are tracked on a clean commit, create this formal
batch directly with the no-override commands below:

```bash
BATCH=vln-r2r-four-method-low-lr-v1-seed0
python3 vln/scripts/run_r2r_local_refinement.py \
  --batch-id "$BATCH" --plan-only --print-commands
python3 vln/scripts/run_r2r_local_refinement.py \
  --batch-id "$BATCH" --resume --confirm-reviewed --gpu 0
python3 vln/scripts/run_r2r_local_refinement.py \
  --batch-id "$BATCH" --status
python3 vln/scripts/run_r2r_local_refinement.py \
  --batch-id "$BATCH" --watch --watch-interval 10
```

`--phase-max-workers PHASE_ID=N` is repeatable and may select only a worker
count declared for that exact phase in the spec. For example, a future increase
of HAMT FSTTA from its default 11 to 12 requires grouped calibration; the
approved arguments must then be repeated unchanged on resume:

```bash
PHASE_ARGS=(
  --phase-max-workers 05-hamt-r2r-fstta=12
)
CALIBRATION_ARGS=(
  --phase-calibration 05-hamt-r2r-fstta=/absolute/path/to/CALIBRATION.json
)
python3 vln/scripts/run_r2r_local_refinement.py \
  --batch-id "$BATCH" --plan-only \
  "${PHASE_ARGS[@]}" "${CALIBRATION_ARGS[@]}"
python3 vln/scripts/run_r2r_local_refinement.py \
  --batch-id "$BATCH" --resume --confirm-reviewed --gpu 0 \
  "${PHASE_ARGS[@]}" "${CALIBRATION_ARGS[@]}"
```

Any phase override above its default safe cap is rejected unless
`--phase-calibration PHASE_ID=CALIBRATION.json` names a completely successful
100-episode grouped run for that exact model, method, GPU, commit, spec, worker
count, and candidate-config prefix. The formal `PLAN.json` and `PHASE.json`
pin the calibration summary, sibling plan, and resource evidence digests; the
same arguments must be repeated on every resume.

`--max-workers` is only a global downward cap; it cannot raise a phase above a
declared value. Worker overrides and the other runtime limits are pinned in
`PLAN.json`; changing them requires a new batch ID. Failed jobs require
`--resume --retry-failed` with the same pinned runtime arguments.

Future concurrency increases start from a five-worker measurement and use the
grouped validation protocol above. ATENA is enabled only for GOAT and fixed at
five workers. Production plans must remain at or below 29,000 MiB; an observed
30,000 MiB is an emergency rollback line, not a planning target. Per-method
pre-launch gates remain 23,500--25,000 MiB and are secondary guards, not
protection against cold-start memory lag.

## Superseded: R2R FeedTTA low-learning-rate refinement

The 108-point common-grid plan below is retained for provenance but is
superseded by the 167-point model-specific FeedTTA section of
`r2r_five_method_local_refinement_v1.json`.  Do not launch it for the next
round.

`r2r_feedtta_low_lr_refinement_v1.json` is a separate follow-up to the completed
250-point-per-model FeedTTA grid. It does not modify or resume the parent batch.
The refinement keeps only the lower learning-rate boundary and the SGR profiles
that were either a parent winner or the no-SGR control:

```text
LR          = [1e-7, 3e-7, 1e-6]
gamma       = [0.5, 0.8, 1.0]
SGR profile = [no_sgr, p005_a005, p010_a010, p010_a020]
```

This is 36 candidates per model and 108 FeedTTA jobs across DUET, HAMT, and
GOAT. The `1e-6` level overlaps the parent grid so that cross-batch drift can be
distinguished from a genuine lower-LR effect. Three standard argmax Source jobs
and three sampled no-update controls make the executable refinement 114 jobs.
The lower rates test stability; they are not assumed to recover the performance
gap caused by FeedTTA's required sampled-action protocol.

Use the independent batch ID below, only after any active ATENA scheduler has
released the GPU:

```bash
cd /root/autodl-tmp/code/NavTTA
SPEC=vln/experiments/r2r_feedtta_low_lr_refinement_v1.json
BATCH=vln-r2r-feedtta-low-lr-refinement-v1-seed0
LAUNCH_DIR="$PWD/vln/results/logs/r2r/hparam_search/$BATCH/_launcher"

mkdir -p "$LAUNCH_DIR"
for METHOD in source feedtta_control feedtta; do
  /root/miniconda3/bin/python3 \
    vln/scripts/run_r2r_cartesian_hparam_search.py "$METHOD" \
    --spec "$SPEC" --batch-id "$BATCH" --gpu 0 --plan-only
done

screen -dmS navtta-feedtta-low-lr-v1 \
  env BATCH="$BATCH" SPEC="$SPEC" LAUNCH_DIR="$LAUNCH_DIR" bash -lc '
    cd /root/autodl-tmp/code/NavTTA || exit 97
    export PYTHONUNBUFFERED=1
    for METHOD in source feedtta_control feedtta; do
      /root/miniconda3/bin/python3 \
        vln/scripts/run_r2r_cartesian_hparam_search.py "$METHOD" \
        --spec "$SPEC" --batch-id "$BATCH" --gpu 0 --resume --fail-fast \
        >>"$LAUNCH_DIR/console.log" 2>&1 || exit $?
    done
  '
```

The generic campaign `--status` total is tied to the complete v2 campaign.
Judge this focused batch by `source`, `feedtta_control`, and `feedtta`
`progress.json`, followed by FeedTTA `SUMMARY.json` and `WINNER.json`.

## Legacy staged multi-benchmark search

`tta_hparam_search_v1.json` is the immutable design input for the five-model,
five-method `val_seen` hyperparameter search.  Search jobs use canonical order
seed 0.  After each setting freezes one configuration, that configuration is
run on exactly order seeds 0, 1, and 2.  The canonical seed-0 result is the
future main-table value; the three-order mean is a separate robustness result.

Raw consoles, job directories, checkpoints, and scheduler state belong below
`vln/results/logs/` and are ignored by Git.  Only compact summaries with full
run-manifest provenance may be added to tracked result files.

New campaigns use the same method/batch/job organization as the AVN search
logs.  Administrative evidence and raw model outputs are separated:

```text
vln/results/logs/hparam_search/<method>/<batch>/
  batch.json
  stages/<stage>/
    stage_manifest.json  grid.csv  metrics.csv  SUMMARY.json
    jobs/<setting>/<run-tag>/

vln/results/tuning/<method>/<batch>/<stage>/<setting>/<run-tag>/val_seen/
```

`batch.json` identifies the method, split, seed, Git commit, search-spec hash,
and result-layout version.  The run tag remains globally unique, while the
directory hierarchy makes the method, stage, and model/benchmark visible
without decoding the tag.  Migrated legacy controls shared by two searches
live under `vln/results/tuning/_shared/`; their migration ledger is stored in
`vln/results/tuning/_migrations/`.

The scheduler's default complete workflow ends after the five full canonical
`val_seen` finalists per setting and freezes the best configuration.  This
keeps the current hyperparameter search independent of the later robustness
section.  Add `--with-orders` to run the frozen configuration under exactly
order seeds 0, 1, and 2; that option requires matching `--order-seed` support
in the baseline runner.  Seed 0 consumes the unchanged canonical manifest;
seeds 1/2 consume the tracked, parent-pinned SHA256-ranked manifests.  The
runner also sets the model/runtime/formal-manifest seed to the order seed.
FeedTTA's action and SGR seeds follow it explicitly, while EAM replay consumes
the global model seed.  This never silently substitutes three ordinary RNG
seeds for three episode permutations.

`--order-seed` is deliberately unavailable to smoke/prefix jobs, Source-only
controls, StreamVLN, split `all`, and native CE v1.2.  The job config must say
`stage=orders`, `episodes=-1`, and carry the identical seed.  All screening,
final-control, final-selection, and frozen-winner decisions remain canonical
seed 0.

Typical invocations are:

```text
python3 vln/scripts/run_tta_hparam_search.py all --batch-id SEARCH_ID
python3 vln/scripts/run_tta_hparam_search.py all --batch-id SEARCH_ID --resume
python3 vln/scripts/run_tta_hparam_search.py all --batch-id SEARCH_ID --status
python3 vln/scripts/run_tta_hparam_search.py all --batch-id SEARCH_ID --watch
python3 vln/scripts/run_tta_hparam_search.py all --batch-id SEARCH_ID --with-orders
```

Each method is completed before the next method starts, while settings are
round-robin scheduled within a stage.  The default limits permit at most three
jobs overall, one job for any one model, and one continuous VLN-CE job.  These
caps, live GPU/RAM launch guards, and the launch stagger can be overridden only
after smoke measurements justify doing so.  A failed attempt is never erased:
`--resume --retry-failed` archives its console, result root, and any formal run
manifest before assigning a new run tag.

Newly planned run tags include the search method, so every method owns a
distinct tuning-result hierarchy even for identical argmax Source jobs in
`controls` and `final_controls`.  Retries always add `-retryN` to the recorded
base tag and remain under the same method/batch/stage/setting hierarchy.

`tta_adapter_parity_audit_v1.json` is a separate, post-search evidence
protocol. It consumes (but cannot modify or promote) the five frozen winner
files and fixes a 56-job canonical-prefix plan: 40 zero-write adapters, eight
argmax Source controls, and eight sampled Source controls. Its dedicated
runner flag, job schema, and `results/audits/adapter_parity/` namespace are
mandatory; ordinary search jobs cannot opt into audit mode. See
`vln/README.md` for the plan/run/validate commands and pass criteria.
