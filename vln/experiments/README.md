# VLN experiment specifications

## R2R model-wise direct Cartesian full-val search

`r2r_modelwise_cartesian_hparam_v2.json` is the immutable design input for the
active DUET/HAMT/GOAT R2R search.  It is intentionally separate from the older
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
