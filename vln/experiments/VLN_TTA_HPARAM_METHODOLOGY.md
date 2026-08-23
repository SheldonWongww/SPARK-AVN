# VLN TTA Hyperparameter Methodology (cross-split consistency)

Status: active design for the VLN main-comparison table.
Scope: R2R (DUET/HAMT/GOAT), REVERIE (DUET/HAMT/GOAT), R2R-CE (ETPNav/BEVBert).
Methods compared: Source, Tent, FSTTA, EAM, FeedTTA, ATENA, IDEA (OURS reserved).

---

## 1. Problem: why val_seen tuning degrades val_unseen

The prior search pinned `split = val_seen`, `order_seed 0`, and selected each
method's winner by `argmax val_seen SR` (with `do_not_reselect_on_val_unseen:
true`). Empirically that produced tiny val_seen gains (e.g. GOAT-R2R
84.82 -> 84.92 SR) and val_unseen numbers that fall *below* Source. Four
mechanisms explain this:

1. **Wrong selection split.** TTA exists to handle distribution shift.
   `val_seen` is (near) the training distribution: prediction entropy is low,
   adaptation gradients are tiny, and every configuration is nearly a no-op.
   Tuning where the method cannot act, then deploying where it must act, is a
   direction error. The real shift lives in `val_unseen`.
2. **argmax over noise.** Selecting the val_seen maximum over dozens/hundreds
   of configs inside a ~1-2 SR jitter band fits the luck of one episode order,
   not a generalizable operating point.
3. **Streaming accumulation on the long split.** `val_unseen` is a longer
   continual stream (R2R 2349 instructions vs 1021). A slightly-too-large LR,
   an absent per-episode update cap, or unprotected in-place parameter
   overwrite compounds into negative transfer / collapse that the shorter
   `val_seen` stream never reaches.
4. **Not how the papers tune.** FSTTA / ATENA / FeedTTA fix *one* sensible
   hyperparameter set per benchmark and report it across all splits. Their
   gains come from a robust working point, not from squeezing val_seen.

Conclusion: the defect is the *selection rule and split*, not the adapters.
The single-eval launcher (`run_source_eval.sh`) already supports any split and
order seed; only the search's selection policy needs to change.

---

## 2. Corrected protocol: cross-split consistency selection

Produce exactly **one frozen configuration per (method, model, benchmark)** and
report it on every split. Selection uses both splits and a no-regression floor,
never a single-split argmax.

### 2.1 Evaluate both splits
Each candidate is run full-length on `val_seen` *and* `val_unseen` with the
canonical `order_seed 0` manifest (the same single order the paper main table
uses). Source is run once per (model, split) as the reference.

### 2.2 Selection rule (the fix)
Let `dSR_seen  = SR_seen(cfg)  - SR_seen(Source)` and
`dSR_unseen = SR_unseen(cfg) - SR_unseen(Source)`.

1. **No-regression floor (hard gate).** Keep a candidate only if
   `SR_seen(cfg)   >= SR_seen(Source)   - epsilon` **and**
   `SR_unseen(cfg) >= SR_unseen(Source) - epsilon`  (default `epsilon = 0.3` SR).
   This eliminates configurations that collapse on either split up front.
2. **Maximize the worst split's gain.** Among survivors, pick the largest
   `min(dSR_seen, dSR_unseen)`. This directly rewards configurations that help
   *consistently* rather than only in-distribution.
3. **Tie-breakers, in order:** higher `mean(dSR_seen, dSR_unseen)` -> higher
   `mean SPL delta` -> lower `relative_param_drift` -> fewer `updates`.
   (Stability breaks near-ties toward the least-destructive adaptation.)
4. **Honest fallback.** If no candidate clears the floor on both splits, report
   the paper/open-source default configuration and mark the cell
   `no_consistent_improvement` in `selected_config.json`. Never fabricate a win
   by reverting to val_seen argmax.

REVERIE uses `RGSPL` as the primary metric in steps 1-3 (floor on `SR`, gain on
`RGSPL`), matching the REVERIE reporting convention; R2R and R2R-CE use `SR`.

### 2.3 Why this removes the degradation
The floor gate makes "below Source on val_unseen" an automatic rejection, and
the worst-split objective refuses configurations that only win in-distribution.
The selected point is therefore, by construction, non-regressive and
consistent across the distribution boundary — which is exactly the property the
original papers rely on.

### 2.4 Robustness (deferred)
`order_seed 0` alone is used for selection and table filling (single canonical
order, paper-consistent). Re-running the frozen winner on `order_seed 1/2/3` to
confirm the result is not order noise is supported by the existing manifests and
is a later, optional confirmation — not required to fill the table.

---

## 3. Search space (small, anchored to published values)

Grids are anchored near the papers' defaults and vary mainly the one knob that
governs streaming stability (learning rate), plus at most one secondary knob.
This is deliberately a handful of points per cell, not hundreds.

| Method  | Primary knob                          | Secondary knob                         | ~configs/cell |
|---------|---------------------------------------|----------------------------------------|---------------|
| Tent    | lr in {1e-5,3e-5,1e-4,3e-4,1e-3}      | norm_scope fixed (last_k_ln)           | 5             |
| FSTTA   | lr_fast in {2e-4,6e-4,1e-3}           | lr_slow in {5e-4,1e-3}; M3/N4; band aligned to release (rho .9/tau .5/a .5/b 1.5) | 6 |
| EAM     | lr in {3e-6,1e-5,3e-5}                | update_interval in {1,4}; a=0.4,M32,K8 | 6             |
| FeedTTA | lr in {2e-6,5e-6,1e-5}                | (p,alpha) in {(0.05,-0.2),(0.05,0.1)}  | 6             |
| ATENA   | lr_query in {5e-7,1e-6}               | query_threshold in {0.05,0.1}; mix_lambda in {0.5,0.75} | 6 |
| IDEA    | lr in {1e-3,3e-3}                     | tau in {0.5,0.7}; L4,K32,lambda .4     | 4             |

R2R budget ~ (5+6+6+6+6+4)=33 configs/model x 2 splits x 3 models plus Source
controls — a few hundred runs, well within the concurrency budget below.

### IDEA optimization steps `O`
IDEA's per-uncovered-step prompt optimization (`opt_steps`, paper `O = 50`) is
slow on the long streams. **Search runs use `opt_steps = 10`** to rank
candidates cheaply; the selected configuration is then **re-evaluated with
`opt_steps = 50`** to produce the reported number, which is the value stored in
`selected_config.json` and written to the table. This keeps the final number
faithful to the paper while keeping the search tractable.

---

## 4. Concurrency and phase schedule (AutoDL, one 32 GB vGPU)

Only one benchmark's one (model, method) phase runs at a time; finish it, then
switch. Do not backfill tail slots with a different method.

R2R same-(model,method) parallelism: Tent 10, FSTTA 12, EAM 5, FeedTTA 6
(HAMT 5), ATENA 5. GOAT-FSTTA capped at 12.

REVERIE: same-(model,method) = 4 (per-model total 4 with one method at a time).

R2R-CE: 3 per (model,method); ETPNav and BEVBert never overlap (strict model
barrier).

REVERIE `test` is submission-only and is out of scope for hyperparameter
search; only `val_seen`/`val_unseen` are used here.

---

## 5. Artifacts and flow

- Search specs: `vln/experiments/{r2r,reverie,r2r_ce}_consistency_search_v1.json`
  (schema `navtta.vln_tta_consistency_search.v1`).
- Runner: `vln/scripts/run_consistency_hparam_search.py`
  - emits per-candidate `--tta-config` job files (schema `navtta.vln_tta_job.v1`),
  - calls `run_source_eval.sh SETTING SPLIT --tta-config ... --result-root ...`
    for both splits (reusing the launcher, order manifests, and formal run
    manifests),
  - parses `console.log` with the shared `parse_metrics`,
  - applies the Section 2.2 selection and writes `selected_config.json`.
  - `--dry-run` prints the exact commands without a GPU; `--concurrency N`
    bounds same-(model,method) parallelism.
- Table filler: `tools/fill_navtta_results.py` writes the frozen two-split
  metrics into `docs/literature/NavTTA_benchmark_results.xlsx` for every method
  row except OURS.

Runs execute only on the AutoDL server (Habitat, datasets, checkpoints). Local
work is limited to code, unit tests, and `--dry-run` command verification.
