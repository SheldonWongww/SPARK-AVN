# NavTTA result audit protocol

## 1. Build the evidence ledger

For every planned job, record the plan identity, latest attempt, lifecycle status, compact metric path, validation marker, canonical manifest path/digest, and diagnostics path. Count expected, launched, completed, failed, metric-bearing, validated, and formal jobs independently.

Use the inventory helper before interpreting results:

```bash
python3 .agents/skills/navtta-result-auditor/scripts/manifest_inventory.py \
  TASK/results/runs --verify all --strict
```

For downloaded evidence whose manifests retain server-absolute paths:

```bash
python3 .agents/skills/navtta-result-auditor/scripts/manifest_inventory.py \
  TASK/results/runs --verify all --path-map /remote/NavTTA=/local/NavTTA
```

`--verify results` authenticates only compact result artifacts and can support a provisional local audit. Only `--verify all` can yield the helper's `formal` level because checkpoint, dataset, auxiliary, pinned-manifest, and result hashes must all be checked. Do not copy large assets into Git.

## 2. Authenticate a formal manifest

Derive expected values from the frozen spec, batch plan, and reviewed registry. Do not use the manifest itself as the source of truth for both expected and actual values.

Require all of the following:

1. The path is canonical under `<task>/results/runs/`, not `results/legacy/`, and any registry-pinned manifest SHA256 matches its bytes.
2. `status=completed`, `exit_code=0`, and start/completion timestamps are coherent.
3. Task, benchmark, model, method, source setting/split, seed, run tag, top-level Git commit, config/effective overrides, and attempt identity equal the frozen plan.
4. Recomputed `immutable_identity_sha256` matches the manifest. The immutable payload is the schema in `tools/run_manifest_identity.py`.
5. Checkpoint, auxiliary checkpoints, dataset index/version, episode stream order/content, and pinned asset/environment/order manifests have expected identity and actual size/SHA256.
6. Hardware and environment provenance are present. The recorded commit exists in the top-level repository; do not substitute a baseline submodule/reference commit.
7. `result_artifacts` is non-empty, names are unique, and every compact artifact matches its recorded size/SHA256.
8. `tools/validate_run_manifest.py` succeeds with `--require-immutable-identity --require-result-artifacts` and expected arguments sourced independently.

A self-consistent manifest can still describe the wrong experiment. Authentication therefore includes both cryptographic integrity and semantic agreement with the frozen plan.

## 3. Distinguish evidence levels

| Level | Minimum proof | Does not prove |
|---|---|---|
| process | live PID, scheduler state, lock, or exit marker | metric correctness or completion |
| metrics | parseable per-job/aggregate values and expected episode coverage | runner validation or provenance |
| validated | successful exit plus runner-specific checks for metrics, episodes, diagnostics, and effective config | complete formal provenance |
| formal | validated + canonical authenticated manifest + frozen-plan match | statistical generalization beyond the evaluated protocol |

Never infer a higher level from a lower one. A zero exit code is process evidence; `validation=ok` without its referenced artifacts is not formal evidence; a manifest path string is not the manifest.

## 4. Construct the matched Source key

Match at least:

- task and benchmark;
- navigation model and exact checkpoint SHA256 (including auxiliary checkpoints where relevant);
- source setting, evaluation split, simulator/evaluator protocol, action selection, and episode count;
- dataset version/index SHA256 and episode stream order/content SHA256;
- seed/order seed and any stochastic-action seed;
- frozen non-TTA controls and compatible code commit/environment.

Require exact equality unless the reviewed plan names a deliberate difference. If commits or hardware differ, establish protocol equivalence and report it; do not silently call the Source matched. Never use an upstream paper number, a legacy row, or a Source run from a different episode stream as the denominator.

Report the full match key and the Source manifest digest beside each delta. Keep Source and TTA metric units consistent; percentage-point deltas are not relative-percent deltas.

## 5. Detect no-op and drift

Read the effective parameters and diagnostics artifacts, not only the requested CLI.

For Source or an explicit zero-update control, require zero updates/slow updates, zero relative parameter drift, and equal before/after model hashes when those hashes are recorded. Any mutation invalidates the control.

For an adaptive method:

- require a non-empty expected trainable scope and positive adapted-parameter count;
- require eligible observations/steps and the method-specific update counters expected by the protocol;
- flag `updates=0`, all updates skipped, zero accepted samples when acceptance is required, or exact zero drift as a likely no-op;
- compare before/after parameter or model hashes when present; unchanged hashes with claimed updates are contradictory;
- require drift and loss/gradient diagnostics to be finite;
- apply only spec-defined drift/gradient/update/query caps; absent a cap, mark unusual values for review rather than inventing rejection thresholds;
- verify feedback/query counts and supervision labels for FeedTTA/ATENA, and do not present them as unsupervised.

A method can legitimately perform no updates on a particular stream only when the frozen protocol permits that outcome and diagnostics explain every skipped update. Such a run may be valid evaluation evidence, but it cannot support a claim that adaptation caused the gain.

Detect configuration drift by comparing the spec, expanded plan, per-job parameters, manifest overrides, and diagnostics. Normalize numeric/string representations before comparison. Check method, scope and parameter names/count, optimizer, learning rates, intervals/windows, reset state, action selection, feedback policy/budget, seed, split, episode count, and retry attempt. Hash the exact spec/config files where the campaign contract provides digests.

## 6. Reconcile and decide

Check that aggregate rows reproduce per-job metrics and that each metric row points to the same run tag and manifest identity. Flag reused stale metrics, duplicated run IDs with different identities, partial downloads, noncanonical recovered manifests, and selection performed on the evaluation split.

Use these decisions:

- **accept:** formal authentication passes, matched Source passes, diagnostics are active and within protocol, and metrics reconcile.
- **provisional:** useful metrics or validated evidence exists, but formal provenance, full file verification, or a matched Source is incomplete.
- **reject:** identity/hash/validation failure, wrong protocol, unmatched comparison, configuration drift, forbidden supervision, or unexplained no-op undermines the claim.

Keep rejected and incomplete-provenance values under the task's legacy/provisional reporting convention and out of formal tables.
