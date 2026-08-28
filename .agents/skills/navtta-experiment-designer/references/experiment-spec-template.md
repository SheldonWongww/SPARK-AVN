# Experiment design-contract template

Use this contract to review new experiment families. It is directly usable only when the task launcher adopts this schema. Otherwise map its fields into the launcher's established schema and retain the mapping in the plan; do not create an unexecutable parallel spec. Keep the scientific contract separate from launcher state and results. The companion linter accepts JSON so it can remain Python-stdlib-only.

## Contents

1. Lifecycle rules
2. Canonical JSON template
3. Field semantics
4. Review checklist

## 1. Lifecycle rules

- Give every scientific revision a new immutable `spec_id` and file.
- Use one contract per method or per supervision-homogeneous group. A multi-method launcher may aggregate contracts, but it must retain each method's distinct feedback fields, timing, and budget.
- Give every numeric candidate and threshold a provenance field: paper/official anchor, authenticated prior run, or predeclared scale probe. A plausible value recalled from memory is not provenance.
- Define primary, secondary, safety, and mechanism metrics per task. Do not use a VLN metric such as SPL as an AVN-wide selection rule without an explicit task decision.
- Keep exactly one `active` spec for a campaign decision. An active spec has `superseded_by: null`.
- Preserve an old file with `status: superseded` and a non-empty `superseded_by`; never launch it. The replacement lists the old IDs in `supersedes`.
- Do not use lifecycle status to encode run progress. Put scheduling and completion state in the run registry or batch artifacts.
- Hash the frozen config and evidence inputs. Changing a scientific invariant requires a successor spec, not a convenient command-line override.
- When extending a legacy task-local schema, map its fields to this contract in the review rather than bulk-rewriting historical specs.

## 2. Canonical JSON template

Replace every `REPLACE_*` value. Hashes are lowercase SHA256 hex digests. Use lists even for one model or seed so later expansion remains explicit.

```json
{
  "schema_version": 1,
  "spec_id": "avn-example-method-development-v1",
  "status": "active",
  "supersedes": [],
  "superseded_by": null,
  "purpose": {
    "phase": "development",
    "hypothesis": "REPLACE_WITH_FALSIFIABLE_HYPOTHESIS",
    "decision": "REPLACE_WITH_DECISION_THIS_EXPERIMENT_ENABLES"
  },
  "scope": {
    "task": "avn",
    "benchmark": "REPLACE_BENCHMARK",
    "models": ["smt_audio"],
    "methods": ["REPLACE_METHOD"],
    "source_setting": "single_source"
  },
  "data": {
    "dataset_manifest": "avn/data/manifests/datasets.yaml",
    "dataset_sha256": "REPLACE_WITH_64_HEX_CHARACTERS",
    "split": {
      "name": "val",
      "role": "development"
    },
    "episode_order": {
      "policy": "fixed_manifest",
      "manifest": "REPLACE_EPISODE_ORDER_MANIFEST",
      "sha256": "REPLACE_WITH_64_HEX_CHARACTERS",
      "order_seeds": [0]
    },
    "horizon": {
      "unit": "episodes",
      "value": 2000,
      "max_steps_per_episode": 500,
      "truncation": "environment_native"
    }
  },
  "randomness": {
    "model_seeds": [0],
    "action_rng": "derive_from_run_seed_and_episode_id",
    "adaptation_rng": "independent_derive_from_run_seed",
    "augmentation_rng": "independent_derive_from_run_seed",
    "replay_rng": "independent_derive_from_run_seed",
    "determinism_notes": "REPLACE_WITH_BACKEND_AND_REPLAY_LIMITS"
  },
  "protocol": {
    "processes": 1,
    "environments": 1,
    "batch_size": 1,
    "causal_online": true,
    "action": {
      "selection": "argmax",
      "invalid_action_mask": "apply_before_loss_and_selection",
      "stop_semantics": "task_native"
    },
    "adaptation": {
      "persistence": "continual_across_episodes",
      "reset_boundary": "new_run",
      "trainable_scope": "REPLACE_EXACT_SELECTOR_AND_EXPECTED_COUNTS",
      "optimizer": "REPLACE_OPTIMIZER_AND_ALL_OPTIONS",
      "update_dose": {
        "trigger_unit": "actions",
        "interval": 1,
        "optimizer_steps_per_trigger": 1,
        "backward_passes_per_trigger": 1,
        "samples_per_step": 1,
        "max_optimizer_steps": 2000
      }
    },
    "feedback": {
      "supervision": "unsupervised",
      "signal": "none",
      "environment_fields": [],
      "provider": "none",
      "timing": "none",
      "delay_episodes": 0,
      "budget": {
        "unit": "queries",
        "maximum": 0
      },
      "noise": "none",
      "affects": "none"
    }
  },
  "source_control": {
    "mode": "reuse",
    "manifest": "REPLACE_VALIDATED_SOURCE_RUN_MANIFEST",
    "manifest_sha256": "REPLACE_WITH_64_HEX_CHARACTERS",
    "matched_on": [
      "checkpoint_sha256",
      "dataset_sha256",
      "split",
      "episode_order_sha256",
      "model_seed",
      "action_protocol",
      "evaluator",
      "horizon"
    ],
    "reuse_reason": "REPLACE_WITH_VERIFIED_IDENTITY_OR_ORDER_INVARIANCE_EVIDENCE",
    "rerun_on_mismatch": true
  },
  "selection": {
    "candidate_count": 1,
    "metric": "REPLACE_PRIMARY_METRIC",
    "constraints": ["no_numeric_failure", "no_unauthorized_feedback"],
    "tie_breakers": ["REPLACE_PREDECLARED_TIE_BREAKER"],
    "order_seed_aggregation": "mean_over_all_declared_development_order_seeds",
    "source_constraint": "REPLACE_MATCHED_SOURCE_FLOOR_OR_NA"
  },
  "freeze": {
    "before_evaluation": true,
    "frozen_from": "REPLACE_SELECTION_ARTIFACT_OR_PREDECLARED_CONFIG",
    "config_sha256": "REPLACE_WITH_64_HEX_CHARACTERS",
    "immutable_fields": [
      "data",
      "randomness",
      "protocol",
      "source_control",
      "selection"
    ],
    "allowed_runtime_overrides": ["gpu_ids", "max_concurrency", "resume"]
  },
  "leakage_controls": {
    "allowed_selection_inputs": ["development_metrics", "predeclared_diagnostics"],
    "forbidden_selection_inputs": ["evaluation_metrics", "hidden_test", "future_episode_feedback"],
    "feedback_field_allowlist": [],
    "selection_rule_frozen_before_evaluation": true,
    "posthoc_grid_expansion_forbidden": true,
    "state_transfer_across_splits": false
  },
  "stopping": {
    "infrastructure_retry": "same_manifest_only",
    "algorithmic_failure_is_result": true,
    "abort_conditions": ["NaN_or_Inf", "unauthorized_field_access"]
  },
  "outputs": {
    "run_manifest_required": true,
    "paired_episode_metrics_required": true,
    "diagnostics": ["attempted_updates", "accepted_updates", "parameter_drift"]
  },
  "provenance": {
    "git_commit": "REPLACE_FULL_GIT_COMMIT",
    "checkpoint_sha256": "REPLACE_WITH_64_HEX_CHARACTERS",
    "created_by": "REPLACE_OWNER",
    "created_at": "REPLACE_ISO_8601_TIMESTAMP"
  }
}
```

## 3. Field semantics

### Split and seeds

`data.split.name` identifies the physical split; `role` states what decisions it may influence. Do not relabel an unseen or test split as development. `order_seeds` control episode permutation only. `model_seeds` identify model initialization/checkpoint randomness. Keep action, adaptation, augmentation, and replay RNG derivations separate so adding one stochastic component does not perturb the others.

### Horizon and update dose

Horizon is consumed environment exposure: episode/action count, per-episode cap, and truncation. Update dose is compute and mutation exposure: trigger unit, interval, optimizer steps, backward passes, samples/replays, and total cap. Report both; equal learning rates or episode counts do not imply equal adaptation dose.

### Source reuse

Reuse only a validated, content-addressed Source manifest. Match checkpoint, dataset, physical split, episode identities/order, model seed, action selection and RNG, evaluator, and horizon. For a proven order-invariant deterministic Source, record the proof in `reuse_reason`; never assume invariance. If any required dimension differs or evidence is incomplete, set `mode: rerun` and schedule the matched Source.

### Feedback supervision

Use `unsupervised` only when no evaluator-only outcome enters selection, loss, query logic, reset, or memory. Pseudo-labels derived solely from allowed model inputs use `pseudo_label`. Any success/failure, reward, distance-to-goal, oracle action, human response, or delayed task outcome makes the protocol `feedback_supervised`. Define whether feedback affects the completed episode (normally forbidden) or only future actions/episodes, and report its budget.

### Freeze and leakage

Freeze the winner and selection rule before evaluation. Hash the exact resolved config, list immutable fields, and restrict runtime overrides to operational choices that cannot change predictions. Never tune on unseen/test metrics, move adapted state between splits unless explicitly studied, select a favorable order seed, omit failed candidates, or extend a grid after observing evaluation outcomes.

## 4. Review checklist

- Confirm the candidate count equals the declared Cartesian/explicit matrix and includes the paper anchor when applicable.
- Confirm every method begins from the same complete checkpoint and a fresh process for each split/order seed.
- Confirm Source action selection, stochastic seed semantics, stream, evaluator, and horizon are matched.
- Confirm update timing is causal and episode feedback affects only future episodes.
- Confirm trainable tensor names/counts, persistent state, reset boundaries, and update dose are measurable.
- Confirm feedback methods are labeled and compared within the correct supervision group.
- Confirm smoke, development, transfer, robustness, and formal outputs have distinct roles and paths.
- Confirm every formal run will record Git commit, resolved config, checkpoint/data/order digests, all seeds, and hardware.
- Confirm failures are retained; only infrastructure faults may retry against the identical manifest.
