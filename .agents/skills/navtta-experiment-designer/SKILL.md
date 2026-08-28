---
name: navtta-experiment-designer
description: Design, review, supersede, or freeze auditable NavTTA experiment specifications and search plans. Use when defining AVN, VLN, or ObjectNav splits, episode order and model seeds, stream horizons, update dose, matched Source reuse, feedback supervision, hyperparameter selection, freeze boundaries, or leakage controls before experiments run.
---

# NavTTA Experiment Designer

Produce a decision-complete specification that another operator can execute without inventing scientific choices. Do not launch jobs or interpret completed results as part of this skill.

## Resolve scope and lifecycle

1. Resolve the repository with `git rev-parse --show-toplevel`, then read its `AGENTS.md`, the target task's `README.md` and `STATUS.md`, and all active specs that overlap the proposed decision.
2. Confirm task activation from the current request and repository policy. Do not infer it from existing code, results, or a stale status note.
3. Search for the campaign before creating a spec. For the canonical schema, exactly one version is `active`; retain replaced files as `superseded`, add `superseded_by`, and make the successor list them in `supersedes`. Do not rewrite a superseded protocol as if it had been run.
4. Read [references/experiment-spec-template.md](references/experiment-spec-template.md) completely and use it as a design-contract checklist. When a launcher already has a task-local schema, preserve that schema and map every checklist invariant into it; never create a parallel JSON format that the launcher cannot execute.

## Lock the scientific protocol

- Separate split identity from split role. Tune only on an authorized development split; freeze selection before unseen, robustness, transfer, or test evaluation.
- Record episode-order seeds and model-initialization seeds as different fields. Also declare deterministic derivations for action sampling, augmentation, replay, and method RNGs.
- Define horizon independently from update dose. Horizon states how much environment stream is consumed; dose states triggers, interval, optimizer steps, backward passes, batch/replay samples, and maximum updates.
- Pair every method with the correct Source action protocol and episode stream. Reuse Source only when a validated manifest is content-addressed and all declared matching dimensions agree; otherwise schedule a rerun. Sampling controls require matching RNG semantics.
- Label runtime supervision as `unsupervised`, `pseudo_label`, or `feedback_supervised`. Declare every environment field, provider, timing, delay, noise, and budget. Binary episode success is feedback supervision and must be reported separately from unsupervised TTA.
- Give every method its own supervision and feedback contract. If a campaign mixes methods, do not apply one campaign-level supervision label to all of them.
- Freeze winners, selection rules, hashes, and immutable fields before evaluation. Explicitly forbid evaluation metrics, hidden fields, and result-dependent grid expansion from influencing selection.
- Tie every numeric candidate to a paper/official anchor, an authenticated prior run, or an explicitly labeled scale probe. Do not invent a plausible grid from memory.
- Select task-native metrics and thresholds separately for AVN, discrete VLN, and continuous VLN. Never transplant SPL or another benchmark's metric as a universal objective.

## Lint and review

Run the stdlib-only, read-only JSON summarizer/linter for a new canonical design-contract JSON:

```bash
python3 .agents/skills/navtta-experiment-designer/scripts/lint_experiment_spec.py \
  <task>/experiments/<spec>.json
python3 .agents/skills/navtta-experiment-designer/scripts/lint_experiment_spec.py \
  <task>/experiments/<spec>.json --format json --strict
```

The linter writes only to stdout/stderr. Fix every error; review warnings rather than mechanically suppressing them. It validates the bundled canonical design contract, not every historical launcher schema. For an existing task-local spec, use that launcher's own validator, plan-only mode, and tests, then record the checklist mapping in the accompanying plan.

## Handoff gate

Return the active spec path and digest, supersession chain, hypothesis, exact matrix size, split and seed table, horizon and dose, Source decision, supervision class, freeze point, leakage controls, stop rules, expected manifests, and unresolved blockers. Hand an approved spec to the experiment operator; never promote smoke or development outputs to a formal table.
