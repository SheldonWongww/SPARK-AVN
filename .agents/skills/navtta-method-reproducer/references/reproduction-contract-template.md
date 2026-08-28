# Method reproduction contract template

Copy this template into the target task's tracked experiment or method-review area. Replace every angle-bracket placeholder; use `N/A — <reason>` rather than deleting an inapplicable field.

## Contents

1. Identity and evidence
2. Paper / official / port decision ledger
3. Ownership boundary
4. Deployment semantics
5. Replay contract
6. Parity evidence
7. Deviations, gates, and handoff

## 1. Identity and evidence

| Field | Value |
|---|---|
| Contract ID and status | `<method>-<task>-port-v<version>` / `draft \| blocked \| parity_passed` |
| Method and claimed category | `<name>` / `unsupervised \| pseudo-label \| feedback-supervised` |
| Paper | `<title, authors, venue/year, URL or DOI>` |
| Paper artifact digest | `<path and SHA256, or public identifier>` |
| Official repository | `<URL>` |
| Official pinned revision | `<full commit SHA>` |
| License | `<SPDX/name and retained license path>` |
| Local upstream reference | `references/repos/<path>` (read-only) |
| NavTTA target | `<avn \| vln \| objectnav>/<model>` |
| Port implementation | `<task-relative paths>` |
| Contract owner/date | `<owner>` / `<YYYY-MM-DD>` |

List every secondary implementation, issue, author clarification, or missing artifact. Never infer “official” from repository popularity.

## 2. Paper / official / port decision ledger

Add one row for every behavior that can affect predictions, updates, supervision, or resource use.

| Semantic item | Paper claim with page/equation | Official code with path:line and revision | NavTTA port | Classification | Decision and consequence |
|---|---|---|---|---|---|
| Input and preprocessing | `<...>` | `<...>` | `<...>` | `exact \| task mapping \| ambiguity \| deviation` | `<...>` |
| Objective and reduction | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| Trainable parameters | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| Optimizer and state | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| Update trigger and steps | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| Reset/persistence | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| Memory/replay | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| Action selection | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| Feedback/query | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |

State the precedence used for each ambiguity. A paper/official mismatch is evidence to report, not permission to blend favorable details. If the port changes a defining behavior, assign a variant name and preserve the aligned baseline.

## 3. Ownership boundary

| Component | Location | Reason | Forbidden dependency/write |
|---|---|---|---|
| Task-agnostic algorithm | `core/navtta_core/...` | `<...>` | `Habitat/simulator/task-policy imports` |
| Task adapter and model mapping | `<task>/...` | `<...>` | `changes under references/repos/` |
| Task config and launcher | `<task>/...` | `<...>` | `cross-task assets or environments` |
| Provenance manifest | `<task>/.../manifests/...` | `<...>` | `dataset/checkpoint binaries in Git` |

Declare whether the task is currently active. For an isolated task, stop unless an explicit activation decision is cited.

## 4. Deployment semantics

### Stream and lifecycle

- Evaluation unit and batch size: `<single action / episode / scene; batch size>`
- Causal order: `<observe -> infer -> act -> optional update>`
- Update unit, interval, and steps: `<...>`
- Continual/reset policy: `<weights, optimizer, teacher/anchor, memory, counters>`
- Episode reset policy: `<recurrent/map/simulator state>`
- Start state: `<checkpoint digest and any source preparation>`

### Action contract

- Environment action space: `<fixed/dynamic, action IDs, low/high-level mapping>`
- Invalid-action mask timing: `<before loss and selection>`
- STOP definition and terminal handling: `<...>`
- Selection: `<argmax / categorical sample / beam / other>`
- RNG: `<generator, seed derivation, consumption order>`
- Matched Source: `<manifest/run ID using identical protocol>`

An adaptation update must not influence the action whose data created that update. If method-native action selection differs from the repository default, add a Source control with the same selection rule.

### Supervision and feedback contract

- Supervision class: `<unsupervised / pseudo-label / feedback-supervised>`
- Allowed runtime fields: `<observations and declared feedback fields>`
- Forbidden evaluator fields: `<success/reward/distance/oracle action/etc. as applicable>`
- Feedback signal, provider, timing, delay, noise, and budget: `<...>`
- Credit assignment: `<which past data receive the signal>`
- Query/self-feedback rule: `<...>`

For binary episode feedback, define “success” for the target task and ensure it arrives only after termination. Report feedback-supervised methods separately from unsupervised methods.

### Parameters and mutable state

| State group | Exact keys/selector | Trainable or mutable? | Reset boundary | Hash/audit method |
|---|---|---:|---|---|
| Deployed model parameters | `<...>` | `<...>` | `<...>` | `<...>` |
| Persistent buffers | `<BatchNorm, queues, counters, ...>` | `<...>` | `<...>` | `<...>` |
| Optimizer | `<type and hyperparameters>` | `<...>` | `<...>` | `<...>` |
| Teacher/source/slow branch | `<...>` | `<...>` | `<...>` | `<...>` |
| Replay/memory | `<...>` | `<...>` | `<...>` | `<...>` |

Record selected tensor names, tensor count, scalar count, gradient coverage, and exclusions. Do not claim an encoder is updated when the task uses frozen precomputed features.

## 5. Replay contract

- Replay unit and payload: `<action-step / episode; observations, masks, hidden state, action, logits, RNG state, metadata>`
- Copy/detach/device policy: `<...>`
- Admission, eviction, sampling, and ordering: `<...>`
- Reconstructed call path: `<...>`
- Stochastic modules and RNG restoration: `<dropout/augmentation/sampling>`
- Mutable buffers and recurrent/map state restoration: `<...>`
- Online-versus-replay comparison: `<logits/features/loss/gradients; tolerance>`

If the stored payload cannot reconstruct the original forward pass, describe it as approximate replay and test the approximation. Never claim exact replay solely because losses are close.

## 6. Parity evidence

| Gate | Fixture and command | Expected invariant | Evidence artifact | Result |
|---|---|---|---|---|
| Formula/unit parity | `<...>` | `<loss, mask, reduction, schedule>` | `<...>` | `pass/fail` |
| Official anchor | `<...>` | `<official outputs/gradients>` | `<...>` | `<...>` |
| Matched Source | `<...>` | `<checkpoint, stream, seeds, action, evaluator>` | `<...>` | `<...>` |
| Zero-write | `<...>` | `<0 actual writes; identical full-state hashes>` | `<...>` | `<...>` |
| Action/trajectory parity | `<...>` | `<exact IDs and termination>` | `<...>` | `<...>` |
| Replay parity | `<...>` | `<declared error <= tolerance>` | `<...>` | `<...>` |

### Zero-write procedure

1. Start Source and audit runs from the same complete checkpoint and process boundary.
2. Execute the real adapter forward, loss, backward, memory, gate, and query paths.
3. Intercept optimizer steps and all direct parameter or persistent-buffer writes.
4. Hash every deployed parameter and persistent buffer before and after; include source, auxiliary, teacher/slow, and dynamically created modules.
5. Compare per-step actions, episode trajectories, termination, metrics, RNG counters, attempted writes, and suppressed writes.
6. Require zero actual writes and exact full-state hash equality. Explain every behavioral mismatch before proceeding.

## 7. Deviations, gates, and handoff

| Issue | Evidence | Scientific impact | Resolution/variant label | Owner |
|---|---|---|---|---|
| `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |

- Allowed next stage: `<blocked / unit tests / smoke / development search / frozen evaluation>`
- Required tests: `<paths and commands>`
- Required diagnostics: `<updates, skips, loss, gradient/update norm, drift, memory/query/replay statistics>`
- Formal-run prerequisites: `<clean commit, config, checkpoint/data/order hashes, seeds, hardware, run manifest>`
- Remaining non-equivalences: `<explicit list>`
