# Research artifact templates

Copy these structures into the response or a task-local research note. Keep identifiers stable across revisions.

## Claim ledger

| ID | Atomic claim | Class | Scope | Supporting evidence and exact locator | Counterevidence | Status | Confidence | Verification needed |
|---|---|---|---|---|---|---|---|---|
| C1 |  | fact / prior-claim / local-observation / inference / hypothesis | task, model, shift, protocol |  |  | supported / mixed / contradicted / open | high / medium / low |  |

Rules:

- Split compound claims into separate rows.
- Give every inference its premises and every number its provenance.
- Use `open` when a primary artifact has not been inspected.

## Mechanism matrix

| Method | Adaptation signal | Objective | Updated component | Timing | Memory and reset | Supervision | Source-data need | Key assumption | Nearest prior and functional difference |
|---|---|---|---|---|---|---|---|---|---|
| Source | none | none | none | none | protocol reset | none | checkpoint | fixed policy is reference | reference |
| Proposed |  |  |  | step / episode / stream |  |  |  |  |  |
| Closest baseline |  |  |  |  |  |  |  |  |  |
| Simple control |  |  |  |  |  |  |  |  |  |

Add rows for each mechanistically close baseline. Explicitly note information that would be unavailable when an action is chosen.

## Contradiction log

| ID | Challenged claim | Disconfirming evidence or rival explanation | Exact locator | Consequence if true | Resolving check | State |
|---|---|---|---|---|---|---|
| X1 | C1 |  |  |  |  | open / resolved / accepted limitation |

Do not delete resolved contradictions. Append the resolution and evidence so the decision history remains auditable.

## Hypothesis card

### Identity

- **Title / version:**
- **Decision owner and date:**
- **Task / model / shift:**
- **Evidence cutoff:**
- **Contribution class:** algorithm / task adaptation / protocol / analysis / combination

### Mechanism

- **Problem:**
- **Proposed causal mechanism:**
- **Closest predecessor and functional delta:**
- **Required assumptions:**
- **Adaptation signal, timing, update target, memory, and reset:**
- **Supervision class:**

### Predictions

- **Primary falsifiable prediction:**
- **Prediction that distinguishes the strongest rival:**
- **Expected failure regime:**
- **Metric, direction, and minimum meaningful effect:**

### Protocol

- **Source checkpoint and provenance:**
- **Dataset versions and split roles:**
- **Tuning boundary and frozen choices:**
- **Baselines and simple controls:**
- **Negative control:**
- **Mechanism ablation:**
- **Seeds / stream order / reset policy:**
- **Compute and stopping budget:**

### Decision rules

- **Advance if:**
- **Revise if:**
- **Reject or stop if:**
- **Leakage checks:**
- **Unresolved contradictions:**
- **Cheapest next test:**

### Handoff

- **Verdict:** advance / revise / reject / blocked
- **Required next skill:**
- **Input artifact locators:**
- **Open decisions and owner:**
