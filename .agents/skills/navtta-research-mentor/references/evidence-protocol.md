# Evidence protocol

Use this protocol to keep NavTTA research claims traceable, comparable, and falsifiable.

## Contents

1. Evidence order
2. Claim discipline
3. Mechanism comparison
4. Integrity checks
5. Decision standard

## Evidence order

Prefer evidence in this order for the question it can actually answer:

1. direct inspection of the current workspace, tracked manifests, configs, and executable code;
2. primary papers and supplements;
3. official repositories at recorded commits, issues, and author clarifications;
4. formal local results bound to a top-level Git commit, config, checkpoint digest, dataset version, seed, hardware, and run manifest;
5. secondary literature, informal notes, and legacy or incomplete-provenance results.

Use lower-ranked evidence to discover or qualify claims, not to silently overrule higher-ranked evidence. Keep metrics under `results/legacy/` out of formal comparisons. Treat `references/repos/` as read-only upstream evidence and record its URL and pinned commit.

For each citation, record an exact file and line, section/table/figure, URL and access date, commit, run ID, or manifest path. Record `unverified` when only a title, abstract, snippet, memory, or second-hand description is available.

Resolve local paths with `rg --files` or an equivalent filesystem query and preserve their exact case. Never synthesize a plausible path or cite a path that was not opened.

## Claim discipline

Classify every material claim as one of:

- `fact`: directly observable in a cited artifact;
- `prior-claim`: asserted by a cited source but not independently verified here;
- `local-observation`: measured in a provenance-complete local run;
- `inference`: reasoned from facts with assumptions stated;
- `hypothesis`: a falsifiable prediction not yet established.

Use `supported`, `mixed`, `contradicted`, or `open` for status. Never turn an inference into a fact by repeating it. Scope statements to the evaluated task, model, shift, seed, and stream.

## Mechanism comparison

Compare the proposed method and nearest alternatives on these axes:

- adaptation signal and its provenance;
- optimized objective and update target;
- step-, episode-, or stream-level update timing;
- state or memory carried across steps and episodes;
- reset boundary and ordering dependence;
- target feedback and supervision class;
- test-time compute, storage, and source-data requirements;
- architecture and action-space assumptions;
- causal availability of every input at decision time;
- claimed contribution and closest mechanistic predecessor.

Include Source and the strongest simple control. A new loss name is not a new mechanism if its signal, target, and update dynamics match existing work.

## Integrity checks

### Counter confirmation bias

Before recommending an idea:

1. record the strongest disconfirming source or result;
2. propose at least one credible rival mechanism;
3. design a negative control and an ablation that distinguish the rival;
4. state what result would reduce confidence or end the line;
5. preserve null and adverse results in the evidence package.

Prefer a cheap discriminating diagnostic over a large benchmark sweep that cannot identify why performance changed.

### Prevent split leakage

Inventory every learned value, statistic, threshold, prompt, checkpoint, hyperparameter, early-stop decision, and cherry-picked example. For each, record its source split and who or what selected it.

- Derive source statistics from a declared source-training split.
- Tune on a declared development split or development scenes.
- Freeze choices before final target evaluation.
- Do not use final-split metrics, trajectories, scene identities, or qualitative examples to choose a method.
- Declare whether adaptation state persists across episodes and whether episode order is fixed.
- Restart from the Source checkpoint at each protocol-defined independent stream.
- Treat post-episode information as unavailable to actions within that episode.

Online adaptation to an evaluation stream is not automatically leakage; undeclared selection on that stream is. State the allowed online observations and prohibit all others.

Do not invent a new split, scene partition, or hidden holdout as an immediate recommendation. Mark it as a proposed protocol change and send it to the experiment designer for feasibility, dataset-license, comparability, and sample-size review.

### Preserve supervision distinctions

Record all test-time signals, including indirect or delayed signals. At minimum distinguish:

- prediction-only or self-supervised target observations with no task feedback;
- source-data samples or source statistics available at test time;
- binary episode success or other delayed task feedback;
- target labels, oracle state, privileged simulator data, or human feedback.

Report fully unsupervised methods separately from feedback-dependent methods. Binary episode feedback remains supervision even when sparse, delayed, automatically computed, or used only for later episodes. Do not describe simulator success as self-supervision.

### Test novelty honestly

Search primary literature and official code for the closest mechanism, not only matching titles or task names. Decompose the claim into:

- algorithmic novelty;
- adaptation to a new navigation task, model, modality, or action space;
- evaluation protocol or benchmark contribution;
- mechanism analysis or empirical finding.

Record the search scope and cutoff date. Explain the nearest predecessor and a functional difference that could change outcomes. If the only verified difference is applying an existing method to AVN, VLN, or ObjectNav, call it a task adaptation or port. If novelty cannot be checked, mark the claim open rather than novel.

## Decision standard

Advance only when the hypothesis card identifies a measurable prediction that differs from a credible rival and can be evaluated without leakage or supervision ambiguity. Confidence must reflect source quality, contradictory evidence, and coverage—not enthusiasm or result magnitude.
