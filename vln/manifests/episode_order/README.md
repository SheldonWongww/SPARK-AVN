# Canonical VLN episode order

Formal online-TTA runs use one JSON manifest per benchmark and split.  Each
manifest has schema `navtta.episode_order.v1`, pins the exact loaded annotation
file by SHA256, and stores the complete ordered sequence as
`{episode_id, scene_id}` records.  The order policy is:

1. splits: `val_seen`, `val_unseen`, then `test`;
2. within a split: normalized scene ID, then natural episode/instruction ID;
3. one rank, one environment, and batch size one for stateful online TTA;
4. reset TTA state once at each split boundary;
5. consume exactly `episode_count` records without cycling or wrap-padding.

## Why the upstream order is not reused

The papers and released checkpoints do not publish a single cross-model
episode sequence.  The discrete DUET/HAMT/GOAT loaders inherit annotation-file
order and historically refill a final batch by wrapping to the beginning;
GOAT's instruction-dictionary update makes that sequence stateful.  The
ETPNav/BEVBERT Habitat iterator can group, shuffle, cycle, and interleave
scenes differently as the process/environment count changes.  StreamVLN sorts
scenes but strides episodes by distributed rank, so its effective order also
depends on world size.  Consequently, an exact paper-time sequence cannot be
reconstructed from the released source.

The canonical order above is therefore an explicit new experimental control,
not a claim about the private order used for published numbers.  It is shared
by all later Source and TTA reruns; published metrics remain labelled as legacy
references.

The discrete DUET/HAMT and GOAT annotations have identical scan, start
heading, path, and instruction content for R2R and REVERIE; only their tokenizer
encodings differ.  They are therefore strict episode-level comparisons after
the common ordering is applied.

R2R-CE needs an additional version control.  The public v1.2 and v1.3 files
have identical episode IDs, scenes, start positions, goals, reference paths,
and instruction text, but every episode has a different `start_rotation`.
Consequently, native ETPNav/BEVBert v1.2 and StreamVLN v1.3 results are not the
same experimental stream even though their order hashes match.  The default
cross-model protocol uses `r2r_ce_v1_3_unified`: v1.3 episode fields plus the
pinned legacy BERT token IDs required by ETPNav/BEVBert.  Build those ignored
runtime annotations with `build_r2r_ce_v1_3_bertidx.py`.  Native v1.2 remains
available only for upstream/paper reproduction and must be labelled by
version.  Stateful TTA results always require a canonical rerun; published
Source aggregates may be cited only under `results/legacy/`.

`test` manifests define submission order only.  Test annotations without public
goals must not be locally scored.

Generate a manifest after installing the shared package in the task
environment.  For example:

```sh
python vln/scripts/build_episode_order_manifest.py \
  --input vln/data/datasets/r2r/val_seen/val_seen.json.gz \
  --output vln/manifests/episode_order/r2r_vlnce_v1_3/val_seen.json \
  --benchmark r2r_vlnce_v1_3_streamvln \
  --split val_seen \
  --format habitat \
  --expected-count 778
```

Do not copy annotation content into a manifest beyond the two stable IDs.  The
source annotation itself remains an untracked data asset.

## Frozen-configuration order robustness

The primary experiment and every search/final-selection stage use the
canonical files above as order seed 0.  Those files are not rewritten or
copied.  After a method's full-`val_seen` winner is frozen, order seeds 1 and 2
use the ten tracked files below `order_seed_1/` and `order_seed_2/` for exactly
five manifest families: `r2r_duet_hamt`, `r2r_goat`,
`r2r_ce_v1_3_unified`, `reverie_duet_hamt`, and `reverie_goat`.

For each record, `sha256_rank_v1` hashes the canonical JSON object containing
the domain separator `navtta.episode_order.sha256_rank.v1`, order seed,
canonical parent's `order_sha256`, scene ID, and episode ID.  Records are
sorted by that digest, with scene/episode ID only as a deterministic collision
tie-breaker.  Each derived manifest preserves the annotation path/digest and
pins its canonical parent path, exact parent-file SHA256, and parent order
SHA256.  It is therefore an explicit permutation, not an implicit loader RNG.

Rebuild or verify the complete tracked set without reading annotation assets:

```sh
python3 vln/scripts/build_order_seed_manifests.py
python3 vln/scripts/build_order_seed_manifests.py --check
```

Nonzero order seeds are valid only for complete frozen-configuration
`val_seen` robustness jobs.  Smoke/prefix, Source-only, StreamVLN, native CE
v1.2, and multi-split jobs remain canonical seed-0 protocols.

## Active evaluator hooks

DUET, HAMT, and GOAT accept either a manifest directory or a path containing a
`{split}` placeholder:

```sh
--test --world_size 1 --batch_size 1 \
--eval_splits val_seen val_unseen test \
--episode_order_manifest /absolute/path/to/the/benchmark-manifest-directory
```

The manifest flag reorders the loaded instruction records and replaces the
upstream duplicate-detection loop with exactly one rollout per manifest entry.
Without this flag, upstream training and evaluation behavior is unchanged.

ETPNav and BEVBERT-CE use YACS overrides instead.  Formal cross-model runs use
`r2r_ce_v1_3_unified`; `r2r_ce_v1_2` is retained for native reproduction:

```text
GPU_NUMBERS 1 NUM_ENVIRONMENTS 1
EVAL.EPISODE_ORDER_MANIFEST /absolute/path/to/r2r_ce_v1_3_unified
```

For leaderboard inference, use
`INFERENCE.EPISODE_ORDER_MANIFEST` with the `test` manifest.  The active
VLN-CE dataset now respects the order of `EPISODES_ALLOWED`; the evaluator also
disables shuffle, scene grouping, and cycling and checks every current episode
against the manifest.

StreamVLN uses `--episode_order_manifest` and requires `--world_size 1`.
Online-TTA resume-by-skipping is rejected because restoring only the result
file does not restore adapter/optimizer state.  Start each formal split in an
empty output directory.  `val_seen` and `val_unseen` produce local metrics.
For `test`, the NavTTA loader supplies start-position placeholders only for the
fields required by Habitat's episode schema, disables all goal-based metrics,
and writes R2R-CE trajectories with `--submission_file`.  Those placeholders
must never be used to claim a local test score; the trajectory JSON must be
submitted to the official evaluation server.
