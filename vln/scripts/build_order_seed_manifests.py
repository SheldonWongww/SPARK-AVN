#!/usr/bin/env python3
"""Build the tracked VLN global-shuffle episode-order manifests.

Seed 0 is the existing canonical manifest and is never rewritten here.  Seeds
1, 2, and 3 are complete permutations of both validation splits derived from
those manifests, without reading the untracked annotation dataset.
"""

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = REPO_ROOT / "core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from navtta_core.experiment import (  # noqa: E402
    derive_seeded_episode_order_manifest,
    load_episode_order_manifest,
    sha256_file,
)


ORDER_ROOT = REPO_ROOT / "vln/manifests/episode_order"
FAMILIES = (
    "r2r_duet_hamt",
    "r2r_goat",
    "r2r_ce_v1_3_unified",
    "reverie_duet_hamt",
    "reverie_goat",
)
EVALUATION_SPLITS = ("val_seen", "val_unseen")
ORDER_SEEDS = (1, 2, 3)


def canonical_path(family, split="val_seen"):
    return ORDER_ROOT / family / "{}.json".format(split)


def derived_path(family, order_seed, split="val_seen"):
    return (
        ORDER_ROOT
        / "order_seed_{}".format(order_seed)
        / family
        / "{}.json".format(split)
    )


def expected_manifest(family, order_seed, split="val_seen"):
    if type(order_seed) is not int or order_seed not in ORDER_SEEDS:
        raise ValueError("order_seed must be the exact integer 1, 2, or 3")
    if split not in EVALUATION_SPLITS:
        raise ValueError(
            "split must be exactly val_seen or val_unseen"
        )
    parent_path = canonical_path(family, split)
    parent = load_episode_order_manifest(
        str(parent_path), expected_split=split
    )
    return derive_seeded_episode_order_manifest(
        parent,
        order_seed=order_seed,
        parent_manifest_path=parent_path.relative_to(REPO_ROOT).as_posix(),
        parent_manifest_sha256=sha256_file(str(parent_path)),
    )


def encoded_manifest(family, order_seed, split="val_seen"):
    document = expected_manifest(family, order_seed, split)
    return (
        json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def selected_pairs(family=None, order_seed=None):
    """Return the legacy family/seed matrix used by val-seen callers."""
    families = (family,) if family is not None else FAMILIES
    seeds = (order_seed,) if order_seed is not None else ORDER_SEEDS
    return [(item, seed) for seed in seeds for item in families]


def selected_manifests(family=None, order_seed=None, split=None):
    splits = (split,) if split is not None else EVALUATION_SPLITS
    return [
        (item, seed, selected_split)
        for item, seed in selected_pairs(family, order_seed)
        for selected_split in splits
    ]


def build(*, check=False, family=None, order_seed=None, split=None):
    mismatches = []
    for selected_family, selected_seed, selected_split in selected_manifests(
        family, order_seed, split
    ):
        destination = derived_path(
            selected_family, selected_seed, selected_split
        )
        expected = encoded_manifest(
            selected_family, selected_seed, selected_split
        )
        actual = destination.read_bytes() if destination.is_file() else None
        if actual == expected:
            print("ok {}".format(destination.relative_to(REPO_ROOT)))
            continue
        if check:
            mismatches.append(destination)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(expected)
        print("wrote {}".format(destination.relative_to(REPO_ROOT)))
    if mismatches:
        raise ValueError(
            "missing or stale derived manifests: {}".format(
                ", ".join(str(path.relative_to(REPO_ROOT)) for path in mismatches)
            )
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true",
        help="verify tracked outputs byte-for-byte without writing",
    )
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--order-seed", type=int, choices=ORDER_SEEDS)
    parser.add_argument("--split", choices=EVALUATION_SPLITS)
    return parser.parse_args()


def main():
    args = parse_args()
    build(
        check=args.check,
        family=args.family,
        order_seed=args.order_seed,
        split=args.split,
    )


if __name__ == "__main__":
    main()
