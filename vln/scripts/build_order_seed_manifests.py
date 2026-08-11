#!/usr/bin/env python3
"""Build the tracked VLN val-seen order-robustness manifests.

Seed 0 is the existing canonical manifest and is never rewritten here.  Seeds
1 and 2 are complete permutations derived from that manifest, without reading
the untracked annotation dataset.
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
ORDER_SEEDS = (1, 2)


def canonical_path(family):
    return ORDER_ROOT / family / "val_seen.json"


def derived_path(family, order_seed):
    return ORDER_ROOT / "order_seed_{}".format(order_seed) / family / "val_seen.json"


def expected_manifest(family, order_seed):
    if type(order_seed) is not int or order_seed not in ORDER_SEEDS:
        raise ValueError("order_seed must be the exact integer 1 or 2")
    parent_path = canonical_path(family)
    parent = load_episode_order_manifest(
        str(parent_path), expected_split="val_seen"
    )
    return derive_seeded_episode_order_manifest(
        parent,
        order_seed=order_seed,
        parent_manifest_path=parent_path.relative_to(REPO_ROOT).as_posix(),
        parent_manifest_sha256=sha256_file(str(parent_path)),
    )


def encoded_manifest(family, order_seed):
    document = expected_manifest(family, order_seed)
    return (
        json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def selected_pairs(family=None, order_seed=None):
    families = (family,) if family is not None else FAMILIES
    seeds = (order_seed,) if order_seed is not None else ORDER_SEEDS
    return [(item, seed) for seed in seeds for item in families]


def build(*, check=False, family=None, order_seed=None):
    mismatches = []
    for selected_family, selected_seed in selected_pairs(family, order_seed):
        destination = derived_path(selected_family, selected_seed)
        expected = encoded_manifest(selected_family, selected_seed)
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
    return parser.parse_args()


def main():
    args = parse_args()
    build(
        check=args.check,
        family=args.family,
        order_seed=args.order_seed,
    )


if __name__ == "__main__":
    main()
