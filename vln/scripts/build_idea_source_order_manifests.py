#!/usr/bin/env python3
"""Build deterministic 128-trajectory train manifests for VLN IDEA.

The source annotation files stay outside Git.  Their provenance and expected
digests come from ``vln/manifests/assets/source_train_assets.json``; generated
subset manifests are written below the run output tree and are hash-pinned by
the formal collection manifest.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = REPO_ROOT / "core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from navtta_core.experiment import (  # noqa: E402
    build_episode_order_manifest,
    sha256_file,
    validate_episode_order_manifest,
)


TRAJECTORY_COUNT = 128
SELECTION_DOMAIN = "navtta.idea.source_train_128.sha256_rank.v1"
ASSET_MANIFEST = REPO_ROOT / "vln/manifests/assets/source_train_assets.json"

SETTINGS = {
    "duet-r2r": {
        "asset": "duet_hamt_r2r_train_bert",
        "path": "vln/data/duet/R2R/annotations/R2R_train_enc.json",
        "benchmark": "r2r_discrete_duet_hamt",
        "kind": "r2r",
    },
    "hamt-r2r": {
        "asset": "duet_hamt_r2r_train_bert",
        "path": "vln/data/hamt/R2R/annotations/R2R_train_enc.json",
        "benchmark": "r2r_discrete_duet_hamt",
        "kind": "r2r",
    },
    "goat-r2r": {
        "asset": "goat_r2r_train_roberta",
        "path": "vln/data/goat/R2R/annotations/R2R_train_roberta_enc.json",
        "benchmark": "r2r_discrete_goat",
        "kind": "r2r",
    },
    "duet-reverie": {
        "asset": "duet_hamt_reverie_train_bert",
        "path": "vln/data/duet/REVERIE/annotations/REVERIE_train_enc.json",
        "benchmark": "reverie_discrete_duet_hamt",
        "kind": "reverie",
    },
    "hamt-reverie": {
        "asset": "duet_hamt_reverie_train_bert",
        "path": "vln/data/hamt/REVERIE/annotations/REVERIE_train_enc.json",
        "benchmark": "reverie_discrete_duet_hamt",
        "kind": "reverie",
    },
    "goat-reverie": {
        "asset": "goat_reverie_train_roberta",
        "path": "vln/data/goat/REVERIE/annotations/REVERIE_train_roberta_enc.json",
        "benchmark": "reverie_discrete_goat",
        "kind": "reverie",
    },
    "etpnav-r2r-ce": {
        "asset": "r2r_ce_v1_3_train_bertidx",
        "path": (
            "vln/data/etpnav/datasets/"
            "R2R_VLNCE_v1-3_preprocessed_BERTidx/train/train_bertidx.json.gz"
        ),
        "benchmark": "r2r_ce_v1_3_unified_etpnav_bevbert",
        "kind": "r2r-ce",
    },
    "bevbert-r2r-ce": {
        "asset": "r2r_ce_v1_3_train_bertidx",
        "path": (
            "vln/data/etpnav/datasets/"
            "R2R_VLNCE_v1-3_preprocessed_BERTidx/train/train_bertidx.json.gz"
        ),
        "benchmark": "r2r_ce_v1_3_unified_etpnav_bevbert",
        "kind": "r2r-ce",
    },
}


def _read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _asset_record(manifest, asset_id, dataset_path):
    matches = []
    for record in manifest.get("assets", []):
        paths = record.get("paths", [record.get("path")])
        if record.get("id") == asset_id and dataset_path in paths:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            "source asset manifest does not uniquely bind {} at {}".format(
                asset_id, dataset_path
            )
        )
    return matches[0]


def _discrete_records(payload, kind):
    if not isinstance(payload, list):
        raise ValueError("discrete source annotation must be a JSON list")
    records = []
    for item in payload:
        if not isinstance(item, dict) or not item.get("scan"):
            raise ValueError("discrete source annotation contains a malformed path")
        instructions = item.get("instructions")
        if not isinstance(instructions, list) or not instructions:
            raise ValueError("discrete source path has no instructions")
        if kind == "reverie" and "objId" in item:
            prefix = "{}_{}".format(item["path_id"], item["objId"])
        elif kind == "reverie" and "id" in item:
            prefix = str(item["id"])
        else:
            prefix = str(item["path_id"])
        records.extend(
            {"episode_id": "{}_{}".format(prefix, index), "scene_id": item["scan"]}
            for index in range(len(instructions))
        )
    return records


def _continuous_records(payload):
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("continuous source annotation has no episodes")
    return [
        {"episode_id": str(item["episode_id"]), "scene_id": item["scene_id"]}
        for item in episodes
    ]


def _rank(setting, record):
    material = "\0".join((
        SELECTION_DOMAIN,
        setting,
        str(record["scene_id"]),
        str(record["episode_id"]),
    )).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def build_manifest(setting, repo_root=REPO_ROOT, asset_manifest=ASSET_MANIFEST):
    if setting not in SETTINGS:
        raise ValueError("unsupported IDEA source setting: {}".format(setting))
    repo_root = Path(repo_root)
    source_assets = _read_json(asset_manifest)
    descriptor = SETTINGS[setting]
    relative_path = descriptor["path"]
    source_record = _asset_record(
        source_assets, descriptor["asset"], relative_path
    )
    dataset_path = repo_root / relative_path
    if not dataset_path.is_file():
        raise FileNotFoundError("missing source-training dataset: {}".format(dataset_path))
    actual_sha256 = sha256_file(str(dataset_path))
    if (
        actual_sha256 != source_record.get("sha256")
        or dataset_path.stat().st_size != source_record.get("size")
    ):
        raise ValueError("source-training dataset does not match its asset manifest")
    payload = _read_json(dataset_path)
    records = (
        _continuous_records(payload)
        if descriptor["kind"] == "r2r-ce"
        else _discrete_records(payload, descriptor["kind"])
    )
    episode_ids = [str(item["episode_id"]) for item in records]
    if len(set(episode_ids)) != len(records):
        # SourceStatisticsAccumulator keys trajectories by their public episode
        # ID, so reject cross-scene ID reuse here instead of failing halfway
        # through a costly GPU collection run.
        raise ValueError("source-training trajectories do not have globally unique IDs")
    if len(records) < TRAJECTORY_COUNT:
        raise ValueError("source-training dataset has fewer than 128 trajectories")
    selected = sorted(records, key=lambda item: (_rank(setting, item), item["episode_id"]))[
        :TRAJECTORY_COUNT
    ]
    manifest = build_episode_order_manifest(
        selected,
        benchmark=descriptor["benchmark"],
        split="train",
        dataset_path=relative_path,
        dataset_sha256=actual_sha256,
        source_id_field=("episode_id" if descriptor["kind"] == "r2r-ce" else "instr_id"),
    )
    manifest["selection"] = {
        "algorithm": "sha256_rank_without_replacement_v1",
        "domain_separator": SELECTION_DOMAIN,
        "setting": setting,
        "population_count": len(records),
        "sample_count": TRAJECTORY_COUNT,
        "asset_manifest_path": Path(asset_manifest).resolve().relative_to(
            repo_root.resolve()
        ).as_posix(),
        "asset_manifest_sha256": sha256_file(str(asset_manifest)),
        "asset_id": descriptor["asset"],
    }
    validate_episode_order_manifest(
        manifest, expected_split="train",
        expected_benchmark=descriptor["benchmark"],
    )
    return manifest


def write_manifest(setting, output, *, check=False, repo_root=REPO_ROOT,
                   asset_manifest=ASSET_MANIFEST):
    manifest = build_manifest(setting, repo_root, asset_manifest)
    encoded = (json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n").encode("utf-8")
    output = Path(output)
    if check:
        if not output.is_file() or output.read_bytes() != encoded:
            raise ValueError("missing or stale IDEA source order manifest: {}".format(output))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_bytes(encoded)
        os.replace(str(temporary), str(output))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True, choices=tuple(SETTINGS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    manifest = write_manifest(args.setting, args.output, check=args.check)
    print(json.dumps({
        "setting": args.setting,
        "output": str(Path(args.output).resolve()),
        "episode_count": manifest["episode_count"],
        "order_sha256": manifest["order_sha256"],
        "dataset_sha256": manifest["dataset"]["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
