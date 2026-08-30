#!/usr/bin/env python3
"""Fail-closed verification of the native BEVBert Source asset pairing."""

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_MANIFEST = Path("vln/manifests/assets/eval_assets.json")
ORDER_MANIFEST_ROOT = Path("vln/manifests/episode_order/r2r_ce_v1_2")
PROTOCOL = "r2r_ce_v1_2_etpnav_bevbert"
MANIFEST_SCHEMA = "navtta.episode_order.v1"
ORDER_POLICY = "scene_id_then_natural_episode_id_v1"
SOURCE_ID_FIELD = "episode_id"
DEFAULT_CLIP_RUNTIME_PATH = Path(
    "/data1/wxy/exp_data/NavTTA/vln/cache/clip/ViT-B-16.pt"
)

WEIGHT_ASSETS = {
    "bevbert_ce_checkpoint": {
        "path": "vln/checkpoints/bevbert/ckpt.iter9600.pth",
        "source": "bevbert_ce_checkpoint",
        "size": 2551084857,
        "sha256": "70dfdfff153f9d54888215e492dae6bff69b674616f32c94a4d51538a67cc0e8",
    },
    "etpnav_waypoint_predictor": {
        "path": "vln/data/etpnav/wp_pred/check_cwp_bestdist_hfov90",
        "source": "etpnav_assets",
        "size": 201954133,
        "sha256": "09d0f42cbd801e05b0fa0212b901d409f033f4d0bce2fce1fa8a1331b502159f",
    },
    "ce_ddppo_depth_encoder": {
        "path": "vln/data/etpnav/ddppo-models/gibson-2plus-resnet50.pth",
        "source": "etpnav_assets",
        "size": 49853716,
        "sha256": "a6a600277efacf5fd98e293267221185d843eb3012aeff62fabfeee24c2bcdad",
    },
}

CLIP_ASSET = {
    "id": "openai_clip_vit_b16",
    "path": "/root/autodl-tmp/cache/clip/ViT-B-16.pt",
    "source": "openai_clip",
    "size": 350837078,
    "sha256": "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
}

SPLIT_SPECS = {
    "val_seen": {
        "asset_id": "etpnav_val_seen_bertidx",
        "path": (
            "vln/data/etpnav/datasets/"
            "R2R_VLNCE_v1-2_preprocessed_BERTidx/val_seen/"
            "val_seen_bertidx.json.gz"
        ),
        "episodes": 778,
        "ordinal": 0,
        "size": 758273,
        "sha256": "83b231674ae3a18cb02b3d65847377e978f45cc35682f807c03becc38b09410c",
        "gt": {
            "asset_id": "etpnav_val_seen_gt",
            "path": (
                "vln/data/etpnav/datasets/"
                "R2R_VLNCE_v1-2_preprocessed/val_seen/val_seen_gt.json.gz"
            ),
            "source": "etpnav_assets",
            "size": 482513,
            "sha256": "2c1df3c1f857b5f974192ea60da5d7dc71ccdd61cdfbdcd004258d5c57b2d0d0",
        },
    },
    "val_unseen": {
        "asset_id": "etpnav_val_unseen_bertidx",
        "path": (
            "vln/data/etpnav/datasets/"
            "R2R_VLNCE_v1-2_preprocessed_BERTidx/val_unseen/"
            "val_unseen_bertidx.json.gz"
        ),
        "episodes": 1839,
        "ordinal": 1,
        "size": 928029,
        "sha256": "7db0a39bef374cf5ce986473f8b35c9721f962c6c1a9af069257aa710df16365",
        "gt": {
            "asset_id": "etpnav_val_unseen_gt",
            "path": (
                "vln/data/etpnav/datasets/"
                "R2R_VLNCE_v1-2_preprocessed/val_unseen/val_unseen_gt.json.gz"
            ),
            "source": "etpnav_assets",
            "size": 1618489,
            "sha256": "1b9497dc1aa6ab68073976f9678ac42a154b393edd9d0b0450d0ac90841f684c",
        },
    },
}

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class PairingError(RuntimeError):
    """Raised when an installed asset does not match the frozen pairing."""


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _load_json(path, label):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairingError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(document, Mapping):
        raise PairingError("{} must contain a JSON object: {}".format(label, path))
    return document


def _require_string(value, label):
    if not isinstance(value, str) or not value:
        raise PairingError("{} must be a non-empty string".format(label))
    return value


def _require_sha256(value, label):
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise PairingError("{} must be a lowercase SHA256 digest".format(label))
    return value


def _require_positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PairingError("{} must be a positive integer".format(label))
    return value


def _asset_index(document):
    if document.get("schema_version") != 1:
        raise PairingError("eval_assets.json schema_version must be 1")
    assets = document.get("assets")
    if not isinstance(assets, list):
        raise PairingError("eval_assets.json assets must be a list")
    index = {}
    for position, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            raise PairingError("eval asset {} must be an object".format(position))
        asset_id = _require_string(asset.get("id"), "eval asset id")
        if asset_id in index:
            raise PairingError("duplicate eval asset id: {}".format(asset_id))
        index[asset_id] = asset
    return index


def _resolve(repo_root, path_text):
    path = Path(path_text)
    return path if path.is_absolute() else Path(repo_root) / path


def _verify_asset_spec(index, asset_id, expected):
    if asset_id not in index:
        raise PairingError("eval_assets.json is missing asset {}".format(asset_id))
    asset = index[asset_id]
    path_text = _require_string(asset.get("path"), "{}.path".format(asset_id))
    if path_text != expected["path"]:
        raise PairingError(
            "{} path mismatch: expected {}, got {}".format(
                asset_id, expected["path"], path_text
            )
        )
    if asset.get("source") != expected["source"]:
        raise PairingError(
            "{} source mismatch: expected {}, got {}".format(
                asset_id, expected["source"], asset.get("source")
            )
        )
    required_for = asset.get("required_for")
    if not isinstance(required_for, list) or "bevbert_ce" not in required_for:
        raise PairingError("{} is not declared required_for bevbert_ce".format(asset_id))
    manifest_size = _require_positive_int(
        asset.get("size"), "{}.size".format(asset_id)
    )
    manifest_sha256 = _require_sha256(
        asset.get("sha256"), "{}.sha256".format(asset_id)
    )
    if manifest_size != expected["size"]:
        raise PairingError(
            "{} frozen size mismatch: expected {}, manifest records {}".format(
                asset_id, expected["size"], manifest_size
            )
        )
    if manifest_sha256 != expected["sha256"]:
        raise PairingError(
            "{} frozen SHA256 mismatch: expected {}, manifest records {}".format(
                asset_id, expected["sha256"], manifest_sha256
            )
        )
    return asset


def _verify_installed_file(repo_root, asset_id, expected, path_override=None):
    path = (
        Path(path_override).expanduser()
        if path_override is not None
        else _resolve(repo_root, expected["path"])
    )
    if not path.is_file():
        raise PairingError("missing {}: {}".format(asset_id, path))
    try:
        actual_size = path.stat().st_size
    except OSError as error:
        raise PairingError("cannot stat {} {}: {}".format(asset_id, path, error))
    if actual_size != expected["size"]:
        raise PairingError(
            "{} size mismatch: expected {}, got {} ({})".format(
                asset_id, expected["size"], actual_size, path
            )
        )
    try:
        actual_sha256 = sha256_file(path)
    except OSError as error:
        raise PairingError("cannot hash {} {}: {}".format(asset_id, path, error))
    if actual_sha256 != expected["sha256"]:
        raise PairingError(
            "{} SHA256 mismatch: expected {}, got {} ({})".format(
                asset_id, expected["sha256"], actual_sha256, path
            )
        )
    return {
        "id": asset_id,
        "path": str(path),
        "size": actual_size,
        "sha256": actual_sha256,
    }


def _natural_key(value):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", str(value))
        if part
    )


def _record_key(record):
    return (_natural_key(record["scene_id"]), _natural_key(record["episode_id"]))


def _normalize_scene_id(value):
    if value is None:
        raise PairingError("episode scene_id cannot be null")
    raw = str(value).replace("\\", "/").rstrip("/")
    scene = Path(raw.rsplit("/", 1)[-1]).stem
    if not scene:
        raise PairingError("episode scene_id cannot be empty")
    return scene


def _canonical_records(episodes, label, exact_fields=False):
    if not isinstance(episodes, list):
        raise PairingError("{} episodes must be a list".format(label))
    records = []
    seen = set()
    for position, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise PairingError("{} episode {} must be an object".format(label, position))
        if exact_fields and set(episode) != {"episode_id", "scene_id"}:
            raise PairingError(
                "{} episode {} fields must be episode_id and scene_id".format(
                    label, position
                )
            )
        if "episode_id" not in episode or "scene_id" not in episode:
            raise PairingError(
                "{} episode {} is missing episode_id or scene_id".format(
                    label, position
                )
            )
        if episode["episode_id"] is None:
            raise PairingError("{} episode_id cannot be null".format(label))
        episode_id = str(episode["episode_id"])
        if not episode_id:
            raise PairingError("{} episode_id cannot be empty".format(label))
        record = {
            "episode_id": episode_id,
            "scene_id": _normalize_scene_id(episode["scene_id"]),
        }
        key = (record["scene_id"], record["episode_id"])
        if key in seen:
            raise PairingError(
                "{} contains duplicate episode {}/{}".format(label, *key)
            )
        seen.add(key)
        records.append(record)
    return sorted(records, key=_record_key)


def _records_sha256(records):
    encoded = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_dataset(path, label):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairingError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(document, Mapping):
        raise PairingError("{} root must be a JSON object".format(label))
    return document.get("episodes")


def _verify_split(repo_root, asset_index, split, spec):
    asset_id = spec["asset_id"]
    expected_dataset = dict(spec)
    expected_dataset["source"] = "etpnav_assets"
    asset = _verify_asset_spec(asset_index, asset_id, expected_dataset)
    if asset.get("episodes") != spec["episodes"]:
        raise PairingError(
            "{} episode count mismatch: expected {}, got {}".format(
                asset_id, spec["episodes"], asset.get("episodes")
            )
        )
    installed = _verify_installed_file(repo_root, asset_id, expected_dataset)

    gt_spec = spec["gt"]
    gt_asset_id = gt_spec["asset_id"]
    _verify_asset_spec(asset_index, gt_asset_id, gt_spec)
    installed_gt = _verify_installed_file(repo_root, gt_asset_id, gt_spec)

    manifest_path = Path(repo_root) / ORDER_MANIFEST_ROOT / (split + ".json")
    manifest = _load_json(manifest_path, "{} order manifest".format(split))
    fixed_fields = {
        "schema": MANIFEST_SCHEMA,
        "benchmark": PROTOCOL,
        "split": split,
        "split_ordinal": spec["ordinal"],
        "order_policy": ORDER_POLICY,
        "source_id_field": SOURCE_ID_FIELD,
    }
    for field, expected in fixed_fields.items():
        if manifest.get(field) != expected:
            raise PairingError(
                "{} manifest {} mismatch: expected {!r}, got {!r}".format(
                    split, field, expected, manifest.get(field)
                )
            )

    manifest_count = manifest.get("episode_count")
    if manifest_count != spec["episodes"]:
        raise PairingError(
            "{} manifest episode_count mismatch: expected {}, got {}".format(
                split, spec["episodes"], manifest_count
            )
        )
    manifest_records = _canonical_records(
        manifest.get("episodes"), "{} manifest".format(split), exact_fields=True
    )
    if manifest.get("episodes") != manifest_records:
        raise PairingError("{} manifest does not follow {}".format(split, ORDER_POLICY))
    if len(manifest_records) != manifest_count:
        raise PairingError("{} manifest episode list/count mismatch".format(split))
    order_sha256 = _records_sha256(manifest_records)
    if manifest.get("order_sha256") != order_sha256:
        raise PairingError("{} manifest order_sha256 mismatch".format(split))

    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise PairingError("{} manifest dataset metadata is missing".format(split))
    if dataset.get("path") != spec["path"] or dataset.get("path") != asset["path"]:
        raise PairingError("{} manifest/asset dataset path mismatch".format(split))
    dataset_sha256 = _require_sha256(
        dataset.get("sha256"), "{} manifest dataset.sha256".format(split)
    )
    if (
        dataset_sha256 != spec["sha256"]
        or dataset_sha256 != asset["sha256"]
        or dataset_sha256 != installed["sha256"]
    ):
        raise PairingError("{} manifest/asset dataset SHA256 mismatch".format(split))

    dataset_episodes = _load_dataset(
        Path(installed["path"]), "{} dataset".format(split)
    )
    dataset_records = _canonical_records(
        dataset_episodes, "{} dataset".format(split)
    )
    if len(dataset_records) != spec["episodes"]:
        raise PairingError(
            "{} dataset episode count mismatch: expected {}, got {}".format(
                split, spec["episodes"], len(dataset_records)
            )
        )
    if dataset_records != manifest_records:
        raise PairingError("{} dataset episodes do not match its order manifest".format(split))

    return {
        "split": split,
        "episodes": len(dataset_records),
        "dataset_path": installed["path"],
        "dataset_size": installed["size"],
        "dataset_sha256": dataset_sha256,
        "order_sha256": order_sha256,
        "gt_path": installed_gt["path"],
        "gt_size": installed_gt["size"],
        "gt_sha256": installed_gt["sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def verify_pairing(repo_root=REPO_ROOT, clip_path=DEFAULT_CLIP_RUNTIME_PATH):
    repo_root = Path(repo_root).expanduser().resolve()
    eval_assets_path = repo_root / ASSET_MANIFEST
    eval_assets = _load_json(eval_assets_path, "evaluation asset manifest")
    index = _asset_index(eval_assets)

    weights = []
    for asset_id, expected in WEIGHT_ASSETS.items():
        _verify_asset_spec(index, asset_id, expected)
        weights.append(_verify_installed_file(repo_root, asset_id, expected))

    clip_asset_id = CLIP_ASSET["id"]
    _verify_asset_spec(index, clip_asset_id, CLIP_ASSET)
    clip = _verify_installed_file(
        repo_root, clip_asset_id, CLIP_ASSET, path_override=clip_path
    )

    splits = []
    for split, spec in SPLIT_SPECS.items():
        splits.append(_verify_split(repo_root, index, split, spec))

    return {
        "protocol": PROTOCOL,
        "eval_assets_path": str(eval_assets_path),
        "eval_assets_sha256": sha256_file(eval_assets_path),
        "shared_checkpoint": "bevbert_ce_checkpoint",
        "shared_splits": list(SPLIT_SPECS),
        "weights": weights,
        "clip": clip,
        "splits": splits,
    }


def _print_report(report):
    for weight in report["weights"]:
        print(
            "PASS weight={} size={} sha256={}".format(
                weight["id"], weight["size"], weight["sha256"]
            )
        )
    print(
        "PASS runtime_asset={} size={} sha256={} path={}".format(
            report["clip"]["id"],
            report["clip"]["size"],
            report["clip"]["sha256"],
            report["clip"]["path"],
        )
    )
    for split in report["splits"]:
        print(
            "PASS split={} episodes={} dataset_sha256={} gt_sha256={} "
            "order_sha256={}".format(
                split["split"],
                split["episodes"],
                split["dataset_sha256"],
                split["gt_sha256"],
                split["order_sha256"],
            )
        )
    print(
        "BEVBERT SOURCE PAIRING OK protocol={} checkpoint={} shared_splits={}".format(
            report["protocol"],
            report["shared_checkpoint"],
            ",".join(report["shared_splits"]),
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Verify BEVBert's native R2R-CE v1.2 annotations, ground truth, "
            "order manifests, checkpoint, CLIP encoder, waypoint predictor, "
            "and depth encoder."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="NavTTA repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--clip-path",
        type=Path,
        default=DEFAULT_CLIP_RUNTIME_PATH,
        help=(
            "runtime OpenAI CLIP ViT-B/16 path "
            "(default: {})".format(DEFAULT_CLIP_RUNTIME_PATH)
        ),
    )
    args = parser.parse_args(argv)
    try:
        report = verify_pairing(args.repo_root, clip_path=args.clip_path)
    except PairingError as error:
        print("BEVBERT SOURCE PAIRING FAILED: {}".format(error), file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
