"""Validation and dataset binding for AVN IDEA source collection."""

import hashlib
import json
import os
import re
from pathlib import Path


SCHEMA = "navtta.avn.idea_source_selection.v2"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scene_id(value):
    raw = getattr(value, "scene_id", value)
    return os.path.splitext(str(raw).replace("\\", "/").rstrip("/").split("/")[-1])[0]


def stable_episode_id(episode):
    return "{}/{}".format(scene_id(episode), str(episode.episode_id))


def load_source_manifest(path, expected_sha256, model=None, source_setting=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("IDEA source manifest is missing: {}".format(path))
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(expected_sha256)):
        raise ValueError("IDEA source manifest SHA256 is required")
    actual = file_sha256(path)
    if actual.lower() != str(expected_sha256).lower():
        raise ValueError("IDEA source manifest SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("task") != "avn":
        raise ValueError("invalid AVN IDEA source manifest schema")
    if payload.get("benchmark") != "mp3d":
        raise ValueError("IDEA source manifest benchmark must be mp3d")
    if model is not None and payload.get("model") != model:
        raise ValueError("IDEA source manifest model mismatch")
    if source_setting is not None and payload.get("source_setting") != source_setting:
        raise ValueError("IDEA source manifest source-setting mismatch")
    dataset = payload.get("dataset", {})
    checkpoint = payload.get("checkpoint", {})
    selection = payload.get("selection", {})
    if dataset.get("split") != "train" or dataset.get("version") != "v1":
        raise ValueError("IDEA source manifest must bind AVN v1 train")
    for value, label in (
        (dataset.get("bundle_sha256"), "dataset bundle"),
        (checkpoint.get("sha256"), "checkpoint"),
        (payload.get("protocol_sha256"), "protocol"),
        (payload.get("episode_order_sha256"), "episode order"),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
            raise ValueError("invalid {} SHA256".format(label))
    if selection.get("count") != 128 or payload.get("episode_count") != 128:
        raise ValueError("canonical IDEA source manifest requires 128 episodes")
    if selection.get("action_selection") != "sample":
        raise ValueError("IDEA source collection must use native sampled actions")
    if canonical_sha256(selection) != payload["protocol_sha256"]:
        raise ValueError("IDEA source protocol SHA256 mismatch")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 128:
        raise ValueError("IDEA source episode list is incomplete")
    if canonical_sha256(episodes) != payload["episode_order_sha256"]:
        raise ValueError("IDEA source episode-order SHA256 mismatch")
    ids = [str(record.get("trajectory_id", "")) for record in episodes]
    if any(not value for value in ids) or len(set(ids)) != 128:
        raise ValueError("IDEA source trajectory ids are invalid")
    return payload, actual


def select_manifest_episodes(episodes, manifest):
    indexed = {}
    for episode in episodes:
        key = stable_episode_id(episode)
        if key in indexed:
            raise ValueError("duplicate AVN source episode {}".format(key))
        indexed[key] = episode
    ordered = []
    for record in manifest["episodes"]:
        key = str(record["trajectory_id"])
        episode = indexed.get(key)
        if episode is None:
            raise ValueError("IDEA source episode is missing: {}".format(key))
        if scene_id(episode) != str(record["scene_id"]):
            raise ValueError("IDEA source scene id mismatch: {}".format(key))
        ordered.append(episode)
    if len(ordered) != 128:
        raise ValueError("IDEA source selection did not yield 128 episodes")
    return ordered


def verify_manifest_assets(manifest, dataset_index_path, checkpoint_path):
    """Verify the exact source annotation bundle and frozen checkpoint bytes."""
    checkpoint_actual = file_sha256(checkpoint_path)
    if checkpoint_actual != manifest["checkpoint"]["sha256"]:
        raise ValueError("IDEA source checkpoint SHA256 mismatch")
    dataset_root = Path(dataset_index_path).resolve().parent
    actual_files = []
    for record in manifest["dataset"].get("files", []):
        relative = str(record.get("path", ""))
        path = dataset_root / relative
        if not relative or not path.is_file():
            raise FileNotFoundError("IDEA source dataset member is missing: {}".format(path))
        digest = file_sha256(path)
        if digest != record.get("sha256"):
            raise ValueError("IDEA source dataset member SHA256 mismatch: {}".format(path))
        actual_files.append({"path": relative, "sha256": digest})
    if not actual_files:
        raise ValueError("IDEA source dataset bundle is empty")
    if canonical_sha256(actual_files) != manifest["dataset"]["bundle_sha256"]:
        raise ValueError("IDEA source dataset bundle SHA256 mismatch")


def collection_provenance(manifest, manifest_sha256, model_state_sha256):
    for value, label in (
        (manifest_sha256, "episode selection manifest"),
        (model_state_sha256, "effective model state"),
    ):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(value)):
            raise ValueError("{} SHA256 is required".format(label))
    return {
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "checkpoint_path": manifest["checkpoint"]["path"],
        # This binds the actual post-load actor, including ENMuS' separately
        # loaded audio/visual/SELD encoders, rather than trusting the primary
        # checkpoint filename alone.
        "model_state_sha256": str(model_state_sha256).lower(),
        "dataset": manifest["dataset"]["name"],
        "dataset_version": manifest["dataset"]["version"],
        "dataset_sha256": manifest["dataset"]["bundle_sha256"],
        "split": manifest["dataset"]["split"],
        "episode_selection_manifest_sha256": str(manifest_sha256).lower(),
        "episode_order_sha256": manifest["episode_order_sha256"],
        "selection_protocol_sha256": manifest["protocol_sha256"],
        "selection_protocol": dict(manifest["selection"]),
        "action_selection": manifest["selection"]["action_selection"],
        "model": manifest["model"],
        "source_setting": manifest["source_setting"],
    }
