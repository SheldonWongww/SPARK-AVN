"""Deterministic immutable identity for NavTTA run manifests."""

import hashlib
import json


IMMUTABLE_IDENTITY_SHA256_FIELD = "immutable_identity_sha256"
IMMUTABLE_IDENTITY_FIELDS = (
    "run_id",
    "task",
    "benchmark",
    "model",
    "method",
    "run_tag",
    "source_setting",
    "seed",
    "git_commit",
    "config",
    "config_overrides",
    "checkpoint",
    "auxiliary_checkpoints",
    "dataset",
    "pinned_manifests",
    "hardware",
    "started_at",
)


def immutable_identity_payload(manifest):
    """Return the complete start-time identity in a fixed top-level schema."""
    return {field: manifest.get(field) for field in IMMUTABLE_IDENTITY_FIELDS}


def immutable_identity_sha256(manifest):
    """Hash canonical JSON for every immutable start-time identity field."""
    payload = json.dumps(
        immutable_identity_payload(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
