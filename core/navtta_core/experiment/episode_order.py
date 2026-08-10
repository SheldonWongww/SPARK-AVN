"""Canonical episode-order manifests for reproducible online evaluation.

The helpers in this module deliberately know nothing about Habitat, MatterSim,
or a particular navigation task.  A caller supplies episode-like objects (or
dictionaries); the default field lookup merely covers the names commonly used
by the active baselines.

An online TTA run must not silently reshuffle an evaluation split or pad its
last batch by wrapping to the start.  The manifest pins both the source dataset
bytes and the exact sequence consumed by the model.
"""

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence


EPISODE_ORDER_SCHEMA = "navtta.episode_order.v1"
CANONICAL_SPLIT_ORDER = ("val_seen", "val_unseen", "test")
CANONICAL_ORDER_POLICY = "scene_id_then_natural_episode_id_v1"


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of ``path`` without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _field(value: Any, names: Sequence[str]) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    raise ValueError(
        "Episode has none of the required fields: {}".format(
            ", ".join(names)
        )
    )


def episode_id(value: Any) -> str:
    """Return the stable ID used by an evaluator for an episode/instruction."""
    return str(_field(value, ("episode_id", "instr_id", "instruction_id")))


def normalize_scene_id(value: Any) -> str:
    """Normalize a scan ID or a Habitat scene path to its portable scene ID."""
    raw = str(_field(value, ("scene_id", "scan"))).replace("\\", "/")
    raw = raw.rstrip("/")
    scene = os.path.splitext(raw.rsplit("/", 1)[-1])[0]
    if not scene:
        raise ValueError("Episode has an empty scene ID")
    return scene


def _natural_key(value: str) -> tuple:
    """Return a type-stable natural-sort key (``2`` precedes ``10``)."""
    parts = re.split(r"(\d+)", str(value))
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part != ""
    )


def _record_key(record: Mapping[str, Any]) -> tuple:
    return (
        _natural_key(str(record["scene_id"])),
        _natural_key(str(record["episode_id"])),
    )


def canonical_episode_records(episodes: Iterable[Any]) -> List[Dict[str, str]]:
    """Build unique records ordered by scene and then natural episode ID."""
    records = [
        {
            "episode_id": episode_id(item),
            "scene_id": normalize_scene_id(item),
        }
        for item in episodes
    ]
    seen = set()
    duplicates = []
    for record in records:
        key = (record["scene_id"], record["episode_id"])
        if key in seen:
            duplicates.append("{}/{}".format(*key))
        seen.add(key)
    if duplicates:
        raise ValueError(
            "Duplicate canonical episode keys: {}".format(
                ", ".join(sorted(set(duplicates)))
            )
        )
    return sorted(records, key=_record_key)


def _records_sha256(records: Sequence[Mapping[str, str]]) -> str:
    payload = json.dumps(
        list(records), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_episode_order_manifest(
    episodes: Iterable[Any],
    *,
    benchmark: str,
    split: str,
    dataset_path: str,
    dataset_sha256: str,
    source_id_field: str,
) -> Dict[str, Any]:
    """Return a self-validating canonical order manifest.

    ``dataset_path`` should be repository-relative (or otherwise portable),
    while ``dataset_sha256`` must describe the exact annotation file consumed
    by the evaluator, including compression when the input is compressed.
    """
    if split not in CANONICAL_SPLIT_ORDER:
        raise ValueError(
            "Unsupported evaluation split {!r}; expected one of {}".format(
                split, CANONICAL_SPLIT_ORDER
            )
        )
    if not re.fullmatch(r"[0-9a-fA-F]{64}", dataset_sha256):
        raise ValueError("dataset_sha256 must be a 64-character hex digest")
    records = canonical_episode_records(episodes)
    return {
        "schema": EPISODE_ORDER_SCHEMA,
        "benchmark": str(benchmark),
        "split": split,
        "split_ordinal": CANONICAL_SPLIT_ORDER.index(split),
        "order_policy": CANONICAL_ORDER_POLICY,
        "source_id_field": str(source_id_field),
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256.lower(),
        },
        "episode_count": len(records),
        "order_sha256": _records_sha256(records),
        "episodes": records,
    }


def validate_episode_order_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_split: Optional[str] = None,
    expected_benchmark: Optional[str] = None,
) -> None:
    """Raise ``ValueError`` if a manifest is malformed or internally changed."""
    if manifest.get("schema") != EPISODE_ORDER_SCHEMA:
        raise ValueError("Unsupported episode-order manifest schema")
    split = manifest.get("split")
    if split not in CANONICAL_SPLIT_ORDER:
        raise ValueError("Manifest contains an unsupported split")
    if expected_split is not None and split != expected_split:
        raise ValueError(
            "Manifest split {!r} does not match requested split {!r}".format(
                split, expected_split
            )
        )
    if manifest.get("split_ordinal") != CANONICAL_SPLIT_ORDER.index(split):
        raise ValueError("Manifest split_ordinal is inconsistent")
    if manifest.get("order_policy") != CANONICAL_ORDER_POLICY:
        raise ValueError("Manifest contains an unsupported order policy")
    benchmark = manifest.get("benchmark")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("Manifest benchmark is missing")
    if expected_benchmark is not None and benchmark != expected_benchmark:
        raise ValueError(
            "Manifest benchmark {!r} does not match expected benchmark {!r}".format(
                benchmark, expected_benchmark
            )
        )
    source_id_field = manifest.get("source_id_field")
    if not isinstance(source_id_field, str) or not source_id_field:
        raise ValueError("Manifest source_id_field is missing")
    records = manifest.get("episodes")
    if not isinstance(records, list):
        raise ValueError("Manifest episodes must be a list")
    normalized = []
    seen = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Every manifest episode must be an object")
        if set(record) != {"episode_id", "scene_id"}:
            raise ValueError(
                "Manifest episode fields must be episode_id and scene_id"
            )
        normalized_record = {
            "episode_id": str(record["episode_id"]),
            "scene_id": str(record["scene_id"]),
        }
        key = (
            normalized_record["scene_id"],
            normalized_record["episode_id"],
        )
        if key in seen:
            raise ValueError("Manifest contains duplicate episode keys")
        seen.add(key)
        normalized.append(normalized_record)
    if manifest.get("episode_count") != len(normalized):
        raise ValueError("Manifest episode_count is inconsistent")
    if normalized != sorted(normalized, key=_record_key):
        raise ValueError("Manifest episodes do not follow the declared order policy")
    if manifest.get("order_sha256") != _records_sha256(normalized):
        raise ValueError("Manifest order_sha256 does not match episodes")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("Manifest dataset metadata is missing")
    if not dataset.get("path"):
        raise ValueError("Manifest dataset path is missing")
    digest = str(dataset.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Manifest dataset SHA256 is invalid")


def load_episode_order_manifest(
    path: str,
    *,
    expected_split: Optional[str] = None,
    expected_benchmark: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and validate an episode-order manifest."""
    with open(path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    validate_episode_order_manifest(
        manifest,
        expected_split=expected_split,
        expected_benchmark=expected_benchmark,
    )
    return manifest


def resolve_episode_order_manifest_path(spec: str, split: str) -> str:
    """Resolve a manifest directory or a ``{split}`` path template."""
    if not spec:
        raise ValueError("episode-order manifest path is empty")
    if os.path.isdir(spec):
        return os.path.join(spec, split + ".json")
    try:
        return spec.format(split=split)
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Invalid episode-order manifest template {!r}: {}".format(spec, error)
        )


def verify_manifest_dataset(
    manifest: Mapping[str, Any], dataset_path: str
) -> None:
    """Verify the exact annotation bytes pinned by ``manifest``."""
    validate_episode_order_manifest(manifest)
    actual = sha256_file(dataset_path)
    expected = manifest["dataset"]["sha256"]
    if actual != expected:
        raise ValueError(
            "Dataset SHA256 mismatch for {}: expected {}, got {}".format(
                dataset_path, expected, actual
            )
        )


def reorder_episodes(
    episodes: Sequence[Any], manifest: Mapping[str, Any]
) -> List[Any]:
    """Return exactly the manifest episodes, in order, with no extras/missing IDs."""
    validate_episode_order_manifest(manifest)
    indexed = {}
    for item in episodes:
        key = (normalize_scene_id(item), episode_id(item))
        if key in indexed:
            raise ValueError("Loaded dataset contains duplicate episode key {}".format(key))
        indexed[key] = item

    ordered = []
    missing = []
    for record in manifest["episodes"]:
        key = (record["scene_id"], record["episode_id"])
        item = indexed.pop(key, None)
        if item is None:
            missing.append("{}/{}".format(*key))
        else:
            ordered.append(item)
    if missing or indexed:
        extras = ["{}/{}".format(*key) for key in sorted(indexed)]
        raise ValueError(
            "Episode manifest/dataset mismatch; missing=[{}], extra=[{}]".format(
                ", ".join(missing), ", ".join(extras)
            )
        )
    return ordered


def prefix_episode_order_manifest(
    manifest: Mapping[str, Any], episode_count: int
) -> Dict[str, Any]:
    """Return a self-consistent prefix for non-formal lifecycle smoke tests."""
    validate_episode_order_manifest(manifest)
    count = int(episode_count)
    if count <= 0 or count > manifest["episode_count"]:
        raise ValueError(
            "Smoke episode count must be in [1, {}]".format(
                manifest["episode_count"]
            )
        )
    prefix = dict(manifest)
    prefix["episodes"] = list(manifest["episodes"][:count])
    prefix["episode_count"] = count
    prefix["order_sha256"] = _records_sha256(prefix["episodes"])
    validate_episode_order_manifest(prefix)
    return prefix


def select_allowed_episodes_in_order(
    episodes: Sequence[Any], allowed_episode_ids: Sequence[Any]
) -> List[Any]:
    """Filter episodes while preserving the caller's explicit ID sequence.

    Missing IDs are ignored because Habitat first partitions a dataset by
    scene and then applies the same global ``EPISODES_ALLOWED`` list inside
    each worker.  Duplicate IDs in the loaded partition are still an error.
    """
    indexed = {}
    for item in episodes:
        identifier = episode_id(item)
        if identifier in indexed:
            raise ValueError(
                "Loaded dataset contains duplicate episode ID {!r}".format(
                    identifier
                )
            )
        indexed[identifier] = item
    return [
        indexed[str(identifier)]
        for identifier in allowed_episode_ids
        if str(identifier) in indexed
    ]


def configure_exact_episode_env(
    env: Any, manifest: Mapping[str, Any]
) -> None:
    """Install a manifest order on a simple batched evaluator environment.

    DUET/HAMT/GOAT expose ``data``, ``batch_size``, ``ix``, and
    ``reset_epoch``.  Requiring batch size one lets their unmodified minibatch
    method consume the final episode without entering its wrap-padding branch.
    The exact agent loop below then performs precisely ``episode_count``
    rollouts.
    """
    validate_episode_order_manifest(manifest)
    if int(getattr(env, "batch_size", 0)) != 1:
        raise ValueError(
            "Canonical online evaluation requires batch_size=1; got {}".format(
                getattr(env, "batch_size", None)
            )
        )
    if not hasattr(env, "data") or not hasattr(env, "reset_epoch"):
        raise ValueError("Environment does not expose the exact-evaluation protocol")
    ordered_data = reorder_episodes(list(env.data), manifest)
    runtime_manifest = manifest
    smoke_count = os.environ.get("NAVTTA_SMOKE_EPISODES")
    if smoke_count:
        runtime_manifest = prefix_episode_order_manifest(manifest, int(smoke_count))
        ordered_data = ordered_data[: runtime_manifest["episode_count"]]
    env.data = ordered_data
    env.ix = 0
    env._navtta_episode_order = list(runtime_manifest["episodes"])


def run_exact_agent_epoch(
    agent: Any, rollout_fn: Any, result_transform: Optional[Any] = None
) -> bool:
    """Run a DUET-style agent once per configured manifest record.

    Return ``False`` when the environment has no manifest, allowing upstream
    evaluation behavior to remain unchanged.  When active, validate both the
    environment batch and emitted trajectory before recording it.
    """
    env = agent.env
    records = getattr(env, "_navtta_episode_order", None)
    if records is None:
        return False

    env.reset_epoch(shuffle=False)
    agent.losses = []
    agent.results = {}
    agent.loss = 0
    tta_adapter = getattr(agent, "tta_adapter", None)
    if tta_adapter is not None:
        tta_adapter.reset()
    for ordinal, expected in enumerate(records):
        trajectories = list(rollout_fn())
        batch = list(getattr(env, "batch", []))
        if len(batch) != 1 or len(trajectories) != 1:
            raise RuntimeError(
                "Exact episode {} expected one batch item and one trajectory; "
                "got {} and {}".format(ordinal, len(batch), len(trajectories))
            )
        actual = {
            "episode_id": episode_id(batch[0]),
            "scene_id": normalize_scene_id(batch[0]),
        }
        if actual != expected:
            raise RuntimeError(
                "Exact episode {} order mismatch: expected {}, got {}".format(
                    ordinal, expected, actual
                )
            )
        trajectory = trajectories[0]
        trajectory_id = str(_field(trajectory, ("instr_id", "episode_id")))
        if trajectory_id != expected["episode_id"]:
            raise RuntimeError(
                "Trajectory ID {!r} does not match expected {!r}".format(
                    trajectory_id, expected["episode_id"]
                )
            )
        if trajectory_id in agent.results:
            raise RuntimeError("Duplicate trajectory ID {!r}".format(trajectory_id))
        agent.loss = 0
        agent.results[trajectory_id] = (
            trajectory
            if result_transform is None
            else result_transform(trajectory)
        )

    if env.ix != len(env.data):
        raise RuntimeError(
            "Exact evaluation consumed index {}, expected {}".format(
                env.ix, len(env.data)
            )
        )
    return True


def iter_exact_batches(
    episodes: Sequence[Any], batch_size: int
) -> Iterator[List[Any]]:
    """Yield the final short batch as-is; never wrap-pad from the stream head."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(episodes), batch_size):
        yield list(episodes[start : start + batch_size])


def canonicalize_eval_splits(splits: Iterable[str]) -> List[str]:
    """Validate and return a subset in val_seen -> val_unseen -> test order."""
    requested = list(splits)
    if len(requested) != len(set(requested)):
        raise ValueError("Evaluation splits must not contain duplicates")
    unknown = set(requested).difference(CANONICAL_SPLIT_ORDER)
    if unknown:
        raise ValueError(
            "Unsupported evaluation splits: {}".format(", ".join(sorted(unknown)))
        )
    return [split for split in CANONICAL_SPLIT_ORDER if split in requested]
