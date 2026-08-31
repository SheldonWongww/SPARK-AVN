"""Write ordered per-episode evidence from the native discrete evaluator.

The navigation baselines already compute one metric vector per episode while
producing their aggregate validation line.  This module serializes that
native output for formal campaign validation; it does not reimplement or
rerun an evaluator.
"""

import hashlib
import json
import math
import os
from pathlib import Path


SCHEMA = "navtta.vln_per_episode_metrics.v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(value, split):
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise ValueError("episode-order manifest is required")
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / (str(split) + ".json")
    if not path.is_file():
        raise ValueError("missing episode-order manifest: {}".format(path))
    return path


def _number(value, metric, episode_id):
    # NumPy scalar values expose item(); converting here keeps the evidence
    # JSON independent of NumPy and rejects arrays or non-finite values.
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(
            "invalid per-episode metric {} for {}".format(metric, episode_id)
        )
    return float(value)


def _records_sha256(records):
    normalized = [
        {
            "episode_id": str(item["episode_id"]),
            "scene_id": str(item["scene_id"]),
        }
        for item in records
    ]
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _active_episode_order(manifest):
    """Return the exact order installed by ``configure_exact_episode_env``.

    The core environment helper applies ``NAVTTA_SMOKE_EPISODES`` after it
    loads the immutable parent manifest.  The same variable is used for both
    explicit smoke runs and scheduler ``--episode-limit`` jobs.  Sidecar
    validation must therefore use that active prefix rather than rejecting a
    correct short run against the full parent stream.
    """
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("episode-order manifest has no episodes list")
    raw_limit = os.environ.get("NAVTTA_SMOKE_EPISODES")
    if raw_limit is None or raw_limit == "":
        return episodes, None
    if not raw_limit.isdigit() or int(raw_limit) <= 0:
        raise ValueError("runtime episode limit must be a positive integer")
    limit = int(raw_limit)
    if limit > len(episodes):
        raise ValueError(
            "runtime episode limit {} exceeds manifest size {}".format(
                limit, len(episodes)
            )
        )
    return episodes[:limit], limit


def write_discrete_per_episode_metrics(
        args, setting, split, evaluator_metrics):
    """Persist native evaluator vectors next to the TTA diagnostics file.

    Source-only and training calls have no ``tta_diagnostics`` destination and
    intentionally remain unchanged.  Formal TTA validation/test runs use one
    process and one split, so a single sidecar has an unambiguous identity.
    """
    diagnostics = getattr(args, "tta_diagnostics", None)
    if not diagnostics:
        return None
    if str(split).lower() in ("test", "test_unseen"):
        raise ValueError("hidden-test metrics must never be materialized")
    if not isinstance(evaluator_metrics, dict):
        raise ValueError("native evaluator metrics must be a mapping")

    manifest_path = _manifest_path(
        getattr(args, "episode_order_manifest", None), split
    )
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    episodes, runtime_limit = _active_episode_order(manifest)
    expected_ids = [str(item.get("episode_id")) for item in episodes]
    actual_ids = [str(value) for value in evaluator_metrics.get("instr_id", ())]
    if actual_ids != expected_ids:
        raise ValueError(
            "native evaluator episode order does not match the pinned manifest"
        )

    metric_names = sorted(
        name for name in evaluator_metrics if name != "instr_id"
    )
    if not metric_names:
        raise ValueError("native evaluator returned no per-episode metrics")
    for name in metric_names:
        values = evaluator_metrics[name]
        if not hasattr(values, "__len__") or len(values) != len(expected_ids):
            raise ValueError(
                "native evaluator metric {} has the wrong length".format(name)
            )

    rows = []
    for ordinal, episode_id in enumerate(expected_ids):
        rows.append({
            "ordinal": ordinal,
            "episode_id": episode_id,
            "metrics": {
                name: _number(
                    evaluator_metrics[name][ordinal], name, episode_id
                )
                for name in metric_names
            },
        })
    document = {
        "schema": SCHEMA,
        "setting": str(setting),
        "split": str(split),
        "run_tag": os.environ.get("NAVTTA_RUN_TAG", ""),
        "episode_count": len(rows),
        "order_sha256": manifest.get("order_sha256"),
        "episode_order_manifest_sha256": _sha256(manifest_path),
        "source": "native_discrete_evaluator_return",
        "episodes": rows,
    }
    if runtime_limit is not None:
        document["parent_order_sha256"] = manifest.get("order_sha256")
        document["order_sha256"] = _records_sha256(episodes)
        document["runtime_episode_limit"] = runtime_limit
    output = Path(diagnostics).expanduser().resolve().parent / (
        "per_episode_metrics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            document, stream, indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(output))
    return output
