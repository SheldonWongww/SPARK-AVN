#!/usr/bin/env python3
"""Prepare or run canonical VLN IDEA source-statistics collection.

Each selected policy is evaluated with its frozen Source checkpoint and native
argmax actions on a deterministic 128-trajectory subset of the source-training
split.  The model-facing runner performs the final artifact/provenance checks;
this launcher writes a compact binding file that can later be copied into the
corresponding consistency-search specification.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402


BUILDER = REPO_ROOT / "vln/scripts/build_idea_source_order_manifests.py"
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
RESULTS_ROOT = REPO_ROOT / "vln/results/idea_source_statistics"
TRAJECTORY_COUNT = 128
SETTINGS = (
    "duet-r2r",
    "hamt-r2r",
    "goat-r2r",
    "duet-reverie",
    "hamt-reverie",
    "goat-reverie",
    "etpnav-r2r-ce",
    "bevbert-r2r-ce",
)
CHECKPOINTS = {
    "duet-r2r": "vln/checkpoints/duet/R2R/best_val_unseen",
    "hamt-r2r": "vln/checkpoints/hamt/R2R/vitbase-finetune-e2e/best_val_unseen",
    "goat-r2r": "vln/checkpoints/goat/R2R/best_val_unseen.pt",
    "duet-reverie": "vln/checkpoints/duet/REVERIE/best_val_unseen",
    "hamt-reverie": "vln/checkpoints/hamt/REVERIE/best_val_unseen",
    "goat-reverie": "vln/checkpoints/goat/REVERIE/best_val_unseen.pt",
    "etpnav-r2r-ce": "vln/checkpoints/etpnav/ckpt.iter12000.pth",
    "bevbert-r2r-ce": "vln/checkpoints/bevbert/ckpt.iter9600.pth",
}
SPEC_GROUPS = {
    "vln/experiments/r2r_consistency_search_v2.json": (
        "duet-r2r", "hamt-r2r", "goat-r2r",
    ),
    "vln/experiments/reverie_consistency_search_v2.json": (
        "duet-reverie", "hamt-reverie", "goat-reverie",
    ),
    "vln/experiments/r2r_ce_consistency_search_v2.json": (
        "etpnav-r2r-ce", "bevbert-r2r-ce",
    ),
}


class CollectionError(RuntimeError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _relative(path):
    return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def prepare_collection(setting, run_tag):
    if setting not in SETTINGS:
        raise CollectionError("unsupported IDEA source setting: {}".format(setting))
    checkpoint = REPO_ROOT / CHECKPOINTS[setting]
    if not checkpoint.is_file():
        raise CollectionError("missing source checkpoint: {}".format(checkpoint))

    input_root = RESULTS_ROOT / "_inputs" / run_tag / setting
    order_dir = input_root / "episode_order"
    order_path = order_dir / "train.json"
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--setting", setting,
            "--output", str(order_path),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    order = _read_json(order_path)
    if order.get("episode_count") != TRAJECTORY_COUNT:
        raise CollectionError("source-order builder did not produce 128 trajectories")

    result_root = RESULTS_ROOT / run_tag / setting / "train"
    artifact_path = result_root / "source_statistics.json"
    config_path = input_root / "collection_config.json"
    config = {
        "schema": "navtta.vln_tta_job.v1",
        "namespace": "idea_source_statistics",
        "stage": "source_statistics",
        "episodes": TRAJECTORY_COUNT,
        "method": "idea",
        "parameters": {
            "action_selection": "argmax",
            "collection_policy": "frozen_source_argmax_rollout",
            "collect_source_stats": True,
            "opt_steps": 50,
            "prompt_length": 4,
            "source_checkpoint_sha256": _sha256(checkpoint),
            "source_dataset": order["dataset"]["path"],
            "source_dataset_version": "sha256:" + order["dataset"]["sha256"],
            "source_split": "train",
            "source_stats_output": str(artifact_path.resolve()),
            "source_trajectories": TRAJECTORY_COUNT,
        },
    }
    _atomic_json(config_path, config)
    command = [
        "bash",
        str(RUNNER),
        setting,
        "train",
        "0",  # overwritten by caller
        "--run-tag",
        run_tag,
        "--tta-config",
        str(config_path.resolve()),
        "--source-order-manifest",
        str(order_dir.resolve()),
    ]
    if setting in ("etpnav-r2r-ce", "bevbert-r2r-ce"):
        command.extend(["--ce-data-version", "v1.3-unified"])
    return {
        "setting": setting,
        "run_tag": run_tag,
        "input_root": str(input_root.resolve()),
        "order_manifest": str(order_path.resolve()),
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": config["parameters"]["source_checkpoint_sha256"],
        "result_root": str(result_root.resolve()),
        "artifact": str(artifact_path.resolve()),
        "command": command,
    }


def binding_from_completed(record):
    artifact_path = Path(record["artifact"])
    diagnostics_path = Path(record["result_root"]) / "tta_diagnostics.json"
    if not artifact_path.is_file() or not diagnostics_path.is_file():
        raise CollectionError(
            "collection is incomplete for {}".format(record["setting"])
        )
    artifact = _read_json(artifact_path)
    provenance = artifact.get("provenance", {})
    if (
        artifact.get("schema") != "navtta.idea.source_statistics"
        or artifact.get("version") != 1
        or provenance.get("setting") != record["setting"]
        or provenance.get("checkpoint_sha256") != record["checkpoint_sha256"]
        or provenance.get("trajectory_count") != TRAJECTORY_COUNT
        or provenance.get("collection_policy")
        != "frozen_source_argmax_rollout"
    ):
        raise CollectionError(
            "source-statistics identity is invalid for {}".format(record["setting"])
        )
    data_version = (
        "v1.3-unified" if record["setting"] in (
            "etpnav-r2r-ce", "bevbert-r2r-ce"
        ) else "native"
    )
    run_id = "{}-{}-train-{}".format(
        record["run_tag"], record["setting"], data_version
    )
    formal_path = REPO_ROOT / "vln/results/runs" / run_id / "manifest.json"
    if not formal_path.is_file():
        raise CollectionError(
            "missing formal collection manifest for {}".format(record["setting"])
        )
    formal = _read_json(formal_path)
    expected_formal = {
        "run_id": run_id,
        "task": "vln",
        "model": record["setting"].split("-", 1)[0],
        "method": "idea",
        "run_tag": record["run_tag"],
        "source_setting": "{}:train:{}:idea".format(
            record["setting"], data_version
        ),
        "status": "completed",
        "exit_code": 0,
    }
    for key, expected in expected_formal.items():
        if formal.get(key) != expected:
            raise CollectionError(
                "formal collection manifest {} mismatch for {}".format(
                    key, record["setting"]
                )
            )
    if (
        formal.get("checkpoint", {}).get("sha256")
        != record["checkpoint_sha256"]
        or formal.get("immutable_identity_sha256")
        != immutable_identity_sha256(formal)
    ):
        raise CollectionError("formal collection identity is invalid")
    order = _read_json(record["order_manifest"])
    dataset = formal.get("dataset", {})
    pinned_order = formal.get("pinned_manifests", {}).get("episode_order", {})
    if (
        dataset.get("stream_order_sha256") != order.get("order_sha256")
        or dataset.get("stream_content_sha256")
        != order.get("dataset", {}).get("sha256")
        or pinned_order.get("sha256") != _sha256(record["order_manifest"])
    ):
        raise CollectionError("formal collection dataset/order binding is invalid")
    auxiliary = {
        item.get("name"): item
        for item in formal.get("auxiliary_checkpoints", [])
        if isinstance(item, dict)
    }
    if auxiliary.get("tta_job_config", {}).get("sha256") != _sha256(
        record["config"]
    ):
        raise CollectionError("formal collection config binding is invalid")
    authenticated = {
        item.get("name"): item
        for item in formal.get("result_artifacts", [])
        if isinstance(item, dict)
    }
    for path in (artifact_path, diagnostics_path):
        name = path.resolve().relative_to(
            Path(record["result_root"]).resolve()
        ).as_posix()
        evidence = authenticated.get(name, {})
        if (
            evidence.get("sha256") != _sha256(path)
            or evidence.get("size") != path.stat().st_size
        ):
            raise CollectionError(
                "formal collection manifest does not authenticate {}".format(name)
            )
    return {
        "path": _relative(artifact_path),
        "sha256": _sha256(artifact_path),
        "trajectory_count": TRAJECTORY_COUNT,
        "collection_policy": "frozen_source_argmax_rollout",
        "checkpoint_sha256": record["checkpoint_sha256"],
        "order_manifest": _relative(record["order_manifest"]),
        "order_manifest_sha256": _sha256(record["order_manifest"]),
        "collection_config": _relative(record["config"]),
        "collection_config_sha256": _sha256(record["config"]),
        "diagnostics": _relative(diagnostics_path),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "formal_manifest": _relative(formal_path),
        "formal_manifest_sha256": _sha256(formal_path),
        "formal_immutable_identity_sha256": formal["immutable_identity_sha256"],
    }


def apply_bindings_to_specs(bindings):
    """Atomically mark IDEA ready after every setting has a valid binding."""
    if set(bindings) != set(SETTINGS):
        missing = sorted(set(SETTINGS).difference(bindings))
        raise CollectionError(
            "--apply-specs requires all eight settings; missing {}".format(
                ", ".join(missing)
            )
        )
    updated = []
    for relative_spec, settings in SPEC_GROUPS.items():
        spec_path = REPO_ROOT / relative_spec
        spec = _read_json(spec_path)
        idea = spec.get("methods", {}).get("idea")
        if not isinstance(idea, dict):
            raise CollectionError("spec has no IDEA method: {}".format(spec_path))
        idea["availability"] = {
            "status": "ready",
            "reason": (
                "offline source-train statistics are bound by path, SHA256, "
                "checkpoint, and 128-trajectory provenance"
            ),
        }
        idea["source_statistics"] = {
            setting: dict(bindings[setting])
            for setting in settings
        }
        _atomic_json(spec_path, spec)
        updated.append(str(spec_path.resolve()))
    return updated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--run-tag", default="idea-source-train128-v1")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--apply-specs",
        action="store_true",
        help="after all eight real collections pass, bind them into the v2 specs",
    )
    parser.add_argument(
        "--bindings-output",
        default=None,
        help="default: vln/results/idea_source_statistics/RUN_TAG/bindings.json",
    )
    args = parser.parse_args(argv)
    if args.gpu < 0:
        raise CollectionError("--gpu must be nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_tag):
        raise CollectionError("invalid run tag")
    if len(args.settings) != len(set(args.settings)):
        raise CollectionError("--settings contains duplicates")
    if args.apply_specs and args.dry_run:
        raise CollectionError("--apply-specs cannot be combined with --dry-run")
    if args.apply_specs and set(args.settings) != set(SETTINGS):
        raise CollectionError("--apply-specs requires the complete eight-setting collection")

    records = []
    bindings = {}
    for setting in SETTINGS:
        if setting not in args.settings:
            continue
        record = prepare_collection(setting, args.run_tag)
        record["command"][4] = str(args.gpu)
        records.append(record)
        command = list(record["command"])
        if args.dry_run:
            command.append("--dry-run")
        print(" ".join(command), flush=True)
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)
        if not args.dry_run:
            bindings[setting] = binding_from_completed(record)

    output = Path(args.bindings_output) if args.bindings_output else (
        RESULTS_ROOT / args.run_tag / "bindings.json"
    )
    summary = {
        "schema": "navtta.idea.source_statistics_bindings.v1",
        "run_tag": args.run_tag,
        "dry_run": args.dry_run,
        "settings": bindings,
        "prepared": records,
    }
    _atomic_json(output, summary)
    updated_specs = apply_bindings_to_specs(bindings) if args.apply_specs else []
    print(json.dumps({
        "bindings": str(output.resolve()),
        "dry_run": args.dry_run,
        "setting_count": len(records),
        "updated_specs": updated_specs,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CollectionError, subprocess.CalledProcessError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
