#!/usr/bin/env python3
"""Safely migrate one persisted VLN TTA search batch to result layout v2.

The utility is intentionally dry-run by default.  It discovers current jobs
from every method campaign for one batch, groups jobs that refer to the same
legacy result root, and only mutates the workspace when ``--apply`` is used.
Archived ``attempts/`` evidence is never discovered or rewritten.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys


SOURCE_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_REPO_ROOT))

from tools.run_manifest_identity import (  # noqa: E402
    IMMUTABLE_IDENTITY_SHA256_FIELD,
    immutable_identity_sha256,
)


MIGRATION_SCHEMA = "navtta.vln_tta_layout_migration.v1"
RESULT_LAYOUT = "method_batch_stage_setting_run_v2"
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


class MigrationError(RuntimeError):
    """Raised before an unsafe or ambiguous migration can be applied."""


def _absolute(path):
    return Path(path).expanduser().absolute()


def _safe_component(value, label):
    if (not isinstance(value, str) or not value
            or value in (".", "..") or SAFE_COMPONENT.fullmatch(value) is None
            or Path(value).name != value):
        raise MigrationError("{} is not one safe path component: {!r}".format(
            label, value
        ))
    return value


def _read_json(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise MigrationError("{} is not a regular file: {}".format(label, path))
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MigrationError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(document, dict):
        raise MigrationError("{} is not a JSON object: {}".format(label, path))
    return document


def _json_bytes(document):
    return (
        json.dumps(
            document, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _path_is_within(path, root):
    try:
        Path(path).relative_to(Path(root))
        return True
    except ValueError:
        return False


def _reject_symlink_components(path, root, label):
    path = _absolute(path)
    root = _absolute(root)
    if not _path_is_within(path, root):
        raise MigrationError("{} escapes {}: {}".format(label, root, path))
    current = root
    if current.is_symlink():
        raise MigrationError("{} contains a symlink: {}".format(label, current))
    for component in path.relative_to(root).parts:
        current = current / component
        if current.is_symlink():
            raise MigrationError("{} contains a symlink: {}".format(
                label, current
            ))


def _replace_path_strings(value, old_root, new_root):
    """Replace an absolute result-root substring in nested JSON values."""
    old_text = str(old_root)
    new_text = str(new_root)
    if isinstance(value, str):
        return value.replace(old_text, new_text)
    if isinstance(value, list):
        return [
            _replace_path_strings(item, old_root, new_root) for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _replace_path_strings(item, old_root, new_root)
            for key, item in value.items()
        }
    return value


def _command_with_result_root(command, result_root, label):
    if (not isinstance(command, list)
            or not all(isinstance(item, str) for item in command)):
        raise MigrationError("{} command is not a string list".format(label))
    command = list(command)
    separate = [
        index for index, item in enumerate(command) if item == "--result-root"
    ]
    attached = [
        index for index, item in enumerate(command)
        if item.startswith("--result-root=")
    ]
    if len(separate) + len(attached) > 1:
        raise MigrationError("{} command has duplicate --result-root".format(
            label
        ))
    if separate:
        index = separate[0]
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            raise MigrationError("{} command has an invalid --result-root".format(
                label
            ))
        command[index + 1] = str(result_root)
    elif attached:
        index = attached[0]
        command[index:index + 1] = ["--result-root", str(result_root)]
    else:
        config_positions = [
            index for index, item in enumerate(command)
            if item == "--tta-config"
        ]
        if (len(config_positions) != 1
                or config_positions[0] + 1 >= len(command)):
            raise MigrationError(
                "{} command has no canonical --tta-config".format(label)
            )
        insert_at = config_positions[0] + 2
        command[insert_at:insert_at] = [
            "--result-root", str(result_root)
        ]
    return command


class JobRecord:
    def __init__(self, path, method, batch_id, stage, document, tuning_root):
        self.path = Path(path)
        self.method = method
        self.batch_id = batch_id
        self.stage = stage
        self.document = document

        checks = (
            ("search_method", method),
            ("stage", stage),
        )
        for field, expected in checks:
            if document.get(field) != expected:
                raise MigrationError(
                    "{} {} does not match its campaign path: {!r} != {!r}".format(
                        self.path, field, document.get(field), expected
                    )
                )
        # The first persisted search plans predate the redundant batch_id
        # field.  Their campaign directory remains authoritative; reject a
        # contradictory value, but allow the migration to backfill omission.
        if document.get("batch_id") not in (None, batch_id):
            raise MigrationError(
                "{} batch_id does not match its campaign path: {!r} != {!r}"
                .format(self.path, document.get("batch_id"), batch_id)
            )
        self.run_tag = _safe_component(document.get("run_tag"), "job run_tag")
        self.setting = _safe_component(document.get("setting"), "job setting")
        expected_job_dir = self.path.parent.absolute()
        declared_job_dir = document.get("job_dir")
        if (not isinstance(declared_job_dir, str)
                or not Path(declared_job_dir).is_absolute()
                or Path(declared_job_dir).absolute() != expected_job_dir):
            raise MigrationError(
                "job_dir does not identify the current job directory: {}".format(
                    self.path
                )
            )
        result_root = document.get("result_root")
        if not isinstance(result_root, str) or not Path(result_root).is_absolute():
            raise MigrationError("job result_root is not absolute: {}".format(
                self.path
            ))
        self.current_root = Path(result_root).absolute()
        self.legacy_root = (
            Path(tuning_root) / self.run_tag / self.setting / "val_seen"
        ).absolute()
        _command_with_result_root(
            document.get("command"), self.current_root, str(self.path)
        )


class ResultGroup:
    def __init__(self, records, tuning_root):
        self.records = sorted(records, key=lambda item: str(item.path))
        first = self.records[0]
        self.legacy_root = first.legacy_root
        values = {
            "batch_id": {item.batch_id for item in self.records},
            "stage": {item.stage for item in self.records},
            "setting": {item.setting for item in self.records},
            "run_tag": {item.run_tag for item in self.records},
        }
        ambiguous = [key for key, items in values.items() if len(items) != 1]
        if ambiguous:
            raise MigrationError(
                "shared legacy root has ambiguous {}: {}".format(
                    ", ".join(ambiguous), self.legacy_root
                )
            )
        self.batch_id = first.batch_id
        self.stage = first.stage
        self.setting = first.setting
        self.run_tag = first.run_tag
        self.methods = sorted({item.method for item in self.records})
        self.shared = len(self.methods) > 1
        self.namespace = "_shared" if self.shared else self.methods[0]
        if self.shared and any(
                item.document.get("config_method") != "source"
                for item in self.records):
            raise MigrationError(
                "only Source jobs may share one legacy result root: {}".format(
                    self.legacy_root
                )
            )
        self.destination = (
            Path(tuning_root) / self.namespace / self.batch_id / self.stage
            / self.setting / self.run_tag / "val_seen"
        ).absolute()
        for record in self.records:
            if record.current_root not in (self.legacy_root, self.destination):
                raise MigrationError(
                    "job result_root is neither legacy nor canonical v2: {}"
                    .format(record.path)
                )


class Migration:
    def __init__(self, repo_root, batch_id, search_root=None,
                 tuning_root=None, runs_root=None):
        self.repo_root = _absolute(repo_root)
        self.batch_id = _safe_component(batch_id, "batch id")
        self.search_root = _absolute(
            search_root or self.repo_root / "vln/results/logs/hparam_search"
        )
        self.tuning_root = _absolute(
            tuning_root or self.repo_root / "vln/results/tuning"
        )
        self.runs_root = _absolute(
            runs_root or self.repo_root / "vln/results/runs"
        )
        self.ledger_path = (
            self.tuning_root / "_migrations" / (self.batch_id + ".json")
        )
        self.records = []
        self.groups = []
        self.pending_writes = {}
        self.write_evidence = []

    def discover(self):
        if self.search_root.is_symlink() or not self.search_root.is_dir():
            raise MigrationError("search root is not a directory: {}".format(
                self.search_root
            ))
        campaign_roots = sorted(
            path for path in self.search_root.glob("*/{}".format(self.batch_id))
            if path.is_dir()
        )
        if not campaign_roots:
            raise MigrationError("no campaigns found for batch {} under {}".format(
                self.batch_id, self.search_root
            ))
        seen = set()
        for campaign in campaign_roots:
            method = _safe_component(campaign.parent.name, "campaign method")
            if method not in METHODS:
                raise MigrationError("unsupported campaign method: {}".format(
                    method
                ))
            _reject_symlink_components(campaign, self.search_root, "campaign")
            for path in sorted(campaign.glob("stages/*/jobs/**/job.json")):
                relative = path.relative_to(campaign)
                if "attempts" in relative.parts:
                    continue
                if path in seen:
                    continue
                seen.add(path)
                parts = relative.parts
                if len(parts) < 5 or parts[0] != "stages" or parts[2] != "jobs":
                    raise MigrationError("unexpected current job path: {}".format(
                        path
                    ))
                stage = _safe_component(parts[1], "campaign stage")
                _reject_symlink_components(path, campaign, "current job path")
                document = _read_json(path, "current job")
                self.records.append(JobRecord(
                    path, method, self.batch_id, stage, document,
                    self.tuning_root,
                ))
        if not self.records:
            raise MigrationError("batch {} has no current jobs".format(
                self.batch_id
            ))
        grouped = {}
        for record in self.records:
            grouped.setdefault(record.legacy_root, []).append(record)
        self.groups = [
            ResultGroup(items, self.tuning_root)
            for _, items in sorted(grouped.items(), key=lambda item: str(item[0]))
        ]
        destinations = {}
        for group in self.groups:
            owner = destinations.setdefault(group.destination, group.legacy_root)
            if owner != group.legacy_root:
                raise MigrationError(
                    "multiple legacy roots map to destination {}".format(
                        group.destination
                    )
                )
        return self

    def _result_action(self, group):
        _reject_symlink_components(
            group.legacy_root, self.tuning_root, "legacy result root"
        )
        _reject_symlink_components(
            group.destination, self.tuning_root, "v2 result root"
        )
        source_exists = group.legacy_root.exists()
        destination_exists = group.destination.exists()
        if source_exists and not group.legacy_root.is_dir():
            raise MigrationError("legacy result root is not a directory: {}".format(
                group.legacy_root
            ))
        if destination_exists and not group.destination.is_dir():
            raise MigrationError("v2 result root is not a directory: {}".format(
                group.destination
            ))
        if source_exists and destination_exists:
            raise MigrationError(
                "both legacy and v2 result roots exist; refusing to merge {} and {}"
                .format(group.legacy_root, group.destination)
            )
        if source_exists:
            return "move", source_exists, destination_exists
        if destination_exists:
            return "already_moved", source_exists, destination_exists
        return "metadata_only", source_exists, destination_exists

    def _queue_bytes(self, path, value, label):
        path = Path(path)
        if path.is_symlink():
            raise MigrationError("{} is a symlink: {}".format(label, path))
        try:
            before = path.read_bytes()
        except OSError as error:
            raise MigrationError("cannot read {} {}: {}".format(label, path, error))
        if before == value:
            return
        previous = self.pending_writes.get(path)
        if previous is not None and previous != value:
            raise MigrationError("conflicting rewrites for {}".format(path))
        self.pending_writes[path] = value
        self.write_evidence.append({
            "path": str(path),
            "sha256_before": _sha256_bytes(before),
            "sha256_after": _sha256_bytes(value),
        })

    def _prepare_job_files(self):
        for group in self.groups:
            for record in group.records:
                job = _replace_path_strings(
                    record.document, group.legacy_root, group.destination
                )
                job["result_root"] = str(group.destination)
                job["result_layout"] = RESULT_LAYOUT
                job["result_namespace"] = group.namespace
                job["batch_id"] = self.batch_id
                job["command"] = _command_with_result_root(
                    job.get("command"), group.destination, str(record.path)
                )
                self._queue_bytes(
                    record.path, _json_bytes(job), "current job"
                )

                config_path = Path(record.document.get("config_path", ""))
                if not config_path.is_absolute():
                    raise MigrationError("job config_path is not absolute: {}".format(
                        record.path
                    ))
                if config_path.absolute() != (
                        record.path.parent / "parameters.json").absolute():
                    raise MigrationError(
                        "job config_path does not identify current parameters.json: {}"
                        .format(record.path)
                    )
                _reject_symlink_components(
                    config_path, self.search_root, "current job config"
                )
                config = _read_json(config_path, "current job config")
                config.update({
                    "batch_id": self.batch_id,
                    "setting": record.setting,
                    "run_tag": record.run_tag,
                    "result_layout": RESULT_LAYOUT,
                    "result_namespace": group.namespace,
                })
                self._queue_bytes(
                    config_path, _json_bytes(config), "current job config"
                )

                metrics_path = record.path.parent / "metrics.json"
                if metrics_path.exists() or metrics_path.is_symlink():
                    metrics = _read_json(metrics_path, "current metrics")
                    metrics = _replace_path_strings(
                        metrics, group.legacy_root, group.destination
                    )
                    metrics["result_root"] = str(group.destination)
                    metrics["result_layout"] = RESULT_LAYOUT
                    metrics["result_namespace"] = group.namespace
                    metrics["batch_id"] = self.batch_id
                    metrics["command"] = _command_with_result_root(
                        metrics.get("command", record.document.get("command")),
                        group.destination, str(metrics_path),
                    )
                    self._queue_bytes(
                        metrics_path, _json_bytes(metrics), "current metrics"
                    )

    def _grid_path(self, record):
        current = record.path.parent
        while current != self.search_root and current.name != "jobs":
            current = current.parent
        if current.name != "jobs":
            raise MigrationError("cannot locate jobs directory for {}".format(
                record.path
            ))
        return current.parent / "grid.csv"

    def _prepare_grids(self):
        by_grid = {}
        for group in self.groups:
            for record in group.records:
                by_grid.setdefault(self._grid_path(record), []).append(
                    (record, group)
                )
        for grid_path, items in sorted(by_grid.items(), key=lambda item: str(item[0])):
            if grid_path.is_symlink() or not grid_path.is_file():
                raise MigrationError("stage grid is not a regular file: {}".format(
                    grid_path
                ))
            try:
                with grid_path.open("r", encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream)
                    fieldnames = reader.fieldnames
                    rows = list(reader)
            except (OSError, UnicodeError, csv.Error) as error:
                raise MigrationError("cannot read stage grid {}: {}".format(
                    grid_path, error
                ))
            required = {"run_tag", "job_dir", "result_root", "command"}
            if not fieldnames or not required.issubset(fieldnames):
                raise MigrationError("stage grid lacks required columns: {}".format(
                    grid_path
                ))
            item_by_job_dir = {
                str(record.path.parent.absolute()): (record, group)
                for record, group in items
            }
            matched = set()
            for row in rows:
                pair = item_by_job_dir.get(row.get("job_dir"))
                if pair is None:
                    continue
                record, group = pair
                if record.path in matched:
                    raise MigrationError("duplicate current job row in {}".format(
                        grid_path
                    ))
                matched.add(record.path)
                for key, value in list(row.items()):
                    if isinstance(value, str):
                        row[key] = value.replace(
                            str(group.legacy_root), str(group.destination)
                        )
                row["run_tag"] = record.run_tag
                row["result_root"] = str(group.destination)
                try:
                    recorded_command = json.loads(row["command"])
                except (TypeError, json.JSONDecodeError) as error:
                    raise MigrationError("invalid grid command in {}: {}".format(
                        grid_path, error
                    ))
                _command_with_result_root(
                    recorded_command, group.destination,
                    "{} recorded row {}".format(
                        grid_path, row.get("ordinal")
                    ),
                )
                command = _replace_path_strings(
                    record.document["command"],
                    group.legacy_root,
                    group.destination,
                )
                row["command"] = json.dumps(
                    _command_with_result_root(
                        command, group.destination,
                        "{} row {}".format(grid_path, row.get("ordinal")),
                    ),
                    ensure_ascii=False,
                )
                if "result_layout" in row:
                    row["result_layout"] = RESULT_LAYOUT
                if "result_namespace" in row:
                    row["result_namespace"] = group.namespace
            expected = {record.path for record, _ in items}
            if matched != expected:
                missing = sorted(str(path) for path in expected.difference(matched))
                raise MigrationError("stage grid misses current jobs: {}".format(
                    ", ".join(missing)
                ))
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            self._queue_bytes(
                grid_path, output.getvalue().encode("utf-8"), "stage grid"
            )

    def _formal_manifest_paths(self):
        if not self.runs_root.exists():
            return []
        if self.runs_root.is_symlink() or not self.runs_root.is_dir():
            raise MigrationError("formal runs root is not a directory: {}".format(
                self.runs_root
            ))
        identities = {(item.run_tag, item.setting) for item in self.records}
        paths = []
        for path in sorted(self.runs_root.glob("*/manifest.json")):
            directory = path.parent.name
            if not any(directory.startswith(
                    "{}-{}-val_seen-".format(run_tag, setting)
            ) for run_tag, setting in identities):
                continue
            paths.append(path)
        return paths

    def _prepare_formal_manifests(self):
        groups = {
            (group.run_tag, group.setting): group for group in self.groups
        }
        for path in self._formal_manifest_paths():
            _reject_symlink_components(
                path, self.runs_root, "current formal manifest"
            )
            manifest = _read_json(path, "current formal manifest")
            recorded_identity = manifest.get(IMMUTABLE_IDENTITY_SHA256_FIELD)
            if (recorded_identity is not None
                    and recorded_identity != immutable_identity_sha256(manifest)):
                raise MigrationError(
                    "current formal manifest immutable identity is invalid: {}"
                    .format(path)
                )
            matches = [
                group for group in groups.values()
                if (manifest.get("run_tag") == group.run_tag
                    and path.parent.name.startswith(
                        "{}-{}-val_seen-".format(
                            group.run_tag, group.setting
                        )
                    ))
            ]
            if len(matches) != 1:
                raise MigrationError(
                    "formal manifest does not identify exactly one current result: {}"
                    .format(path)
                )
            group = matches[0]
            migrated = _replace_path_strings(
                manifest, group.legacy_root, group.destination
            )
            if migrated != manifest:
                if IMMUTABLE_IDENTITY_SHA256_FIELD in migrated:
                    migrated[IMMUTABLE_IDENTITY_SHA256_FIELD] = (
                        immutable_identity_sha256(migrated)
                    )
                self._queue_bytes(
                    path, _json_bytes(migrated), "current formal manifest"
                )

    def _reject_active_workers(self):
        for record in self.records:
            state_path = record.path.parent / "worker_state.json"
            if not state_path.exists() and not state_path.is_symlink():
                continue
            state = _read_json(state_path, "current worker state")
            if state.get("status") in ("starting", "running"):
                raise MigrationError(
                    "current worker is still active; stop the batch before "
                    "migration: {}".format(record.path)
                )

    def prepare(self):
        if not self.groups:
            self.discover()
        self.pending_writes = {}
        self.write_evidence = []
        self._reject_active_workers()
        actions = []
        for group in self.groups:
            action, source_exists, destination_exists = self._result_action(group)
            actions.append((group, action, source_exists, destination_exists))
        self._prepare_job_files()
        self._prepare_grids()
        self._prepare_formal_manifests()
        return actions

    def _existing_ledger(self):
        _reject_symlink_components(
            self.ledger_path, self.tuning_root, "layout migration ledger"
        )
        if not self.ledger_path.exists() and not self.ledger_path.is_symlink():
            return None
        ledger = _read_json(self.ledger_path, "layout migration ledger")
        if ledger.get("schema") != MIGRATION_SCHEMA:
            raise MigrationError("unsupported migration ledger schema: {}".format(
                self.ledger_path
            ))
        if ledger.get("batch_id") != self.batch_id:
            raise MigrationError("migration ledger batch mismatch: {}".format(
                self.ledger_path
            ))
        if ledger.get("result_layout") != RESULT_LAYOUT:
            raise MigrationError("migration ledger layout mismatch: {}".format(
                self.ledger_path
            ))
        expected = {
            (
                str(group.legacy_root),
                str(group.destination),
                tuple(str(record.path.relative_to(self.search_root))
                      for record in group.records),
            )
            for group in self.groups
        }
        recorded = {
            (
                item.get("source_result_root"),
                item.get("destination_result_root"),
                tuple(item.get("current_jobs", [])),
            )
            for item in ledger.get("groups", []) if isinstance(item, dict)
        }
        if recorded != expected:
            raise MigrationError(
                "existing migration ledger does not match current jobs: {}".format(
                    self.ledger_path
                )
            )
        return ledger

    def _document(self, actions, applied, existing=None):
        groups = []
        for group, action, source_exists, destination_exists in actions:
            groups.append({
                "source_result_root": str(group.legacy_root),
                "destination_result_root": str(group.destination),
                "namespace": group.namespace,
                "shared": group.shared,
                "methods": group.methods,
                "batch_id": group.batch_id,
                "stage": group.stage,
                "setting": group.setting,
                "run_tag": group.run_tag,
                "source_present_before": source_exists,
                "destination_present_before": destination_exists,
                "result_action": action,
                "current_jobs": [
                    str(record.path.relative_to(self.search_root))
                    for record in group.records
                ],
            })
        document = {
            "schema": MIGRATION_SCHEMA,
            "migration": "LAYOUT_MIGRATION",
            "result_layout": RESULT_LAYOUT,
            "status": "applied" if applied else "dry_run",
            "batch_id": self.batch_id,
            "search_root": str(self.search_root),
            "tuning_root": str(self.tuning_root),
            "runs_root": str(self.runs_root),
            "ledger_path": str(self.ledger_path),
            "groups": groups,
            "updated_files": sorted(
                self.write_evidence, key=lambda item: item["path"]
            ),
        }
        if applied:
            document["applied_at"] = (
                existing.get("applied_at") if existing else
                datetime.now(timezone.utc).isoformat()
            )
        return document

    def _move_group(self, group):
        group.destination.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(
            group.destination.parent, self.tuning_root, "v2 result parent"
        )
        try:
            group.legacy_root.rename(group.destination)
        except OSError as error:
            raise MigrationError("cannot move {} to {}: {}".format(
                group.legacy_root, group.destination, error
            ))
        current = group.legacy_root.parent
        while current != self.tuning_root:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    def run(self, apply=False):
        actions = self.prepare()
        existing = self._existing_ledger()
        if not apply:
            return self._document(actions, applied=False, existing=existing)

        move_groups = [
            group for group, action, _, _ in actions if action == "move"
        ]
        no_changes = not move_groups and not self.pending_writes
        if existing is not None and no_changes:
            result = dict(existing)
            result["idempotent_noop"] = True
            return result

        for group in move_groups:
            self._move_group(group)
        for path, value in sorted(
                self.pending_writes.items(), key=lambda item: str(item[0])):
            _atomic_write(path, value)
        document = self._document(actions, applied=True, existing=existing)
        document["idempotent_noop"] = False
        _atomic_write(self.ledger_path, _json_bytes(document))
        return document


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_id", nargs="?", help="persisted search batch id")
    parser.add_argument("--batch-id", dest="batch_id_option")
    parser.add_argument(
        "--repo-root", default=str(SOURCE_REPO_ROOT),
        help="NavTTA checkout root (primarily useful for isolated verification)",
    )
    parser.add_argument("--search-root")
    parser.add_argument("--tuning-root")
    parser.add_argument("--runs-root")
    parser.add_argument(
        "--apply", action="store_true",
        help="perform moves and metadata rewrites; default is a read-only dry run",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_id and args.batch_id_option:
        parser.error("specify the batch id either positionally or with --batch-id")
    batch_id = args.batch_id_option or args.batch_id
    if not batch_id:
        parser.error("a batch id is required")
    try:
        migration = Migration(
            args.repo_root,
            batch_id,
            search_root=args.search_root,
            tuning_root=args.tuning_root,
            runs_root=args.runs_root,
        ).discover()
        document = migration.run(apply=args.apply)
    except MigrationError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    print(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
