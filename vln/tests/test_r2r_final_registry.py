from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln/scripts"))

import build_r2r_final_registry as builder  # noqa: E402
from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402


SELECTION_PATH = REPO_ROOT / "vln/results/final/r2r/selected_winners.json"


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parameters(method):
    if method == "tent":
        return {"lr": 1e-5, "update_interval": 1}
    if method == "fstta":
        return {"lr_fast": 1e-3, "lr_slow": 1e-4}
    if method == "eam":
        return {"lr": 1e-6, "memory_size": 32}
    if method == "feedtta":
        return {"lr": 1e-6, "action_selection": "argmax"}
    if method == "atena":
        return {"lr_query": 1e-7, "action_selection": "argmax"}
    raise AssertionError(method)


def _manifest(run_tag, setting, method, checkpoint_sha256):
    model = builder.MODEL_FOR_SETTING[setting]
    manifest = {
        "run_id": "{}-{}-val_seen-native".format(run_tag, setting),
        "task": "vln",
        "benchmark": "r2r_test_fixture",
        "model": model,
        "method": method,
        "run_tag": run_tag,
        "source_setting": "{}:val_seen:native:{}".format(setting, method),
        "seed": 0,
        "git_commit": "a" * 40,
        "config": "/fixture/config.json",
        "config_overrides": ["python", "main.py"],
        "checkpoint": {
            "path": "/fixture/checkpoint",
            "size": 1,
            "sha256": checkpoint_sha256,
        },
        "auxiliary_checkpoints": [],
        "dataset": {
            "path": "/fixture/dataset.json",
            "version": "fixture",
            "size": 1,
            "index_sha256": "b" * 64,
            "stream_order_sha256": builder.EXPECTED_ORDER_SHA256,
            "stream_content_sha256": "b" * 64,
        },
        "pinned_manifests": {},
        "hardware": {"gpu_name": "fixture"},
        "started_at": "2026-08-22T00:00:00+00:00",
        "completed_at": "2026-08-22T00:01:00+00:00",
        "status": "completed",
        "exit_code": 0,
        "result_artifacts": [
            {
                "name": "logs/valid.txt",
                "path": "/fixture/result/logs/valid.txt",
                "size": 1,
                "sha256": "c" * 64,
            }
        ],
    }
    manifest["immutable_identity_sha256"] = immutable_identity_sha256(manifest)
    return manifest


def _complete_fixture(root):
    root = Path(root)
    source_records = {}
    selection_records = {}
    manifest_paths = {}
    source_values = {
        "duet-r2r": (78.84, 72.88),
        "hamt-r2r": (75.61, 72.18),
        "goat-r2r": (84.82, 80.05),
    }

    for setting_index, setting in enumerate(builder.SETTINGS):
        checkpoint_sha256 = str(setting_index + 1) * 64
        source_run_tag = "fixture-source-{}".format(setting)
        source_relative = (
            "vln/results/runs/{}-{}-val_seen-native/manifest.json".format(
                source_run_tag, setting
            )
        )
        source_path = root / source_relative
        _write_json(
            source_path,
            _manifest(source_run_tag, setting, "source", checkpoint_sha256),
        )
        source_sr, source_spl = source_values[setting]
        source_records[setting] = {
            "model": builder.MODEL_FOR_SETTING[setting],
            "run_tag": source_run_tag,
            "parameters": {"action_selection": "argmax", "action_seed": 0},
            "metrics": {"SR": source_sr, "SPL": source_spl},
            "checkpoint_sha256": checkpoint_sha256,
            "formal_manifest_path": source_relative,
            "formal_manifest_sha256": _sha256(source_path),
        }
        manifest_paths[(setting, "source")] = source_path

        selection_records[setting] = {}
        for method_index, method in enumerate(builder.METHODS):
            run_tag = "fixture-{}-{}".format(setting, method)
            relative = (
                "vln/results/runs/{}-{}-val_seen-native/manifest.json".format(
                    run_tag, setting
                )
            )
            manifest_path = root / relative
            _write_json(
                manifest_path,
                _manifest(run_tag, setting, method, checkpoint_sha256),
            )
            selection_records[setting][method] = {
                "selection_status": "ready",
                "supervision_category": builder.SUPERVISION_FOR_METHOD[method],
                "parameters": _parameters(method),
                "run_tag": run_tag,
                "metrics": {
                    "SR": round(source_sr + 0.1 * (method_index + 1), 2),
                    "SPL": round(source_spl + 0.2 * (method_index + 1), 2),
                },
                "formal_manifest_path": relative,
                "formal_manifest_sha256": _sha256(manifest_path),
            }
            manifest_paths[(setting, method)] = manifest_path

    source_ledger_path = root / "vln/manifests/r2r_source.json"
    source_ledger = {
        "schema": builder.SOURCE_SCHEMA,
        "benchmark": "r2r",
        "split": "val_seen",
        "source_protocol": "standard_argmax",
        "episode_count": builder.EXPECTED_EPISODES,
        "order_seed": builder.EXPECTED_ORDER_SEED,
        "episode_order_sha256": builder.EXPECTED_ORDER_SHA256,
        "records": source_records,
    }
    _write_json(source_ledger_path, source_ledger)

    selection_path = root / "vln/results/final/r2r/selected_winners.json"
    selection = {
        "schema": builder.SELECTION_SCHEMA,
        "selection_status": "complete",
        "protocol": {
            "benchmark": "r2r",
            "split": "val_seen",
            "episode_count": builder.EXPECTED_EPISODES,
            "order_seed": builder.EXPECTED_ORDER_SEED,
            "episode_order_sha256": builder.EXPECTED_ORDER_SHA256,
            "selection_scope": "val_seen_seed0_hyperparameter_selection",
        },
        "source_ledger": {
            "path": source_ledger_path.relative_to(root).as_posix(),
            "sha256": _sha256(source_ledger_path),
        },
        "records": selection_records,
    }
    _write_json(selection_path, selection)
    return {
        "selection": selection,
        "selection_path": selection_path,
        "source_ledger_path": source_ledger_path,
        "manifest_paths": manifest_paths,
    }


class R2RFinalRegistryTest(unittest.TestCase):
    def test_workspace_selection_is_complete(self):
        _, selection, pending = builder.load_selection(SELECTION_PATH)
        self.assertEqual(pending, [])
        self.assertEqual(selection["selection_status"], "complete")
        fstta = selection["records"]["duet-r2r"]["fstta"]
        self.assertEqual(fstta["metrics"], {"SR": 79.14, "SPL": 73.1})
        self.assertEqual(fstta["parameters"]["lr_fast"], 1.8e-3)
        self.assertEqual(fstta["parameters"]["lr_slow"], 3e-4)
        self.assertEqual(fstta["parameters"]["m"], 8)
        self.assertEqual(fstta["parameters"]["n"], 4)

    def test_pending_selection_writes_no_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
            pending_cells = (
                ("duet-r2r", "eam"),
                ("duet-r2r", "atena"),
                ("hamt-r2r", "eam"),
                ("hamt-r2r", "atena"),
                ("goat-r2r", "eam"),
            )
            selection["selection_status"] = "awaiting_formal_confirmation"
            for setting, method in pending_cells:
                record = selection["records"][setting][method]
                record["selection_status"] = "awaiting_formal_confirmation"
                for field in (
                    "run_tag",
                    "metrics",
                    "formal_manifest_path",
                    "formal_manifest_sha256",
                ):
                    record[field] = None
            selection_path = Path(directory) / "selected_winners.json"
            _write_json(selection_path, selection)
            output = Path(directory) / "registry.json"
            with self.assertRaisesRegex(
                builder.RegistryError, "awaiting 5 formal confirmations"
            ):
                builder.write_registry(output, selection_path, REPO_ROOT)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name("registry.json.tmp").exists())

    def test_complete_registry_has_source_and_five_tta_records(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _complete_fixture(directory)
            registry = builder.build_registry(
                fixture["selection_path"], directory
            )
            self.assertEqual(registry["schema"], builder.REGISTRY_SCHEMA)
            self.assertEqual(registry["registry_status"], "complete")
            self.assertEqual(
                registry["publication_status"],
                "selection_registry_not_publication_final",
            )
            self.assertEqual(set(registry["records"]), set(builder.SETTINGS))
            for setting in builder.SETTINGS:
                records = registry["records"][setting]
                self.assertEqual(set(records), set(builder.ALL_METHODS))
                self.assertEqual(
                    records["source"]["delta_vs_source_pp"],
                    {"SR": 0.0, "SPL": 0.0},
                )
                self.assertEqual(
                    records["feedtta"]["supervision_category"],
                    "binary_episode_feedback_tta",
                )
                self.assertEqual(
                    records["fstta"]["delta_vs_source_pp"],
                    {"SR": 0.2, "SPL": 0.4},
                )
                for record in records.values():
                    self.assertTrue(
                        {
                            "parameters",
                            "run_tag",
                            "metrics",
                            "formal_manifest_path",
                            "formal_manifest_sha256",
                        }.issubset(record)
                    )

    def test_written_registry_revalidates_against_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _complete_fixture(directory)
            output = Path(directory) / "vln/results/final/r2r/registry.json"
            path, expected = builder.write_registry(
                output, fixture["selection_path"], directory
            )
            actual = builder.validate_registry(
                path, fixture["selection_path"], directory
            )
            self.assertEqual(actual, expected)

    def test_tampered_formal_manifest_is_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _complete_fixture(directory)
            manifest_path = fixture["manifest_paths"][("duet-r2r", "tent")]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["completed_at"] = "2026-08-22T00:02:00+00:00"
            _write_json(manifest_path, manifest)
            output = Path(directory) / "registry.json"
            with self.assertRaisesRegex(builder.RegistryError, "SHA256 mismatch"):
                builder.write_registry(
                    output, fixture["selection_path"], directory
                )
            self.assertFalse(output.exists())

    def test_builder_does_not_fall_back_to_logs_evidence_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _complete_fixture(directory)
            selection = fixture["selection"]
            record = selection["records"]["duet-r2r"]["tent"]
            canonical = fixture["manifest_paths"][("duet-r2r", "tent")]
            evidence = Path(directory) / "vln/results/logs/evidence/manifest.json"
            _write_json(
                evidence, json.loads(canonical.read_text(encoding="utf-8"))
            )
            record["formal_manifest_evidence_copy_path"] = evidence.relative_to(
                directory
            ).as_posix()
            _write_json(fixture["selection_path"], selection)
            canonical.unlink()
            with self.assertRaisesRegex(
                builder.RegistryError, "canonical formal manifest is missing"
            ):
                builder.build_registry(fixture["selection_path"], directory)

    def test_registry_document_rejects_incomplete_or_missing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _complete_fixture(directory)
            registry = builder.build_registry(
                fixture["selection_path"], directory
            )
            incomplete = deepcopy(registry)
            incomplete["registry_status"] = "pending"
            with self.assertRaisesRegex(builder.RegistryError, "registry_status"):
                builder.validate_registry_document(incomplete)

            missing = deepcopy(registry)
            del missing["records"]["duet-r2r"]["tent"][
                "formal_manifest_sha256"
            ]
            with self.assertRaisesRegex(
                builder.RegistryError, "lacks formal_manifest_sha256"
            ):
                builder.validate_registry_document(missing)


if __name__ == "__main__":
    unittest.main()
