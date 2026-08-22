import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln/scripts"))
sys.path.insert(0, str(REPO_ROOT))

import build_r2r_benchmark_results as builder  # noqa: E402
from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.order_seen = _digest("seen-order")
        self.order_unseen = _digest("unseen-order")
        self.dataset_seen = _digest("seen-dataset")
        self.dataset_unseen = _digest("unseen-dataset")
        self.commit = "a" * 40
        self.checkpoints = {
            setting: _digest(setting + "-checkpoint")
            for setting in builder.SETTINGS
        }
        self.parameters = {
            "source": {"action_selection": "argmax", "action_seed": 0},
            "tent": {"lr": 1e-5, "update_interval": 1},
            "fstta": {"lr_fast": 1e-4, "lr_slow": 1e-5, "m": 8},
            "eam": {"lr": 2e-6, "memory_size": 64},
            "feedtta": {"lr": 1e-6, "action_selection": "argmax"},
            "atena": {"lr_self": 1e-7, "action_selection": "argmax"},
        }
        self.registry_path = (
            self.root / "vln/results/final/r2r/registry.json"
        )
        self.spec_path = (
            self.root / "vln/experiments/r2r_val_unseen_frozen_eval_v1.json"
        )
        self.source_path = (
            self.root / "vln/manifests/r2r_val_unseen_reused_source_controls.json"
        )
        self.batch_root = (
            self.root
            / "vln/results/logs/r2r/frozen_val_unseen/unit-test-batch"
        )
        self._build()

    def _manifest(
        self,
        setting,
        method,
        split,
        run_tag,
        order_digest,
        dataset_digest,
        artifacts,
    ):
        run_id = "{}-{}-{}-native".format(run_tag, setting, split)
        manifest = {
            "run_id": run_id,
            "task": "vln",
            "benchmark": builder.BENCHMARK_FOR_SETTING[setting],
            "model": builder.MODEL_FOR_SETTING[setting],
            "method": method,
            "run_tag": run_tag,
            "source_setting": "{}:{}:native{}".format(
                setting,
                split,
                "" if method == "source" and split == "val_unseen" else ":" + method,
            ),
            "seed": 0,
            "git_commit": self.commit,
            "config": "/remote/config.json",
            "config_overrides": [],
            "checkpoint": {
                "path": "/remote/checkpoint",
                "size": 10,
                "sha256": self.checkpoints[setting],
            },
            "auxiliary_checkpoints": [],
            "dataset": {
                "stream_order_sha256": order_digest,
                "stream_content_sha256": dataset_digest,
            },
            "pinned_manifests": {},
            "hardware": {"gpu_name": "fixture GPU"},
            "started_at": "2026-08-22T00:00:00+00:00",
            "completed_at": "2026-08-22T00:01:00+00:00",
            "status": "completed",
            "exit_code": 0,
            "result_artifacts": [
                {
                    "name": name,
                    "path": "/remote/output/{}".format(name),
                    "size": index + 1,
                    "sha256": digest,
                }
                for index, (name, digest) in enumerate(artifacts)
            ],
        }
        manifest["immutable_identity_sha256"] = immutable_identity_sha256(
            manifest
        )
        relative = "vln/results/runs/{}/manifest.json".format(run_id)
        path = self.root / relative
        _write_json(path, manifest)
        return {
            "path": relative,
            "remote_path": "/root/autodl-tmp/code/NavTTA/{}".format(relative),
            "sha256": builder._sha256(path),
            "identity": manifest["immutable_identity_sha256"],
        }

    def _seen_metrics(self, setting_index, method_index):
        source_sr = 70.0 + setting_index * 3.0
        source_spl = 60.0 + setting_index * 2.0
        if method_index < 0:
            return {"SR": source_sr, "SPL": source_spl}
        return {
            "SR": source_sr + (method_index + 1) * 0.2,
            "SPL": source_spl + (method_index + 1) * 0.1,
        }

    def _unseen_metrics(self, setting_index, method_index):
        source_sr = 62.0 + setting_index * 3.0
        source_spl = 52.0 + setting_index * 2.0
        if method_index < 0:
            return {"SR": source_sr, "SPL": source_spl}
        return {
            "SR": source_sr + (method_index - 1) * 0.15,
            "SPL": source_spl + method_index * 0.12,
        }

    def _build_registry(self):
        records = {}
        for setting_index, setting in enumerate(builder.SETTINGS):
            setting_records = {}
            source_metrics = self._seen_metrics(setting_index, -1)
            for method_index, method in enumerate(builder.ALL_METHODS):
                metrics = (
                    source_metrics
                    if method == "source"
                    else self._seen_metrics(setting_index, method_index - 1)
                )
                run_tag = "seen-{}-{}".format(setting, method)
                formal = self._manifest(
                    setting,
                    method,
                    "val_seen",
                    run_tag,
                    self.order_seen,
                    self.dataset_seen,
                    [("logs/valid.txt", _digest(run_tag + "-metric"))],
                )
                setting_records[method] = {
                    "method": method,
                    "supervision_category": builder.SUPERVISION_FOR_METHOD[method],
                    "parameters": self.parameters[method],
                    "run_tag": run_tag,
                    "metrics": metrics,
                    "delta_vs_source_pp": builder._delta(metrics, source_metrics),
                    "checkpoint_sha256": self.checkpoints[setting],
                    "formal_manifest_path": formal["path"],
                    "formal_manifest_sha256": formal["sha256"],
                    "formal_immutable_identity_sha256": formal["identity"],
                }
            records[setting] = setting_records
        registry = {
            "schema": builder.REGISTRY_SCHEMA,
            "registry_status": "complete",
            "protocol_status": "complete_val_seen_seed0_selection",
            "provenance_status": "complete_formal_manifest_identity_verified",
            "publication_status": "selection_registry_not_publication_final",
            "benchmark": "r2r",
            "split": "val_seen",
            "protocol": {
                "episode_count": 1021,
                "order_seed": 0,
                "episode_order_sha256": self.order_seen,
                "source_protocol": "standard_argmax",
                "metric_unit": "percentage_points",
            },
            "supervision_categories": {
                "source_no_adaptation": {
                    "methods": ["source"],
                    "uses_episode_feedback": False,
                },
                "unsupervised_tta": {
                    "methods": ["tent", "fstta", "eam"],
                    "uses_episode_feedback": False,
                },
                "binary_episode_feedback_tta": {
                    "methods": ["feedtta", "atena"],
                    "uses_episode_feedback": True,
                    "feedback": "binary_navigation_success",
                },
            },
            "records": records,
        }
        _write_json(self.registry_path, registry)
        return registry

    def _build_source_ledger(self):
        records = {}
        for setting_index, setting in enumerate(builder.SETTINGS):
            run_tag = "unseen-{}-source".format(setting)
            metric_digest = _digest(run_tag + "-metric")
            formal = self._manifest(
                setting,
                "source",
                "val_unseen",
                run_tag,
                self.order_unseen,
                self.dataset_unseen,
                [("logs/valid.txt", metric_digest)],
            )
            records[setting] = {
                "model": builder.MODEL_FOR_SETTING[setting],
                "run_tag": run_tag,
                "parameters": self.parameters["source"],
                "metrics": self._unseen_metrics(setting_index, -1),
                "checkpoint_sha256": self.checkpoints[setting],
                "dataset_sha256": self.dataset_unseen,
                "episode_order_sha256": self.order_unseen,
                "evidence_status": "ready",
                "formal_manifest_path": formal["path"],
                "formal_manifest_sha256": formal["sha256"],
                "immutable_identity_sha256": formal["identity"],
                "metrics_artifact_path": "/remote/output/logs/valid.txt",
                "metrics_artifact_sha256": metric_digest,
            }
        ledger = {
            "schema": builder.SOURCE_SCHEMA,
            "benchmark": "r2r",
            "split": "val_unseen",
            "source_protocol": "standard_argmax",
            "episode_count": 2349,
            "canonical_order_seed": 0,
            "episode_order_sha256": self.order_unseen,
            "dataset_version": "fixture",
            "hardware": "fixture GPU",
            "source_execution_policy": {
                "execution": "reuse_only",
                "rerun_forbidden": True,
                "recovery_policy": "fixture",
            },
            "records": records,
        }
        _write_json(self.source_path, ledger)
        return ledger

    def _build_orders(self):
        bindings = {}
        for name, settings in (
            ("duet_hamt", ["duet-r2r", "hamt-r2r"]),
            ("goat", ["goat-r2r"]),
        ):
            path = self.root / "vln/manifests/episode_order" / name / "val_unseen.json"
            _write_json(path, {
                "schema": "navtta.episode_order.v1",
                "split": "val_unseen",
                "episode_count": 2349,
                "order_sha256": self.order_unseen,
            })
            bindings[name] = {
                "path": path.relative_to(self.root).as_posix(),
                "sha256": builder._sha256(path),
                "settings": settings,
            }
        return bindings

    def _build_spec(self, registry, order_bindings):
        jobs = []
        for setting in builder.SETTINGS:
            for method in builder.METHODS:
                record = registry["records"][setting][method]
                jobs.append({
                    "setting": setting,
                    "model": builder.MODEL_FOR_SETTING[setting],
                    "method": method,
                    "selected_run_tag": record["run_tag"],
                    "selected_formal_manifest_sha256": record[
                        "formal_manifest_sha256"
                    ],
                })
        spec = {
            "schema": builder.SPEC_SCHEMA,
            "experiment_id": "fixture-r2r-unseen",
            "registry_dependency": {
                "path": self.registry_path.relative_to(self.root).as_posix(),
                "sha256": builder._sha256(self.registry_path),
                "schema": builder.REGISTRY_SCHEMA,
                "required_status": "complete",
            },
            "source_control": {
                "manifest": self.source_path.relative_to(self.root).as_posix(),
                "sha256": builder._sha256(self.source_path),
                "execution": "reuse_only",
                "rerun_forbidden": True,
            },
            "protocol": {
                "benchmark": "r2r",
                "split": "val_unseen",
                "episode_count": 2349,
                "canonical_order_seed": 0,
                "episode_order_sha256": self.order_unseen,
                "order_seed_cli_forbidden": True,
                "full_split_only": True,
                "frozen_from": "fixture",
                "selection_on_val_unseen": False,
                "report_all_cells": True,
                "order_manifests": order_bindings,
            },
            "matrix": {
                "model_order": list(builder.MODEL_FOR_SETTING.values()),
                "setting_order": list(builder.SETTINGS),
                "method_order": list(builder.METHODS),
                "strict_model_barrier": True,
                "parallel_methods_within_model": True,
                "jobs": jobs,
            },
        }
        _write_json(self.spec_path, spec)
        return spec

    def _build_batch(self, registry, spec):
        batch = {
            "schema": builder.BATCH_SCHEMA,
            "batch_id": "unit-test-batch",
            "experiment_id": spec["experiment_id"],
            "spec_path": "/remote/spec.json",
            "spec_sha256": builder._sha256(self.spec_path),
            "registry": spec["registry_dependency"],
            "source_ledger": {
                "path": spec["source_control"]["manifest"],
                "sha256": spec["source_control"]["sha256"],
            },
            "gpu": 0,
            "source_execution_jobs": 0,
            "tta_jobs": 15,
        }
        _write_json(self.batch_root / "BATCH.json", batch)
        results = []
        ordinal = 0
        for setting_index, setting in enumerate(builder.SETTINGS):
            source_metrics = self._unseen_metrics(setting_index, -1)
            for method_index, method in enumerate(builder.METHODS):
                registry_record = registry["records"][setting][method]
                run_tag = "unseen-{}-{}".format(setting, method)
                metrics = self._unseen_metrics(setting_index, method_index)
                metric_digest = _digest(run_tag + "-metric")
                diagnostics_digest = _digest(run_tag + "-diagnostics")
                formal = self._manifest(
                    setting,
                    method,
                    "val_unseen",
                    run_tag,
                    self.order_unseen,
                    self.dataset_unseen,
                    [
                        ("logs/valid.txt", metric_digest),
                        ("tta_diagnostics.json", diagnostics_digest),
                    ],
                )
                summary_row = {
                    "setting": setting,
                    "method": method,
                    "run_tag": run_tag,
                    "parameters": self.parameters[method],
                    "metrics": metrics,
                    "feedback_endpoint": builder.FEEDBACK_ENDPOINT.get(method),
                    "formal_manifest": formal["remote_path"],
                    "formal_manifest_sha256": formal["sha256"],
                }
                results.append(summary_row)
                attempt = (
                    self.batch_root
                    / "models"
                    / "{:02d}-{}".format(setting_index, builder.MODEL_FOR_SETTING[setting])
                    / "jobs"
                    / run_tag
                    / "attempt-00"
                )
                row = {
                    "schema": builder.JOB_SCHEMA,
                    "batch_id": batch["batch_id"],
                    "spec_path": "/remote/spec.json",
                    "spec_sha256": batch["spec_sha256"],
                    "attempt": 0,
                    "run_tag": run_tag,
                    "base_run_tag": run_tag,
                    "model_index": setting_index,
                    "method_index": method_index,
                    "ordinal": ordinal,
                    "setting": setting,
                    "model": builder.MODEL_FOR_SETTING[setting],
                    "method": method,
                    "parameters": self.parameters[method],
                    "selected_anchor": registry_record,
                    "episode_count": 2349,
                    "canonical_order_seed": 0,
                    "git_commit": self.commit,
                    "expected_benchmark": builder.BENCHMARK_FOR_SETTING[setting],
                    "expected_checkpoint_sha256": self.checkpoints[setting],
                    "expected_dataset_sha256": self.dataset_unseen,
                    "expected_episode_order_sha256": self.order_unseen,
                    "source_ledger_path": "/remote/source.json",
                    "source_ledger_sha256": spec["source_control"]["sha256"],
                    "result_root": "/remote/result",
                    "formal_manifest": formal["remote_path"],
                    "command": ["runner"],
                    "metrics": metrics,
                    "metric_artifact": "/remote/output/logs/valid.txt",
                    "metric_artifact_sha256": metric_digest,
                    "metric_line": "Env name: val_unseen, sr: {}, spl: {}".format(
                        metrics["SR"], metrics["SPL"]
                    ),
                    "diagnostics_path": "/remote/output/tta_diagnostics.json",
                    "diagnostics_sha256": diagnostics_digest,
                    "feedback_endpoint": builder.FEEDBACK_ENDPOINT.get(method),
                    "adapter_diagnostics": {},
                    "formal_manifest_sha256": formal["sha256"],
                    "formal_immutable_identity_sha256": formal["identity"],
                }
                _write_json(attempt / "metrics.json", row)
                _write_json(attempt / "parameters.json", {
                    "schema": "navtta.vln_tta_job.v1",
                    "namespace": "tuning",
                    "batch_id": batch["batch_id"],
                    "stage": "frozen_val_unseen",
                    "setting": setting,
                    "method": method,
                    "search_method": method,
                    "episodes": -1,
                    "parameters": self.parameters[method],
                    "frozen_evaluation_provenance": {
                        "selection_benchmark": "r2r",
                        "selection_split": "val_seen",
                        "evaluation_split": "val_unseen",
                        "selection_on_val_unseen": False,
                        "canonical_order_seed": 0,
                        "registry": spec["registry_dependency"],
                        "selected_anchor": registry_record,
                        "source_ledger": {
                            "path": spec["source_control"]["manifest"],
                            "sha256": spec["source_control"]["sha256"],
                        },
                    },
                })
                ordinal += 1
        _write_json(self.batch_root / "SUMMARY.json", {
            "schema": builder.SUMMARY_SCHEMA,
            "experiment_id": spec["experiment_id"],
            "source_execution_jobs": 0,
            "total_jobs": 15,
            "complete": True,
            "states": {
                "completed": 15,
                "finished": 0,
                "failed": 0,
                "invalid": 0,
                "running": 0,
                "orphaned": 0,
                "pending": 0,
            },
            "source_evidence_blockers": [],
            "results": results,
        })

    def _build(self):
        registry = self._build_registry()
        self._build_source_ledger()
        order_bindings = self._build_orders()
        spec = self._build_spec(registry, order_bindings)
        self._build_batch(registry, spec)


class R2RBenchmarkResultsTest(unittest.TestCase):
    def test_complete_matrix_builds_and_only_explicit_outputs_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            before = {
                path.relative_to(fixture.root).as_posix(): builder._sha256(path)
                for path in fixture.root.rglob("*")
                if path.is_file()
            }
            document = builder.build_results(
                fixture.batch_root, repo_root=fixture.root
            )
            self.assertEqual(document["schema"], builder.RESULT_SCHEMA)
            self.assertFalse(
                document["selection_policy"]["selection_on_val_unseen"]
            )
            self.assertEqual(set(document["records"]), set(builder.SETTINGS))
            for setting in builder.SETTINGS:
                methods = document["records"][setting]["methods"]
                self.assertEqual(set(methods), set(builder.ALL_METHODS))
                for method in builder.ALL_METHODS:
                    self.assertEqual(
                        methods[method]["frozen_parameters"],
                        fixture.parameters[method],
                    )
                    self.assertIn("formal_manifest_sha256", methods[method]["val_seen"]["provenance"])
                    self.assertIn("formal_manifest_sha256", methods[method]["val_unseen"]["provenance"])

            output = fixture.root / "final-output/results.json"
            report = fixture.root / "final-output/RESULTS.md"
            builder.write_results(document, output, report)
            builder.validate_outputs(document, output, report)
            self.assertIn("## Main comparison", report.read_text(encoding="utf-8"))
            self.assertIn("## Frozen parameters", report.read_text(encoding="utf-8"))
            after = {
                path.relative_to(fixture.root).as_posix(): builder._sha256(path)
                for path in fixture.root.rglob("*")
                if path.is_file() and "final-output/" not in path.as_posix()
            }
            self.assertEqual(before, after)

    def test_parameter_drift_is_rejected_even_when_downloaded_files_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            summary_path = fixture.batch_root / "SUMMARY.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            target = summary["results"][0]
            target["parameters"] = dict(target["parameters"], lr=9e-2)
            _write_json(summary_path, summary)
            metrics_path = next(fixture.batch_root.rglob("metrics.json"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["parameters"] = target["parameters"]
            _write_json(metrics_path, metrics)
            config_path = metrics_path.with_name("parameters.json")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["parameters"] = target["parameters"]
            _write_json(config_path, config)
            with self.assertRaisesRegex(builder.ArchiveError, "parameters drifted"):
                builder.build_results(fixture.batch_root, repo_root=fixture.root)

    def test_unauthenticated_metric_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            metrics_path = next(fixture.batch_root.rglob("metrics.json"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["metric_artifact_sha256"] = _digest("not-in-manifest")
            _write_json(metrics_path, metrics)
            with self.assertRaisesRegex(
                builder.ArchiveError, "metric_artifact_sha256 is not manifest-authenticated"
            ):
                builder.build_results(fixture.batch_root, repo_root=fixture.root)

    def test_incomplete_batch_is_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            summary_path = fixture.batch_root / "SUMMARY.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["complete"] = False
            summary["states"]["completed"] = 14
            summary["states"]["running"] = 1
            _write_json(summary_path, summary)
            output = fixture.root / "never-written.json"
            with self.assertRaisesRegex(builder.ArchiveError, "complete mismatch"):
                document = builder.build_results(
                    fixture.batch_root, repo_root=fixture.root
                )
                builder.write_results(document, output, fixture.root / "never.md")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
