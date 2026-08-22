import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln/scripts"))
sys.path.insert(0, str(REPO_ROOT))

import build_reverie_benchmark_results as builder  # noqa: E402
from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.commit = "a" * 40
        self.order_seen = _digest("reverie-seen-order")
        self.order_unseen = _digest("reverie-unseen-order")
        self.order_test = _digest("reverie-test-order")
        self.dataset_seen = {
            setting: _digest(setting + "-seen-dataset")
            for setting in builder.SETTINGS
        }
        self.dataset_unseen = {
            "duet-reverie": _digest("duet-hamt-unseen-dataset"),
            "hamt-reverie": _digest("duet-hamt-unseen-dataset"),
            "goat-reverie": _digest("goat-unseen-dataset"),
        }
        self.dataset_test = {
            "duet-reverie": _digest("duet-hamt-test-dataset"),
            "hamt-reverie": _digest("duet-hamt-test-dataset"),
            "goat-reverie": _digest("goat-test-dataset"),
        }
        self.checkpoints = {
            setting: _digest(setting + "-checkpoint")
            for setting in builder.SETTINGS
        }
        self.parameters = {
            "source": {"action_selection": "argmax", "action_seed": 0},
            "tent": {"lr": 1e-5, "update_interval": 1},
            "fstta": {"lr_fast": 1e-4, "lr_slow": 1e-5, "m": 2},
            "eam": {"lr": 2e-6, "memory_size": 8},
            "feedtta": {"lr": 1e-6, "action_selection": "argmax"},
            "atena": {"lr_self": 1e-7, "action_selection": "argmax"},
        }
        self.registry_path = self.root / "vln/results/final/reverie/registry.json"
        self.spec_path = self.root / "vln/experiments/reverie_val_unseen_frozen_eval_v1.json"
        self.val_source_path = self.root / "vln/manifests/reverie_val_unseen_reused_source_controls.json"
        self.test_source_path = self.root / "vln/manifests/reverie_test_reused_source_submissions.json"
        self.val_batch_root = self.root / "vln/results/logs/reverie/frozen_val_unseen/unit-val"
        self.test_batch_root = self.root / "vln/results/logs/reverie/test_submissions/unit-test"
        self.val_orders, self.test_orders = self._build_orders()
        registry = self._build_registry()
        self._build_val_source_ledger()
        self._build_test_source_ledger()
        spec = self._build_spec(registry)
        self._build_val_batch(registry, spec)
        self._build_test_batch(registry, spec)

    def _metrics(self, setting_index, method_index, split):
        offset = 0.0 if split == "val_seen" else -10.0
        base = {
            "SR": 60.0 + setting_index * 2.0 + offset,
            "SPL": 52.0 + setting_index * 2.0 + offset,
            "RGS": 44.0 + setting_index * 2.0 + offset,
            "RGSPL": 38.0 + setting_index * 2.0 + offset,
        }
        if method_index < 0:
            return base
        return {
            key: value + (method_index + 1) * (0.5 if key != "SR" else 1.0)
            for key, value in base.items()
        }

    def _artifact(self, run_tag, setting, split, relative, content):
        path = self.root / "vln/results/tuning" / run_tag / setting / split / relative
        if isinstance(content, str):
            _write_text(path, content)
        else:
            _write_json(path, content)
        return {
            "name": str(relative).replace("\\", "/"),
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": builder._sha256(path),
        }, path

    def _manifest(
        self, setting, method, split, run_tag, order_sha, dataset_sha,
        artifacts, commit=None, parameters=None, order_binding=None,
    ):
        run_id = "{}-{}-{}-native".format(run_tag, setting, split)
        result_root = Path(artifacts[0]["path"]).parents[1]
        overrides = []
        if method != "source":
            order_directory = (
                (self.root / order_binding["path"]).resolve().parent
                if order_binding is not None
                else self.root / "vln/manifests/episode_order/fixture_seen"
            )
            overrides = [
                "--test", "--eval_splits", split, "--seed", "0",
                "--output_dir", str(result_root),
                "--episode_order_manifest", str(order_directory),
                *builder.tta_config_cli._discrete(
                    method,
                    json.loads(json.dumps(parameters, sort_keys=True)),
                    str(result_root / "tta_diagnostics.json"),
                ),
            ]
        manifest = {
            "run_id": run_id,
            "task": "vln",
            "benchmark": builder.BENCHMARK_FOR_SETTING[setting],
            "model": builder.MODEL_FOR_SETTING[setting],
            "method": method,
            "run_tag": run_tag,
            "source_setting": "{}:{}:native{}".format(
                setting, split, "" if method == "source" else ":" + method
            ),
            "seed": 0,
            "git_commit": commit or self.commit,
            "config": "/remote/config.json",
            "config_overrides": overrides,
            "checkpoint": {
                "path": "/remote/checkpoint",
                "size": 10,
                "sha256": self.checkpoints[setting],
            },
            "auxiliary_checkpoints": [],
            "dataset": {
                "stream_order_sha256": order_sha,
                "stream_content_sha256": dataset_sha,
            },
            "pinned_manifests": ({
                "episode_order": {
                    "path": str((self.root / order_binding["path"]).resolve()),
                    "size": (self.root / order_binding["path"]).stat().st_size,
                    "sha256": order_binding["sha256"],
                }
            } if order_binding is not None else {}),
            "hardware": {"gpu_name": "fixture GPU"},
            "started_at": "2026-08-22T00:00:00+00:00",
            "completed_at": "2026-08-22T00:01:00+00:00",
            "status": "completed",
            "exit_code": 0,
            "result_artifacts": artifacts,
        }
        manifest["immutable_identity_sha256"] = immutable_identity_sha256(manifest)
        relative = "vln/results/runs/{}/manifest.json".format(run_id)
        path = self.root / relative
        _write_json(path, manifest)
        return {
            "path": relative,
            "remote_path": "/root/autodl-tmp/code/NavTTA/{}".format(relative),
            "sha256": builder._sha256(path),
            "identity": manifest["immutable_identity_sha256"],
        }

    def _metric_line(self, split, metrics):
        return "Env name: {}, sr: {}, spl: {}, rgs: {}, rgspl: {}\n".format(
            split, metrics["SR"], metrics["SPL"], metrics["RGS"],
            metrics["RGSPL"],
        )

    def _build_orders(self):
        val_bindings, test_bindings = {}, {}
        for key, settings, benchmark in (
            (
                "duet_hamt", builder.SETTINGS[:2],
                builder.BENCHMARK_FOR_SETTING[builder.SETTINGS[0]],
            ),
            (
                "goat", builder.SETTINGS[2:],
                builder.BENCHMARK_FOR_SETTING[builder.SETTINGS[2]],
            ),
        ):
            val_path = self.root / "vln/manifests/episode_order/reverie_{}/val_unseen.json".format(key)
            val_dataset = self.dataset_unseen[settings[0]]
            _write_json(val_path, {
                "schema": "navtta.episode_order.v1",
                "benchmark": benchmark,
                "split": "val_unseen",
                "episode_count": builder.VAL_UNSEEN_EPISODES,
                "order_sha256": self.order_unseen,
                "dataset": {"sha256": val_dataset},
                "episodes": [
                    {"episode_id": str(index)}
                    for index in range(builder.VAL_UNSEEN_EPISODES)
                ],
            })
            val_bindings[key] = {
                "path": val_path.relative_to(self.root).as_posix(),
                "sha256": builder._sha256(val_path),
                "settings": list(settings),
            }
            test_path = self.root / "vln/manifests/episode_order/reverie_{}/test.json".format(key)
            test_dataset = self.dataset_test[settings[0]]
            _write_json(test_path, {
                "schema": "navtta.episode_order.v1",
                "benchmark": benchmark,
                "split": "test",
                "episode_count": builder.TEST_EPISODES,
                "order_sha256": self.order_test,
                "dataset": {"sha256": test_dataset},
                "episodes": [
                    {"episode_id": str(index)}
                    for index in range(builder.TEST_EPISODES)
                ],
            })
            test_bindings[key] = {
                "path": test_path.relative_to(self.root).as_posix(),
                "sha256": builder._sha256(test_path),
                "settings": list(settings),
            }
        return val_bindings, test_bindings

    def _build_registry(self):
        records = {}
        protocol = {
            "benchmark": "reverie",
            "split": "val_seen",
            "episode_count": builder.VAL_SEEN_EPISODES,
            "order_seed": 0,
            "episode_order_sha256": self.order_seen,
            "primary_metric": "RGSPL",
            "selection_scope": "val_seen_seed0_hyperparameter_selection",
        }
        for setting_index, setting in enumerate(builder.SETTINGS):
            source_metrics = self._metrics(setting_index, -1, "val_seen")
            records[setting] = {}
            for method_index, method in enumerate(builder.ALL_METHODS):
                metrics = (
                    source_metrics if method == "source"
                    else self._metrics(setting_index, method_index - 1, "val_seen")
                )
                run_tag = (
                    "seen-source-batch" if method == "source"
                    else "seen-{}-{}".format(setting, method)
                )
                metric, _ = self._artifact(
                    run_tag, setting, "val_seen", "logs/valid.txt",
                    self._metric_line("val_seen", metrics),
                )
                formal = self._manifest(
                    setting, method, "val_seen", run_tag, self.order_seen,
                    self.dataset_seen[setting], [metric],
                    parameters=self.parameters[method],
                )
                records[setting][method] = {
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
                    "metric_artifact_sha256": metric["sha256"],
                }
        seen_source_path = self.root / "vln/manifests/reverie_reused_source_controls.json"
        selected_path = self.root / "vln/results/final/reverie/selected_winners.json"
        selected_records = {}
        for setting in builder.SETTINGS:
            selected_records[setting] = {}
            for method in builder.METHODS:
                record = records[setting][method]
                selected_records[setting][method] = {
                    **record,
                    "selection_status": "ready",
                    "source_metrics": records[setting]["source"]["metrics"],
                    "origin": "new_full_candidate",
                }
        source_records = {}
        for setting in builder.SETTINGS:
            record = records[setting]["source"]
            metric_path = (
                self.root / "vln/results/tuning" / record["run_tag"]
                / setting / "val_seen/logs/valid.txt"
            )
            source_records[setting] = {
                "model": builder.MODEL_FOR_SETTING[setting],
                "run_tag": record["run_tag"],
                "parameters": record["parameters"],
                "metrics": record["metrics"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "dataset_sha256": self.dataset_seen[setting],
                "formal_manifest_path": record["formal_manifest_path"],
                "formal_manifest_sha256": record["formal_manifest_sha256"],
                "immutable_identity_sha256": record[
                    "formal_immutable_identity_sha256"
                ],
                "metrics_artifact_path": str(metric_path),
                "metrics_artifact_sha256": record["metric_artifact_sha256"],
            }
        _write_json(seen_source_path, {
            "schema": builder.VAL_SEEN_SOURCE_SCHEMA,
            "benchmark": "reverie",
            "split": "val_seen",
            "source_protocol": "standard_argmax",
            "episode_count": builder.VAL_SEEN_EPISODES,
            "order_seed": 0,
            "episode_order_sha256": self.order_seen,
            "dataset_version": "fixture REVERIE val_seen",
            "hardware": "fixture GPU",
            "source_batch": {
                "batch_id": "seen-source-batch", "git_commit": self.commit,
            },
            "records": source_records,
        })
        source_binding = {
            "path": seen_source_path.relative_to(self.root).as_posix(),
            "sha256": builder._sha256(seen_source_path),
        }
        search_id = "fixture-search"
        incumbent_path = self.root / "vln/experiments/fixture_incumbent.json"
        _write_json(incumbent_path, {"schema": "fixture.incumbent.v1"})
        search_spec_path = self.root / "vln/experiments/fixture_search.json"
        _write_json(search_spec_path, {
            "schema": builder.SEARCH_SPEC_SCHEMA,
            "experiment_id": search_id,
            "dependencies": {
                "source_registry": source_binding,
                "incumbent_spec": {
                    "path": incumbent_path.relative_to(self.root).as_posix(),
                    "sha256": builder._sha256(incumbent_path),
                },
            },
        })
        search_root = (
            self.root / "vln/results/logs/reverie/hparam_search" / search_id
        )
        raw_records = {
            setting: {
                method: {
                    key: selected_records[setting][method][key]
                    for key in (
                        "origin", "parameters", "run_tag", "metrics",
                        "source_metrics", "delta_vs_source_pp",
                        "formal_manifest_sha256",
                    )
                }
                for method in builder.METHODS
            }
            for setting in builder.SETTINGS
        }
        raw_selection_path = search_root / "FINAL_SELECTION.json"
        _write_json(raw_selection_path, {
            "schema": builder.SEARCH_SELECTION_SCHEMA,
            "experiment_id": search_id,
            "split": "val_seen",
            "primary_metric": "RGSPL",
            "spec_sha256": builder._sha256(search_spec_path),
            "git_commit": self.commit,
            "records": raw_records,
        })
        search_batch_path = search_root / "BATCH.json"
        _write_json(search_batch_path, {
            "schema": builder.SEARCH_BATCH_SCHEMA,
            "batch_id": search_id,
            "experiment_id": search_id,
            "spec_sha256": builder._sha256(search_spec_path),
            "git_commit": self.commit,
            "source_registry_sha256": source_binding["sha256"],
            "incumbent_spec_sha256": builder._sha256(incumbent_path),
        })
        _write_json(selected_path, {
            "schema": "navtta.vln_reverie_final_selection.v1",
            "selection_status": "complete",
            "purpose": "fixture",
            "protocol": protocol,
            "source_ledger": source_binding,
            "search_evidence": {
                "experiment_id": search_id,
                "batch_id": search_id,
                "git_commit": self.commit,
                "batch_binding_sha256": builder._sha256(search_batch_path),
                "final_selection_sha256": builder._sha256(raw_selection_path),
                "search_spec_path": search_spec_path.relative_to(self.root).as_posix(),
                "search_spec_sha256": builder._sha256(search_spec_path),
                "spec_sha256": builder._sha256(search_spec_path),
            },
            "records": selected_records,
        })
        registry = {
            "schema": builder.REGISTRY_SCHEMA,
            "registry_status": "complete",
            "protocol_status": "complete_val_seen_seed0_selection",
            "provenance_status": "complete_formal_manifest_identity_verified",
            "publication_status": "selection_registry_not_test_result",
            "benchmark": "reverie",
            "split": "val_seen",
            "protocol": protocol,
            "source_ledger": source_binding,
            "selected_winners": {
                "path": selected_path.relative_to(self.root).as_posix(),
                "sha256": builder._sha256(selected_path),
            },
            "supervision_categories": {
                "source_no_adaptation": {
                    "methods": ["source"], "uses_episode_feedback": False,
                },
                "unsupervised_tta": {
                    "methods": list(builder.TEST_METHODS),
                    "uses_episode_feedback": False,
                },
                "binary_episode_feedback_tta": {
                    "methods": list(builder.UNAVAILABLE_TEST_METHODS),
                    "uses_episode_feedback": True,
                    "feedback": "binary_navigation_success",
                },
            },
            "records": records,
        }
        _write_json(self.registry_path, registry)
        return registry

    def _build_val_source_ledger(self):
        records = {}
        for setting_index, setting in enumerate(builder.SETTINGS):
            metrics = self._metrics(setting_index, -1, "val_unseen")
            run_tag = "val-unseen-{}-source".format(setting)
            metric, metric_path = self._artifact(
                run_tag, setting, "val_unseen", "logs/valid.txt",
                self._metric_line("val_unseen", metrics),
            )
            formal = self._manifest(
                setting, "source", "val_unseen", run_tag,
                self.order_unseen, self.dataset_unseen[setting], [metric],
                order_binding=self.val_orders[
                    "goat" if setting == "goat-reverie" else "duet_hamt"
                ],
            )
            records[setting] = {
                "model": builder.MODEL_FOR_SETTING[setting],
                "run_tag": run_tag,
                "parameters": self.parameters["source"],
                "metrics": metrics,
                "checkpoint_sha256": self.checkpoints[setting],
                "dataset_sha256": self.dataset_unseen[setting],
                "episode_order_sha256": self.order_unseen,
                "evidence_status": "ready",
                "formal_manifest_path": formal["path"],
                "formal_manifest_sha256": formal["sha256"],
                "immutable_identity_sha256": formal["identity"],
                "metrics_artifact_path": str(metric_path),
                "metrics_artifact_sha256": metric["sha256"],
            }
        ledger = {
            "schema": builder.VAL_SOURCE_SCHEMA,
            "benchmark": "reverie",
            "split": "val_unseen",
            "source_protocol": "standard_argmax",
            "episode_count": builder.VAL_UNSEEN_EPISODES,
            "canonical_order_seed": 0,
            "episode_order_sha256": self.order_unseen,
            "source_execution_policy": {
                "execution": "reuse_only", "rerun_forbidden": True,
            },
            "source_batch": {"batch_id": "source", "git_commit": self.commit},
            "records": records,
        }
        _write_json(self.val_source_path, ledger)

    def _submission_payload(self):
        return [
            {"instr_id": str(index), "trajectory": [], "predObjId": 0}
            for index in range(builder.TEST_EPISODES)
        ]

    def _build_test_source_ledger(self):
        records = {}
        for setting in builder.SETTINGS:
            run_tag = "test-{}-source".format(setting)
            submission, submission_path = self._artifact(
                run_tag, setting, "test", "preds/submit_test.json",
                self._submission_payload(),
            )
            log, _ = self._artifact(
                run_tag, setting, "test", "logs/valid.txt", "test complete\n"
            )
            formal = self._manifest(
                setting, "source", "test", run_tag, self.order_test,
                self.dataset_test[setting], [log, submission],
                order_binding=self.test_orders[
                    "goat" if setting == "goat-reverie" else "duet_hamt"
                ],
            )
            records[setting] = {
                "model": builder.MODEL_FOR_SETTING[setting],
                "run_tag": run_tag,
                "checkpoint_sha256": self.checkpoints[setting],
                "dataset_sha256": self.dataset_test[setting],
                "episode_order_sha256": self.order_test,
                "formal_manifest_path": formal["path"],
                "formal_manifest_sha256": formal["sha256"],
                "immutable_identity_sha256": formal["identity"],
                "submission_artifact_path": str(submission_path),
                "submission_artifact_sha256": submission["sha256"],
                "submission_artifact_size": submission["size"],
            }
        ledger = {
            "schema": builder.TEST_SOURCE_SCHEMA,
            "benchmark": "reverie",
            "split": "test",
            "episode_count": builder.TEST_EPISODES,
            "canonical_order_seed": 0,
            "episode_order_sha256": self.order_test,
            "hidden_ground_truth": True,
            "local_metrics_available": False,
            "source_execution_policy": {
                "execution": "reuse_existing_submission_only",
                "rerun_forbidden": True,
            },
            "source_batch": {"batch_id": "source", "git_commit": self.commit},
            "records": records,
        }
        _write_json(self.test_source_path, ledger)

    def _build_spec(self, registry):
        jobs = []
        for setting in builder.SETTINGS:
            for method in builder.METHODS:
                record = registry["records"][setting][method]
                jobs.append({
                    "setting": setting,
                    "model": builder.MODEL_FOR_SETTING[setting],
                    "method": method,
                    "selected_run_tag": record["run_tag"],
                    "selected_formal_manifest_sha256": record["formal_manifest_sha256"],
                })
        spec = {
            "schema": builder.SPEC_SCHEMA,
            "experiment_id": "fixture-reverie-frozen-eval",
            "status": "materialized_from_complete_val_seen_registry",
            "registry_dependency": {
                "path": self.registry_path.relative_to(self.root).as_posix(),
                "sha256": builder._sha256(self.registry_path),
                "schema": builder.REGISTRY_SCHEMA,
                "required_status": "complete",
            },
            "source_control": {
                "manifest": self.val_source_path.relative_to(self.root).as_posix(),
                "sha256": builder._sha256(self.val_source_path),
                "execution": "reuse_only",
                "rerun_forbidden": True,
            },
            "protocol": {
                "benchmark": "reverie",
                "split": "val_unseen",
                "episode_count": builder.VAL_UNSEEN_EPISODES,
                "canonical_order_seed": 0,
                "episode_order_sha256": self.order_unseen,
                "order_seed_cli_forbidden": True,
                "full_split_only": True,
                "frozen_from": "reverie_val_seen_seed0_final_registry",
                "selection_on_val_unseen": False,
                "reported_metrics": list(builder.METRICS),
                "order_manifests": self.val_orders,
                "binary_feedback": {
                    "label": "navigation_success_only",
                    "grounding_feedback_forbidden": True,
                    "simulator_distance_fallback_forbidden": True,
                    "feedtta_timing": "eager_once_per_episode",
                    "atena_timing": "lazy_only_when_entropy_gate_queries",
                },
            },
            "matrix": {
                "model_order": list(builder.MODELS),
                "setting_order": list(builder.SETTINGS),
                "method_order": list(builder.METHODS),
                "strict_model_barrier": True,
                "parallel_methods_within_model": True,
                "jobs": jobs,
            },
            "execution": {
                "model_phase_order": "duet_then_hamt_then_goat",
                "max_workers_by_model": {"duet": 2, "hamt": 2, "goat": 4},
                "estimated_gpu_memory_mib_by_method": {
                    "tent": 4096, "fstta": 5120, "eam": 6144,
                    "feedtta": 8192, "atena": 10240,
                },
                "max_aggregate_gpu_memory_mib": 29000,
                "minimum_free_gpu_memory_mib": 4096,
                "emergency_abort_used_gpu_memory_mib": 30500,
                "launch_stagger_seconds": 5,
                "poll_seconds": 2,
                "formal_requires_clean_tracked_tree": True,
                "formal_requires_review_confirmation": True,
                "retry_assigns_new_run_tag": True,
            },
            "budget": {
                "source_execution_jobs": 0,
                "tta_jobs": 15,
                "total_executed_jobs": 15,
            },
            "test_submission_policy": {
                "benchmark": "reverie",
                "native_split": "test",
                "episode_count": builder.TEST_EPISODES,
                "canonical_order_seed": 0,
                "episode_order_sha256": self.order_test,
                "hidden_ground_truth": True,
                "execution_purpose": "submission_generation_only",
                "local_metric_computation_forbidden": True,
                "selection_or_ranking_on_test_forbidden": True,
                "eligible_tta_methods": list(builder.TEST_METHODS),
                "eligible_tta_submission_jobs": 9,
                "legal_online_feedback_interface": None,
                "source": {
                    "execution": "reuse_existing_submission_only",
                    "rerun_forbidden": True,
                    "ledger": self.test_source_path.relative_to(self.root).as_posix(),
                    "sha256": builder._sha256(self.test_source_path),
                },
                "order_manifests": self.test_orders,
                "unavailable_without_legal_online_feedback": {
                    "feedtta": {
                        "status": "N/A", "launch_policy": "fail_closed",
                        "reason": "hidden test lacks online feedback",
                    },
                    "atena": {
                        "status": "N/A", "launch_policy": "fail_closed",
                        "reason": "hidden test lacks online queried feedback",
                    },
                },
            },
        }
        _write_json(self.spec_path, spec)
        return spec

    def _diagnostics(self, method, episodes, metrics=None, hidden_test=False):
        adapter = {
            "episodes": episodes, "updates": episodes,
            "relative_param_drift": 0.01,
        }
        if hidden_test or method in builder.TEST_METHODS:
            supervision, endpoint = "unsupervised", None
        elif method == "feedtta":
            supervision = "binary_navigation_success_feedback"
            endpoint = builder.FEEDBACK_ENDPOINT[method]
            successes = int(round(float(metrics["SR"]) * episodes / 100.0))
            adapter.update({
                "feedback_episodes": episodes,
                "successful_feedback_episodes": successes,
                "failed_feedback_episodes": episodes - successes,
                "action_selection_protocol": "target_native_argmax",
            })
        else:
            supervision = "binary_navigation_success_feedback"
            endpoint = builder.FEEDBACK_ENDPOINT[method]
            adapter.update({
                "queries": episodes // 2,
                "self_label_episodes": episodes - episodes // 2,
                "feedback_observed_episodes": episodes // 2,
                "query_gate_evaluations": episodes,
                "self_prediction_evaluations": episodes,
            })
        return {
            "method": method,
            "episode_count": episodes,
            "action_selection": "target_native_argmax",
            "supervision": supervision,
            "binary_feedback_endpoint": endpoint,
            "adapter": adapter,
        }

    def _config(self, batch_id, setting, method, parameters, registry_record,
                spec, split):
        provenance = {
            "selection_benchmark": "reverie",
            "selection_split": "val_seen",
            "evaluation_split": split,
            "canonical_order_seed": 0,
            "registry": spec["registry_dependency"],
            "selected_anchor": registry_record,
        }
        if split == "val_unseen":
            provenance.update({
                "selection_on_val_unseen": False,
                "source_ledger": {
                    "path": spec["source_control"]["manifest"],
                    "sha256": spec["source_control"]["sha256"],
                },
            })
        else:
            provenance.update({
                "selection_on_test": False,
                "hidden_ground_truth": True,
                "submission_generation_only": True,
                "source_submission_ledger": spec["test_submission_policy"]["source"],
            })
        return {
            "schema": "navtta.vln_tta_job.v1",
            "namespace": "tuning",
            "batch_id": batch_id,
            "stage": "frozen_val_unseen" if split == "val_unseen" else "frozen_test_submission",
            "setting": setting,
            "method": method,
            "search_method": method,
            "episodes": -1,
            "parameters": parameters,
            "frozen_evaluation_provenance": provenance,
        }

    def _build_val_batch(self, registry, spec):
        batch_id = "unit-val"
        batch = {
            "schema": builder.BATCH_SCHEMA,
            "batch_id": batch_id,
            "experiment_id": spec["experiment_id"],
            "spec_path": str(self.spec_path),
            "spec_sha256": builder._sha256(self.spec_path),
            "git_commit": self.commit,
            "registry": spec["registry_dependency"],
            "gpu": 0,
            "mode": "val_unseen",
            "source_execution_jobs": 0,
            "tta_jobs": 15,
            "source_ledger": spec["source_control"],
        }
        _write_json(self.val_batch_root / "BATCH.json", batch)
        results = []
        ordinal = 0
        for setting_index, setting in enumerate(builder.SETTINGS):
            for method_index, method in enumerate(builder.METHODS):
                registry_record = registry["records"][setting][method]
                metrics = self._metrics(setting_index, method_index, "val_unseen")
                base_run_tag = (
                    "{}-frozen-{:02d}-{}-{:02d}-{}-{}".format(
                        batch_id, setting_index,
                        builder.MODEL_FOR_SETTING[setting], method_index, method,
                        builder._job_digest(setting, method, registry_record),
                    )
                )
                run_tag = base_run_tag
                metric, metric_path = self._artifact(
                    run_tag, setting, "val_unseen", "logs/valid.txt",
                    self._metric_line("val_unseen", metrics),
                )
                diagnostics_value = self._diagnostics(
                    method, builder.VAL_UNSEEN_EPISODES, metrics
                )
                diagnostics, diagnostics_path = self._artifact(
                    run_tag, setting, "val_unseen", "tta_diagnostics.json",
                    diagnostics_value,
                )
                formal = self._manifest(
                    setting, method, "val_unseen", run_tag,
                    self.order_unseen, self.dataset_unseen[setting],
                    [metric, diagnostics], parameters=self.parameters[method],
                    order_binding=self.val_orders[
                        "goat" if setting == "goat-reverie" else "duet_hamt"
                    ],
                )
                attempt = (
                    self.val_batch_root / "models"
                    / "{:02d}-{}".format(
                        setting_index, builder.MODEL_FOR_SETTING[setting]
                    )
                    / "jobs" / base_run_tag / "attempt-00"
                )
                config = self._config(
                    batch_id, setting, method, self.parameters[method],
                    registry_record, spec, "val_unseen",
                )
                config_path = attempt / "parameters.json"
                _write_json(config_path, config)
                metadata = {
                    "schema": builder.VAL_JOB_SCHEMA,
                    "batch_id": batch_id,
                    "spec_path": str(self.spec_path),
                    "spec_sha256": batch["spec_sha256"],
                    "attempt": 0,
                    "run_tag": run_tag,
                    "base_run_tag": base_run_tag,
                    "model_index": setting_index,
                    "method_index": method_index,
                    "ordinal": ordinal,
                    "gpu": 0,
                    "setting": setting,
                    "model": builder.MODEL_FOR_SETTING[setting],
                    "method": method,
                    "parameters": self.parameters[method],
                    "selected_anchor": registry_record,
                    "episode_count": builder.VAL_UNSEEN_EPISODES,
                    "canonical_order_seed": 0,
                    "git_commit": self.commit,
                    "expected_benchmark": builder.BENCHMARK_FOR_SETTING[setting],
                    "expected_checkpoint_sha256": self.checkpoints[setting],
                    "expected_dataset_sha256": self.dataset_unseen[setting],
                    "expected_episode_order_sha256": self.order_unseen,
                    "episode_order_manifest": str((
                        self.root / self.val_orders[
                            "goat" if setting == "goat-reverie" else "duet_hamt"
                        ]["path"]
                    ).resolve()),
                    "source_ledger_path": str(self.val_source_path),
                    "source_ledger_sha256": spec["source_control"]["sha256"],
                    "result_root": str(metric_path.parents[1]),
                    "formal_manifest": formal["remote_path"],
                    "parameters_path": str(config_path),
                    "parameters_sha256": builder._sha256(config_path),
                    "command": [
                        str(self.root / "vln/scripts/run_source_eval.sh"),
                        setting, "val_unseen", "0", "--run-tag", run_tag,
                        "--tta-config", str(config_path),
                    ],
                }
                _write_json(attempt / "job.json", metadata)
                _write_text(attempt / "exitcode", "0\n")
                row = {
                    **metadata,
                    "metrics": metrics,
                    "metric_artifact": str(metric_path),
                    "metric_artifact_sha256": metric["sha256"],
                    "metric_line": self._metric_line("val_unseen", metrics).strip(),
                    "diagnostics_path": str(diagnostics_path),
                    "diagnostics_sha256": diagnostics["sha256"],
                    "feedback_endpoint": builder.FEEDBACK_ENDPOINT.get(method),
                    "adapter_diagnostics": diagnostics_value["adapter"],
                    "formal_manifest_sha256": formal["sha256"],
                    "formal_immutable_identity_sha256": formal["identity"],
                }
                _write_json(attempt / "metrics.json", row)
                results.append({
                    key: row[key] for key in (
                        "setting", "method", "run_tag", "parameters", "metrics",
                        "feedback_endpoint", "formal_manifest",
                        "formal_manifest_sha256",
                    )
                })
                ordinal += 1
        _write_json(self.val_batch_root / "SUMMARY.json", {
            "schema": builder.VAL_SUMMARY_SCHEMA,
            "experiment_id": spec["experiment_id"],
            "source_execution_jobs": 0,
            "total_jobs": 15,
            "complete": True,
            "states": {
                "completed": 15, "finished": 0, "failed": 0,
                "invalid": 0, "running": 0, "orphaned": 0, "pending": 0,
            },
            "results": results,
        })

    def _build_test_batch(self, registry, spec):
        batch_id = "unit-test"
        batch = {
            "schema": builder.BATCH_SCHEMA,
            "batch_id": batch_id,
            "experiment_id": spec["experiment_id"],
            "spec_path": str(self.spec_path),
            "spec_sha256": builder._sha256(self.spec_path),
            "git_commit": self.commit,
            "registry": spec["registry_dependency"],
            "gpu": 0,
            "mode": "test_submission",
            "source_execution_jobs": 0,
            "tta_jobs": 9,
            "test_source_submission_ledger": spec["test_submission_policy"]["source"],
        }
        _write_json(self.test_batch_root / "BATCH.json", batch)
        submissions = []
        ordinal = 0
        for setting_index, setting in enumerate(builder.SETTINGS):
            for method_index, method in enumerate(builder.TEST_METHODS):
                registry_record = registry["records"][setting][method]
                base_run_tag = (
                    "{}-submission-{:02d}-{}-{:02d}-{}-{}".format(
                        batch_id, setting_index,
                        builder.MODEL_FOR_SETTING[setting], method_index, method,
                        builder._job_digest(setting, method, registry_record),
                    )
                )
                run_tag = base_run_tag
                submission, submission_path = self._artifact(
                    run_tag, setting, "test", "preds/submit_test.json",
                    self._submission_payload(),
                )
                diagnostics_value = self._diagnostics(
                    method, builder.TEST_EPISODES, hidden_test=True
                )
                diagnostics, diagnostics_path = self._artifact(
                    run_tag, setting, "test", "tta_diagnostics.json",
                    diagnostics_value,
                )
                log, _ = self._artifact(
                    run_tag, setting, "test", "logs/valid.txt", "test complete\n"
                )
                formal = self._manifest(
                    setting, method, "test", run_tag, self.order_test,
                    self.dataset_test[setting], [log, submission, diagnostics],
                    parameters=self.parameters[method],
                    order_binding=self.test_orders[
                        "goat" if setting == "goat-reverie" else "duet_hamt"
                    ],
                )
                attempt = (
                    self.test_batch_root / "models"
                    / "{:02d}-{}".format(
                        setting_index, builder.MODEL_FOR_SETTING[setting]
                    )
                    / "jobs" / base_run_tag / "attempt-00"
                )
                config = self._config(
                    batch_id, setting, method, self.parameters[method],
                    registry_record, spec, "test",
                )
                config_path = attempt / "parameters.json"
                _write_json(config_path, config)
                metadata = {
                    "schema": builder.TEST_JOB_SCHEMA,
                    "batch_id": batch_id,
                    "spec_path": str(self.spec_path),
                    "spec_sha256": batch["spec_sha256"],
                    "attempt": 0,
                    "run_tag": run_tag,
                    "base_run_tag": base_run_tag,
                    "model_index": setting_index,
                    "method_index": method_index,
                    "ordinal": ordinal,
                    "gpu": 0,
                    "setting": setting,
                    "model": builder.MODEL_FOR_SETTING[setting],
                    "method": method,
                    "parameters": self.parameters[method],
                    "selected_anchor": registry_record,
                    "episode_count": builder.TEST_EPISODES,
                    "canonical_order_seed": 0,
                    "hidden_ground_truth": True,
                    "submission_generation_only": True,
                    "git_commit": self.commit,
                    "expected_benchmark": builder.BENCHMARK_FOR_SETTING[setting],
                    "expected_checkpoint_sha256": self.checkpoints[setting],
                    "expected_dataset_sha256": self.dataset_test[setting],
                    "expected_episode_order_sha256": self.order_test,
                    "episode_order_manifest": str((
                        self.root / self.test_orders[
                            "goat" if setting == "goat-reverie" else "duet_hamt"
                        ]["path"]
                    ).resolve()),
                    "test_source_ledger_path": str(self.test_source_path),
                    "test_source_ledger_sha256": spec[
                        "test_submission_policy"
                    ]["source"]["sha256"],
                    "result_root": str(submission_path.parents[1]),
                    "formal_manifest": formal["remote_path"],
                    "parameters_path": str(config_path),
                    "parameters_sha256": builder._sha256(config_path),
                    "command": [
                        str(self.root / "vln/scripts/run_source_eval.sh"),
                        setting, "test", "0", "--run-tag", run_tag,
                        "--tta-config", str(config_path),
                    ],
                }
                _write_json(attempt / "job.json", metadata)
                _write_text(attempt / "exitcode", "0\n")
                row = {
                    **metadata,
                    "submission_path": str(submission_path),
                    "submission_sha256": submission["sha256"],
                    "submission_size": submission["size"],
                    "diagnostics_path": str(diagnostics_path),
                    "diagnostics_sha256": diagnostics["sha256"],
                    "formal_manifest_sha256": formal["sha256"],
                    "formal_immutable_identity_sha256": formal["identity"],
                    "metrics": None,
                }
                _write_json(attempt / "submission.json", row)
                submissions.append({
                    key: row[key] for key in (
                        "setting", "method", "run_tag", "parameters",
                        "submission_path", "submission_sha256",
                        "submission_size", "formal_manifest",
                        "formal_manifest_sha256",
                    )
                })
                ordinal += 1
        _write_json(self.test_batch_root / "SUBMISSIONS.json", {
            "schema": builder.TEST_SUMMARY_SCHEMA,
            "experiment_id": spec["experiment_id"],
            "split": "test",
            "hidden_ground_truth": True,
            "submission_generation_only": True,
            "local_metrics": None,
            "source_execution_jobs": 0,
            "reused_source_submissions": 3,
            "tta_submission_jobs": 9,
            "unavailable_methods": {
                "feedtta": "N/A: no legal online binary feedback",
                "atena": "N/A: no legal online binary feedback",
            },
            "complete": True,
            "states": {
                "completed": 9, "finished": 0, "failed": 0,
                "invalid": 0, "running": 0, "orphaned": 0, "pending": 0,
            },
            "submissions": submissions,
        })


class ReverieBenchmarkResultsTest(unittest.TestCase):
    def _fixture(self, directory):
        stack = mock.patch.multiple(
            builder, VAL_SEEN_EPISODES=3, VAL_UNSEEN_EPISODES=4,
            TEST_EPISODES=3,
            VAL_SEEN_ORDER=_digest("reverie-seen-order"),
            VAL_UNSEEN_ORDER=_digest("reverie-unseen-order"),
            TEST_ORDER=_digest("reverie-test-order"),
        )
        stack.start()
        self.addCleanup(stack.stop)
        return Fixture(directory)

    def test_complete_evidence_builds_metric_and_submission_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            before = {
                path.relative_to(fixture.root).as_posix(): builder._sha256(path)
                for path in fixture.root.rglob("*") if path.is_file()
            }
            document = builder.build_results(
                fixture.val_batch_root, fixture.test_batch_root,
                repo_root=fixture.root,
            )
            self.assertFalse(document["selection_policy"]["selection_on_val_unseen"])
            self.assertFalse(document["selection_policy"]["selection_or_ranking_on_test"])
            self.assertEqual(document["protocols"]["test"]["native_split"], "test")
            self.assertFalse(document["protocols"]["test"]["local_metrics_available"])
            for setting in builder.SETTINGS:
                methods = document["records"][setting]["methods"]
                self.assertEqual(set(methods), set(builder.ALL_METHODS))
                for method in builder.ALL_METHODS:
                    self.assertIsNone(methods[method]["test"]["metrics"])
                for method in ("source",) + builder.TEST_METHODS:
                    self.assertIsNotNone(methods[method]["test"]["submission"])
                for method in builder.UNAVAILABLE_TEST_METHODS:
                    self.assertEqual(methods[method]["test"]["status"], "N/A")
                    self.assertIsNone(methods[method]["test"]["submission"])
            output = fixture.root / "final/results.json"
            report = fixture.root / "final/RESULTS.md"
            builder.write_results(document, output, report)
            builder.validate_outputs(document, output, report)
            text = report.read_text(encoding="utf-8")
            self.assertIn("official hidden split is named `test`", text)
            self.assertIn("submission ready; no local metrics", text)
            self.assertIn("N/A (no legal online feedback)", text)
            after = {
                path.relative_to(fixture.root).as_posix(): builder._sha256(path)
                for path in fixture.root.rglob("*")
                if path.is_file()
                and not path.relative_to(fixture.root).as_posix().startswith("final/")
            }
            self.assertEqual(before, after)

    def test_incomplete_val_unseen_batch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            path = fixture.val_batch_root / "SUMMARY.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["complete"] = False
            value["states"]["completed"] = 14
            value["states"]["running"] = 1
            _write_json(path, value)
            with self.assertRaisesRegex(builder.ArchiveError, "complete mismatch"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_incomplete_test_submission_batch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            path = fixture.test_batch_root / "SUBMISSIONS.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["complete"] = False
            value["states"]["completed"] = 8
            value["states"]["pending"] = 1
            _write_json(path, value)
            with self.assertRaisesRegex(builder.ArchiveError, "complete mismatch"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_hidden_test_metric_fabrication_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            path = fixture.test_batch_root / "SUBMISSIONS.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["local_metrics"] = {"RGSPL": 99.0}
            _write_json(path, value)
            with self.assertRaisesRegex(builder.ArchiveError, "local_metrics mismatch"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_feedback_method_in_hidden_test_batch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            path = fixture.test_batch_root / "SUBMISSIONS.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["submissions"][0]["method"] = "feedtta"
            _write_json(path, value)
            with self.assertRaisesRegex(builder.ArchiveError, "matrix is incomplete"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_val_unseen_parameter_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            path = fixture.val_batch_root / "SUMMARY.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["results"][0]["parameters"] = {"lr": 0.5}
            _write_json(path, value)
            with self.assertRaisesRegex(builder.ArchiveError, "parameters drifted"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_selected_winner_digest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            path = fixture.root / "vln/results/final/reverie/selected_winners.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["purpose"] = "tampered"
            _write_json(path, value)
            with self.assertRaisesRegex(builder.ArchiveError, "selected-winner binding"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_noncanonical_downstream_order_digests_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            registry_path, registry, _ = builder._load_registry(
                fixture.root, fixture.registry_path
            )
            spec = json.loads(fixture.spec_path.read_text(encoding="utf-8"))
            spec["protocol"]["episode_order_sha256"] = _digest("other-val-order")
            _write_json(fixture.spec_path, spec)
            with self.assertRaisesRegex(builder.ArchiveError, "noncanonical"):
                builder._load_spec(
                    fixture.root, fixture.spec_path, registry_path, registry
                )

            spec["protocol"]["episode_order_sha256"] = fixture.order_unseen
            spec["test_submission_policy"]["episode_order_sha256"] = _digest(
                "other-test-order"
            )
            _write_json(fixture.spec_path, spec)
            with self.assertRaisesRegex(builder.ArchiveError, "noncanonical"):
                builder._load_spec(
                    fixture.root, fixture.spec_path, registry_path, registry
                )

    def test_unknown_frozen_parameter_is_rejected(self):
        with self.assertRaisesRegex(builder.ArchiveError, "unknown parameters"):
            builder._validate_parameters(
                "tent", {"lr": 1e-5, "update_interval": 1, "rogue": 7},
                "fixture Tent",
            )

    def test_zero_threshold_atena_must_query_every_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            diagnostics_path = fixture.root / "atena-diagnostics.json"
            metrics = fixture._metrics(0, 4, "val_unseen")
            _write_json(
                diagnostics_path,
                fixture._diagnostics(
                    "atena", builder.VAL_UNSEEN_EPISODES, metrics
                ),
            )
            row = {
                "run_tag": "zero-threshold-atena",
                "method": "atena",
                "parameters": {"query_threshold": 0.0},
            }
            with self.assertRaisesRegex(builder.ArchiveError, "must query every"):
                builder._validate_diagnostics(diagnostics_path, row, metrics)

    def test_invalid_val_seen_source_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            registry = json.loads(fixture.registry_path.read_text(encoding="utf-8"))
            ledger = json.loads(fixture.root.joinpath(
                "vln/manifests/reverie_reused_source_controls.json"
            ).read_text(encoding="utf-8"))
            ledger["schema"] = "fixture.invalid"
            _write_json(
                fixture.root / "vln/manifests/reverie_reused_source_controls.json",
                ledger,
            )
            with self.assertRaisesRegex(builder.ArchiveError, "schema mismatch"):
                builder._load_val_seen_source_ledger(
                    fixture.root / "vln/manifests/reverie_reused_source_controls.json",
                    registry, registry["protocol"],
                )

    def test_noncanonical_job_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            cache_path = next(fixture.val_batch_root.rglob("metrics.json"))
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache["ordinal"] = 99
            _write_json(cache_path, cache)
            job_path = cache_path.with_name("job.json")
            job = json.loads(job_path.read_text(encoding="utf-8"))
            job["ordinal"] = 99
            _write_json(job_path, job)
            with self.assertRaisesRegex(builder.ArchiveError, "ordinal mismatch"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_extra_val_metric_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            cache = json.loads(next(
                fixture.val_batch_root.rglob("metrics.json")
            ).read_text(encoding="utf-8"))
            rogue = Path(cache["result_root"]) / "rogue/valid.txt"
            _write_text(rogue, cache["metric_line"] + "\n")
            with self.assertRaisesRegex(builder.ArchiveError, "result tree"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_extra_test_submission_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            cache = json.loads(next(
                fixture.test_batch_root.rglob("submission.json")
            ).read_text(encoding="utf-8"))
            rogue = Path(cache["result_root"]) / "rogue/submit_test_copy.json"
            _write_json(rogue, fixture._submission_payload())
            with self.assertRaisesRegex(builder.ArchiveError, "result tree"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_cached_diagnostics_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            cache_path = next(fixture.val_batch_root.rglob("metrics.json"))
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache["adapter_diagnostics"]["updates"] += 1
            _write_json(cache_path, cache)
            with self.assertRaisesRegex(builder.ArchiveError, "cached adapter"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_cached_metric_line_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            cache_path = next(fixture.val_batch_root.rglob("metrics.json"))
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache["metric_line"] = "forged metric line"
            _write_json(cache_path, cache)
            with self.assertRaisesRegex(builder.ArchiveError, "cached metric line"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_summary_feedback_endpoint_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            summary_path = fixture.val_batch_root / "SUMMARY.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["results"][0]["feedback_endpoint"] = "forged"
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(builder.ArchiveError, "summary feedback"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_changed_nonselected_manifest_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            cache = json.loads(next(
                fixture.test_batch_root.rglob("submission.json")
            ).read_text(encoding="utf-8"))
            Path(cache["result_root"]).joinpath("logs/valid.txt").write_text(
                "changed but still metric-free\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(builder.ArchiveError, "artifact changed"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_val_seen_tta_dataset_must_match_source_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            registry = json.loads(fixture.registry_path.read_text(encoding="utf-8"))
            record = registry["records"]["duet-reverie"]["tent"]
            manifest_path = fixture.root / record["formal_manifest_path"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset"]["stream_content_sha256"] = _digest(
                "different-dataset-snapshot"
            )
            manifest["immutable_identity_sha256"] = immutable_identity_sha256(
                manifest
            )
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(builder.ArchiveError, "dataset digest"):
                builder._validate_manifest(
                    fixture.root,
                    record["formal_manifest_path"],
                    builder._sha256(manifest_path),
                    "duet-reverie", "tent", "val_seen", record["run_tag"],
                    record["checkpoint_sha256"], fixture.order_seen,
                    dataset_sha256=fixture.dataset_seen["duet-reverie"],
                    immutable_sha256=manifest["immutable_identity_sha256"],
                    parameters=record["parameters"],
                )

    def test_attempt_configuration_digest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            cache = next(fixture.val_batch_root.rglob("metrics.json"))
            config_path = cache.with_name("parameters.json")
            value = json.loads(config_path.read_text(encoding="utf-8"))
            value["episodes"] = 1
            _write_json(config_path, value)
            with self.assertRaisesRegex(builder.ArchiveError, "path/digest mismatch"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_new_process_group_token_is_verified_but_legacy_cache_stays_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            cache_path = next(fixture.val_batch_root.rglob("metrics.json"))
            attempt = cache_path.parent
            metadata_path = attempt / "job.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            token = builder._expected_process_group_token(metadata, attempt)
            metadata["process_group_token"] = token
            _write_json(metadata_path, metadata)
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache["process_group_token"] = token
            _write_json(cache_path, cache)
            builder.build_results(
                fixture.val_batch_root, fixture.test_batch_root,
                repo_root=fixture.root,
            )
            metadata["process_group_token"] = "0" * 64
            cache["process_group_token"] = "0" * 64
            _write_json(metadata_path, metadata)
            _write_json(cache_path, cache)
            with self.assertRaisesRegex(builder.ArchiveError, "process-group token"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_changed_test_submission_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            cache = next(fixture.test_batch_root.rglob("submission.json"))
            row = json.loads(cache.read_text(encoding="utf-8"))
            Path(row["submission_path"]).write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(builder.ArchiveError, "artifact changed"):
                builder.build_results(
                    fixture.val_batch_root, fixture.test_batch_root,
                    repo_root=fixture.root,
                )

    def test_final_document_rejects_invented_test_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            document = builder.build_results(
                fixture.val_batch_root, fixture.test_batch_root,
                repo_root=fixture.root,
            )
            document["records"]["duet-reverie"]["methods"]["tent"]["test"][
                "metrics"
            ] = {name: 99.0 for name in builder.METRICS}
            with self.assertRaisesRegex(builder.ArchiveError, "status is invalid"):
                builder.validate_results_document(document)


if __name__ == "__main__":
    unittest.main()
