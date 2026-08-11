import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/run_tta_adapter_parity_audit.py"
SPEC = importlib.util.spec_from_file_location("adapter_parity_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

from navtta_core.experiment.episode_order import (  # noqa: E402
    build_episode_order_manifest,
    prefix_episode_order_manifest,
)


class AdapterParityAuditTest(unittest.TestCase):
    @staticmethod
    def _write_json(path, document):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(document, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _frozen_search(self, root, batch_id="search-batch"):
        search_spec_path = (
            REPO_ROOT / "vln/experiments/tta_hparam_search_v1.json"
        )
        search_sha = MODULE.sha256(search_spec_path)
        search_spec = MODULE.read_json(search_spec_path)
        commit = MODULE.git("rev-parse", "HEAD")
        for method in MODULE.METHODS:
            method_root = Path(root) / method / batch_id
            parameters = {
                setting: ({
                    "lr_query": 1e-6,
                    "lr_self": 1e-7,
                    "mix_lambda": 0.5,
                    "query_threshold": 0.1,
                    "self_loss_weight": 0.1,
                } if method == "atena" else {
                    "lr": 1e-6,
                })
                for setting in MODULE.SETTINGS
            }
            final_tags = {}
            control_tags = {}
            stage_metrics = {"final": {}, "final_controls": {}}
            for stage in ("final", "final_controls"):
                stage_root = method_root / "stages" / stage
                candidate_count = (
                    int(search_spec["protocol"][
                        "full_val_seen_finalists_per_setting"
                    ])
                    if stage == "final" else 1
                )
                ordinal = 0
                for candidate_index in range(candidate_count):
                    for setting in MODULE.SETTINGS:
                        tag = "{}-{}-{}-{}".format(
                            method, stage, setting, candidate_index
                        )
                        job_dir = stage_root / "jobs" / str(ordinal)
                        config_path = job_dir / "parameters.json"
                        result_root = job_dir / "result"
                        config_method = (
                            method if stage == "final" else "source"
                        )
                        if stage == "final":
                            job_parameters = dict(parameters[setting])
                            varied_lr = (
                                "lr_query" if method == "atena" else "lr"
                            )
                            job_parameters[varied_lr] *= candidate_index + 1
                        else:
                            job_parameters = {
                                "action_selection": (
                                    "sample"
                                    if method == "feedtta" else "argmax"
                                ),
                                "action_seed": 0,
                            }
                        job = {
                            "batch_id": batch_id,
                            "ordinal": ordinal,
                            "base_run_tag": tag,
                            "run_tag": tag,
                            "attempt": 0,
                            "setting": setting,
                            "model": MODULE.MODEL[setting],
                            "family": (
                                "continuous"
                                if setting in MODULE.CONTINUOUS else "discrete"
                            ),
                            "benchmark": (
                                "reverie" if setting.endswith("reverie")
                                else "r2r-ce" if setting.endswith("r2r-ce")
                                else "r2r"
                            ),
                            "search_method": method,
                            "config_method": config_method,
                            "stage": stage,
                            "episodes": -1,
                            "order_seed": None,
                            "parameters": job_parameters,
                            "parent_run_tags": [],
                            "config_path": str(config_path),
                            "job_dir": str(job_dir),
                            "result_root": str(result_root),
                            "command": ["unit-test-runner", tag],
                        }
                        self._write_json(config_path, {
                            "schema": "navtta.vln_tta_job.v1",
                            "method": config_method,
                            "search_method": method,
                            "stage": stage,
                            "episodes": -1,
                            "order_seed": None,
                            "parameters": job_parameters,
                        })
                        self._write_json(job_dir / "job.json", job)
                        (job_dir / "exitcode").write_text(
                            "0\n", encoding="utf-8"
                        )
                        self._write_json(job_dir / "worker_state.json", {
                            "status": "finished",
                            "exit_code": 0,
                        })

                        metric_values = {
                            "SR": 50.0 - candidate_index,
                            "SPL": 45.0 - candidate_index,
                            "ORACLE_SR": 55.0 - candidate_index,
                            "OSR": 55.0 - candidate_index,
                            "RGS": 40.0 - candidate_index,
                            "RGSPL": 35.0 - candidate_index,
                        }
                        diagnostics_path = None
                        diagnostics_sha256 = None
                        adapter_diagnostics = None
                        if stage == "final":
                            diagnostics_path = (
                                result_root / "tta_diagnostics.json"
                            )
                            adapter_diagnostics = {
                                "relative_param_drift": 0.01,
                                "updates": 1,
                            }
                            self._write_json(diagnostics_path, {
                                "episode_count": int(search_spec[
                                    "setting_episode_counts"
                                ][setting]),
                                "adapter": adapter_diagnostics,
                            })
                            diagnostics_sha256 = MODULE.sha256(
                                diagnostics_path
                            )
                            diagnostics_path = str(diagnostics_path)
                        metrics = dict(job)
                        metrics.update({
                            "metrics": metric_values,
                            "expected_episodes": int(search_spec[
                                "setting_episode_counts"
                            ][setting]),
                            "diagnostics_path": diagnostics_path,
                            "diagnostics_sha256": diagnostics_sha256,
                            "adapter_diagnostics": adapter_diagnostics,
                            "requires_posthoc_late_collapse_check": True,
                        })
                        self._write_json(job_dir / "metrics.json", metrics)
                        if candidate_index == 0:
                            stage_metrics[stage][setting] = metric_values
                            if stage == "final":
                                final_tags[setting] = tag
                            else:
                                control_tags[setting] = tag
                        ordinal += 1
                self._write_json(stage_root / "stage_manifest.json", {
                    "schema": "navtta.vln_tta_search_stage.v1",
                    "batch_id": batch_id,
                    "git_commit": commit,
                    "spec_sha256": search_sha,
                    "method": method,
                    "stage": stage,
                    "episodes": -1,
                    "settings": list(MODULE.SETTINGS),
                    "job_count": ordinal,
                })
                self._write_json(stage_root / "SUMMARY.json", {
                    "schema": "navtta.vln_tta_stage_summary.v1",
                    "planned": ordinal,
                    "validated": ordinal,
                    "errors": [],
                    "terminal": True,
                    "promotion_ready": None,
                    "complete": True,
                })
            self._write_json(method_root / "FINAL_SELECTION.json", {
                "schema": "navtta.vln_tta_final_selection.v1",
                "method": method,
                "split": "val_seen",
                "finalist_stage": "final",
                "matched_source_stage": "final_controls",
                "git_commit": commit,
                "spec_sha256": search_sha,
                "settings": {
                    setting: {
                        "winner_run_tag": final_tags[setting],
                        "source_run_tag": control_tags[setting],
                        "frozen_parameters": parameters[setting],
                        "winner_metrics": stage_metrics["final"][setting],
                        "source_metrics": stage_metrics[
                            "final_controls"
                        ][setting],
                    }
                    for setting in MODULE.SETTINGS
                },
            })
            self._write_json(method_root / "FROZEN_HPARAMETERS.json", {
                "schema": "navtta.vln_tta_frozen_hparams.v1",
                "method": method,
                "split": "val_seen",
                "generated_from": "FINAL_SELECTION.json",
                "git_commit": commit,
                "spec_sha256": search_sha,
                "settings": parameters,
            })

    def test_spec_requires_exact_40_plus_8_plus_8_design(self):
        spec = MODULE.load_spec()
        self.assertEqual(spec["canonical_prefix_episodes"], 256)
        self.assertEqual(spec["expected_jobs"]["total"], 56)
        self.assertTrue(
            spec["protocol"]["ordinary_hparam_search_entry_forbidden"]
        )

    def test_plan_is_separate_immutable_and_uses_frozen_winners(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            search = root / "search"
            audit = root / "audit"
            self._frozen_search(search)
            with mock.patch.object(MODULE, "AUDIT_ROOT", audit):
                campaign, plan = MODULE.create_plan(
                    "audit-batch", "search-batch", search_root=search
                )
                jobs = MODULE.validate_plan(campaign, plan)

            self.assertEqual(len(jobs), 56)
            self.assertEqual(
                sum(job["kind"] == "adapter_zero_update" for job in jobs), 40
            )
            self.assertEqual(
                sum(job["kind"] == "source_argmax" for job in jobs), 8
            )
            self.assertEqual(
                sum(job["kind"] == "source_sampled" for job in jobs), 8
            )
            self.assertTrue(all(
                "run_tta_hparam_search.py" not in " ".join(job["command"])
                for job in jobs
            ))
            adapter = next(
                job for job in jobs if job["kind"] == "adapter_zero_update"
            )
            config = MODULE.read_json(adapter["config_path"])
            self.assertEqual(
                config["schema"], "navtta.vln_tta_adapter_parity_job.v1"
            )
            self.assertEqual(config["namespace"], "adapter_parity_audit")
            self.assertTrue(config["audit_zero_update"])
            self.assertFalse(config["audit_control"])
            self.assertEqual(config["episodes"], 256)

    def test_plan_rejects_tampered_audit_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            search = root / "search"
            audit = root / "audit"
            self._frozen_search(search)
            with mock.patch.object(MODULE, "AUDIT_ROOT", audit):
                campaign, plan = MODULE.create_plan(
                    "audit-batch", "search-batch", search_root=search
                )
                config_path = Path(plan["job_records"][0]["config_path"])
                config = MODULE.read_json(config_path)
                config["episodes"] = 255
                self._write_json(config_path, config)
                with self.assertRaisesRegex(
                    MODULE.AuditError, "audit config changed"
                ):
                    MODULE.validate_plan(campaign, plan)

    def test_frozen_loader_requires_selection_and_final_stage_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            search = Path(directory) / "search"
            self._frozen_search(search)
            selection_path = (
                search / "tent" / "search-batch" / "FINAL_SELECTION.json"
            )
            selection = MODULE.read_json(selection_path)
            selection["settings"]["duet-r2r"]["winner_run_tag"] = "missing"
            self._write_json(selection_path, selection)
            with self.assertRaisesRegex(
                MODULE.AuditError, "frozen parameters are invalid"
            ):
                MODULE.load_frozen(
                    search, "search-batch", MODULE.load_spec()
                )

    def test_frozen_loader_requires_all_five_finalists_per_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            search = Path(directory) / "search"
            self._frozen_search(search)
            job_dir = (
                search / "tent" / "search-batch" / "stages"
                / "final" / "jobs" / "8"
            )
            job_path = job_dir / "job.json"
            metrics_path = job_dir / "metrics.json"
            job = MODULE.read_json(job_path)
            metrics = MODULE.read_json(metrics_path)
            self.assertEqual(job["setting"], "duet-r2r")
            job["setting"] = "duet-reverie"
            metrics["setting"] = "duet-reverie"
            metrics["expected_episodes"] = 1423
            self._write_json(job_path, job)
            self._write_json(metrics_path, metrics)
            with self.assertRaisesRegex(
                MODULE.AuditError, "final setting coverage is invalid"
            ):
                MODULE.load_frozen(
                    search, "search-batch", MODULE.load_spec()
                )

    def test_frozen_loader_rejects_joint_document_parameter_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            search = Path(directory) / "search"
            self._frozen_search(search)
            method_root = search / "tent" / "search-batch"
            frozen_path = method_root / "FROZEN_HPARAMETERS.json"
            selection_path = method_root / "FINAL_SELECTION.json"
            frozen = MODULE.read_json(frozen_path)
            selection = MODULE.read_json(selection_path)
            arbitrary = {"lr": 0.123456789}
            frozen["settings"]["duet-r2r"] = arbitrary
            selection["settings"]["duet-r2r"][
                "frozen_parameters"
            ] = arbitrary
            self._write_json(frozen_path, frozen)
            self._write_json(selection_path, selection)

            with self.assertRaisesRegex(
                MODULE.AuditError, "frozen parameters are invalid"
            ):
                MODULE.load_frozen(
                    search, "search-batch", MODULE.load_spec()
                )

    def test_frozen_loader_rejects_unpaired_source_diagnostics_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            search = Path(directory) / "search"
            self._frozen_search(search)
            metrics_path = (
                search / "tent" / "search-batch" / "stages"
                / "final_controls" / "jobs" / "0" / "metrics.json"
            )
            metrics = MODULE.read_json(metrics_path)
            self.assertIsNone(metrics["diagnostics_path"])
            metrics["diagnostics_sha256"] = "0" * 64
            self._write_json(metrics_path, metrics)
            with self.assertRaisesRegex(
                MODULE.AuditError, "Source diagnostics path/hash mismatch"
            ):
                MODULE.load_frozen(
                    search, "search-batch", MODULE.load_spec()
                )

    def test_plan_rechecks_search_summary_and_all_job_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            search = root / "search"
            audit = root / "audit"
            self._frozen_search(search)
            with mock.patch.object(MODULE, "AUDIT_ROOT", audit):
                campaign, plan = MODULE.create_plan(
                    "audit-batch", "search-batch", search_root=search
                )
                stage = plan["frozen_hyperparameters"]["tent"][
                    "final_stage"
                ]
                summary_path = Path(stage["summary_path"])
                original_summary = summary_path.read_bytes()
                summary_path.write_bytes(original_summary + b" ")
                with self.assertRaisesRegex(
                    MODULE.AuditError, "stage summary changed"
                ):
                    MODULE.validate_plan(campaign, plan)
                summary_path.write_bytes(original_summary)

                evidence_path = next(
                    Path(path)
                    for files in stage["evidence_sha256"].values()
                    for path in files
                    if path.endswith("/metrics.json")
                )
                evidence_path.write_bytes(evidence_path.read_bytes() + b" ")
                with self.assertRaisesRegex(
                    MODULE.AuditError, "stage evidence changed"
                ):
                    MODULE.validate_plan(campaign, plan)

    def _formal_fixture(self, root):
        root = Path(root)
        dataset = root / "dataset.json"
        dataset.write_text("[]\n", encoding="utf-8")
        dataset_sha = MODULE.sha256(dataset)
        episodes = [
            {"episode_id": str(index), "scene_id": "scene"}
            for index in range(256)
        ]
        parent = build_episode_order_manifest(
            episodes,
            benchmark="unit-r2r",
            split="val_seen",
            dataset_path=str(dataset),
            dataset_sha256=dataset_sha,
            source_id_field="episode_id",
        )
        parent_path = root / "parent_order.json"
        self._write_json(parent_path, parent)
        prefix = prefix_episode_order_manifest(parent, 256)
        prefix["canonical_parent"] = {
            "path": str(parent_path.resolve()),
            "sha256": MODULE.sha256(parent_path),
            "episode_count": 256,
            "order_sha256": parent["order_sha256"],
        }
        prefix["audit_prefix"] = {
            "protocol": "zero_update_adapter_parity",
            "canonical_prefix": True,
            "episode_count": 256,
        }

        result_root = root / "result"
        prefix_path = result_root / "episode_order_prefix.json"
        diagnostics_path = result_root / "tta_diagnostics.json"
        output_path = result_root / "output.json"
        self._write_json(prefix_path, prefix)
        self._write_json(diagnostics_path, {"ok": True})
        self._write_json(output_path, {"episodes": 256})
        config_path = root / "audit_config.json"
        self._write_json(config_path, {"schema": "unit"})
        checkpoint = root / "checkpoint.bin"
        checkpoint.write_bytes(b"checkpoint")
        asset_manifest = root / "assets.json"
        environment_manifest = root / "environment.json"
        self._write_json(asset_manifest, {"assets": []})
        self._write_json(environment_manifest, {"environments": []})

        def metadata(path, hash_field="sha256"):
            path = Path(path).resolve()
            return {
                "path": str(path),
                "size": path.stat().st_size,
                hash_field: MODULE.sha256(path),
            }

        run_tag = "audit-unit"
        setting = "duet-r2r"
        formal_path = (
            root / "formal" /
            "{}-{}-val_seen-native".format(run_tag, setting) /
            "manifest.json"
        )
        commit = MODULE.git("rev-parse", "HEAD")
        manifest = {
            "run_id": "{}-{}-val_seen-native".format(run_tag, setting),
            "task": "vln",
            "benchmark": "unit-r2r",
            "model": "duet",
            "method": "tent",
            "run_tag": run_tag,
            "source_setting": "duet-r2r:val_seen:native:tent",
            "seed": 0,
            "git_commit": commit,
            "config": str(config_path),
            "config_overrides": ["python", "evaluate.py"],
            "checkpoint": metadata(checkpoint),
            "auxiliary_checkpoints": [
                dict(name="audit_job_config", **metadata(config_path)),
                dict(
                    name="canonical_episode_order_parent",
                    **metadata(parent_path)
                ),
            ],
            "dataset": {
                **metadata(dataset, "index_sha256"),
                "version": "unit-r2r",
                "stream_order_sha256": prefix["order_sha256"],
                "stream_content_sha256": dataset_sha,
            },
            "pinned_manifests": {
                "assets": metadata(asset_manifest),
                "environment": metadata(environment_manifest),
                "episode_order": metadata(prefix_path),
            },
            "hardware": {
                "hostname": "unit",
                "platform": "unit",
                "python": "3.9",
                "cuda_visible_devices": "0",
                "torch": "unit",
                "torch_cuda": "unit",
                "cuda_available": True,
                "cudnn": 1,
                "gpu_name": "unit-gpu",
                "gpu_capability": [0, 0],
            },
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:01:00+00:00",
            "status": "completed",
            "exit_code": 0,
            "result_artifacts": [
                dict(name=path.name, **metadata(path))
                for path in (prefix_path, diagnostics_path, output_path)
            ],
        }
        manifest[MODULE.IMMUTABLE_IDENTITY_SHA256_FIELD] = (
            MODULE.immutable_identity_sha256(manifest)
        )
        self._write_json(formal_path, manifest)
        job = {
            "run_tag": run_tag,
            "setting": setting,
            "model": "duet",
            "method": "tent",
            "run_data_version": "native",
            "config_path": str(config_path),
            "result_root": str(result_root),
            "prefix_manifest_path": str(prefix_path),
            "formal_manifest_path": str(formal_path),
        }
        plan = {
            "git_commit": commit,
            "identity_files": {
                "asset_manifest": {
                    "path": str(asset_manifest.resolve()),
                    "sha256": MODULE.sha256(asset_manifest),
                },
                "environment_manifest": {
                    "path": str(environment_manifest.resolve()),
                    "sha256": MODULE.sha256(environment_manifest),
                },
            },
            "episode_orders": {
                setting: {
                    "path": str(parent_path.resolve()),
                    "manifest_sha256": MODULE.sha256(parent_path),
                    "benchmark": "unit-r2r",
                    "dataset_path": str(dataset),
                    "dataset_sha256": dataset_sha,
                    "full_order_sha256": parent["order_sha256"],
                    "prefix_episode_ids": [
                        str(item["episode_id"])
                        for item in prefix["episodes"]
                    ],
                    "prefix_order_sha256": prefix["order_sha256"],
                }
            },
        }
        return job, plan, {
            "config": config_path,
            "prefix": prefix_path,
            "artifact": output_path,
            "manifest": formal_path,
        }

    def test_formal_manifest_rejects_config_order_artifact_and_identity_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            job, plan, paths = self._formal_fixture(directory)
            evidence = MODULE.validate_formal_manifest(job, plan)
            self.assertEqual(evidence["sha256"], MODULE.sha256(paths["manifest"]))

            for name in ("config", "prefix", "artifact"):
                with self.subTest(tamper=name):
                    path = paths[name]
                    original = path.read_bytes()
                    path.write_bytes(original + b" ")
                    with self.assertRaises(MODULE.AuditError):
                        MODULE.validate_formal_manifest(job, plan)
                    path.write_bytes(original)

            manifest = MODULE.read_json(paths["manifest"])
            manifest["config"] = "tampered.json"
            self._write_json(paths["manifest"], manifest)
            with self.assertRaisesRegex(MODULE.AuditError, "identity"):
                MODULE.validate_formal_manifest(job, plan)

    def test_job_validation_requires_positive_suppressed_attempts(self):
        spec = MODULE.load_spec()
        order = MODULE.order_identity("duet-r2r", spec)
        expected_ids = order["prefix_episode_ids"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / "result"
            job_dir = root / "job"
            result_root.mkdir()
            job_dir.mkdir()
            (job_dir / "exitcode").write_text("0\n", encoding="utf-8")
            predictions = [
                {"instr_id": identifier, "trajectory": [["vp", 0.0, 0.0]]}
                for identifier in expected_ids
            ]
            (result_root / "submit_val_seen.json").write_text(
                json.dumps(predictions), encoding="utf-8"
            )
            (result_root / "console.log").write_text(
                "Env name: val_seen, sr: 75.0, spl: 70.0, oracle_sr: 80.0\n",
                encoding="utf-8",
            )
            state_hash = "b" * 64
            diagnostics = {
                "schema": "navtta.vln_discrete_tta.v1",
                "method": "tent",
                "episode_count": 256,
                "action_steps": 256,
                "trajectory_steps": 256,
                "trajectory_sha256": "c" * 64,
                "audit_zero_update": True,
                "audit_control": False,
                "adapter": {
                    "updates": 0,
                    "episodes": 256,
                    "slow_updates": 0,
                    "relative_param_drift": 0.0,
                    "audit_mode": "zero_update_adapter_parity",
                    "audit_expected_episodes": 256,
                    "parameter_writes_suppressed": True,
                    "action_steps": 256,
                    "parameter_write_attempts": 3,
                    "suppressed_parameter_write_attempts": 3,
                    "optimizer_step_attempts": 2,
                    "suppressed_optimizer_step_attempts": 2,
                    "parameter_state_before_sha256": state_hash,
                    "parameter_state_after_sha256": state_hash,
                    "parameter_state_hash_match": True,
                    "model_state_before_sha256": state_hash,
                    "model_state_after_sha256": state_hash,
                    "model_state_hash_match": True,
                    "deployed_model_states_hash_match": True,
                },
            }
            diagnostics_path = result_root / "tta_diagnostics.json"
            diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
            job = {
                "run_tag": "unit",
                "ordinal": 0,
                "kind": "adapter_zero_update",
                "setting": "duet-r2r",
                "family": "discrete",
                "method": "tent",
                "job_dir": str(job_dir),
                "result_root": str(result_root),
            }
            plan = {"episode_orders": {"duet-r2r": order}}
            with mock.patch.object(
                MODULE, "validate_formal_manifest", return_value={
                    "path": "manifest.json", "sha256": "d" * 64,
                }
            ):
                result = MODULE.validate_job(job, plan)
            self.assertEqual(result["adapter"]["updates"], 0)

            diagnostics["adapter"]["model_state_hash_match"] = False
            diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.AuditError, "zero-update adapter invariants"
            ), mock.patch.object(
                MODULE, "validate_formal_manifest", return_value={
                    "path": "manifest.json", "sha256": "d" * 64,
                }
            ):
                MODULE.validate_job(job, plan)
            diagnostics["adapter"]["model_state_hash_match"] = True

            diagnostics["adapter"]["parameter_write_attempts"] = 0
            diagnostics["adapter"]["suppressed_parameter_write_attempts"] = 0
            diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.AuditError, "suppression/state-hash"
            ), mock.patch.object(
                MODULE, "validate_formal_manifest", return_value={
                    "path": "manifest.json", "sha256": "d" * 64,
                }
            ):
                MODULE.validate_job(job, plan)


if __name__ == "__main__":
    unittest.main()
