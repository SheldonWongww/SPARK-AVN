from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln/scripts"))

import run_reverie_frozen_transfer as runner  # noqa: E402
from tta_config_cli import translate  # noqa: E402
from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402


SPEC_PATH = REPO_ROOT / "vln/experiments/reverie_r2r_frozen_transfer_v1.json"


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class ReverieFrozenTransferTest(unittest.TestCase):
    def test_source_registry_uses_canonical_self_validating_manifests(self):
        spec = runner.load_spec(SPEC_PATH)
        _, source = runner._validate_source_registry(spec)
        for setting, record in source["records"].items():
            self.assertTrue(
                record["formal_manifest_path"].startswith("vln/results/runs/")
            )
            self.assertNotIn("/results/source/", record["formal_manifest_path"])
            manifest_path = REPO_ROOT / record["formal_manifest_path"]
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(
                runner._sha256(manifest_path), record["formal_manifest_sha256"]
            )

    def test_source_registry_rejects_manifest_identity_mismatch(self):
        spec = runner.load_spec(SPEC_PATH)
        source_path = REPO_ROOT / spec["source_control"]["manifest"]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(
            dir=runner.FORMAL_ROOT
        ) as formal_directory, tempfile.TemporaryDirectory(
            dir=REPO_ROOT / "vln/results/logs"
        ) as directory:
            directory = Path(directory)
            bad_manifest = json.loads(
                (
                    REPO_ROOT
                    / source["records"]["duet-reverie"]["formal_manifest_path"]
                ).read_text(encoding="utf-8")
            )
            bad_manifest["model"] = "wrong-model"
            bad_manifest_path = Path(formal_directory) / "manifest.json"
            _write_json(bad_manifest_path, bad_manifest)
            source["records"]["duet-reverie"]["formal_manifest_path"] = str(
                bad_manifest_path
            )
            source["records"]["duet-reverie"]["formal_manifest_sha256"] = (
                runner._sha256(bad_manifest_path)
            )
            source_copy = directory / "source.json"
            _write_json(source_copy, source)
            candidate = deepcopy(spec)
            candidate["source_control"]["manifest"] = str(source_copy)
            candidate["source_control"]["sha256"] = runner._sha256(source_copy)
            with self.assertRaisesRegex(runner.UserError, "model mismatch"):
                runner._validate_source_registry(candidate)

    def test_source_metrics_artifacts_are_optional_only_for_planning(self):
        spec = runner.load_spec(SPEC_PATH)
        source_path = REPO_ROOT / spec["source_control"]["manifest"]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(
            dir=REPO_ROOT / "vln/results/logs"
        ) as directory:
            directory = Path(directory)
            for record in source["records"].values():
                record["metrics_artifact_path"] = str(
                    directory / "missing-valid.txt"
                )
            source_copy = directory / "source.json"
            _write_json(source_copy, source)
            candidate = deepcopy(spec)
            candidate["source_control"]["manifest"] = str(source_copy)
            candidate["source_control"]["sha256"] = runner._sha256(source_copy)
            runner._validate_source_registry(
                candidate, require_metrics_artifacts=False
            )
            with self.assertRaisesRegex(
                runner.UserError, "metrics artifact is unavailable"
            ):
                runner._validate_source_registry(
                    candidate, require_metrics_artifacts=True
                )

    def test_spec_is_complete_frozen_matrix(self):
        spec = runner.load_spec(SPEC_PATH)
        jobs = runner.expand_jobs(spec, "test-batch", gpu=0)
        self.assertEqual(len(jobs), 15)
        self.assertEqual(
            {(job["setting"], job["method"]) for job in jobs},
            {
                (setting, method)
                for setting in runner.SETTINGS
                for method in runner.METHODS
            },
        )
        self.assertEqual(
            spec["execution"]["default_concurrency_profile"],
            "shared_gpu_with_r2r_ce",
        )
        self.assertTrue(spec["execution"]["joint_launch_required"])
        self.assertTrue(spec["execution"]["joint_readiness_ack_required"])
        self.assertTrue(spec["execution"]["campaign_lifetime_lock_required"])
        self.assertTrue(spec["execution"]["shared_active_reservation_required"])
        self.assertEqual(
            spec["execution"]["concurrency_profiles"]
            ["shared_gpu_with_r2r_ce"]["max_workers_by_method"],
            {method: 1 for method in runner.METHODS},
        )
        self.assertEqual(
            spec["execution"]["concurrency_profiles"]
            ["shared_gpu_with_r2r_ce"]["max_aggregate_gpu_memory_mib"],
            26000,
        )
        self.assertEqual(
            spec["execution"]["concurrency_profiles"]
            ["exclusive_gpu"]["max_workers_by_method"],
            {"tent": 3, "fstta": 3, "eam": 3, "feedtta": 2, "atena": 2},
        )
        self.assertNotIn("source", {job["method"] for job in jobs})

    def test_latest_r2r_anchor_parameters_are_frozen(self):
        spec = runner.load_spec(SPEC_PATH)
        jobs = {
            (job["r2r_setting"], job["method"]): job["parameters"]
            for job in spec["jobs"]
        }
        self.assertEqual(jobs[("duet-r2r", "tent")]["last_k_ln"], 9)
        self.assertEqual(jobs[("duet-r2r", "tent")]["lr"], 1e-5)
        self.assertEqual(jobs[("hamt-r2r", "tent")]["update_interval"], 1)
        self.assertEqual(jobs[("goat-r2r", "feedtta")]["lr"], 1e-6)
        self.assertEqual(jobs[("goat-r2r", "feedtta")]["p"], 0.1)
        self.assertEqual(jobs[("goat-r2r", "feedtta")]["alpha"], -0.1)
        self.assertEqual(jobs[("goat-r2r", "atena")]["lr_query"], 3.75e-7)
        self.assertEqual(
            jobs[("goat-r2r", "atena")]["lr_self"], 3.515625e-8
        )

    def test_all_generated_configs_translate_for_reverie(self):
        spec = runner.load_spec(SPEC_PATH)
        jobs = runner.expand_jobs(spec, "test-batch", gpu=0)
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory)
            for job in jobs:
                attempt_dir, metadata = runner.materialize_attempt(
                    spec, SPEC_PATH, "test-batch", batch_root, job, 0
                )
                config_path = attempt_dir / "parameters.json"
                method, tokens = translate(
                    job["setting"], config_path, attempt_dir / "diag.json"
                )
                self.assertEqual(method, job["method"])
                self.assertIn("--tta_method", tokens)
                command = metadata["command"]
                self.assertNotIn("--episode-limit", command)
                self.assertNotIn("--order-seed", command)

    def test_registry_is_required_and_parameters_must_match(self):
        spec = runner.load_spec(SPEC_PATH)
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "registry.json"
            candidate = deepcopy(spec)
            candidate["r2r_registry_dependency"]["path"] = str(registry_path)
            with self.assertRaisesRegex(runner.UserError, "not ready"):
                runner.validate_r2r_registry(candidate)

            records = {}
            for job in spec["jobs"]:
                records.setdefault(job["r2r_setting"], {})[job["method"]] = {
                    "parameters": job["parameters"],
                    "run_tag": "formal-{}-{}".format(
                        job["r2r_setting"], job["method"]
                    ),
                    "metrics": {"SR": 80.0, "SPL": 75.0},
                    "formal_manifest_path": "vln/results/runs/example/manifest.json",
                    "formal_manifest_sha256": "a" * 64,
                }
            registry = {
                "schema": runner.REGISTRY_SCHEMA,
                "registry_status": "complete",
                "selected_winners": {
                    "path": "vln/results/final/r2r/selected_winners.json",
                    "sha256": runner._sha256(
                        REPO_ROOT / "vln/results/final/r2r/selected_winners.json"
                    ),
                },
                "source_ledger": {
                    "path": "vln/manifests/r2r_reused_source_controls.json",
                    "sha256": runner._sha256(
                        REPO_ROOT / "vln/manifests/r2r_reused_source_controls.json"
                    ),
                },
                "records": records,
            }
            _write_json(registry_path, registry)
            rebuild = mock.patch.object(
                runner.r2r_registry_builder,
                "validate_registry",
                side_effect=lambda path, *unused: json.loads(
                    Path(path).read_text(encoding="utf-8")
                ),
            )
            with rebuild:
                _, loaded = runner.validate_r2r_registry(candidate)
            bound = runner.bind_jobs_to_registry(
                runner.expand_jobs(candidate, "test-batch"), loaded
            )
            self.assertEqual(
                bound[0]["r2r_anchor"]["run_tag"],
                "formal-duet-r2r-tent",
            )

            registry["records"]["duet-r2r"]["tent"]["parameters"]["lr"] = 9e-6
            _write_json(registry_path, registry)
            with rebuild, self.assertRaisesRegex(
                runner.UserError, "parameters differ"
            ):
                runner.validate_r2r_registry(candidate)

    def test_feedback_attempt_validation_rejects_distance_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt-00"
            result = root / "result"
            formal_root = root / "formal"
            run_id = "test-run-duet-reverie-val_seen-native"
            formal = formal_root / run_id / "manifest.json"
            metadata = {
                "episode_count": 1423,
                "method": "feedtta",
                "model": "duet",
                "setting": "duet-reverie",
                "run_tag": "test-run",
                "result_root": str(result),
                "formal_manifest": str(formal),
                "ordinal": 0,
                "parameters": {"lr": 1e-6},
                "git_commit": "a" * 40,
                "expected_benchmark": "reverie_discrete_duet_hamt",
                "expected_checkpoint_sha256": "b" * 64,
                "expected_dataset_sha256": "c" * 64,
                "expected_episode_order_sha256": "d" * 64,
            }
            _write_json(attempt / "job.json", metadata)
            (attempt / "exitcode").write_text("0\n", encoding="utf-8")
            manifest = {
                "run_id": run_id,
                "task": "vln",
                "benchmark": "reverie_discrete_duet_hamt",
                "model": "duet",
                "method": "feedtta",
                "run_tag": "test-run",
                "source_setting": "duet-reverie:val_seen:native:feedtta",
                "seed": 0,
                "git_commit": "a" * 40,
                "status": "completed",
                "exit_code": 0,
                "checkpoint": {"sha256": "b" * 64},
                "dataset": {
                    "stream_content_sha256": "c" * 64,
                    "stream_order_sha256": "d" * 64,
                },
            }
            (result / "logs").mkdir(parents=True)
            (result / "logs/valid.txt").write_text(
                "Env name: val_seen, sr: 72.0, spl: 64.0, "
                "rgs: 58.0, rgspl: 52.0\n",
                encoding="utf-8",
            )
            diagnostics = {
                "method": "feedtta",
                "episode_count": 1423,
                "supervision": "binary_navigation_success_feedback",
                "binary_feedback_endpoint": (
                    "reverie_submitted_trajectory_evaluator_navigation_"
                    "success_every_episode"
                ),
                "adapter": {
                    "episodes": 1423,
                    "feedback_episodes": 1423,
                    "successful_feedback_episodes": 1000,
                    "failed_feedback_episodes": 423,
                },
            }
            _write_json(result / "tta_diagnostics.json", diagnostics)
            artifacts = []
            for artifact_path in (
                result / "logs/valid.txt", result / "tta_diagnostics.json"
            ):
                artifacts.append({
                    "name": artifact_path.relative_to(result).as_posix(),
                    "path": str(artifact_path),
                    "size": artifact_path.stat().st_size,
                    "sha256": runner._sha256(artifact_path),
                })
            manifest["result_artifacts"] = artifacts
            manifest["immutable_identity_sha256"] = (
                immutable_identity_sha256(manifest)
            )
            _write_json(formal, manifest)
            with mock.patch.object(runner, "FORMAL_ROOT", formal_root):
                validated = runner.validate_attempt(attempt)
            self.assertEqual(validated["metrics"]["RGSPL"], 52.0)

            diagnostics["binary_feedback_endpoint"] = (
                "final_simulator_observation_distance"
            )
            _write_json(result / "tta_diagnostics.json", diagnostics)
            (attempt / "metrics.json").unlink()
            with mock.patch.object(
                runner, "FORMAL_ROOT", formal_root
            ), self.assertRaisesRegex(
                runner.UserError, "binary-feedback endpoint"
            ):
                runner.validate_attempt(attempt)

    def test_reverie_agents_do_not_pass_final_observation_distance(self):
        paths = [
            REPO_ROOT / "vln/baselines/duet/map_nav_src/reverie/agent_obj.py",
            REPO_ROOT / "vln/baselines/hamt/finetune_src/reverie/agent.py",
            REPO_ROOT / "vln/baselines/goat/map_nav_src/reverie/agent_obj_goat.py",
        ]
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertIn("tta_reverie_episode_stats", source)
            self.assertNotIn("tta_episode_end(observations=obs)", source)


if __name__ == "__main__":
    unittest.main()
