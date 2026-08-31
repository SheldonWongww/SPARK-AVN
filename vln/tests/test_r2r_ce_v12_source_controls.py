import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/run_r2r_ce_v12_source_controls.py"
SPEC = importlib.util.spec_from_file_location("v12_source_controls", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class R2RCEV12SourceControlsTest(unittest.TestCase):
    def test_plan_is_exact_native_v12_four_job_matrix(self):
        plan = MODULE.build_plan(
            "native-v12-source-unit", [2, 3],
            created_at="2026-08-31T00:00:00+00:00",
        )
        self.assertEqual(plan["protocol"]["data_version"], "v1.2-native")
        self.assertEqual(plan["protocol"]["model_seed"], 0)
        self.assertEqual(plan["protocol"]["episode_order_seed"], 0)
        self.assertEqual(plan["protocol"]["split_order_within_model"], [
            "val_unseen", "val_seen",
        ])
        self.assertFalse(plan["protocol"]["cross_split_state_reuse"])
        self.assertEqual(len(plan["jobs"]), 4)
        self.assertEqual(
            [(item["setting"], item["split"], item["gpu"])
             for item in plan["jobs"]],
            [
                ("etpnav-r2r-ce", "val_unseen", 2),
                ("etpnav-r2r-ce", "val_seen", 2),
                ("bevbert-r2r-ce", "val_unseen", 3),
                ("bevbert-r2r-ce", "val_seen", 3),
            ],
        )
        for job in plan["jobs"]:
            command = job["command_template"]
            self.assertIn("--ce-data-version", command)
            self.assertEqual(
                command[command.index("--ce-data-version") + 1],
                "v1.2-native",
            )
            self.assertEqual(job["seed"], 0)
            self.assertEqual(job["order_seed"], 0)
            config_path = REPO_ROOT / job["config_path"]
            self.assertTrue(config_path.is_file())
            self.assertEqual(
                job["config_sha256"], MODULE.sha256_file(config_path)
            )
            self.assertIn("attempt-{attempt:03d}", job["run_tag_template"])

    def test_source_manifest_uses_native_version_config_digest_and_log_capture(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"--dataset-version", PROTOCOL', source)
        self.assertIn('"evaluation_config={}".format', source)
        self.assertIn("stdout=log", source)
        self.assertIn("stderr=subprocess.STDOUT", source)

    def test_terminal_source_attempt_is_not_retried_in_same_batch(self):
        job = {"key": "etpnav-r2r-ce:source:val_seen"}
        with mock.patch.object(
            MODULE, "job_status", return_value=("failed", 1, {"status": "failed"})
        ):
            with self.assertRaisesRegex(
                MODULE.WorkflowError, "not retryable in the same batch"
            ):
                MODULE.execute_job(job, {}, Path("/tmp/unused"), Path("/tmp"), Path("/tmp/env"))

    def _synthetic_result(self, root, ids, aggregate_success=0.5):
        result_root = root / "result"
        metric_root = result_root / "metrics/source_val_seen"
        metric_root.mkdir(parents=True)
        per_episode = {}
        successes = [1.0, 0.0]
        spls = [0.8, 0.0]
        for index, episode_id in enumerate(ids):
            per_episode[episode_id] = {
                "success": successes[index],
                "spl": spls[index],
                "distance_to_goal": 1.0 + index,
            }
        aggregate = {
            "success": aggregate_success,
            "spl": 0.4,
            "distance_to_goal": 1.5,
        }
        (metric_root / "stats_ckpt_1_val_seen.json").write_text(
            json.dumps(aggregate), encoding="utf-8"
        )
        (metric_root / "stats_ep_ckpt_1_val_seen_r0_w1.json").write_text(
            json.dumps(per_episode), encoding="utf-8"
        )
        (result_root / "console.log").write_text(
            "TASK_CONFIG.DATASET.DATA_PATH R2R_VLNCE_v1-2_preprocessed\n"
            "Episodes evaluated: 2\n",
            encoding="utf-8",
        )
        return result_root

    def test_result_validation_binds_order_and_recomputes_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = self._synthetic_result(root, ["10", "11"])
            paths = {
                "result_root": result_root,
                "validation": root / "validation.json",
                "run_id": "unit-run",
                "run_tag": "unit-tag",
            }
            job = {
                "setting": "etpnav-r2r-ce", "model": "etpnav",
                "split": "val_seen", "expected_episodes": 2,
            }
            order = {
                "episode_ids": ["10", "11"], "episode_count": 2,
                "order_sha256": "a" * 64, "dataset_sha256": "b" * 64,
            }
            with mock.patch.object(MODULE, "order_binding", return_value=order):
                report, aggregate, per_episode = MODULE.validate_result(job, paths)
            self.assertEqual(report["episode_count"], 2)
            self.assertEqual(report["metrics"], {"SR": 50.0, "SPL": 40.0})
            self.assertTrue(report["ordered_episode_ids_match"])
            self.assertTrue(aggregate.is_file())
            self.assertTrue(per_episode.is_file())

    def test_result_validation_rejects_order_or_aggregate_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = self._synthetic_result(
                root, ["10", "11"], aggregate_success=0.75
            )
            paths = {
                "result_root": result_root,
                "validation": root / "validation.json",
                "run_id": "unit-run",
                "run_tag": "unit-tag",
            }
            job = {
                "setting": "etpnav-r2r-ce", "model": "etpnav",
                "split": "val_seen", "expected_episodes": 2,
            }
            order = {
                "episode_ids": ["10", "11"], "episode_count": 2,
                "order_sha256": "a" * 64, "dataset_sha256": "b" * 64,
            }
            with mock.patch.object(MODULE, "order_binding", return_value=order):
                with self.assertRaisesRegex(
                    MODULE.WorkflowError, "aggregate/per-episode"
                ):
                    MODULE.validate_result(job, paths)

            aggregate = next(result_root.rglob("stats_ckpt_*.json"))
            aggregate.write_text(json.dumps({
                "success": 0.5, "spl": 0.4, "distance_to_goal": 1.5,
            }), encoding="utf-8")
            order["episode_ids"] = ["11", "10"]
            with mock.patch.object(MODULE, "order_binding", return_value=order):
                with self.assertRaisesRegex(MODULE.WorkflowError, "IDs/order"):
                    MODULE.validate_result(job, paths)

    def test_attempt_directories_are_append_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = {
                "setting": "etpnav-r2r-ce", "split": "val_unseen",
                "run_tag_template": "batch-etp-{attempt:03d}",
            }
            first = MODULE.attempt_paths(root, job, 1)
            second = MODULE.attempt_paths(root, job, 2)
            first["directory"].mkdir(parents=True)
            MODULE.atomic_json(first["attempt"], {"status": "failed"})
            second["directory"].mkdir(parents=True)
            MODULE.atomic_json(second["attempt"], {"status": "ready"})
            self.assertEqual(MODULE.attempt_numbers(root, job), [1, 2])
            self.assertEqual(
                MODULE.latest_attempt(root, job)[1]["status"], "ready"
            )
            self.assertEqual(
                json.loads(first["attempt"].read_text())["status"], "failed"
            )

    def test_candidate_ledgers_are_two_content_addressed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            for setting in MODULE.SETTINGS:
                for split in MODULE.SPLITS:
                    jobs.append({
                        "key": "{}:{}".format(setting, split),
                        "setting": setting,
                        "model": MODULE.MODEL[setting],
                        "split": split,
                        "gpu": 0,
                    })
            plan = {
                "batch_id": "unit", "git_commit": "c" * 40,
                "plan_identity_sha256": "d" * 64, "jobs": jobs,
                "orders": {
                    split: {
                        "path": "vln/manifests/order/{}.json".format(split),
                        "sha256": "a" * 64, "order_sha256": "b" * 64,
                        "episodes": MODULE.EXPECTED_EPISODES[split],
                    } for split in MODULE.SPLITS
                },
            }
            state = {
                "status": "ready", "completed_at": "2026-08-31T00:00:00Z",
            }
            def record(job, ignored_state):
                del ignored_state
                return {
                    "model": job["model"], "evidence_status": "ready",
                    "checkpoint_sha256": "1" * 64,
                    "dataset_sha256": "2" * 64,
                    "episode_order_sha256": "b" * 64,
                    "metrics": {"SR": 1.0, "SPL": 1.0},
                }
            with mock.patch.object(
                MODULE, "job_status", return_value=("ready", 1, state)
            ), mock.patch.object(MODULE, "ledger_record", side_effect=record):
                outputs = MODULE.emit_candidate_ledgers(plan, root)
            self.assertEqual(len(outputs), 2)
            self.assertEqual({item["split"] for item in outputs}, set(MODULE.SPLITS))
            for item in outputs:
                path = Path(item["path"])
                self.assertTrue(path.is_file())
                self.assertIn(item["sha256"], path.name)
                self.assertEqual(MODULE.sha256_file(path), item["sha256"])
                document = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(document["data_version"], "v1.2-native")
                self.assertEqual(set(document["records"]), set(MODULE.SETTINGS))
                self.assertFalse(
                    document["promotion_policy"]["existing_v1_3_ledgers_overwritten"]
                )


if __name__ == "__main__":
    unittest.main()
