import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln/scripts"))

import joint_campaign_contract as contract  # noqa: E402
import run_joint_reverie_r2r_ce as joint  # noqa: E402
import shared_gpu_launch_guard as gpu_guard  # noqa: E402


class JointCampaignTest(unittest.TestCase):
    def test_plan_is_exact_parallel_zero_source_launch(self):
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            self.assertEqual(joint.main(["--plan-only"]), 0)
        plan = json.loads(output.getvalue())
        self.assertTrue(plan["parallel_required"])
        self.assertEqual(plan["source_jobs"], 0)
        self.assertEqual(plan["reverie_tta_jobs"], 15)
        self.assertEqual(plan["r2r_ce_screening_jobs"], 40)
        self.assertEqual(plan["r2r_ce_full_jobs"], 10)
        self.assertIn("--joint-launch-manifest", plan["reverie_command"])
        self.assertIn("--joint-launch-manifest", plan["r2r_ce_command"])
        self.assertIn("shared_gpu_with_r2r_ce", plan["reverie_command"])
        self.assertNotIn("source", plan["r2r_ce_command"])
        spec_index = plan["r2r_ce_command"].index("--spec")
        self.assertEqual(
            Path(plan["r2r_ce_command"][spec_index + 1]).resolve(),
            joint.DEFAULT_CE_SPEC.resolve(),
        )

    def test_contract_requires_both_live_launcher_owned_pids(self):
        root = REPO_ROOT / "vln/results/logs/joint_campaigns"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            directory = Path(directory)
            spec = directory / "spec.json"
            spec.write_text("{}\n", encoding="utf-8")
            manifest = directory / "launch.json"
            document = {
                "schema": contract.SCHEMA,
                "status": "children_spawned",
                "gpu": 0,
                "concurrency_profile": "shared_gpu_with_r2r_ce",
                "reverie_worker_cap": 1,
                "r2r_ce_worker_cap": 1,
                "batch_ids": {"reverie": "rev", "r2r_ce": "ce"},
                "specs": {
                    "reverie": {"path": str(spec), "sha256": "a" * 64},
                    "r2r_ce": {"path": str(spec), "sha256": "a" * 64},
                },
                "pids": {"reverie": 123, "r2r_ce": 456},
                "processes": {
                    "reverie": {"pid": 123, "start_token": "a", "cmdline_sha256": "b"},
                    "r2r_ce": {"pid": 456, "start_token": "c", "cmdline_sha256": "d"},
                },
            }
            manifest.write_text(json.dumps(document), encoding="utf-8")
            with mock.patch.object(contract.os, "getpid", return_value=123), \
                    mock.patch.object(
                        contract, "process_identity_alive", return_value=True
                    ):
                contract.validate_joint_launch(
                    manifest,
                    repo_root=REPO_ROOT,
                    role="reverie",
                    batch_id="rev",
                    gpu=0,
                    spec_path=spec,
                    spec_sha256="a" * 64,
                    timeout_seconds=0.2,
                )
            document["pids"]["reverie"] = 999
            document["processes"]["reverie"]["pid"] = 999
            manifest.write_text(json.dumps(document), encoding="utf-8")
            with mock.patch.object(contract.os, "getpid", return_value=123), \
                    mock.patch.object(
                        contract, "process_identity_alive", return_value=True
                    ), \
                    self.assertRaisesRegex(
                        contract.JointLaunchError, "not spawned"
                    ):
                contract.validate_joint_launch(
                    manifest,
                    repo_root=REPO_ROOT,
                    role="reverie",
                    batch_id="rev",
                    gpu=0,
                    spec_path=spec,
                    spec_sha256="a" * 64,
                    timeout_seconds=0.2,
                )

    def test_campaign_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.lock"
            with contract.campaign_lifetime_lock(
                path, role="test", batch_id="batch"
            ):
                with self.assertRaisesRegex(
                    contract.JointLaunchError, "already held"
                ):
                    with contract.campaign_lifetime_lock(
                        path, role="test", batch_id="batch"
                    ):
                        pass

    def test_gpu_reservation_hides_cold_start_allocation(self):
        owner = {
            "pid": 123,
            "start_token": "start",
            "cmdline_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            gpu_guard, "process_identity_alive", return_value=True
        ):
            with gpu_guard.shared_gpu_launch_guard(
                0, lock_root=directory
            ) as ledger:
                ledger.reserve(
                    "first",
                    gpu_memory_mib=8000,
                    cgroup_memory_gib=20.0,
                    observed_gpu_memory_mib=500,
                    observed_cgroup_memory_gib=5.0,
                    owner=owner,
                )
            with gpu_guard.shared_gpu_launch_guard(
                0, lock_root=directory
            ) as ledger:
                snapshot = ledger.snapshot(700, 6.0)
                self.assertEqual(snapshot["effective_gpu_memory_mib"], 8500)
                self.assertEqual(snapshot["effective_cgroup_memory_gib"], 25.0)
                ledger.release("first")

    def test_corrupt_gpu_reservation_ledger_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu-0.reservations.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(
                gpu_guard.ReservationLedgerError, "cannot authenticate"
            ):
                with gpu_guard.shared_gpu_launch_guard(
                    0, lock_root=directory
                ):
                    pass

    def test_resume_refuses_a_live_prior_scheduler(self):
        identity = {
            "schema": contract.SCHEMA,
            "joint_id": "joint",
            "git_commit": "a" * 40,
            "gpu": 0,
            "concurrency_profile": "shared_gpu_with_r2r_ce",
            "reverie_worker_cap": 1,
            "r2r_ce_worker_cap": 1,
            "batch_ids": {"reverie": "rev", "r2r_ce": "ce"},
            "specs": {
                "reverie": {"path": "/rev", "sha256": "b" * 64},
                "r2r_ce": {"path": "/ce", "sha256": "c" * 64},
            },
        }
        existing = dict(identity)
        existing["processes"] = {
            "reverie": {
                "pid": 123,
                "start_token": "start",
                "cmdline_sha256": "cmd",
            }
        }
        with mock.patch.object(
            joint, "process_identity_alive", return_value=True
        ), self.assertRaisesRegex(joint.UserError, "prior schedulers are alive"):
            joint._validate_resume_identity(existing, identity)

    def test_ready_ack_is_attempt_and_process_bound(self):
        root = REPO_ROOT / "vln/results/logs/joint_campaigns"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            directory = Path(directory)
            manifest = directory / "launch.json"
            ack_path = directory / "acks/attempt-00-reverie.json"
            own = {
                "pid": os.getpid(),
                "start_token": "start",
                "cmdline_sha256": "cmd",
            }
            document = {
                "active_attempt": 0,
                "status": "children_spawned",
                "processes": {"reverie": own},
                "acks": {"reverie": {"path": str(ack_path)}},
            }
            manifest.write_text(json.dumps(document), encoding="utf-8")
            with mock.patch.object(
                contract, "process_identity_alive", return_value=True
            ), mock.patch.object(contract, "process_identity", return_value=own):
                contract.write_ready_ack(
                    manifest,
                    document,
                    role="reverie",
                    batch_id="rev",
                    spec_sha256="a" * 64,
                )
                ack = json.loads(ack_path.read_text(encoding="utf-8"))
                self.assertEqual(ack["active_attempt"], 0)
                self.assertEqual(ack["process"], own)
                document["status"] = "both_ready"
                manifest.write_text(json.dumps(document), encoding="utf-8")
                released = contract.wait_for_joint_release(
                    manifest, document, role="reverie", timeout_seconds=0.2
                )
                self.assertEqual(released["status"], "both_ready")


if __name__ == "__main__":
    unittest.main()
