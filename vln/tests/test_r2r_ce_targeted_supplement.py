import copy
from contextlib import contextmanager, ExitStack
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/run_r2r_ce_targeted_supplement.py"
SPEC_PATH = REPO_ROOT / "vln/experiments/r2r_ce_targeted_supplement_v1.json"
MODULE_SPEC = importlib.util.spec_from_file_location("r2r_ce_supplement", SCRIPT)
RUNNER = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(RUNNER)

TRANSLATOR_PATH = REPO_ROOT / "vln/scripts/tta_config_cli.py"
TRANSLATOR_SPEC = importlib.util.spec_from_file_location(
    "r2r_ce_supplement_config", TRANSLATOR_PATH
)
TRANSLATOR = importlib.util.module_from_spec(TRANSLATOR_SPEC)
TRANSLATOR_SPEC.loader.exec_module(TRANSLATOR)


class R2RCETargetedSupplementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = RUNNER.load_spec(SPEC_PATH)

    def _result(
        self, method, setting, index, parameters, spl, sr, updates,
        drift, changes=0,
    ):
        return {
            "run_tag": "{}-{}-{}".format(setting, method, index),
            "point_index": index,
            "setting": setting,
            "config_method": method,
            "parameters": parameters,
            "metrics": {"SPL": spl, "SR": sr},
            "adapter_diagnostics": {
                "updates": updates,
                "relative_param_drift": drift,
            },
            "navigation_record_change_count": changes,
        }

    def test_exact_minimal_scope_and_budget(self):
        self.assertEqual(set(self.spec["methods"]), {"tent", "fstta", "feedtta"})
        self.assertNotIn("eam", self.spec["methods"])
        self.assertNotIn("atena", self.spec["methods"])
        self.assertEqual(
            RUNNER.enabled_methods("etpnav-r2r-ce", self.spec),
            ("tent", "fstta", "feedtta"),
        )
        self.assertEqual(
            RUNNER.enabled_methods("bevbert-r2r-ce", self.spec),
            ("fstta", "feedtta"),
        )
        self.assertEqual(self.spec["budget"], {
            "screening_tta_jobs": 15,
            "full_val_seen_tta_jobs_max": 5,
            "source_execution_jobs": 0,
            "total_executed_jobs_max": 20,
        })
        count = sum(
            len(RUNNER.expand_candidates(method, setting, self.spec))
            for setting in RUNNER.SETTINGS
            for method in RUNNER.enabled_methods(setting, self.spec)
        )
        self.assertEqual(count, 15)

    def test_reviewed_candidate_values_and_fstta_windows(self):
        tent = RUNNER.expand_candidates("tent", "etpnav-r2r-ce", self.spec)
        self.assertEqual([item["lr"] for item in tent], [3e-7, 1e-6, 3e-6])
        self.assertTrue(all(item["update_interval"] == 1 for item in tent))
        self.assertTrue(all(item["norm_scope"] == "last_k_ln" for item in tent))

        expected_fstta = [
            (3e-7, 1e-5, 4, 16),
            (1e-6, 3e-5, 4, 16),
            (1e-6, 3e-5, 1, 16),
        ]
        for setting in RUNNER.SETTINGS:
            points = RUNNER.expand_candidates("fstta", setting, self.spec)
            self.assertEqual(
                [(p["lr_fast"], p["lr_slow"], p["m"], p["n"]) for p in points],
                expected_fstta,
            )
            self.assertTrue(all(point["m"] <= 15 for point in points))

            feedtta = RUNNER.expand_candidates("feedtta", setting, self.spec)
            self.assertEqual([item["lr"] for item in feedtta], [3e-7, 1e-6, 2e-6])
            self.assertTrue(all(item["scope_profile"] == "last_crossmodal"
                                for item in feedtta))
            self.assertTrue(all(item["action_selection"] == "argmax"
                                for item in feedtta))

    def test_fstta_spec_fails_closed_above_single_episode_bound(self):
        changed = copy.deepcopy(self.spec)
        changed["methods"]["fstta"]["settings"]["etpnav-r2r-ce"][
            "candidates"
        ][0]["m"] = 16
        with self.assertRaisesRegex(RUNNER.UserError, "episode bound"):
            RUNNER._validate_search_grid(changed)

    def test_model_major_phases_and_concurrency_are_fixed(self):
        phases = RUNNER.phase_sequence(self.spec)
        execution = self.spec["execution"]
        self.assertEqual(
            execution["parallel_peer_campaign"],
            "reverie_frozen_evaluation",
        )
        self.assertTrue(execution["shared_gpu_launch_guard_required"])
        self.assertTrue(execution["shared_active_reservation_required"])
        self.assertEqual(
            [(item["setting"], item["kind"], item["max_workers"])
             for item in phases],
            [
                ("etpnav-r2r-ce", "screening", 3),
                ("etpnav-r2r-ce", "full_confirmation", 3),
                ("bevbert-r2r-ce", "screening", 3),
                ("bevbert-r2r-ce", "full_confirmation", 2),
            ],
        )
        limits = self.spec["execution"]["resource_limits"]
        self.assertLessEqual(
            limits["max_gpu_memory_mib_before_launch"]
            + limits["estimated_job_gpu_memory_mib"],
            limits["max_aggregate_gpu_memory_mib"],
        )
        runtime = RUNNER.runtime_args(
            SimpleNamespace(
                batch_id="runtime-unit", gpu=0, resume=False,
                retry_failed=False,
            ),
            phases[0], self.spec,
        )
        self.assertEqual(runtime.max_workers, 3)
        self.assertEqual(runtime.max_continuous_workers, 3)
        self.assertEqual(runtime.max_discrete_workers, 1)

        changed = copy.deepcopy(self.spec)
        changed["execution"]["shared_active_reservation_required"] = False
        with self.assertRaisesRegex(RUNNER.UserError, "shared-GPU coordination"):
            RUNNER._validate_execution(changed)

    def test_peer_reservations_participate_in_atomic_resource_gate(self):
        args = SimpleNamespace(
            gpu=0,
            max_gpu_memory_mib=22000,
            estimated_job_gpu_memory_mib=8000,
            max_aggregate_gpu_memory_mib=30000,
            max_memory_gib=55.0,
            estimated_job_memory_gib=20.0,
            max_aggregate_memory_gib=75.0,
        )
        ledger = mock.Mock()
        ledger.snapshot.return_value = {
            "effective_gpu_memory_mib": 25000,
            "effective_cgroup_memory_gib": 60.0,
        }
        with mock.patch.object(
            RUNNER.staged, "gpu_stats", return_value=(1000, 10)
        ), mock.patch.object(
            RUNNER.staged, "cgroup_memory_gib", return_value=5.0
        ):
            okay, gpu, memory, snapshot = RUNNER._reserved_resources_ok(
                args, ledger
            )
        self.assertFalse(okay)
        self.assertEqual((gpu, memory), (1000, 5.0))
        self.assertEqual(snapshot["effective_gpu_memory_mib"], 25000)
        ledger.snapshot.assert_called_once_with(1000, 5.0)

    def test_resume_reclaims_shared_reservation_for_worker_and_runner(self):
        scheduler = {
            "pid": 10, "start_token": "scheduler", "cmdline_sha256": "a" * 64
        }
        worker = {
            "pid": 20, "start_token": "worker", "cmdline_sha256": "b" * 64
        }
        child = {
            "pid": 30, "start_token": "runner", "cmdline_sha256": "c" * 64
        }

        class Ledger:
            def __init__(self):
                self.document = {"reservations": {}}

            def reserve(self, token, **kwargs):
                self.document["reservations"][token] = {
                    "gpu_memory_mib": kwargs["gpu_memory_mib"],
                    "cgroup_memory_gib": kwargs["cgroup_memory_gib"],
                    "owners": [kwargs["owner"]],
                    "metadata": kwargs["metadata"],
                }

            def add_owner(self, token, owner):
                self.document["reservations"][token]["owners"].append(owner)

        ledger = Ledger()

        @contextmanager
        def guard(_gpu):
            yield ledger

        args = SimpleNamespace(
            gpu=0, batch_id="batch", estimated_job_gpu_memory_mib=8000,
            estimated_job_memory_gib=20.0,
        )
        job = {
            "batch_id": "batch", "run_tag": "candidate",
            "setting": "etpnav-r2r-ce", "gpu": 0,
            "config_method": "tent", "phase_id": "00-screening",
        }
        with mock.patch.object(
            RUNNER, "shared_gpu_launch_guard", side_effect=guard
        ), mock.patch.object(
            RUNNER.staged, "gpu_stats", return_value=(12000, 50)
        ), mock.patch.object(
            RUNNER.staged, "cgroup_memory_gib", return_value=40.0
        ), mock.patch.object(
            RUNNER.staged, "process_identity", return_value=scheduler
        ):
            token = RUNNER._claim_running_reservation(
                args, job, [worker, child]
            )
            ledger.document["reservations"][token]["metadata"]["role"] = "peer"
            with self.assertRaisesRegex(
                RUNNER.UserError, "reservation binding mismatch"
            ):
                RUNNER._claim_running_reservation(args, job, [worker, child])
            ledger.document["reservations"][token]["metadata"][
                "role"
            ] = RUNNER.RESERVATION_ROLE
        record = ledger.document["reservations"][token]
        self.assertEqual(record["owners"], [scheduler, worker, child])
        self.assertEqual(record["metadata"]["role"], RUNNER.RESERVATION_ROLE)

    def test_live_descendant_keeps_job_running_after_both_wrappers_exit(self):
        child = {
            "pid": 30, "start_token": "runner", "cmdline_sha256": "c" * 64
        }
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory)
            job = {
                "batch_id": "batch", "run_tag": "candidate",
                "job_dir": str(job_dir), "setting": "etpnav-r2r-ce",
                "config_method": "tent", "phase_id": "00-screening",
            }
            with mock.patch.object(
                RUNNER, "_proc_token_processes", return_value=[{
                    "identity": child, "pgid": 29, "sid": 29,
                }]
            ):
                self.assertEqual(
                    RUNNER._live_job_identities(job),
                    [child],
                )

    def test_recorded_worker_and_runner_are_both_preserved(self):
        worker = {
            "pid": 20, "start_token": "worker", "cmdline_sha256": "b" * 64
        }
        runner = {
            "pid": 30, "start_token": "runner", "cmdline_sha256": "c" * 64
        }
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory)
            (job_dir / "worker_state.json").write_text(json.dumps({
                "status": "running",
                "worker_pid": 20,
                "worker_process": worker,
                "runner_pid": 30,
                "runner_process": runner,
            }), encoding="utf-8")
            with mock.patch.object(
                RUNNER.staged, "process_alive", return_value=True
            ), mock.patch.object(
                RUNNER.os, "getpgid", side_effect=lambda pid: pid
            ), mock.patch.object(
                RUNNER.os, "getsid", side_effect=lambda pid: pid
            ), mock.patch.object(
                RUNNER, "_proc_process_state", return_value="S"
            ):
                records = RUNNER._recorded_live_processes({
                    "job_dir": str(job_dir)
                })
        self.assertEqual(
            records,
            [
                {"identity": worker, "pgid": 20, "sid": 20},
                {"identity": runner, "pgid": 30, "sid": 30},
            ],
        )

    def test_proc_scan_requires_exact_inherited_token(self):
        token = "d" * 64
        identity = {
            "pid": 30, "start_token": "python", "cmdline_sha256": "e" * 64
        }
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            matched = proc / "30"
            matched.mkdir()
            (matched / "stat").write_text(
                "30 (python worker) S 1 29 29 0 0 0\n", encoding="utf-8"
            )
            (matched / "environ").write_bytes(
                "{}={}".format(RUNNER.PROCESS_TOKEN_ENV, token).encode("ascii")
                + b"\0"
            )
            wrong = proc / "31"
            wrong.mkdir()
            (wrong / "stat").write_text(
                "31 (python worker) S 1 29 29 0 0 0\n", encoding="utf-8"
            )
            (wrong / "environ").write_bytes(b"OTHER=value\0")
            with mock.patch.object(
                RUNNER.staged, "process_identity",
                side_effect=lambda pid: identity if pid == 30 else None,
            ), mock.patch.object(
                RUNNER.staged, "process_alive", return_value=True
            ):
                self.assertEqual(
                    RUNNER._proc_token_processes(token, proc),
                    [{"identity": identity, "pgid": 29, "sid": 29}],
                )

    def test_exitcode_with_live_descendant_does_not_release_retry_or_cross_barrier(self):
        identity = {
            "pid": 30, "start_token": "python", "cmdline_sha256": "e" * 64
        }
        record = {"identity": identity, "pgid": 29, "sid": 29}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir = root / "job"
            job_dir.mkdir()
            (job_dir / "exitcode").write_text("0\n", encoding="utf-8")
            job = {
                "batch_id": "batch", "run_tag": "candidate",
                "setting": "etpnav-r2r-ce", "job_dir": str(job_dir), "gpu": 0,
                "config_method": "tent", "phase_id": "00-screening",
            }
            args = SimpleNamespace(
                batch_id="batch", gpu=0, resume=True, retry_failed=True,
                fail_fast=False, max_workers=1, max_gpu_memory_mib=22000,
                estimated_job_gpu_memory_mib=8000,
                max_aggregate_gpu_memory_mib=30000,
                max_memory_gib=55.0, estimated_job_memory_gib=20.0,
                max_aggregate_memory_gib=75.0, launch_stagger=0,
                resource_wait_timeout=10,
            )
            with mock.patch.object(
                RUNNER, "assert_campaign_head"
            ), mock.patch.object(
                RUNNER, "_live_job_processes", return_value=[record]
            ), mock.patch.object(
                RUNNER, "_claim_running_reservation",
                side_effect=RUNNER.UserError("audit stop after claim"),
            ) as claim, mock.patch.object(
                RUNNER, "_release_reservation"
            ) as release, mock.patch.object(
                RUNNER.staged, "_bump_attempt"
            ) as retry, mock.patch.object(
                RUNNER.subprocess, "Popen"
            ) as popen, mock.patch.object(
                RUNNER.staged, "write_summary"
            ) as summary, mock.patch.object(RUNNER.signal, "signal"):
                with self.assertRaisesRegex(RUNNER.UserError, "audit stop"):
                    RUNNER.run_phase_batch(
                        args, {"kind": "screening"}, root, [job], {}
                    )
            claim.assert_called_once_with(args, job, [identity])
            release.assert_not_called()
            retry.assert_not_called()
            popen.assert_not_called()
            summary.assert_not_called()

    def test_phase_barrier_advances_only_after_exitcode_descendants_finish(self):
        identity = {
            "pid": 30, "start_token": "python", "cmdline_sha256": "e" * 64
        }
        record = {"identity": identity, "pgid": 29, "sid": 29}
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir = root / "job"
            job_dir.mkdir()
            (job_dir / "exitcode").write_text("0\n", encoding="utf-8")
            job = {
                "batch_id": "batch", "run_tag": "candidate",
                "setting": "etpnav-r2r-ce", "job_dir": str(job_dir), "gpu": 0,
                "config_method": "tent", "phase_id": "01-full",
            }
            args = SimpleNamespace(
                batch_id="batch", gpu=0, resume=True, retry_failed=False,
                fail_fast=True, max_workers=1, max_gpu_memory_mib=22000,
                estimated_job_gpu_memory_mib=8000,
                max_aggregate_gpu_memory_mib=30000,
                max_memory_gib=55.0, estimated_job_memory_gib=20.0,
                max_aggregate_memory_gib=75.0, launch_stagger=0,
                resource_wait_timeout=10,
            )

            def claim(*_args):
                events.append("claim_live_descendant")

            def release(*_args):
                events.append("release")

            def summary(*_args):
                events.append("summary")
                return ([{}], [])

            with mock.patch.object(
                RUNNER, "assert_campaign_head"
            ), mock.patch.object(
                RUNNER, "_live_job_processes",
                side_effect=[[record], [record], []],
            ), mock.patch.object(
                RUNNER, "_claim_running_reservation", side_effect=claim
            ), mock.patch.object(
                RUNNER, "_release_reservation", side_effect=release
            ), mock.patch.object(
                RUNNER.staged, "append_resource"
            ), mock.patch.object(
                RUNNER.staged, "_progress"
            ), mock.patch.object(
                RUNNER.staged, "write_summary", side_effect=summary
            ), mock.patch.object(
                RUNNER.time, "sleep"
            ), mock.patch.object(RUNNER.signal, "signal"):
                RUNNER.run_phase_batch(
                    args, {"kind": "full_confirmation"}, root, [job], {}
                )
        self.assertEqual(events, ["claim_live_descendant", "release", "summary"])

    def test_screening_jobs_are_method_round_robin_canonical_prefixes(self):
        phases = RUNNER.phase_sequence(self.spec)
        etp = RUNNER.build_screening_jobs(
            phases[0], "supplement-unit", self.spec, gpu=2
        )
        bev = RUNNER.build_screening_jobs(
            phases[2], "supplement-unit", self.spec, gpu=2
        )
        self.assertEqual(len(etp), 9)
        self.assertEqual(len(bev), 6)
        self.assertEqual(
            [job["config_method"] for job in etp[:3]],
            ["tent", "fstta", "feedtta"],
        )
        self.assertEqual(
            [job["config_method"] for job in bev[:2]],
            ["fstta", "feedtta"],
        )
        for job in etp + bev:
            self.assertNotEqual(job["config_method"], "source")
            self.assertEqual(job["episodes"], 100)
            self.assertEqual(job["canonical_order_seed"], 0)
            self.assertIsNone(job["order_seed"])
            self.assertNotIn("--order-seed", job["command"])
            self.assertEqual(
                job["command"][job["command"].index("--episode-limit") + 1],
                "100",
            )
            self.assertTrue(job["restart_from_source_checkpoint"])

    def test_full_jobs_are_dynamic_capped_and_restart_from_source(self):
        phases = RUNNER.phase_sequence(self.spec)
        promotions = {"setting": "etpnav-r2r-ce", "cells": {}}
        for method in ("tent", "fstta", "feedtta"):
            point = RUNNER.expand_candidates(method, "etpnav-r2r-ce", self.spec)[1]
            promotions["cells"][method] = {
                "selected": {
                    "run_tag": "screen-{}".format(method),
                    "point_index": 1,
                    "parameters": point,
                }
            }
        jobs = RUNNER.build_full_jobs(
            phases[1], "supplement-unit", promotions, self.spec, gpu=0
        )
        self.assertEqual(len(jobs), 3)
        for job in jobs:
            self.assertEqual(job["episodes"], -1)
            self.assertNotIn("--episode-limit", job["command"])
            self.assertNotIn("--order-seed", job["command"])
            self.assertEqual(job["parent_run_tags"], [
                "screen-{}".format(job["config_method"])
            ])
            self.assertTrue(job["restart_from_source_checkpoint"])

        promotions["cells"]["feedtta"]["selected"] = None
        self.assertEqual(
            len(RUNNER.build_full_jobs(
                phases[1], "supplement-unit", promotions, self.spec
            )),
            2,
        )

    def test_fstta_zero_updates_and_large_m_cannot_be_promoted(self):
        setting = "etpnav-r2r-ce"
        source = RUNNER.staged.reused_source_result(setting, 100, self.spec)
        results = [
            self._result(
                "fstta", setting, 0,
                {"lr_fast": 3e-7, "lr_slow": 1e-5, "m": 4, "n": 16},
                55.0, 60.0, 0, 0.0,
            ),
            self._result(
                "fstta", setting, 1,
                {"lr_fast": 1e-6, "lr_slow": 3e-5, "m": 16, "n": 16},
                56.0, 60.0, 100, 0.01,
            ),
            self._result(
                "fstta", setting, 2,
                {"lr_fast": 1e-6, "lr_slow": 3e-5, "m": 1, "n": 16},
                52.0, 60.0, 900, 0.02,
            ),
        ]
        selected, record = RUNNER.select_finalist(
            "fstta", setting, results, source, self.spec
        )
        self.assertEqual(selected["run_tag"], results[2]["run_tag"])
        reasons = {item["run_tag"]: item["reasons"] for item in record["ranked"]}
        self.assertIn("no_effective_updates", reasons[results[0]["run_tag"]])
        self.assertIn(
            "fast_window_exceeds_episode_bound", reasons[results[1]["run_tag"]]
        )

    def test_feedtta_metric_tie_chooses_highest_safe_nonzero_behavior_lr(self):
        setting = "bevbert-r2r-ce"
        source = RUNNER.staged.reused_source_result(setting, 100, self.spec)
        spl = source["metrics"]["SPL"]
        sr = source["metrics"]["SR"]
        results = [
            self._result(
                "feedtta", setting, 0, {"lr": 3e-7}, spl, sr, 100, 1e-6, 0
            ),
            self._result(
                "feedtta", setting, 1, {"lr": 1e-6}, spl, sr, 100, 1e-4, 2
            ),
            self._result(
                "feedtta", setting, 2, {"lr": 2e-6}, spl, sr, 100, 2e-4, 1
            ),
        ]
        selected, record = RUNNER.select_finalist(
            "feedtta", setting, results, source, self.spec
        )
        self.assertEqual(selected["parameters"]["lr"], 2e-6)
        self.assertEqual(
            record["decision"], "metric_tie_highest_safe_nonzero_behavior_lr"
        )

    def test_feedtta_all_noop_tie_promotes_no_full_candidate(self):
        setting = "etpnav-r2r-ce"
        source = RUNNER.staged.reused_source_result(setting, 100, self.spec)
        results = [
            self._result(
                "feedtta", setting, index, {"lr": lr},
                source["metrics"]["SPL"], source["metrics"]["SR"],
                100, lr, 0,
            )
            for index, lr in enumerate((3e-7, 1e-6, 2e-6))
        ]
        selected, record = RUNNER.select_finalist(
            "feedtta", setting, results, source, self.spec
        )
        self.assertIsNone(selected)
        self.assertEqual(
            record["decision"], "all_top_candidates_are_navigation_noops"
        )

    def test_navigation_record_change_is_exact_and_field_pinned(self):
        fields = self.spec["selection"]["feedtta"]["navigation_record_fields"]
        base = {field: float(index) for index, field in enumerate(fields)}
        source = {"1": dict(base), "2": dict(base)}
        candidate = copy.deepcopy(source)
        candidate["2"]["steps_taken"] += 1.0
        self.assertEqual(
            RUNNER.count_navigation_record_changes(
                candidate, source, ["1", "2"], fields
            ),
            1,
        )

    def test_generated_configs_translate_and_never_execute_source(self):
        phases = RUNNER.phase_sequence(self.spec)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(mock.patch.object(RUNNER, "LOG_ROOT", root / "logs"))
            stack.enter_context(mock.patch.object(
                RUNNER, "TUNING_ROOT", root / "tuning"
            ))
            jobs = RUNNER.build_screening_jobs(
                phases[0], "translation-unit", self.spec, gpu=0
            )
            for job in jobs[:3]:
                RUNNER.staged._write_job(job)
                config = json.loads(
                    Path(job["config_path"]).read_text(encoding="utf-8")
                )
                method, tokens = TRANSLATOR.translate(
                    job["setting"], job["config_path"],
                    str(root / "diagnostics.json"),
                )
                self.assertEqual(method, job["config_method"])
                self.assertNotEqual(method, "source")
                self.assertTrue(tokens)
                self.assertNotIn("order_seed", config)
                self.assertEqual(config["episodes"], 100)

    def test_full_contract_binds_config_tokens_and_authenticated_aggregate(self):
        phase = RUNNER.phase_sequence(self.spec)[1]
        point = RUNNER.expand_candidates(
            "tent", "etpnav-r2r-ce", self.spec
        )[0]
        promotions = {
            "setting": "etpnav-r2r-ce",
            "cells": {
                "tent": {"selected": {
                    "run_tag": "screen-tent", "point_index": 0,
                    "parameters": point,
                }},
                "fstta": {"selected": None},
                "feedtta": {"selected": None},
            },
        }
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(mock.patch.object(RUNNER, "LOG_ROOT", root / "logs"))
            stack.enter_context(mock.patch.object(
                RUNNER, "TUNING_ROOT", root / "tuning"
            ))
            job = RUNNER.build_full_jobs(
                phase, "formal-unit", promotions, self.spec
            )[0]
            RUNNER.staged._write_job(job)
            result_root = Path(job["result_root"])
            aggregate = result_root / "metrics/source_val_seen/stats_ckpt_59.json"
            aggregate.parent.mkdir(parents=True)
            raw = {
                "steps_taken": 10.0, "distance_to_goal": 1.0,
                "success": 0.6, "oracle_success": 0.7,
                "path_length": 9.0, "collisions": 0.1,
                "spl": 0.5, "ndtw": 0.65, "sdtw": 0.45,
                "ghost_cnt": 2.0,
            }
            aggregate.write_text(json.dumps(raw), encoding="utf-8")
            method, tokens = TRANSLATOR.translate(
                job["setting"], job["config_path"],
                str(result_root / "tta_diagnostics.json"),
            )
            self.assertEqual(method, "tent")
            manifest = {
                "config": job["config_path"],
                "config_overrides": ["python", "run.py"] + tokens,
                "hardware": {"gpu": "unit"},
                "result_artifacts": [{
                    "name": "metrics/source_val_seen/stats_ckpt_59.json",
                    "path": str(aggregate),
                }],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            parsed = {
                "formal_manifest_path": str(manifest_path),
                "metrics": {"SR": 60.0, "SPL": 50.0},
            }
            contract = RUNNER.validate_full_result_contract(job, parsed)
            self.assertEqual(contract["metrics"]["SR"], 60.0)
            self.assertEqual(contract["metrics"]["SPL"], 50.0)
            self.assertEqual(
                contract["job_config_sha256"], RUNNER.sha256(job["config_path"])
            )

    def test_campaign_plan_is_immutable_and_resume_validates_jobs(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            head = RUNNER.git("rev-parse", "HEAD")
            stack.enter_context(mock.patch.object(
                RUNNER, "git",
                side_effect=lambda *args: "" if args[0] == "status" else head,
            ))
            stack.enter_context(mock.patch.object(RUNNER, "LOG_ROOT", root / "logs"))
            stack.enter_context(mock.patch.object(
                RUNNER, "TUNING_ROOT", root / "tuning"
            ))
            cli = SimpleNamespace(
                batch_id="plan-unit", gpu=0, resume=False,
            )
            campaign, planned = RUNNER.ensure_campaign_plan(cli, self.spec)
            self.assertEqual(sum(len(item[2]) for item in planned), 15)
            plan = json.loads((campaign / "PLAN.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["source_execution_jobs"], 0)
            self.assertEqual(plan["full_jobs_max"], 5)
            self.assertEqual(plan["shared_gpu_coordination"], {
                "peer_campaign": "reverie_frozen_evaluation",
                "launch_guard_required": True,
                "active_reservation_required": True,
                "reservation_role": RUNNER.RESERVATION_ROLE,
            })
            cli.resume = True
            _, resumed = RUNNER.ensure_campaign_plan(cli, self.spec)
            self.assertEqual(sum(len(item[2]) for item in resumed), 15)

            cli.gpu = 1
            with self.assertRaisesRegex(RUNNER.UserError, "job identity changed"):
                RUNNER.ensure_campaign_plan(cli, self.spec)
            cli.gpu = 0

            job_path = Path(resumed[0][2][0]["job_dir"]) / "job.json"
            original_job = job_path.read_text(encoding="utf-8")
            job = json.loads(original_job)
            job["retry_result_root_parent"] = str(root / "escaped-retries")
            job_path.write_text(json.dumps(job), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.UserError, "job identity changed"):
                RUNNER.ensure_campaign_plan(cli, self.spec)
            job_path.write_text(original_job, encoding="utf-8")

            job = json.loads(original_job)
            job["command"][0] = "/bin/echo"
            job_path.write_text(json.dumps(job), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.UserError, "worker command changed"):
                RUNNER.ensure_campaign_plan(cli, self.spec)
            job_path.write_text(original_job, encoding="utf-8")

            config_path = Path(resumed[0][2][0]["config_path"])
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["parameters"]["lr"] = 9e-3
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.UserError, "runtime config changed"):
                RUNNER.ensure_campaign_plan(cli, self.spec)

    def test_campaign_head_guard_rejects_commit_or_spec_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            head = RUNNER.git("rev-parse", "HEAD")
            plan = {
                "schema": RUNNER.PLAN_SCHEMA,
                "git_commit": "0" * 40,
                "spec_path": str(SPEC_PATH),
                "spec_sha256": RUNNER.sha256(SPEC_PATH),
            }
            RUNNER.staged.atomic_json(root / "PLAN.json", plan)
            with self.assertRaisesRegex(RUNNER.UserError, "Git commit changed"):
                RUNNER.assert_campaign_head(root)
            plan["git_commit"] = head
            plan["spec_sha256"] = "0" * 64
            RUNNER.staged.atomic_json(root / "PLAN.json", plan)
            with self.assertRaisesRegex(RUNNER.UserError, "spec changed"):
                RUNNER.assert_campaign_head(root)

    def test_retry_stays_in_campaign_result_parent_and_revalidates(self):
        phase = RUNNER.phase_sequence(self.spec)[0]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(mock.patch.object(RUNNER, "LOG_ROOT", root / "logs"))
            stack.enter_context(mock.patch.object(
                RUNNER, "TUNING_ROOT", root / "tuning"
            ))
            original = RUNNER.build_screening_jobs(
                phase, "retry-unit", self.spec, gpu=0
            )[0]
            RUNNER.staged._write_job(original)
            current = json.loads(
                (Path(original["job_dir"]) / "job.json").read_text(encoding="utf-8")
            )
            RUNNER.staged._bump_attempt(current)
            self.assertEqual(current["attempt"], 1)
            self.assertEqual(
                Path(current["result_root"]),
                Path(original["retry_result_root_parent"])
                / (original["base_run_tag"] + "-retry1") / "val_seen",
            )
            validated = RUNNER._validate_persisted_jobs(
                Path(original["job_dir"]).parents[1], [original]
            )
            self.assertEqual(validated[0]["run_tag"], current["run_tag"])

    def test_invalid_screening_postcondition_is_not_considered_complete(self):
        job = {"stage": RUNNER.SCREENING_STAGE, "run_tag": "candidate"}
        with mock.patch.object(
            RUNNER, "_load_valid_results", return_value=[{"run_tag": "candidate"}]
        ), mock.patch.object(
            RUNNER, "enrich_screening_result",
            side_effect=RUNNER.UserError("bad canonical records"),
        ):
            self.assertFalse(RUNNER._phase_complete("unused", [job], self.spec))

    def test_launch_requires_review_and_clean_tracked_tree(self):
        cli = SimpleNamespace(confirm_reviewed=False)
        with self.assertRaisesRegex(RUNNER.UserError, "confirm-reviewed"):
            RUNNER._assert_launchable(cli, self.spec)
        cli.confirm_reviewed = True
        with mock.patch.object(RUNNER, "git", return_value=" M tracked.py"):
            with self.assertRaisesRegex(RUNNER.UserError, "clean tracked"):
                RUNNER._assert_launchable(cli, self.spec)

        def untracked_git(*arguments):
            if arguments[0] == "status":
                return ""
            if arguments[:2] == ("ls-files", "--error-unmatch"):
                raise RUNNER.subprocess.CalledProcessError(1, arguments)
            return ""

        with mock.patch.object(RUNNER, "git", side_effect=untracked_git):
            with self.assertRaisesRegex(RUNNER.UserError, "tracked implementation"):
                RUNNER._assert_launchable(cli, self.spec)

    def test_launch_reserves_inside_shared_guard_before_process_creation(self):
        events = []
        scheduler = {
            "pid": 10, "start_token": "scheduler", "cmdline_sha256": "a" * 64
        }
        worker = {
            "pid": 20, "start_token": "worker", "cmdline_sha256": "b" * 64
        }

        class Ledger:
            document = {"reservations": {}}

            def snapshot(self, gpu, memory):
                events.append(("snapshot", gpu, memory))
                return {
                    "effective_gpu_memory_mib": gpu,
                    "effective_cgroup_memory_gib": memory,
                }

            def reserve(self, token, **_kwargs):
                events.append(("reserve", token))

            def add_owner(self, token, identity):
                events.append(("add_owner", token, identity["pid"]))

            def release(self, token):
                events.append(("rollback", token))

        @contextmanager
        def guard(_gpu):
            events.append(("guard_enter",))
            try:
                yield Ledger()
            finally:
                events.append(("guard_exit",))

        class Process:
            pid = 20

            @staticmethod
            def poll():
                return 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir = root / "job"
            job_dir.mkdir()
            job = {
                "batch_id": "batch", "run_tag": "candidate",
                "setting": "etpnav-r2r-ce", "job_dir": str(job_dir), "gpu": 0,
                "config_method": "tent", "phase_id": "00-screening",
            }
            args = SimpleNamespace(
                batch_id="batch", gpu=0, resume=False, retry_failed=False,
                fail_fast=True, max_workers=1, max_gpu_memory_mib=22000,
                estimated_job_gpu_memory_mib=8000,
                max_aggregate_gpu_memory_mib=30000,
                max_memory_gib=55.0, estimated_job_memory_gib=20.0,
                max_aggregate_memory_gib=75.0, launch_stagger=0,
                resource_wait_timeout=10,
            )

            def launch(*_args, **kwargs):
                events.append((
                    "popen", kwargs["env"][RUNNER.PROCESS_TOKEN_ENV]
                ))
                (job_dir / "exitcode").write_text("0\n", encoding="utf-8")
                return Process()

            def identity(pid=None):
                return scheduler if pid is None else worker

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    RUNNER, "assert_campaign_head"
                ))
                stack.enter_context(mock.patch.object(
                    RUNNER, "shared_gpu_launch_guard", side_effect=guard
                ))
                stack.enter_context(mock.patch.object(
                    RUNNER, "release_shared_gpu_reservation"
                ))
                stack.enter_context(mock.patch.object(
                    RUNNER, "_live_job_processes", return_value=[]
                ))
                stack.enter_context(mock.patch.object(
                    RUNNER.staged, "gpu_stats", return_value=(1000, 10)
                ))
                stack.enter_context(mock.patch.object(
                    RUNNER.staged, "cgroup_memory_gib", return_value=5.0
                ))
                stack.enter_context(mock.patch.object(
                    RUNNER.staged, "process_identity", side_effect=identity
                ))
                stack.enter_context(mock.patch.object(
                    RUNNER.staged, "append_resource"
                ))
                stack.enter_context(mock.patch.object(RUNNER.staged, "_progress"))
                stack.enter_context(mock.patch.object(
                    RUNNER.staged, "write_summary", return_value=([{}], [])
                ))
                stack.enter_context(mock.patch.object(
                    RUNNER.subprocess, "Popen", side_effect=launch
                ))
                stack.enter_context(mock.patch.object(RUNNER.signal, "signal"))
                stack.enter_context(mock.patch.object(RUNNER.time, "sleep"))
                RUNNER.run_phase_batch(
                    args, {"kind": "full_confirmation"}, root, [job], {}
                )

        names = [item[0] for item in events]
        self.assertLess(names.index("guard_enter"), names.index("snapshot"))
        self.assertLess(names.index("snapshot"), names.index("reserve"))
        self.assertLess(names.index("reserve"), names.index("popen"))
        self.assertLess(names.index("popen"), names.index("add_owner"))
        self.assertLess(names.index("add_owner"), names.index("guard_exit"))
        popen_event = next(item for item in events if item[0] == "popen")
        self.assertEqual(popen_event[1], RUNNER._job_process_token(job))

    def test_failed_launch_waits_for_process_group_before_reservation_release(self):
        events = []

        class Process:
            pid = 20

            @staticmethod
            def poll():
                return None

        process = Process()
        job = {"run_tag": "candidate"}
        with mock.patch.object(
            RUNNER, "_cleanup_process_groups",
            side_effect=[([{"pgid": 20}], {20}), ([], set())],
        ), mock.patch.object(
            RUNNER.os, "killpg",
            side_effect=lambda pid, signum: events.append(("kill", pid, signum)),
        ):
            self.assertTrue(RUNNER._terminate_launched_process(process, job))
        self.assertEqual(events[0], ("kill", 20, RUNNER.signal.SIGTERM))

    def test_cleanup_kills_token_descendant_after_direct_wrapper_exits(self):
        class Process:
            pid = 20

            @staticmethod
            def poll():
                return 1

        descendant = {
            "identity": {
                "pid": 31, "start_token": "child",
                "cmdline_sha256": "c" * 64,
            },
            "pgid": 30,
            "sid": 30,
        }
        with mock.patch.object(
            RUNNER, "_cleanup_process_groups",
            side_effect=[([descendant], {30}), ([], set())],
        ), mock.patch.object(RUNNER.os, "killpg") as killpg:
            self.assertTrue(RUNNER._terminate_launched_process(
                Process(), {"run_tag": "candidate"}
            ))
        killpg.assert_called_once_with(30, RUNNER.signal.SIGTERM)

    def test_failed_launch_retains_reservation_until_all_descendants_exit(self):
        ledger = mock.Mock()
        process = mock.Mock()
        job = {"run_tag": "candidate"}
        with mock.patch.object(
            RUNNER, "_terminate_launched_process", return_value=False
        ):
            with self.assertRaisesRegex(
                RUNNER.UserError, "reservation retained"
            ):
                RUNNER._rollback_failed_launch(
                    ledger, "reservation", process, job
                )
        ledger.release.assert_not_called()
        with mock.patch.object(
            RUNNER, "_terminate_launched_process", return_value=True
        ):
            RUNNER._rollback_failed_launch(
                ledger, "reservation", process, job
            )
        ledger.release.assert_called_once_with("reservation")

    def test_results_are_aggregated_while_campaign_lock_is_held(self):
        phases = RUNNER.phase_sequence(self.spec)
        screening = [phases[0], phases[2]]
        active = {"value": False}

        @contextmanager
        def lock(_root):
            active["value"] = True
            try:
                yield
            finally:
                active["value"] = False

        def promotion(phase, *_args):
            return {
                "setting": phase["setting"],
                "cells": {
                    method: {"selected": None}
                    for method in RUNNER.enabled_methods(phase["setting"], self.spec)
                },
            }

        def confirmation(phase, *_args):
            return {"setting": phase["setting"], "cells": {}}

        def aggregate(*_args):
            self.assertTrue(active["value"])

        initial = [
            (phase, Path("unused") / phase["phase_id"], [])
            for phase in screening
        ]
        cli = SimpleNamespace(
            batch_id="lock-unit", plan_only=False, gpu=0, resume=False
        )
        with mock.patch.object(
            RUNNER, "_assert_launchable"
        ), mock.patch.object(
            RUNNER, "ensure_campaign_plan", return_value=(Path("unused"), initial)
        ), mock.patch.object(
            RUNNER, "campaign_lock", side_effect=lock
        ), mock.patch.object(
            RUNNER, "_phase_complete", return_value=True
        ), mock.patch.object(
            RUNNER, "_load_valid_results", return_value=[]
        ), mock.patch.object(
            RUNNER, "summarize_screening", side_effect=promotion
        ), mock.patch.object(
            RUNNER, "build_full_jobs", return_value=[]
        ), mock.patch.object(
            RUNNER, "ensure_phase_plan",
            side_effect=lambda phase, *_args: (
                Path("unused") / phase["phase_id"], []
            )
        ), mock.patch.object(
            RUNNER, "_mark_empty_phase"
        ), mock.patch.object(
            RUNNER, "summarize_full", side_effect=confirmation
        ), mock.patch.object(
            RUNNER, "aggregate_campaign", side_effect=aggregate
        ) as aggregate_mock:
            RUNNER.execute_campaign(cli, self.spec)
        aggregate_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
