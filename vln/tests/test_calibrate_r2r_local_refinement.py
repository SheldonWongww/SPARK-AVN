import copy
import importlib.util
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/calibrate_r2r_local_refinement.py"
SPEC = importlib.util.spec_from_file_location(
    "calibrate_r2r_local_refinement", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


PHASE_ID = "01-duet-r2r-tent"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_campaign(root, count=3):
    root = Path(root)
    phase = {
        "index": 1,
        "phase_id": PHASE_ID,
        "setting": "duet-r2r",
        "model": "duet",
        "method": "tent",
        "job_count": count,
        "global_ordinal_start": 1,
    }
    plan = {
        "schema": MODULE.PLAN_SCHEMA,
        "batch_id": "calibration-test",
        "git_commit": MODULE._git("rev-parse", "--verify", "HEAD"),
        "spec_sha256": "b" * 64,
        "episode_count": 1021,
        "gpu": 0,
        "phases": [phase],
    }
    write_json(root / "PLAN.json", plan)
    write_json(root / "phases" / PHASE_ID / "PHASE.json", {
        "schema": MODULE.PHASE_SCHEMA,
        "batch_id": plan["batch_id"],
        "git_commit": plan["git_commit"],
        "spec_sha256": plan["spec_sha256"],
        "gpu": plan["gpu"],
        **phase,
    })
    jobs = []
    for index in range(count):
        run_tag = f"formal-tent-{index:02d}"
        job_dir = root / "phases" / PHASE_ID / "jobs" / run_tag
        config_path = job_dir / "parameters.json"
        formal_result = (
            root.parent / "formal-tuning" / run_tag / "val_seen"
        ).resolve()
        config = {
            "schema": "navtta.vln_tta_job.v1",
            "method": "tent",
            "search_method": "tent",
            "episodes": -1,
            "parameters": {"lr": 10 ** (-7 + index)},
            "batch_id": "calibration-test",
            "setting": "duet-r2r",
            "run_tag": run_tag,
            "result_layout": "r2r_benchmark_model_method_local_refinement_v1",
            "result_namespace": "tent",
        }
        job = {
            "batch_id": "calibration-test",
            "ordinal": index,
            "point_index": index,
            "phase_id": PHASE_ID,
            "setting": "duet-r2r",
            "model": "duet",
            "search_method": "tent",
            "config_method": "tent",
            "parameters": config["parameters"],
            "run_tag": run_tag,
            "base_run_tag": run_tag,
            "episodes": -1,
            "job_dir": str(job_dir.resolve()),
            "config_path": str(config_path.resolve()),
            "result_root": str(formal_result),
            "retry_result_root_parent": str(formal_result.parent.parent),
            "command": [
                "/bin/true", "duet-r2r", "val_seen", "0",
                "--run-tag", run_tag,
                "--tta-config", str(config_path.resolve()),
                "--result-root", str(formal_result),
            ],
        }
        write_json(config_path, config)
        write_json(job_dir / "job.json", job)
        jobs.append(job)
    return plan, phase, jobs


def make_prior_calibration(campaign, plan, phase, jobs, cap=2):
    root = Path(campaign) / "calibration" / PHASE_ID / "prior-evidence"
    prior_jobs = []
    for index, job in enumerate(jobs):
        config = json.loads(Path(job["config_path"]).read_text())
        config["batch_id"] = f"{plan['batch_id']}-calibration"
        config["run_tag"] = f"prior-clone-{index}"
        config_path = root / "jobs" / f"prior-{index}" / "parameters.json"
        write_json(config_path, config)
        prior_jobs.append({
            "source_run_tag": job["run_tag"],
            "config_path": str(config_path),
        })
    prior_plan = {
        "schema": MODULE.CALIBRATION_PLAN_SCHEMA,
        "calibration_id": "prior-evidence",
        "batch_id": plan["batch_id"],
        "phase_id": phase["phase_id"],
        "setting": phase["setting"],
        "model": phase["model"],
        "method": phase["method"],
        "source_git_commit": plan["git_commit"],
        "source_spec_sha256": plan["spec_sha256"],
        "gpu": plan["gpu"],
        "jobs": prior_jobs,
    }
    prior = {
        "schema": MODULE.CALIBRATION_SCHEMA,
        "calibration_id": "prior-evidence",
        "batch_id": plan["batch_id"],
        "phase_id": phase["phase_id"],
        "setting": phase["setting"],
        "model": phase["model"],
        "method": phase["method"],
        "status": "inconclusive",
        "recommended_cap": cap,
        "levels": [
            {
                "active_count": count,
                "steady_confirmed": True,
                "steady_gpu_memory_mib": 1000.0 + count * 1000.0,
                "peak_gpu_memory_mib": 1100.0 + count * 1000.0,
            }
            for count in range(1, cap + 1)
        ],
        "jobs": [
            {
                "worker_index": index + 1,
                "source_run_tag": jobs[index]["run_tag"],
            }
            for index in range(cap)
        ],
    }
    write_json(root / "CALIBRATION_PLAN.json", prior_plan)
    write_json(root / "CALIBRATION.json", prior)
    return root / "CALIBRATION.json"


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeProcess:
    next_pid = 41000

    def __init__(self, clock, lifetime=8.0, exit_code=0):
        self.clock = clock
        self.started = clock.now
        self.lifetime = lifetime
        self.normal_exit_code = exit_code
        self.forced_exit_code = None
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1

    def poll(self):
        if self.forced_exit_code is not None:
            return self.forced_exit_code
        if self.clock.now - self.started >= self.lifetime:
            return self.normal_exit_code
        return None

    def deliver(self, signum):
        self.forced_exit_code = -int(signum)


class CalibrationHelperTests(unittest.TestCase):
    def test_cli_defaults_include_adaptive_steady_gate(self):
        args = MODULE.parse_args([
            "--batch-id", "calibration-test",
            "--phase-id", PHASE_ID,
            "--target-workers", "3",
        ])
        self.assertIsNone(args.episode_limit)
        self.assertEqual(args.stagger_seconds, 15.0)
        self.assertEqual(args.sample_interval_seconds, 3.0)
        self.assertEqual(args.steady_seconds, 20.0)
        self.assertEqual(args.steady_samples, 3)
        self.assertEqual(args.load_timeout_seconds, 300.0)
        self.assertEqual(args.min_loaded_memory_mib, 512)
        self.assertEqual(args.steady_relative_tolerance, 0.02)
        self.assertEqual(args.initial_workers, 0)
        self.assertIsNone(args.prior_calibration)

    def test_plan_revision_allows_only_audited_helper_drift_for_continuation(self):
        plan = {"git_commit": "a" * 40}

        def git(*args):
            if args[:2] == ("rev-parse", "--verify"):
                return "b" * 40
            if args[:2] == ("status", "--porcelain"):
                return ""
            raise AssertionError(args)

        with mock.patch.object(MODULE, "_git", side_effect=git), \
                mock.patch.object(
                    MODULE, "_helper_only_commit_diff",
                    return_value=["vln/scripts/calibrate_r2r_local_refinement.py"],
                ):
            audit = MODULE._assert_plan_revision(
                plan, allow_helper_only_drift=True
            )
        self.assertEqual(audit["plan_git_commit"], "a" * 40)
        self.assertEqual(audit["execution_git_commit"], "b" * 40)
        with mock.patch.object(MODULE, "_git", side_effect=git):
            with self.assertRaisesRegex(MODULE.UserError, "commit mismatch"):
                MODULE._assert_plan_revision(plan, allow_helper_only_drift=False)

        def forbidden_git(*args):
            if args[0] == "cat-file":
                return ""
            if args[0] == "diff":
                return "vln/scripts/run_source_eval.sh"
            raise AssertionError(args)

        with mock.patch.object(MODULE, "_git", side_effect=forbidden_git):
            with self.assertRaisesRegex(MODULE.UserError, "outside.*allowlist"):
                MODULE._helper_only_commit_diff("a" * 40, "b" * 40)

    def test_clone_isolates_job_config_result_and_exit_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            campaign = temp / "logs" / "calibration-test"
            make_campaign(campaign)
            plan, phase, _, jobs = MODULE._load_phase(campaign, PHASE_ID)
            selected = MODULE._select_distinct_jobs(jobs, 3)
            formal_snapshots = {
                Path(job["config_path"]): Path(job["config_path"]).read_bytes()
                for job in selected
            }
            root, clones, document = MODULE._clone_jobs(
                campaign, plan, phase, selected, "isolation",
                episode_limit=256, tuning_root=temp / "tuning",
            )

            self.assertTrue(MODULE._is_relative_to(root, campaign / "calibration"))
            self.assertTrue(document["formal_paths_are_read_only"])
            for source, clone in zip(selected, clones):
                self.assertNotEqual(clone["job_dir"], source["job_dir"])
                self.assertNotEqual(clone["config_path"], source["config_path"])
                self.assertNotEqual(clone["result_root"], source["result_root"])
                self.assertIn("/_calibration/", clone["result_root"])
                self.assertTrue(
                    clone["result_root"].endswith(
                        f"/{clone['run_tag']}/val_seen"
                    )
                )
                command = clone["command"]
                self.assertEqual(command.count("--episode-limit"), 1)
                self.assertEqual(
                    command[command.index("--episode-limit") + 1], "256"
                )
                self.assertEqual(
                    command[command.index("--run-tag") + 1], clone["run_tag"]
                )
                self.assertEqual(
                    command[command.index("--tta-config") + 1],
                    clone["config_path"],
                )
                self.assertEqual(
                    command[command.index("--result-root") + 1],
                    clone["result_root"],
                )
                config = json.loads(Path(clone["config_path"]).read_text())
                self.assertEqual(config["episodes"], 256)
                self.assertEqual(config["run_tag"], clone["run_tag"])
                self.assertFalse(Path(source["job_dir"], "exitcode").exists())
            for path, content in formal_snapshots.items():
                self.assertEqual(path.read_bytes(), content)

    def test_cross_phase_job_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory) / "calibration-test"
            _, _, jobs = make_campaign(campaign, count=2)
            path = Path(jobs[1]["job_dir"]) / "job.json"
            value = json.loads(path.read_text())
            value["phase_id"] = "99-goat-r2r-feedtta"
            write_json(path, value)
            with self.assertRaisesRegex(MODULE.UserError, "cross-phase job"):
                MODULE._load_phase(campaign, PHASE_ID)

    def test_prior_evidence_requires_exact_continuous_recommended_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory) / "calibration-test"
            plan, phase, jobs = make_campaign(campaign, count=4)
            prior_path = make_prior_calibration(
                campaign, plan, phase, jobs, cap=2
            )
            evidence = MODULE._load_prior_evidence(
                prior_path, campaign, plan, phase, jobs, 2, 0
            )
            self.assertEqual(evidence["recommended_cap"], 2)
            self.assertTrue(evidence["continuous_steady_levels_verified"])
            self.assertEqual(evidence["level_n_steady_gpu_memory_mib"], 3000.0)

            prior = json.loads(prior_path.read_text())
            prior["recommended_cap"] = 3
            write_json(prior_path, prior)
            with self.assertRaisesRegex(MODULE.UserError, "must equal"):
                MODULE._load_prior_evidence(
                    prior_path, campaign, plan, phase, jobs, 2, 0
                )
            prior["recommended_cap"] = 2
            prior["levels"][0]["steady_confirmed"] = False
            write_json(prior_path, prior)
            with self.assertRaisesRegex(MODULE.UserError, "not continuously"):
                MODULE._load_prior_evidence(
                    prior_path, campaign, plan, phase, jobs, 2, 0
                )
            prior["levels"][0]["steady_confirmed"] = True
            write_json(prior_path, prior)
            current_config_path = Path(jobs[0]["config_path"])
            current_config = json.loads(current_config_path.read_text())
            current_config["parameters"]["lr"] = 0.5
            write_json(current_config_path, current_config)
            with self.assertRaisesRegex(MODULE.UserError, "config identity"):
                MODULE._load_prior_evidence(
                    prior_path, campaign, plan, phase, jobs, 2, 0
                )

    def test_two_segment_chain_cap3_to_cap5_can_seed_initial5(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory) / "calibration-test"
            plan, phase, jobs = make_campaign(campaign, count=7)
            cap3_path = make_prior_calibration(
                campaign, plan, phase, jobs, cap=3
            )
            cap3 = MODULE._load_prior_evidence(
                cap3_path, campaign, plan, phase, jobs[:5], 3, 0
            )
            current_levels = [
                {
                    "active_count": count,
                    "steady_confirmed": count >= 3,
                    "steady_gpu_memory_mib": 1000.0 + count * 1000.0,
                    "peak_gpu_memory_mib": 1100.0 + count * 1000.0,
                }
                for count in range(1, 6)
            ]
            merged = MODULE._merge_level_evidence(current_levels, cap3, 3)
            cap5_root = campaign / "calibration" / PHASE_ID / "cap5"
            cap5_plan_jobs = []
            for index, job in enumerate(jobs[:5]):
                config = json.loads(Path(job["config_path"]).read_text())
                config["batch_id"] = "calibration-test-calibration"
                config["run_tag"] = f"cap5-clone-{index}"
                config_path = cap5_root / "jobs" / str(index) / "parameters.json"
                write_json(config_path, config)
                cap5_plan_jobs.append({
                    "source_run_tag": job["run_tag"],
                    "config_path": str(config_path),
                })
            public_cap3 = {
                key: value for key, value in cap3.items()
                if not key.startswith("_")
            }
            write_json(cap5_root / "CALIBRATION_PLAN.json", {
                "schema": MODULE.CALIBRATION_PLAN_SCHEMA,
                "calibration_id": "cap5",
                "batch_id": plan["batch_id"],
                "phase_id": phase["phase_id"],
                "setting": phase["setting"],
                "model": phase["model"],
                "method": phase["method"],
                "source_git_commit": plan["git_commit"],
                "source_spec_sha256": plan["spec_sha256"],
                "gpu": 0,
                "jobs": cap5_plan_jobs,
            })
            write_json(cap5_root / "CALIBRATION.json", {
                "schema": MODULE.CALIBRATION_SCHEMA,
                "level_evidence_schema": MODULE.LEVEL_EVIDENCE_SCHEMA,
                "calibration_id": "cap5",
                "batch_id": plan["batch_id"],
                "phase_id": phase["phase_id"],
                "setting": phase["setting"],
                "model": phase["model"],
                "method": phase["method"],
                "status": "completed",
                "recommended_cap": 5,
                "initial_workers": 3,
                "initial_group": {"steady_confirmed": True},
                "prior_evidence": public_cap3,
                "levels": merged,
                "jobs": [
                    {
                        "worker_index": index + 1,
                        "source_run_tag": jobs[index]["run_tag"],
                    }
                    for index in range(5)
                ],
                "source_git_commit": plan["git_commit"],
                "source_spec_sha256": plan["spec_sha256"],
                "gpu": 0,
                "execution_git_commit": plan["git_commit"],
            })

            cap5 = MODULE._load_prior_evidence(
                cap5_root / "CALIBRATION.json", campaign, plan, phase,
                jobs[:7], 5, 0,
            )
            self.assertEqual(cap5["recommended_cap"], 5)
            prefix = cap5["_validated_level_prefix"]
            self.assertEqual(
                [level["active_count"] for level in prefix], [1, 2, 3, 4, 5]
            )
            self.assertEqual(prefix[0]["evidence_origin"], "prior")
            cap5_path = cap5_root / "CALIBRATION.json"
            strict_document = json.loads(cap5_path.read_text())
            legacy_document = copy.deepcopy(strict_document)
            legacy_document.pop("level_evidence_schema")
            for level in legacy_document["levels"]:
                if level["active_count"] < 3:
                    level.clear()
                    level.update({
                        "active_count": len([
                            item for item in legacy_document["levels"]
                            if item.get("active_count", 99) < 3
                        ]),
                        "steady_confirmed": False,
                        "steady_gpu_memory_mib": None,
                        "peak_gpu_memory_mib": 1,
                    })
            # Preserve exact active-count identities after simulating the old
            # transient replay rows.
            legacy_document["levels"][0]["active_count"] = 1
            legacy_document["levels"][1]["active_count"] = 2
            write_json(cap5_path, legacy_document)
            legacy_cap5 = MODULE._load_prior_evidence(
                cap5_path, campaign, plan, phase, jobs[:7], 5, 0
            )
            self.assertEqual(
                legacy_cap5["_validated_level_prefix"][0]["evidence_origin"],
                "prior_legacy_resolved",
            )
            write_json(cap5_path, strict_document)

            next_levels = MODULE._merge_level_evidence(
                [
                    {
                        "active_count": count,
                        "steady_confirmed": count >= 5,
                        "steady_gpu_memory_mib": 1000.0 + count * 1000.0,
                        "peak_gpu_memory_mib": 1100.0 + count * 1000.0,
                    }
                    for count in range(1, 8)
                ],
                cap5,
                5,
            )
            by_count = {level["active_count"]: level for level in next_levels}
            self.assertEqual(by_count[1]["evidence_origin"], "prior")
            self.assertEqual(
                by_count[5]["evidence_origin"],
                "current_initial_group_revalidation",
            )
            tampered = json.loads(cap5_path.read_text())
            inherited = next(
                level for level in tampered["levels"]
                if level["active_count"] == 1
            )
            inherited["steady_gpu_memory_mib"] += 1
            write_json(cap5_path, tampered)
            with self.assertRaisesRegex(MODULE.UserError, "binding|bound field"):
                MODULE._load_prior_evidence(
                    cap5_path, campaign, plan, phase, jobs[:7], 5, 0
                )

            with self.assertRaisesRegex(MODULE.UserError, "cycle"):
                MODULE._validated_level_chain(
                    json.loads(cap3_path.read_text()), cap3_path,
                    campaign.parent, visited={cap3_path.resolve()},
                )

    def _prepared_clones(self, temp, count=3):
        campaign = temp / "logs" / "calibration-test"
        make_campaign(campaign, count=count)
        plan, phase, _, jobs = MODULE._load_phase(campaign, PHASE_ID)
        selected = MODULE._select_distinct_jobs(jobs, count)
        root, clones, calibration_plan = MODULE._clone_jobs(
            campaign, plan, phase, selected, "runtime",
            episode_limit=128, tuning_root=temp / "tuning",
        )
        calibration_plan.update({
            "gpu": 0,
            "stagger_seconds": 2.0,
            "sample_interval_seconds": 2.0,
            "steady_seconds": 2.0,
            "steady_samples": 2,
            "load_timeout_seconds": 30.0,
            "min_loaded_memory_mib": 500,
            "steady_relative_tolerance": 0.01,
        })
        return root, phase, clones, calibration_plan

    def test_progressive_launch_and_summary_recommend_target(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root, phase, clones, plan = self._prepared_clones(temp)
            clock = FakeClock()
            launched_at = []
            processes = []

            def launch(_job):
                launched_at.append(clock.now)
                process = FakeProcess(clock, lifetime=20.0)
                processes.append(process)
                return process, io.BytesIO()

            def gpu(_gpu):
                return 1000 + 1000 * len(processes), 91

            with mock.patch.object(MODULE.time, "monotonic", clock.monotonic), \
                    mock.patch.object(MODULE.time, "sleep", clock.sleep), \
                    mock.patch.object(MODULE, "_launch", side_effect=launch), \
                    mock.patch.object(MODULE, "_gpu_stats", side_effect=gpu), \
                    mock.patch.object(MODULE, "_cgroup_memory_gib", return_value=4.0):
                summary = MODULE._run_calibration(
                    root, phase, clones, plan, gpu=0,
                    stagger_seconds=0.0, sample_interval_seconds=2.0,
                    terminate_grace_seconds=0.0,
                    steady_seconds=2.0, steady_samples=2,
                    load_timeout_seconds=30.0,
                    min_loaded_memory_mib=500,
                    steady_relative_tolerance=0.01,
                )

            self.assertEqual(launched_at, [0.0, 4.0, 8.0])
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["max_observed_active_count"], 3)
            self.assertEqual(summary["recommended_cap"], 3)
            self.assertEqual(summary["successful_workers"], 3)
            self.assertTrue(all(job["episode_throughput_eps"] for job in summary["jobs"]))
            levels = {item["active_count"]: item for item in summary["levels"]}
            self.assertEqual(levels[1]["previous_steady_gpu_memory_mib"], 1000.0)
            self.assertEqual(levels[1]["steady_gpu_memory_mib"], 2000.0)
            self.assertEqual(levels[2]["previous_steady_gpu_memory_mib"], 2000.0)
            self.assertEqual(levels[3]["previous_steady_gpu_memory_mib"], 3000.0)
            self.assertEqual(levels[3]["peak_gpu_memory_mib"], 4000)
            self.assertEqual(levels[3]["steady_gpu_memory_mib"], 4000.0)
            self.assertTrue(all(levels[count]["steady_confirmed"] for count in (1, 2, 3)))
            self.assertEqual(levels[3]["load_wait_seconds"], 4.0)
            self.assertTrue((root / "resource.csv").is_file())
            persisted = json.loads((root / "CALIBRATION.json").read_text())
            self.assertEqual(persisted["recommended_cap"], 3)

    def test_thirty_second_cold_start_never_releases_second_worker_early(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root, phase, clones, plan = self._prepared_clones(temp, count=2)
            clock = FakeClock()
            launched_at = []
            processes = []

            def launch(_job):
                launched_at.append(clock.now)
                process = FakeProcess(clock, lifetime=60.0)
                processes.append(process)
                return process, io.BytesIO()

            def gpu(_gpu):
                if not processes:
                    return 1, 0
                if len(processes) == 1:
                    return (1, 0) if clock.now < 30 else (2001, 90)
                return 3001, 90

            with mock.patch.object(MODULE.time, "monotonic", clock.monotonic), \
                    mock.patch.object(MODULE.time, "sleep", clock.sleep), \
                    mock.patch.object(MODULE, "_launch", side_effect=launch), \
                    mock.patch.object(MODULE, "_gpu_stats", side_effect=gpu), \
                    mock.patch.object(MODULE, "_cgroup_memory_gib", return_value=4.0):
                summary = MODULE._run_calibration(
                    root, phase, clones, plan, gpu=0,
                    stagger_seconds=0.0, sample_interval_seconds=2.0,
                    terminate_grace_seconds=0.0,
                    steady_seconds=2.0, steady_samples=2,
                    load_timeout_seconds=50.0,
                    min_loaded_memory_mib=500,
                    steady_relative_tolerance=0.01,
                )

            self.assertEqual(launched_at, [0.0, 32.0])
            self.assertGreater(launched_at[1], 30.0)
            self.assertEqual(summary["recommended_cap"], 2)
            levels = {item["active_count"]: item for item in summary["levels"]}
            self.assertEqual(levels[1]["load_wait_seconds"], 32.0)
            self.assertEqual(levels[2]["previous_steady_gpu_memory_mib"], 2001.0)

    def test_load_timeout_fails_without_launching_next_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root, phase, clones, plan = self._prepared_clones(temp, count=2)
            clock = FakeClock()
            processes = []

            def launch(_job):
                process = FakeProcess(clock, lifetime=100.0)
                processes.append(process)
                return process, io.BytesIO()

            def deliver(process, signum):
                process.deliver(signum)

            with mock.patch.object(MODULE.time, "monotonic", clock.monotonic), \
                    mock.patch.object(MODULE.time, "sleep", clock.sleep), \
                    mock.patch.object(MODULE, "_launch", side_effect=launch), \
                    mock.patch.object(MODULE, "_gpu_stats", return_value=(1, 0)), \
                    mock.patch.object(MODULE, "_cgroup_memory_gib", return_value=4.0), \
                    mock.patch.object(MODULE, "_signal_process_group", side_effect=deliver):
                summary = MODULE._run_calibration(
                    root, phase, clones, plan, gpu=0,
                    stagger_seconds=0.0, sample_interval_seconds=2.0,
                    terminate_grace_seconds=0.0,
                    steady_seconds=2.0, steady_samples=2,
                    load_timeout_seconds=20.0,
                    min_loaded_memory_mib=500,
                    steady_relative_tolerance=0.01,
                )

            self.assertEqual(len(processes), 1)
            self.assertEqual(summary["stop_reason"], "load_timeout")
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["recommended_cap"], 0)
            level = next(item for item in summary["levels"] if item["active_count"] == 1)
            self.assertFalse(level["steady_confirmed"])
            self.assertEqual(level["previous_steady_gpu_memory_mib"], 1.0)
            self.assertEqual(level["load_wait_seconds"], 20.0)

    def test_worker_exit_before_new_level_steady_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root, phase, clones, plan = self._prepared_clones(temp, count=2)
            clock = FakeClock()
            processes = []

            def launch(_job):
                lifetime = 6.0 if not processes else 100.0
                process = FakeProcess(clock, lifetime=lifetime)
                processes.append(process)
                return process, io.BytesIO()

            def gpu(_gpu):
                return 1000 + 1000 * len(processes), 90

            def deliver(process, signum):
                process.deliver(signum)

            with mock.patch.object(MODULE.time, "monotonic", clock.monotonic), \
                    mock.patch.object(MODULE.time, "sleep", clock.sleep), \
                    mock.patch.object(MODULE, "_launch", side_effect=launch), \
                    mock.patch.object(MODULE, "_gpu_stats", side_effect=gpu), \
                    mock.patch.object(MODULE, "_cgroup_memory_gib", return_value=4.0), \
                    mock.patch.object(MODULE, "_signal_process_group", side_effect=deliver):
                summary = MODULE._run_calibration(
                    root, phase, clones, plan, gpu=0,
                    stagger_seconds=0.0, sample_interval_seconds=2.0,
                    terminate_grace_seconds=0.0,
                    steady_seconds=2.0, steady_samples=2,
                    load_timeout_seconds=30.0,
                    min_loaded_memory_mib=500,
                    steady_relative_tolerance=0.01,
                )

            self.assertEqual(summary["stop_reason"], "worker_exited_before_steady")
            self.assertEqual(summary["status"], "inconclusive")
            self.assertEqual(summary["recommended_cap"], 1)
            levels = {item["active_count"]: item for item in summary["levels"]}
            self.assertTrue(levels[1]["steady_confirmed"])
            self.assertFalse(levels[2]["steady_confirmed"])

    def test_prior_safe_initial_group_revalidates_before_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root, phase, clones, plan = self._prepared_clones(temp, count=4)
            clock = FakeClock()
            launched_at = []
            processes = []
            evidence = {
                "calibration_id": "prior",
                "path": "/tmp/prior/CALIBRATION.json",
                "sha256": "a" * 64,
                "recommended_cap": 2,
                "initial_workers": 2,
                "level_n_steady_gpu_memory_mib": 3000.0,
                "_validated_level_prefix": [
                    {
                        "active_count": count,
                        "steady_confirmed": True,
                        "steady_gpu_memory_mib": 1000.0 + count * 1000.0,
                        "peak_gpu_memory_mib": 1100.0 + count * 1000.0,
                    }
                    for count in (1, 2)
                ],
            }
            plan["initial_workers"] = 2
            plan["prior_evidence"] = evidence

            def launch(_job):
                launched_at.append(clock.now)
                process = FakeProcess(clock, lifetime=30.0)
                processes.append(process)
                return process, io.BytesIO()

            def gpu(_gpu):
                return 1000 + 1000 * len(processes), 90

            with mock.patch.object(MODULE.time, "monotonic", clock.monotonic), \
                    mock.patch.object(MODULE.time, "sleep", clock.sleep), \
                    mock.patch.object(MODULE, "_launch", side_effect=launch), \
                    mock.patch.object(MODULE, "_gpu_stats", side_effect=gpu), \
                    mock.patch.object(MODULE, "_cgroup_memory_gib", return_value=4.0):
                summary = MODULE._run_calibration(
                    root, phase, clones, plan, gpu=0,
                    stagger_seconds=2.0, sample_interval_seconds=2.0,
                    terminate_grace_seconds=0.0,
                    steady_seconds=2.0, steady_samples=2,
                    load_timeout_seconds=30.0,
                    min_loaded_memory_mib=500,
                    steady_relative_tolerance=0.01,
                    prior_evidence=evidence, initial_workers=2,
                )

            self.assertEqual(launched_at, [0.0, 2.0, 6.0, 10.0])
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["recommended_cap"], 4)
            self.assertEqual(summary["initial_workers"], 2)
            self.assertEqual(summary["prior_evidence"], evidence)
            self.assertTrue(summary["initial_group"]["steady_confirmed"])
            self.assertEqual(
                summary["initial_group"]["required_loaded_gpu_memory_mib"],
                2940.0,
            )
            levels = {item["active_count"]: item for item in summary["levels"]}
            self.assertEqual(
                levels[2]["level_kind"], "prior_safe_initial_group_recovery"
            )
            self.assertTrue(levels[2]["steady_confirmed"])
            self.assertEqual(
                levels[3]["previous_steady_gpu_memory_mib"], 3000.0
            )

    def test_initial_group_worker_exit_before_recovery_is_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root, phase, clones, plan = self._prepared_clones(temp, count=3)
            clock = FakeClock()
            processes = []
            evidence = {
                "calibration_id": "prior",
                "path": "/tmp/prior/CALIBRATION.json",
                "sha256": "a" * 64,
                "recommended_cap": 2,
                "initial_workers": 2,
                "level_n_steady_gpu_memory_mib": 3000.0,
                "_validated_level_prefix": [
                    {
                        "active_count": count,
                        "steady_confirmed": True,
                        "steady_gpu_memory_mib": 1000.0 + count * 1000.0,
                        "peak_gpu_memory_mib": 1100.0 + count * 1000.0,
                    }
                    for count in (1, 2)
                ],
            }
            plan["initial_workers"] = 2
            plan["prior_evidence"] = evidence

            def launch(_job):
                lifetime = 3.0 if not processes else 100.0
                process = FakeProcess(clock, lifetime=lifetime)
                processes.append(process)
                return process, io.BytesIO()

            def gpu(_gpu):
                return 1000 + 1000 * len(processes), 90

            def deliver(process, signum):
                process.deliver(signum)

            with mock.patch.object(MODULE.time, "monotonic", clock.monotonic), \
                    mock.patch.object(MODULE.time, "sleep", clock.sleep), \
                    mock.patch.object(MODULE, "_launch", side_effect=launch), \
                    mock.patch.object(MODULE, "_gpu_stats", side_effect=gpu), \
                    mock.patch.object(MODULE, "_cgroup_memory_gib", return_value=4.0), \
                    mock.patch.object(MODULE, "_signal_process_group", side_effect=deliver):
                summary = MODULE._run_calibration(
                    root, phase, clones, plan, gpu=0,
                    stagger_seconds=2.0, sample_interval_seconds=2.0,
                    terminate_grace_seconds=0.0,
                    steady_seconds=2.0, steady_samples=2,
                    load_timeout_seconds=30.0,
                    min_loaded_memory_mib=500,
                    steady_relative_tolerance=0.01,
                    prior_evidence=evidence, initial_workers=2,
                )

            self.assertEqual(len(processes), 2)
            self.assertEqual(summary["stop_reason"], "initial_group_worker_exit")
            self.assertEqual(summary["status"], "failed")
            self.assertFalse(summary["initial_group"]["steady_confirmed"])
            self.assertEqual(summary["recommended_cap"], 0)

    def test_29gb_gate_stops_before_next_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root, phase, clones, plan = self._prepared_clones(temp)
            clock = FakeClock()
            processes = []

            def launch(_job):
                process = FakeProcess(clock, lifetime=6.0)
                processes.append(process)
                return process, io.BytesIO()

            def gpu(_gpu):
                return ((1000, 10) if clock.now == 0 else (29000, 95))

            with mock.patch.object(MODULE.time, "monotonic", clock.monotonic), \
                    mock.patch.object(MODULE.time, "sleep", clock.sleep), \
                    mock.patch.object(MODULE, "_launch", side_effect=launch), \
                    mock.patch.object(MODULE, "_gpu_stats", side_effect=gpu), \
                    mock.patch.object(MODULE, "_cgroup_memory_gib", return_value=4.0):
                summary = MODULE._run_calibration(
                    root, phase, clones, plan, gpu=0,
                    stagger_seconds=2.0, sample_interval_seconds=2.0,
                    terminate_grace_seconds=0.0,
                )

            self.assertEqual(len(processes), 1)
            self.assertEqual(summary["status"], "safety_stop")
            self.assertEqual(summary["stop_reason"], "planned_memory_threshold")
            self.assertEqual(summary["unlaunched_workers"], 2)
            self.assertEqual(summary["recommended_cap"], 0)

    def test_30gb_gate_terminates_only_launched_calibration_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root, phase, clones, plan = self._prepared_clones(temp)
            clock = FakeClock()
            processes = []

            def launch(_job):
                process = FakeProcess(clock, lifetime=100.0)
                processes.append(process)
                return process, io.BytesIO()

            def gpu(_gpu):
                return ((1000, 5) if clock.now == 0 else (30000, 99))

            def deliver(process, signum):
                process.deliver(signum)

            with mock.patch.object(MODULE.time, "monotonic", clock.monotonic), \
                    mock.patch.object(MODULE.time, "sleep", clock.sleep), \
                    mock.patch.object(MODULE, "_launch", side_effect=launch), \
                    mock.patch.object(MODULE, "_gpu_stats", side_effect=gpu), \
                    mock.patch.object(MODULE, "_cgroup_memory_gib", return_value=4.0), \
                    mock.patch.object(MODULE, "_signal_process_group", side_effect=deliver):
                summary = MODULE._run_calibration(
                    root, phase, clones, plan, gpu=0,
                    stagger_seconds=2.0, sample_interval_seconds=2.0,
                    terminate_grace_seconds=0.0,
                )

            self.assertEqual(len(processes), 1)
            self.assertEqual(processes[0].forced_exit_code, -signal.SIGTERM)
            self.assertEqual(summary["status"], "emergency_abort")
            self.assertEqual(summary["emergency_observed_gpu_memory_mib"], 30000)
            self.assertEqual(summary["failed_workers"], 1)
            self.assertEqual(summary["recommended_cap"], 0)


if __name__ == "__main__":
    unittest.main()
