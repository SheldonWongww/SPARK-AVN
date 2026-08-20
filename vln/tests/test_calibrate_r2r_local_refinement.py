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
        "git_commit": "a" * 40,
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
