import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "vln/scripts/monitor_r2r_ce_resources.py"
SPEC = importlib.util.spec_from_file_location("monitor_r2r_ce_resources", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MonitorR2RCEResourcesTest(unittest.TestCase):
    def test_parse_nvidia_smi_rows(self):
        apps = MODULE.parse_compute_apps("123, 3242 MiB\ninvalid\n456, 4100\n")
        self.assertEqual(
            apps,
            [
                {"pid": 123, "gpu_memory_mib": 3242.0},
                {"pid": 456, "gpu_memory_mib": 4100.0},
            ],
        )
        state = MODULE.parse_gpu_state("32760, 10087, 22153, 9, 1\n")
        self.assertEqual(state["memory_total_mib"], 32760.0)
        self.assertEqual(state["memory_used_mib"], 10087.0)

    def test_classify_consistency_job(self):
        command = (
            "bash run_source_eval.sh etpnav-r2r-ce val_unseen 0 "
            "--run-tag batch-search-etpnav-r2r-ce-feedtta-lr_2em6-s3 "
            "--tta-config /tmp/r2r-ce/batch/etpnav-r2r-ce/feedtta/"
            "search/lr_2em6/seed_3/tta_config.json"
        )
        identity = MODULE.classify_command(command)
        self.assertEqual(identity["setting"], "etpnav-r2r-ce")
        self.assertEqual(identity["method"], "feedtta")
        self.assertEqual(identity["stage"], "search")
        self.assertEqual(identity["candidate_id"], "lr_2em6")
        self.assertEqual(identity["order_seed"], 3)
        self.assertEqual(
            identity["run_tag"],
            "batch-search-etpnav-r2r-ce-feedtta-lr_2em6-s3",
        )

    def test_classify_concurrency_calibration_worker(self):
        command = (
            "bash run_source_eval.sh etpnav-r2r-ce val_seen 0 "
            "--run-tag batch-calibration-etpnav-r2r-ce-eam-paper_anchor-w5 "
            "--tta-config /tmp/etpnav-r2r-ce/eam/calibration/"
            "paper_anchor/worker_5/tta_config.json"
        )
        identity = MODULE.classify_command(command)
        self.assertEqual(identity["setting"], "etpnav-r2r-ce")
        self.assertEqual(identity["method"], "eam")
        self.assertEqual(identity["stage"], "calibration")
        self.assertEqual(identity["candidate_id"], "paper_anchor")
        self.assertIsNone(identity["order_seed"])
        self.assertEqual(identity["calibration_worker_index"], 5)

    def test_identify_job_through_parent_chain_and_update_peaks(self):
        tag = "calibration-batch"
        table = {
            10: {
                "ppid": 1,
                "rss_mib": 20.0,
                "command": (
                    "bash run_source_eval.sh bevbert-r2r-ce val_unseen 0 "
                    "--run-tag calibration-batch-search-bevbert-r2r-ce-atena-"
                    "paper_anchor-s1 --tta-config /tmp/bevbert-r2r-ce/atena/"
                    "search/paper_anchor/seed_1/tta_config.json"
                ),
            },
            11: {"ppid": 10, "rss_mib": 6000.0, "command": "python"},
            12: {"ppid": 10, "rss_mib": 500.0, "command": "worker"},
        }
        identity = MODULE.identify_job(11, table, tag)
        self.assertEqual(identity["anchor_pid"], 10)
        self.assertEqual(identity["method"], "atena")
        self.assertEqual(MODULE.descendant_rss(10, table), 6520.0)
        summary = {"phases": {}, "total_samples": 0, "observed_at": None}
        state = {
            "memory_total_mib": 32760.0,
            "memory_used_mib": 5000.0,
            "memory_free_mib": 27760.0,
            "gpu_utilization_percent": 80.0,
            "memory_utilization_percent": 30.0,
        }
        MODULE.update_summary(
            summary,
            state,
            [{"pid": 11, "gpu_memory_mib": 4100.0}],
            table,
            tag,
            {"memory_current_gib": 12.5, "memory_limit_gib": 90.0},
        )
        phase = summary["phases"]["bevbert-r2r-ce/atena/search"]
        self.assertEqual(phase["max_concurrent_jobs_observed"], 1)
        self.assertEqual(phase["concurrent_job_sample_counts"], {"1": 1})
        self.assertEqual(phase["peak_sum_process_gpu_memory_mib"], 4100.0)
        self.assertEqual(phase["peak_cgroup_memory_gib"], 12.5)
        self.assertEqual(phase["mean_board_memory_used_mib"], 5000.0)
        self.assertEqual(phase["mean_gpu_utilization_percent"], 80.0)
        job = next(iter(phase["jobs"].values()))
        self.assertEqual(job["peak_host_rss_mib"], 6520.0)

    def test_read_cgroup_state(self):
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "memory.current"
            maximum = Path(directory) / "memory.max"
            current.write_text(str(12 * 2 ** 30), encoding="utf-8")
            maximum.write_text(str(90 * 2 ** 30), encoding="utf-8")
            state = MODULE.read_cgroup_state(current, maximum)
            self.assertEqual(state["memory_current_gib"], 12.0)
            self.assertEqual(state["memory_limit_gib"], 90.0)

    def test_atomic_json_creates_compact_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            MODULE.atomic_json(path, {"schema": MODULE.SCHEMA, "value": 1})
            self.assertIn(MODULE.SCHEMA, path.read_text(encoding="utf-8"))
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_resume_requires_matching_identity(self):
        class Args:
            output = None
            resume = False
            watch_pattern = "batch"
            gpu = 0
            poll_seconds = 1.0

        with tempfile.TemporaryDirectory() as directory:
            Args.output = str(Path(directory) / "summary.json")
            first = MODULE.load_or_create_summary(Args)
            MODULE.atomic_json(Args.output, first)
            with self.assertRaisesRegex(ValueError, "--resume"):
                MODULE.load_or_create_summary(Args)
            Args.resume = True
            resumed = MODULE.load_or_create_summary(Args)
            self.assertEqual(resumed["watch_pattern"], "batch")
            self.assertEqual(len(resumed["resume_times"]), 1)
            document = json.loads(Path(Args.output).read_text(encoding="utf-8"))
            document["gpu_index"] = 1
            MODULE.atomic_json(Args.output, document)
            with self.assertRaisesRegex(ValueError, "gpu_index"):
                MODULE.load_or_create_summary(Args)


if __name__ == "__main__":
    unittest.main()
