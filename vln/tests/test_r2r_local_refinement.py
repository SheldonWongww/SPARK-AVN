from contextlib import ExitStack, nullcontext
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/run_r2r_local_refinement.py"
MODULE_SPEC = importlib.util.spec_from_file_location("r2r_local_refinement", SCRIPT)
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)

TRANSLATOR_SCRIPT = REPO_ROOT / "vln/scripts/tta_config_cli.py"
TRANSLATOR_SPEC = importlib.util.spec_from_file_location(
    "r2r_local_refinement_tta_config_cli", TRANSLATOR_SCRIPT
)
TRANSLATOR = importlib.util.module_from_spec(TRANSLATOR_SPEC)
TRANSLATOR_SPEC.loader.exec_module(TRANSLATOR)


def cli(**overrides):
    values = {
        "batch_id": "local-refinement-test",
        "gpu": 0,
        "resume": False,
        "retry_failed": False,
        "plan_only": False,
        "print_commands": False,
        "confirm_reviewed": True,
        "max_workers": None,
        "max_gpu_memory_mib": None,
        "max_memory_gib": 75.0,
        "launch_stagger": 0.0,
        "resource_wait_timeout": 10.0,
        "phase_max_workers": {},
        "phase_calibrations": {},
    }
    values.update(overrides)
    return type("Args", (), values)()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_worker_calibration(root, spec, phase, requested, args):
    jobs = MODULE.build_phase_jobs(
        phase, args.batch_id, spec, gpu=args.gpu
    )
    policy = MODULE._calibration_policy(phase, spec)
    calibration_root = (
        Path(root) / "calibration" / phase["phase_id"] / f"verify-{requested}"
    )
    parent_root = (
        Path(root) / "calibration" / phase["phase_id"] / "baseline-five"
    )
    parent_summary_path = parent_root / "CALIBRATION.json"
    parent_plan_path = parent_root / "CALIBRATION_PLAN.json"
    write_json(parent_summary_path, {"calibration_id": "baseline-five"})
    write_json(parent_plan_path, {"calibration_id": "baseline-five"})

    planned_jobs = []
    summary_jobs = []
    for index, job in enumerate(jobs[:requested], start=1):
        config = MODULE.staged._job_config(job)
        config.update({
            "batch_id": f"{args.batch_id}-calibration",
            "run_tag": f"calibration-worker-{index}",
            "episodes": policy["episode_limit"],
        })
        job_dir = calibration_root / "jobs" / str(index)
        config_path = job_dir / "parameters.json"
        write_json(config_path, config)
        (job_dir / "exitcode").write_text("0\n", encoding="utf-8")
        planned_jobs.append({
            "job_dir": str(job_dir),
            "config_path": str(config_path),
            "source_run_tag": job["run_tag"],
            "command": list(job["command"]) + [
                "--episode-limit", str(policy["episode_limit"])
            ],
        })
        summary_jobs.append({
            "worker_index": index,
            "source_run_tag": job["run_tag"],
            "exit_code": 0,
            "episode_budget": policy["episode_limit"],
        })

    resource_path = calibration_root / "resource.csv"
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    with resource_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=MODULE.CALIBRATION_RESOURCE_FIELDS
        )
        writer.writeheader()
        writer.writerow({
            "observed_at": "2026-08-20T00:00:00+0800",
            "elapsed_seconds": "30",
            "phase_id": phase["phase_id"],
            "setting": phase["setting"],
            "method": phase["method"],
            "launched_count": str(requested),
            "active_count": str(requested),
            "completed_count": "0",
            "gpu_memory_mib": "24000",
            "gpu_utilization_pct": "95",
            "cgroup_memory_gib": "20",
            "action": "sample",
        })

    common = {
        "calibration_id": f"verify-{requested}",
        "phase_id": phase["phase_id"],
        "setting": phase["setting"],
        "model": phase["model"],
        "method": phase["method"],
        "source_git_commit": MODULE.git("rev-parse", "HEAD"),
        "source_spec_sha256": spec["_sha256"],
        "gpu": args.gpu,
        "calibration_policy": policy,
        "episode_limit": policy["episode_limit"],
        "episode_budget_per_worker": policy["episode_limit"],
        "target_workers": requested,
        "initial_workers": requested,
    }
    calibration_plan = {
        "schema": MODULE.CALIBRATION_PLAN_SCHEMA,
        **common,
        "jobs": planned_jobs,
    }
    summary = {
        "schema": MODULE.CALIBRATION_SCHEMA,
        **common,
        "status": "completed",
        "stop_reason": "target_completed",
        "recommended_cap": requested,
        "launched_workers": requested,
        "successful_workers": requested,
        "failed_workers": 0,
        "unlaunched_workers": 0,
        "peak_observed_gpu_memory_mib": 24_000,
        "resource_csv": "resource.csv",
        "resource_csv_sha256": MODULE.sha256(resource_path),
        "initial_group": {"steady_confirmed": True},
        "initial_group_mode": "sizing_jump",
        "levels": [{
            "active_count": requested,
            "steady_confirmed": True,
            "evidence_origin": "current_sizing_jump_group",
        }],
        "sizing_parent": {
            "path": str(parent_summary_path),
            "sha256": MODULE.sha256(parent_summary_path),
            "plan_path": str(parent_plan_path),
            "plan_sha256": MODULE.sha256(parent_plan_path),
        },
        "sizing_projection": {"accepted": True},
        "jobs": summary_jobs,
    }
    plan_path = calibration_root / "CALIBRATION_PLAN.json"
    summary_path = calibration_root / "CALIBRATION.json"
    write_json(plan_path, calibration_plan)
    write_json(summary_path, summary)
    return jobs, summary_path, plan_path, summary, calibration_plan


class R2RLocalRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = MODULE.load_spec()

    def test_spec_expands_dynamic_enabled_phases_without_eam(self):
        phases = MODULE.phase_sequence(self.spec)
        enabled = MODULE.enabled_phases_by_setting(self.spec)
        self.assertEqual(len(phases), sum(len(value) for value in enabled.values()))
        self.assertEqual(
            sum(phase["job_count"] for phase in phases),
            self.spec["budget"]["total_before_confirmation"],
        )
        self.assertEqual(
            [(phase["setting"], phase["method"]) for phase in phases],
            [
                (setting, method)
                for setting in MODULE.SETTINGS
                for method in enabled[setting]
            ],
        )
        self.assertTrue(all("eam" not in value for value in enabled.values()))
        self.assertNotIn(
            "eam", self.spec["execution"]["method_order_within_model"]
        )
        self.assertNotIn("atena", enabled["duet-r2r"])
        self.assertNotIn("atena", enabled["hamt-r2r"])
        self.assertIn("atena", enabled["goat-r2r"])
        active = MODULE.active_tta_methods(self.spec)
        self.assertNotIn("eam", active)
        for method in active:
            for setting in MODULE.enabled_settings_for_method(self.spec, method):
                points = MODULE.expand_candidates(method, setting, self.spec)
                self.assertEqual(
                    len(points),
                    len({MODULE.canonical(point["parameters"]) for point in points}),
                )
                anchor = self.spec["parent_anchors"][method][setting]["candidate"]
                self.assertEqual(
                    sum(
                        MODULE.canonical(point["candidate"])
                        == MODULE.canonical(anchor)
                        for point in points
                    ),
                    1,
                )

    def test_spec_rejects_duplicate_and_budget_drift(self):
        duplicate = copy.deepcopy(self.spec)
        duplicate.pop("_path", None)
        duplicate.pop("_sha256", None)
        duplicate.pop("setting_episode_counts", None)
        points = duplicate["methods"]["feedtta"]["settings"]["duet-r2r"]["points"]
        points.append(copy.deepcopy(points[0]))
        duplicate["methods"]["feedtta"]["settings"]["duet-r2r"][
            "candidate_count"
        ] += 1
        with self.assertRaisesRegex(MODULE.UserError, "duplicate"):
            MODULE._validate_spec(duplicate)

        drift = copy.deepcopy(self.spec)
        drift.pop("_path", None)
        drift.pop("_sha256", None)
        drift.pop("setting_episode_counts", None)
        drift["budget"]["search_total"] += 1
        with self.assertRaisesRegex(MODULE.UserError, "search_total"):
            MODULE._validate_spec(drift)

    def test_reused_source_manifest_is_pinned_and_source_is_not_a_phase(self):
        binding = self.spec["_reused_source_manifest"]
        self.assertRegex(binding["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            binding["document"]["schema"], MODULE.REUSED_SOURCE_SCHEMA
        )
        self.assertEqual(
            set(binding["document"]["records"]), set(MODULE.SETTINGS)
        )
        plan_binding = MODULE._reused_source_plan_binding(self.spec)
        self.assertEqual(plan_binding["reference"], plan_binding["path"])
        self.assertFalse(Path(plan_binding["path"]).is_absolute())
        self.assertTrue(all(
            "source" not in phases
            for phases in MODULE.enabled_phases_by_setting(self.spec).values()
        ))
        self.assertEqual(
            set(self.spec["execution"]["control_execution"]),
            {"feedtta_control"},
        )

        with mock.patch.object(
            MODULE,
            "_load_validated_results",
            side_effect=AssertionError("historical Source artifacts were read"),
        ):
            record = MODULE._control_result(
                "new-campaign", self.spec, "duet-r2r", "source"
            )
        self.assertEqual(
            record,
            binding["document"]["records"]["duet-r2r"],
        )
        record["metrics"]["SR"] = -1
        self.assertNotEqual(
            MODULE._control_result(
                "new-campaign", self.spec, "duet-r2r", "source"
            )["metrics"]["SR"],
            -1,
        )

    def test_reused_source_manifest_rejects_invalid_records_and_paths(self):
        valid = self.spec["_reused_source_manifest"]["document"]
        invalid_documents = []

        invalid = copy.deepcopy(valid)
        invalid["schema"] = "unsupported"
        invalid_documents.append(("schema", invalid))

        invalid = copy.deepcopy(valid)
        invalid["records"].pop("goat-r2r")
        invalid_documents.append(("keys mismatch", invalid))

        invalid = copy.deepcopy(valid)
        invalid["records"]["duet-r2r"]["parameters"]["action_seed"] = 1
        invalid_documents.append(("argmax with seed 0", invalid))

        invalid = copy.deepcopy(valid)
        invalid["records"]["hamt-r2r"]["metrics"].pop("SPL")
        invalid_documents.append(("metrics.SPL", invalid))

        invalid = copy.deepcopy(valid)
        invalid["records"]["goat-r2r"]["formal_manifest_sha256"] = None
        invalid_documents.append(("formal_manifest_sha256", invalid))

        invalid = copy.deepcopy(valid)
        invalid["source_batch"]["git_commit"] = "0" * 39
        invalid_documents.append(("40-hex Git commit", invalid))

        invalid = copy.deepcopy(valid)
        invalid["records"]["duet-r2r"]["job_json_path"] = "../job.json"
        invalid_documents.append(("safe repository-relative path", invalid))

        for error, document in invalid_documents:
            with self.subTest(error=error), self.assertRaisesRegex(
                MODULE.UserError, error
            ):
                MODULE._validate_reused_source_document(document, "test-manifest")

        outside = copy.deepcopy(self.spec)
        outside["controls"]["standard_argmax_source_manifest"] = "/tmp/source.json"
        with self.assertRaisesRegex(
            MODULE.UserError, "safe repository-relative path"
        ):
            MODULE._validate_spec(outside)

        untracked = copy.deepcopy(self.spec)
        untracked["controls"]["standard_argmax_source_manifest"] = (
            "vln/manifests/not-a-tracked-source-control.json"
        )
        with self.assertRaisesRegex(MODULE.UserError, "tracked by Git"):
            MODULE._validate_spec(untracked)

        source_enabled = copy.deepcopy(self.spec)
        source_enabled["execution"]["method_order_within_model"].insert(0, "source")
        source_enabled["execution"]["enabled_methods_by_setting"][
            "duet-r2r"
        ].insert(0, "source")
        with self.assertRaisesRegex(MODULE.UserError, "Source phases must be disabled"):
            MODULE._validate_spec(source_enabled)

    def test_every_generated_runtime_config_translates(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(mock.patch.object(MODULE, "LOG_ROOT", root / "logs"))
            stack.enter_context(mock.patch.object(MODULE, "TUNING_ROOT", root / "tuning"))
            count = 0
            for phase in MODULE.phase_sequence(self.spec):
                for job in MODULE.build_phase_jobs(
                    phase, "translation-test", self.spec, gpu=2
                ):
                    MODULE.staged._write_job(job)
                    config = json.loads(
                        Path(job["config_path"]).read_text(encoding="utf-8")
                    )
                    self.assertNotIn("stage", config)
                    self.assertNotIn("order_seed", config)
                    translated_method, tokens = TRANSLATOR.translate(
                        job["setting"], job["config_path"],
                        str(root / "tta_diagnostics.json"),
                    )
                    self.assertEqual(translated_method, job["config_method"])
                    self.assertTrue(tokens)
                    count += 1
            self.assertEqual(
                count, self.spec["budget"]["total_before_confirmation"]
            )

    def test_campaign_plan_is_pinned_and_resume_revalidates_configs(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(mock.patch.object(MODULE, "LOG_ROOT", root / "logs"))
            stack.enter_context(mock.patch.object(MODULE, "TUNING_ROOT", root / "tuning"))
            args = cli(batch_id="resume-test")
            campaign, planned = MODULE.ensure_campaign_plan(args, self.spec)
            self.assertEqual(len(planned), len(MODULE.phase_sequence(self.spec)))
            self.assertEqual(
                json.loads((campaign / "PLAN.json").read_text())["job_count"],
                self.spec["budget"]["total_before_confirmation"],
            )
            self.assertEqual(
                json.loads((campaign / "PLAN.json").read_text())[
                    "reused_source_manifest"
                ],
                MODULE._reused_source_plan_binding(self.spec),
            )
            with self.assertRaisesRegex(MODULE.UserError, "use --resume"):
                MODULE.ensure_campaign_plan(args, self.spec)

            resumed_args = cli(batch_id="resume-test", resume=True)
            _, resumed = MODULE.ensure_campaign_plan(resumed_args, self.spec)
            job = resumed[1][2][0]
            config_path = Path(job["config_path"])
            original = config_path.read_text(encoding="utf-8")
            config = json.loads(original)
            config["parameters"]["lr"] = 99
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.UserError, "runtime config mismatch"):
                MODULE.ensure_campaign_plan(resumed_args, self.spec)
            config_path.write_text(original, encoding="utf-8")

            manifest = json.loads((campaign / "PLAN.json").read_text())
            manifest["spec_sha256"] = "0" * 64
            (campaign / "PLAN.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.UserError, "spec_sha256"):
                MODULE.ensure_campaign_plan(resumed_args, self.spec)

    def test_launch_guard_requires_enabled_spec_and_manual_confirmation(self):
        disabled = copy.deepcopy(self.spec)
        disabled["execution"]["launchable"] = False
        with self.assertRaisesRegex(MODULE.UserError, "not launchable"):
            MODULE.assert_launchable(cli(), disabled)

        invalid_barrier = copy.deepcopy(self.spec)
        invalid_barrier["execution"]["barrier_implementation_status"] = "anything"
        with self.assertRaisesRegex(MODULE.UserError, "implemented_validated"):
            MODULE.assert_launchable(cli(), invalid_barrier)
        with self.assertRaisesRegex(MODULE.UserError, "implemented_validated"):
            MODULE._validate_spec(invalid_barrier)

        enabled = copy.deepcopy(self.spec)
        enabled["execution"]["launchable"] = True
        enabled["execution"]["barrier_implementation_status"] = "implemented_validated"
        with self.assertRaisesRegex(MODULE.UserError, "confirm-reviewed"):
            MODULE.assert_launchable(cli(confirm_reviewed=False), enabled)
        MODULE.assert_launchable(cli(confirm_reviewed=True), enabled)

    def test_phase_worker_override_is_declared_and_persistable(self):
        phase = next(
            item for item in MODULE.phase_sequence(self.spec)
            if item["method"] not in MODULE.CONTROL_METHODS
            and self.spec["execution"]["concurrency_calibration"]
            [item["method"]][item["setting"]].get("conditional_test_steps")
        )
        value = self.spec["execution"]["concurrency_calibration"][
            phase["method"]
        ][phase["setting"]]["conditional_test_steps"][0]
        args = cli(phase_max_workers={phase["phase_id"]: value})
        self.assertEqual(
            MODULE._configured_worker_limit(args, phase, self.spec), value
        )
        campaign = MODULE.campaign_manifest(
            "worker-override", MODULE.phase_sequence(self.spec), self.spec, args
        )
        self.assertEqual(
            campaign["runtime"]["phase_max_workers"],
            {phase["phase_id"]: value},
        )
        phase_doc = MODULE.phase_manifest(
            phase, "worker-override", [], self.spec, args
        )
        self.assertEqual(
            phase_doc["reused_source_manifest"],
            MODULE._reused_source_plan_binding(self.spec),
        )
        self.assertEqual(phase_doc["effective_worker_limit"], value)
        self.assertIn(value, phase_doc["declared_worker_limits"])
        self.assertIn(
            self.spec["execution"]["gpu_safety"][
                "calibration_initial_group_workers"
            ],
            phase_doc["declared_worker_limits"],
        )
        calibration_policy = phase_doc["calibration_policy"]
        self.assertEqual(calibration_policy["initial_group_workers"], 5)
        self.assertEqual(calibration_policy["episode_limit"], 100)
        self.assertEqual(calibration_policy["projection_max_mib"], 28500)
        self.assertEqual(calibration_policy["projection_safety_factor"], 1.05)
        self.assertEqual(
            calibration_policy["allowed_worker_counts"],
            phase_doc["declared_worker_limits"],
        )
        self.assertTrue(calibration_policy["independent_steady_validation"])
        invalid = cli(phase_max_workers={phase["phase_id"]: 999})
        with self.assertRaisesRegex(MODULE.UserError, "not declared"):
            MODULE._configured_worker_limit(invalid, phase, self.spec)

        parsed = MODULE.parse_args([
            "--batch-id", "override-test",
            "--phase-max-workers", f"{phase['phase_id']}={value}",
            "--plan-only",
        ])
        self.assertEqual(parsed.phase_max_workers, {phase["phase_id"]: value})

    def test_raised_worker_override_requires_calibration_evidence(self):
        phases = MODULE.phase_sequence(self.spec)
        phase = next(
            item for item in phases
            if item["method"] not in MODULE.CONTROL_METHODS
            and any(
                value > MODULE._safe_worker_limit(item, self.spec)
                for value in self.spec["execution"]["concurrency_calibration"]
                [item["method"]][item["setting"]].get(
                    "conditional_test_steps", []
                )
            )
        )
        requested = next(
            value
            for value in self.spec["execution"]["concurrency_calibration"][
                phase["method"]
            ][phase["setting"]]["conditional_test_steps"]
            if value > MODULE._safe_worker_limit(phase, self.spec)
        )
        args = cli(
            phase_max_workers={phase["phase_id"]: requested},
            phase_calibrations={},
        )

        with self.assertRaisesRegex(MODULE.UserError, "requires.*calibration"):
            MODULE._validate_worker_calibration_overrides(
                args, phases, self.spec
            )

    def test_phase_calibration_cli_parses_and_rejects_duplicate(self):
        phase = next(
            item for item in MODULE.phase_sequence(self.spec)
            if item["method"] not in MODULE.CONTROL_METHODS
        )
        path = Path("/tmp/calibration-evidence/CALIBRATION.json")
        option = f"{phase['phase_id']}={path}"
        parsed = MODULE.parse_args([
            "--batch-id", "calibration-cli-test",
            "--phase-calibration", option,
            "--plan-only",
        ])
        self.assertEqual(
            parsed.phase_calibrations,
            {phase["phase_id"]: str(path.resolve())},
        )

        with self.assertRaises(SystemExit):
            MODULE.parse_args([
                "--batch-id", "calibration-cli-test",
                "--phase-calibration", option,
                "--phase-calibration", option,
                "--plan-only",
            ])

    def test_grouped_worker_calibration_validates_and_binds_digests(self):
        phase = next(
            item for item in MODULE.phase_sequence(self.spec)
            if item["method"] not in MODULE.CONTROL_METHODS
            and self.spec["execution"]["concurrency_calibration"]
            [item["method"]][item["setting"]].get("conditional_test_steps")
        )
        requested = self.spec["execution"]["concurrency_calibration"][
            phase["method"]
        ][phase["setting"]]["conditional_test_steps"][0]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory) / "logs"
            stack.enter_context(mock.patch.object(MODULE, "LOG_ROOT", root))
            stack.enter_context(mock.patch.object(
                MODULE, "TUNING_ROOT", Path(directory) / "tuning"
            ))
            args = cli(batch_id="calibration-binding-test")
            jobs, summary_path, plan_path, _, _ = make_worker_calibration(
                root, self.spec, phase, requested, args
            )

            binding = MODULE._validated_worker_calibration(
                summary_path, phase, jobs, requested, self.spec, args
            )

            self.assertEqual(binding["sha256"], MODULE.sha256(summary_path))
            self.assertEqual(binding["plan_sha256"], MODULE.sha256(plan_path))
            self.assertRegex(binding["sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(binding["plan_sha256"], r"^[0-9a-f]{64}$")

    def test_grouped_worker_calibration_rejects_tampering(self):
        phase = next(
            item for item in MODULE.phase_sequence(self.spec)
            if item["method"] not in MODULE.CONTROL_METHODS
            and self.spec["execution"]["concurrency_calibration"]
            [item["method"]][item["setting"]].get("conditional_test_steps")
        )
        requested = self.spec["execution"]["concurrency_calibration"][
            phase["method"]
        ][phase["setting"]]["conditional_test_steps"][0]
        for tamper in ("status", "cap", "spec", "config", "parent_digest"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory) / "logs"
                stack.enter_context(mock.patch.object(MODULE, "LOG_ROOT", root))
                stack.enter_context(mock.patch.object(
                    MODULE, "TUNING_ROOT", Path(directory) / "tuning"
                ))
                args = cli(batch_id=f"tamper-{tamper}")
                jobs, summary_path, _, summary, calibration_plan = (
                    make_worker_calibration(
                        root, self.spec, phase, requested, args
                    )
                )
                if tamper == "status":
                    summary["status"] = "failed"
                    write_json(summary_path, summary)
                elif tamper == "cap":
                    summary["recommended_cap"] = requested - 1
                    write_json(summary_path, summary)
                elif tamper == "spec":
                    summary["source_spec_sha256"] = "0" * 64
                    write_json(summary_path, summary)
                elif tamper == "config":
                    config_path = Path(
                        calibration_plan["jobs"][0]["config_path"]
                    )
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    config["parameters"] = {"lr": 99}
                    write_json(config_path, config)
                else:
                    summary["sizing_parent"]["sha256"] = "0" * 64
                    write_json(summary_path, summary)

                with self.assertRaises(MODULE.UserError):
                    MODULE._validated_worker_calibration(
                        summary_path, phase, jobs, requested, self.spec, args
                    )

    def test_execute_campaign_calls_run_batch_in_exact_phase_order(self):
        enabled = copy.deepcopy(self.spec)
        enabled["execution"]["launchable"] = True
        enabled["execution"]["barrier_implementation_status"] = "implemented_validated"
        phases = MODULE.phase_sequence(enabled)
        root = Path("/tmp/refinement-order-test")
        planned = [
            (phase, root / phase["phase_id"], [{"phase_id": phase["phase_id"]}])
            for phase in phases
        ]
        completed = set()
        calls = []

        def complete(path, spec):
            del spec
            return str(path) in completed

        def run_batch(args, path, jobs, spec):
            del args, jobs, spec
            calls.append(Path(path).name)
            completed.add(str(path))
            return [{"placeholder": True}]

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                MODULE, "ensure_campaign_plan", return_value=(root, planned)
            ))
            stack.enter_context(mock.patch.object(MODULE, "campaign_lock", return_value=nullcontext()))
            stack.enter_context(mock.patch.object(MODULE, "_phase_complete", side_effect=complete))
            stack.enter_context(mock.patch.object(MODULE, "_assert_no_live_other_phase"))
            stack.enter_context(mock.patch.object(MODULE, "summarize_phase"))
            stack.enter_context(mock.patch.object(MODULE, "aggregate_campaign"))
            stack.enter_context(mock.patch.object(MODULE.staged, "run_batch", side_effect=run_batch))
            MODULE.execute_campaign(cli(), enabled)

        self.assertEqual(calls, [phase["phase_id"] for phase in phases])

    def test_execute_campaign_never_advances_after_phase_failure(self):
        enabled = copy.deepcopy(self.spec)
        enabled["execution"]["launchable"] = True
        enabled["execution"]["barrier_implementation_status"] = "implemented_validated"
        phases = MODULE.phase_sequence(enabled)[:4]
        root = Path("/tmp/refinement-failure-test")
        planned = [
            (phase, root / phase["phase_id"], [{"phase_id": phase["phase_id"]}])
            for phase in phases
        ]
        completed = set()
        calls = []

        def complete(path, spec):
            del spec
            return str(path) in completed

        def run_batch(args, path, jobs, spec):
            del args, jobs, spec
            calls.append(Path(path).name)
            if len(calls) == 3:
                raise MODULE.staged.UserError("synthetic failure")
            completed.add(str(path))
            return [{"placeholder": True}]

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                MODULE, "ensure_campaign_plan", return_value=(root, planned)
            ))
            stack.enter_context(mock.patch.object(MODULE, "campaign_lock", return_value=nullcontext()))
            stack.enter_context(mock.patch.object(MODULE, "_phase_complete", side_effect=complete))
            stack.enter_context(mock.patch.object(MODULE, "_assert_no_live_other_phase"))
            stack.enter_context(mock.patch.object(MODULE, "summarize_phase"))
            stack.enter_context(mock.patch.object(MODULE.staged, "run_batch", side_effect=run_batch))
            with self.assertRaisesRegex(MODULE.staged.UserError, "synthetic failure"):
                MODULE.execute_campaign(cli(), enabled)

        self.assertEqual(calls, [phase["phase_id"] for phase in phases[:3]])

    def test_resume_rejects_downstream_evidence_before_active_barrier(self):
        enabled = copy.deepcopy(self.spec)
        enabled["execution"]["launchable"] = True
        enabled["execution"]["barrier_implementation_status"] = "implemented_validated"
        phases = MODULE.phase_sequence(enabled)[:2]
        root = Path("/tmp/refinement-downstream-test")
        planned = [
            (phase, root / phase["phase_id"], [{"phase_id": phase["phase_id"]}])
            for phase in phases
        ]
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                MODULE, "ensure_campaign_plan", return_value=(root, planned)
            ))
            stack.enter_context(mock.patch.object(
                MODULE, "campaign_lock", return_value=nullcontext()
            ))
            stack.enter_context(mock.patch.object(
                MODULE, "_phase_complete", return_value=False
            ))
            stack.enter_context(mock.patch.object(
                MODULE, "_phase_started",
                side_effect=lambda path: Path(path).name == phases[1]["phase_id"],
            ))
            with self.assertRaisesRegex(MODULE.UserError, "downstream execution"):
                MODULE.execute_campaign(cli(resume=True), enabled)

    def test_status_exposes_per_phase_barriers_and_totals(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(mock.patch.object(MODULE, "LOG_ROOT", root / "logs"))
            stack.enter_context(mock.patch.object(MODULE, "TUNING_ROOT", root / "tuning"))
            args = cli(batch_id="status-test")
            _, planned = MODULE.ensure_campaign_plan(args, self.spec)
            snapshot = MODULE.status_snapshot("status-test")
            self.assertEqual(
                snapshot["totals"]["planned"],
                self.spec["budget"]["total_before_confirmation"],
            )
            self.assertEqual(snapshot["active_phase"], planned[0][0]["phase_id"])
            self.assertTrue(snapshot["phases"][0]["barrier_released"])
            self.assertFalse(snapshot["phases"][1]["barrier_released"])

            first_root = planned[0][1]
            first_job = planned[0][2][0]
            for job in planned[0][2]:
                Path(job["job_dir"], "console.log").write_text(
                    "Env name: val_seen, SR: 78.0, SPL: 72.0\n",
                    encoding="utf-8",
                )
                Path(job["job_dir"], "exitcode").write_text(
                    "0\n", encoding="utf-8"
                )
                write_json(
                    Path(job["result_root"]) / "tta_diagnostics.json",
                    {
                        "episode_count": 1021,
                        "adapter": {
                            "relative_param_drift": 0.01,
                            "updates": 1,
                        },
                    },
                )
            snapshot = MODULE.status_snapshot("status-test")
            self.assertEqual(snapshot["active_phase"], planned[1][0]["phase_id"])
            self.assertTrue(snapshot["phases"][1]["barrier_released"])
            self.assertFalse(snapshot["phases"][2]["barrier_released"])

            Path(first_job["job_dir"], "console.log").write_text(
                "finished without metrics\n", encoding="utf-8"
            )
            snapshot = MODULE.status_snapshot("status-test")
            self.assertEqual(snapshot["totals"]["invalid"], 1)
            self.assertEqual(snapshot["totals"]["failed"], 1)
            self.assertTrue(snapshot["needs_attention"])

    def test_feedtta_summary_uses_source_and_sampled_control(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(mock.patch.object(MODULE, "LOG_ROOT", root / "logs"))
            phase = MODULE._find_phase(self.spec, "duet-r2r", "feedtta")
            MODULE.phase_root("summary-test", phase).mkdir(parents=True)
            source = MODULE._control_result(
                "summary-test", self.spec, "duet-r2r", "source"
            )
            sampled = {
                "run_tag": "sampled", "parameters": {},
                "metrics": {"SR": 77.0, "SPL": 68.0},
                "adapter_diagnostics": None,
            }
            candidates = []
            for index in range(phase["job_count"]):
                candidates.append({
                    "run_tag": f"candidate-{index}",
                    "parameters": {"lr": index + 1},
                    "metrics": {
                        "SR": 78.0 + index / 10.0,
                        "SPL": 70.0 + index / 20.0,
                    },
                    "adapter_diagnostics": {
                        "relative_param_drift": 0.01 * index,
                        "updates": index + 1,
                        "feedback_observed_episodes": 1021,
                    },
                })

            def controls(path, spec):
                del spec
                name = Path(path).name
                if name.endswith("-feedtta_control"):
                    return [sampled]
                raise AssertionError(name)

            stack.enter_context(mock.patch.object(
                MODULE, "_load_validated_results", side_effect=controls
            ))
            document = MODULE.summarize_phase(
                phase, "summary-test", self.spec, results=candidates
            )
            winner_index = phase["job_count"] - 1
            self.assertEqual(
                document["winner_run_tag"], f"candidate-{winner_index}"
            )
            self.assertAlmostEqual(
                document["delta_sr_pp"],
                78.0 + winner_index / 10.0 - source["metrics"]["SR"],
            )
            self.assertAlmostEqual(
                document["delta_sampled_control_sr_pp"],
                1.0 + winner_index / 10.0,
            )
            with (MODULE.phase_root("summary-test", phase) / "TOP5.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]["run_tag"], f"candidate-{winner_index}")

    def test_final_aggregate_writes_winners_frozen_and_top5(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(mock.patch.object(MODULE, "LOG_ROOT", root / "logs"))
            batch_id = "aggregate-test"
            for phase in MODULE.phase_sequence(self.spec):
                phase_dir = MODULE.phase_root(batch_id, phase)
                phase_dir.mkdir(parents=True)
                if phase["method"] in MODULE.CONTROL_METHODS:
                    MODULE.staged.atomic_json(phase_dir / "CONTROL.json", {
                        "control": phase["method"],
                        "setting": phase["setting"],
                    })
                    continue
                MODULE.staged.atomic_json(phase_dir / "WINNER.json", {
                    "method": phase["method"],
                    "setting": phase["setting"],
                    "winner_parameters": {"point": phase["phase_id"]},
                })
                with (phase_dir / "TOP5.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as stream:
                    writer = csv.DictWriter(stream, fieldnames=MODULE._top5_fields())
                    writer.writeheader()
                    writer.writerow({
                        key: (phase["phase_id"] if key == "run_tag" else "")
                        for key in MODULE._top5_fields()
                    })

            document = MODULE.aggregate_campaign(batch_id, self.spec)
            active = MODULE.active_tta_methods(self.spec)
            for setting in MODULE.SETTINGS:
                self.assertEqual(
                    document["controls"][setting]["source"],
                    self.spec["_reused_source_manifest"]["document"]["records"][
                        setting
                    ],
                )
            self.assertEqual(set(document["methods"]), set(active))
            self.assertTrue(all(
                set(document["methods"][method])
                == set(MODULE.enabled_settings_for_method(self.spec, method))
                for method in active
            ))
            frozen = json.loads(
                (MODULE.campaign_root(batch_id) / "FROZEN_HPARAMETERS.json").read_text()
            )
            self.assertEqual(set(frozen["settings"]), set(MODULE.SETTINGS))
            with (MODULE.campaign_root(batch_id) / "TOP5.csv").open() as stream:
                self.assertEqual(
                    len(list(csv.DictReader(stream))),
                    sum(
                        1 for phase in MODULE.phase_sequence(self.spec)
                        if phase["method"] not in MODULE.CONTROL_METHODS
                    ),
                )


if __name__ == "__main__":
    unittest.main()
