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
    }
    values.update(overrides)
    return type("Args", (), values)()


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
        self.assertEqual(phase_doc["effective_worker_limit"], value)
        invalid = cli(phase_max_workers={phase["phase_id"]: 999})
        with self.assertRaisesRegex(MODULE.UserError, "not declared"):
            MODULE._configured_worker_limit(invalid, phase, self.spec)

        parsed = MODULE.parse_args([
            "--batch-id", "override-test",
            "--phase-max-workers", f"{phase['phase_id']}={value}",
            "--plan-only",
        ])
        self.assertEqual(parsed.phase_max_workers, {phase["phase_id"]: value})

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
            Path(first_job["job_dir"], "console.log").write_text(
                "Env name: val_seen, SR: 78.0, SPL: 72.0\n",
                encoding="utf-8",
            )
            Path(first_job["job_dir"], "exitcode").write_text(
                "0\n", encoding="utf-8"
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
            source = {
                "run_tag": "source", "parameters": {},
                "metrics": {"SR": 78.0, "SPL": 72.0},
                "adapter_diagnostics": None,
            }
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
                if name.endswith("-source"):
                    return [source]
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
            self.assertAlmostEqual(document["delta_sr_pp"], winner_index / 10.0)
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
