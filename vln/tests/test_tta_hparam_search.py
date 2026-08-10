from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/run_tta_hparam_search.py"
MODULE_SPEC = importlib.util.spec_from_file_location("tta_search", SCRIPT)
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)


def args(**overrides):
    values = {
        "method": "tent",
        "settings": ["duet-r2r", "etpnav-r2r-ce"],
        "smoke": False,
        "stage": "stage1",
        "batch_id": "unit-test",
        "episodes": 256,
        "gpu": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def result(run_tag, setting, parameters, primary, sr=80.0, drift=0.01,
           updates=10):
    metric = "RGSPL" if setting.endswith("reverie") else "SPL"
    metrics = {metric: primary, "SR": sr, "SPL": primary, "RGS": primary}
    return {
        "run_tag": run_tag,
        "setting": setting,
        "parameters": parameters,
        "metrics": metrics,
        "adapter_diagnostics": {
            "relative_param_drift": drift,
            "updates": updates,
        },
    }


def write_stage_manifest(root, stage, method, settings, commit=None,
                         spec_sha256=None, batch_id="unit-test",
                         episodes=None):
    directory = Path(root) / "stages" / stage
    directory.mkdir(parents=True, exist_ok=True)
    episodes = (
        MODULE.stage_episode_count(stage, MODULE.load_spec())
        if episodes is None else episodes
    )
    for ordinal, setting in enumerate(settings):
        job_dir = directory / "jobs" / str(ordinal)
        job_dir.mkdir(parents=True, exist_ok=True)
        command = ["runner"]
        if episodes > 0:
            command.extend(["--episode-limit", str(episodes)])
        (job_dir / "job.json").write_text(json.dumps({
            "batch_id": batch_id,
            "ordinal": ordinal,
            "run_tag": "{}-{}-{}".format(batch_id, stage, ordinal),
            "setting": setting,
            "search_method": method,
            "stage": stage,
            "episodes": episodes,
            "command": command,
        }), encoding="utf-8")
    (directory / "stage_manifest.json").write_text(json.dumps({
        "schema": "navtta.vln_tta_search_stage.v1",
        "batch_id": batch_id,
        "git_commit": commit or MODULE.git("rev-parse", "HEAD"),
        "spec_sha256": spec_sha256 or MODULE.sha256(MODULE.SPEC_PATH),
        "method": method,
        "stage": stage,
        "episodes": episodes,
        "job_count": len(settings),
        "settings": list(settings),
    }), encoding="utf-8")
    return directory


def write_frozen_artifacts(root, method, settings, commit=None,
                           spec_sha256=None, split="val_seen"):
    root = Path(root)
    commit = commit or MODULE.git("rev-parse", "HEAD")
    spec_sha256 = spec_sha256 or MODULE.sha256(MODULE.SPEC_PATH)
    selection_settings = {
        setting: {
            "winner_run_tag": "winner-{}".format(setting),
            "frozen_parameters": {"lr": 1e-6},
        }
        for setting in settings
    }
    (root / "FINAL_SELECTION.json").write_text(json.dumps({
        "schema": "navtta.vln_tta_final_selection.v1",
        "method": method,
        "split": split,
        "git_commit": commit,
        "spec_sha256": spec_sha256,
        "settings": selection_settings,
    }), encoding="utf-8")
    (root / "FROZEN_HPARAMETERS.json").write_text(json.dumps({
        "schema": "navtta.vln_tta_frozen_hparams.v1",
        "method": method,
        "split": split,
        "git_commit": commit,
        "spec_sha256": spec_sha256,
        "settings": {
            setting: value["frozen_parameters"]
            for setting, value in selection_settings.items()
        },
    }), encoding="utf-8")


class TTAHparamSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = MODULE.load_spec()

    def test_stage1_grids_are_complete_and_round_robin(self):
        expected_per_setting = {
            "tent": 35,
            "fstta": 20,
            "eam": 35,
            "feedtta": 35,
            "atena": 5,
        }
        for method, count in expected_per_setting.items():
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                jobs = MODULE.build_jobs(
                    args(method=method), self.spec, Path(directory)
                )
                self.assertEqual(len(jobs), count * 2)
                self.assertEqual([job["setting"] for job in jobs[:4]], [
                    "duet-r2r", "etpnav-r2r-ce",
                    "duet-r2r", "etpnav-r2r-ce",
                ])

    def test_tent_anchor_includes_fixed_protocol(self):
        anchor = MODULE.anchor_for("tent", "duet-r2r", self.spec)
        self.assertEqual(anchor["optimizer"], "AdamW")
        self.assertEqual(anchor["norm_scope"], "ln")
        self.assertEqual(anchor["max_grad_norm"], 0.0)
        self.assertFalse(anchor["episodic"])

    def test_smoke_has_one_anchor_per_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = MODULE.build_jobs(
                args(smoke=True, stage="smoke", episodes=2),
                self.spec,
                Path(directory),
            )
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(job["stage"] == "smoke" for job in jobs))
        self.assertTrue(all(job["episodes"] == 2 for job in jobs))

    def test_method_specific_stage_expansions(self):
        setting = "duet-r2r"
        cases = (
            ("fstta", "stage2", 8),
            ("fstta", "stage3", 10),
            ("eam", "stage2", 10),
            ("eam", "stage3", 8),
            ("feedtta", "stage2", 78),
            ("atena", "stage2", 23),
            ("atena", "stage3", 10),
        )
        for method, stage, expected in cases:
            anchor = MODULE.clean_parameters(
                MODULE.anchor_for(method, setting, self.spec)
            )
            second = dict(anchor)
            if method == "fstta":
                second["lr_fast"] *= 3
            elif method == "eam":
                second["lr"] *= 3
            elif method == "feedtta":
                second["lr"] *= 3
            elif method == "atena":
                second["lr_query"] *= 2
                second["lr_self"] *= 2
            promoted = [
                result("anchor", setting, anchor, 80),
                result("second", setting, second, 79),
            ]
            with self.subTest(method=method, stage=stage):
                expanded = MODULE.expand_stage(
                    method, stage, setting, promoted, self.spec
                )
                self.assertEqual(len(expanded), expected)

    def test_feedtta_promotes_anchor_and_best_intensity(self):
        self.assertEqual(
            MODULE.promotion_limit("feedtta", "stage2", self.spec), 2
        )

    def test_final_controls_are_full_val_matched_source_jobs(self):
        settings = ["duet-r2r", "etpnav-r2r-ce"]
        candidates, promotion = MODULE._stage_candidates(
            "feedtta", "final_controls", settings, self.spec, Path("unused")
        )
        self.assertFalse(promotion)
        with tempfile.TemporaryDirectory() as directory:
            jobs = MODULE.build_jobs(
                args(
                    method="feedtta", settings=settings,
                    stage="final_controls", episodes=None,
                ),
                self.spec,
                Path(directory),
                stage="final_controls",
                candidates_by_setting=candidates,
            )
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(job["episodes"] == -1 for job in jobs))
        self.assertTrue(all(job["config_method"] == "source" for job in jobs))
        self.assertTrue(all("--episode-limit" not in job["command"] for job in jobs))
        self.assertTrue(all(
            job["parameters"]["action_selection"] == "sample" for job in jobs
        ))

    def test_promotion_enforces_source_floor_and_keeps_anchor(self):
        setting = "duet-r2r"
        anchor = MODULE.clean_parameters(
            MODULE.anchor_for("tent", setting, self.spec)
        )
        other = dict(anchor)
        other["lr"] = 3e-6
        bad = dict(anchor)
        bad["lr"] = 1e-4
        candidates = [
            result("paper", setting, anchor, 70, sr=78),
            result("best", setting, other, 82, sr=80),
            result("unsafe", setting, bad, 95, sr=70),
        ]
        source = result("source", setting, {}, 75, sr=80)
        selected, record = MODULE.rank_and_promote(
            candidates, source, "tent", setting, 2, self.spec
        )
        self.assertEqual({item["run_tag"] for item in selected}, {"best", "paper"})
        unsafe = next(item for item in record["ranked"]
                      if item["run_tag"] == "unsafe")
        self.assertFalse(unsafe["eligible"])
        self.assertIn("below_matched_source_sr_floor", unsafe["reasons"])

    def test_order_jobs_are_exactly_three_and_pass_order_seed(self):
        candidates = {"duet-r2r": [
            MODULE._candidate({"lr": 1e-6}, order_seed=seed)
            for seed in (0, 1, 2)
        ]}
        with tempfile.TemporaryDirectory() as directory:
            jobs = MODULE.build_jobs(
                args(settings=["duet-r2r"], stage="orders", episodes=-1),
                self.spec,
                Path(directory),
                stage="orders",
                candidates_by_setting=candidates,
            )
        self.assertEqual([job["order_seed"] for job in jobs], [0, 1, 2])
        for seed, job in enumerate(jobs):
            self.assertNotIn("--episode-limit", job["command"])
            index = job["command"].index("--order-seed")
            self.assertEqual(job["command"][index + 1], str(seed))

    def test_final_selection_freezes_full_val_winner_before_orders(self):
        setting = "duet-r2r"
        anchor = MODULE.clean_parameters(
            MODULE.anchor_for("tent", setting, self.spec)
        )
        finalists = []
        for index, score in enumerate((70, 71, 75, 73, 72)):
            parameters = dict(anchor)
            parameters["lr"] = (index + 1) * 1e-6
            finalists.append(result(
                "final-{}".format(index), setting, parameters, score, sr=80
            ))
        source = result("full-source", setting, {}, 69, sr=80)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_stage_manifest(
                root, "final_controls", "tent", [setting]
            )
            write_stage_manifest(root, "final", "tent", [setting])

            def load(stage_dir, spec, allow_partial=False):
                del spec, allow_partial
                if Path(stage_dir).name == "final":
                    return finalists
                if Path(stage_dir).name == "final_controls":
                    return [source]
                raise AssertionError(stage_dir)

            with mock.patch.object(MODULE, "load_stage_results", side_effect=load):
                selection = MODULE.summarize_final(
                    root, "unit-test", "tent", [setting], self.spec
                )

            self.assertEqual(
                selection["settings"][setting]["winner_run_tag"], "final-2"
            )
            selection_path = root / "FINAL_SELECTION.json"
            frozen_path = root / "FROZEN_HPARAMETERS.json"
            original_selection = selection_path.read_bytes()
            before = frozen_path.read_bytes()
            with mock.patch.object(
                MODULE, "load_stage_results", side_effect=load
            ):
                MODULE.summarize_final(
                    root, "unit-test", "tent", [setting], self.spec
                )
            self.assertEqual(selection_path.read_bytes(), original_selection)
            self.assertEqual(frozen_path.read_bytes(), before)
            finalists[0]["metrics"]["SPL"] = 99
            with mock.patch.object(
                MODULE, "load_stage_results", side_effect=load
            ), self.assertRaisesRegex(MODULE.UserError, "immutable output"):
                MODULE.summarize_final(
                    root, "unit-test", "tent", [setting], self.spec
                )
            self.assertEqual(selection_path.read_bytes(), original_selection)
            self.assertEqual(frozen_path.read_bytes(), before)
            candidates, promotion = MODULE._stage_candidates(
                "tent", "orders", [setting], self.spec, root
            )
            self.assertFalse(promotion)
            self.assertEqual(
                [item["order_seed"] for item in candidates[setting]], [0, 1, 2]
            )
            self.assertTrue(all(
                item["parent_run_tags"] == ["final-2"]
                for item in candidates[setting]
            ))
            order_results = []
            for seed, candidate in enumerate(candidates[setting]):
                item = result(
                    "order-{}".format(seed), setting,
                    candidate["parameters"], 74 + seed, sr=80,
                )
                item["order_seed"] = seed
                order_results.append(item)
            write_stage_manifest(root, "orders", "tent", [setting])
            with mock.patch.object(
                MODULE, "load_stage_results", return_value=order_results
            ):
                MODULE.summarize_orders(
                    root, "unit-test", "tent", [setting], self.spec
                )
            order_path = root / "ORDER_ROBUSTNESS.json"
            original_order = order_path.read_bytes()
            with mock.patch.object(
                MODULE, "load_stage_results", return_value=order_results
            ):
                MODULE.summarize_orders(
                    root, "unit-test", "tent", [setting], self.spec
                )
            self.assertEqual(order_path.read_bytes(), original_order)
            order_results[0]["metrics"]["SPL"] = 100
            with mock.patch.object(
                MODULE, "load_stage_results", return_value=order_results
            ), self.assertRaisesRegex(MODULE.UserError, "immutable output"):
                MODULE.summarize_orders(
                    root, "unit-test", "tent", [setting], self.spec
                )
            self.assertEqual(order_path.read_bytes(), original_order)
            self.assertEqual(frozen_path.read_bytes(), before)

    def test_retry_uses_new_run_tag_and_preserves_attempt_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory) / "job"
            job_dir.mkdir()
            config = job_dir / "parameters.json"
            config.write_text("{}", encoding="utf-8")
            for name in ("console.log", "exitcode", "worker_state.json"):
                (job_dir / name).write_text("old\n", encoding="utf-8")
            job = {
                "base_run_tag": "base", "run_tag": "base", "attempt": 0,
                "setting": "duet-r2r", "job_dir": str(job_dir),
                "config_path": str(config), "result_root": "/old/result",
                "command": ["runner", "--run-tag", "base"],
                "config_method": "tent", "search_method": "tent",
                "stage": "stage1", "episodes": 2, "order_seed": None,
                "parameters": {},
            }
            MODULE._bump_attempt(job)
            self.assertEqual(job["run_tag"], "base-retry1")
            self.assertIn("base-retry1", job["result_root"])
            self.assertEqual(job["command"][-1], "base-retry1")
            attempt = job_dir / "attempts" / "attempt-00"
            self.assertTrue((attempt / "console.log").is_file())
            self.assertTrue((attempt / "exitcode").is_file())

    def test_parse_metrics_rejects_incomplete_episode_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir = root / "job"
            result_root = root / "result"
            job_dir.mkdir()
            result_root.mkdir()
            (job_dir / "console.log").write_text(
                "Env name: val_seen, sr: 80.0, spl: 75.0\n",
                encoding="utf-8",
            )
            (result_root / "tta_diagnostics.json").write_text(json.dumps({
                "episode_count": 1,
                "adapter": {"episodes": 1, "updates": 1,
                            "relative_param_drift": 0.01},
            }), encoding="utf-8")
            job = {
                "job_dir": str(job_dir), "result_root": str(result_root),
                "episodes": 2, "setting": "duet-r2r", "config_method": "tent",
            }
            with self.assertRaisesRegex(MODULE.UserError, "incomplete episode stream"):
                MODULE.parse_metrics(job, self.spec)

    def test_screening_prerequisite_can_load_terminal_partial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            for ordinal in range(2):
                job_dir = stage / "jobs" / str(ordinal)
                job_dir.mkdir(parents=True)
                (job_dir / "job.json").write_text(json.dumps({
                    "ordinal": ordinal,
                }), encoding="utf-8")
            successful = result(
                "ok", "duet-r2r",
                MODULE.clean_parameters(
                    MODULE.anchor_for("tent", "duet-r2r", self.spec)
                ),
                75,
            )
            with mock.patch.object(
                MODULE, "write_summary",
                return_value=([successful], [{"run_tag": "oom", "exit_code": 1}]),
            ):
                loaded = MODULE.load_stage_results(
                    stage, self.spec, allow_partial=True
                )
                self.assertEqual(loaded, [successful])
                with self.assertRaisesRegex(MODULE.UserError, "incomplete"):
                    MODULE.load_stage_results(stage, self.spec)

    def test_workflow_has_controls_final_and_orders(self):
        self.assertEqual(MODULE.workflow_stages("tent"), (
            "smoke", "controls", "stage1", "final_controls", "final",
        ))
        self.assertEqual(MODULE.workflow_stages("fstta", include_orders=True), (
            "smoke", "controls", "stage1", "stage2", "stage3",
            "final_controls", "final", "orders",
        ))
        self.assertEqual(MODULE.previous_search_stage("tent", "final"), "stage1")
        self.assertEqual(MODULE.previous_search_stage("fstta", "final"), "stage3")

    def test_expected_campaign_job_counts(self):
        expected = {
            "tent": 344,
            "fstta": 368,
            "eam": 488,
            "feedtta": 968,
            "atena": 363,
        }
        stage1 = {"tent": 35, "fstta": 20, "eam": 35,
                  "feedtta": 35, "atena": 5}
        intermediate = {
            "tent": 0,
            "fstta": 8 + 10,
            "eam": 10 + 8,
            "feedtta": 78,
        }
        # ATENA cannot be represented by one per-setting intermediate count.
        totals = {
            method: 8 + 8 + stage1[method] * 8 + 8 + 5 * 8
            for method in expected
        }
        totals["fstta"] += intermediate["fstta"] * 8
        totals["eam"] += intermediate["eam"] * 8
        totals["feedtta"] += intermediate["feedtta"] * 8
        totals["atena"] += 23 * 3 + 22 * 5 + 10 * 8
        self.assertEqual(totals, expected)
        self.assertEqual(sum(totals.values()), 2531)

    def test_status_reports_final_controls_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "tent" / "batch" / "stages" / "final_controls"
            stage.mkdir(parents=True)
            (stage / "progress.json").write_text(json.dumps({
                "planned": 8,
                "pending": 0,
                "running": 0,
                "succeeded": 8,
                "failed": 0,
                "updated_at": "now",
                "running_jobs": [],
            }), encoding="utf-8")
            output = io.StringIO()
            with mock.patch.object(MODULE, "LOG_ROOT", root), \
                    redirect_stdout(output):
                MODULE.campaign_status("tent", "batch")
            status = json.loads(output.getvalue())
            self.assertEqual(
                status["methods"]["tent"]["final_controls"]["succeeded"], 8
            )

    def test_data_dependent_plan_rejects_prerequisite_commit_and_spec(self):
        settings = ["duet-r2r"]
        commit = "a" * 40
        spec_sha256 = MODULE.sha256(MODULE.SPEC_PATH)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(MODULE, "git", return_value=commit):
            root = Path(directory)
            write_stage_manifest(
                root, "controls", "fstta", settings,
                commit=commit, spec_sha256=spec_sha256,
            )
            stage1 = write_stage_manifest(
                root, "stage1", "fstta", settings,
                commit="b" * 40, spec_sha256=spec_sha256,
            )
            request = args(
                method="fstta", settings=settings, stage="stage2",
                resume=False,
            )
            with self.assertRaisesRegex(MODULE.UserError, "git_commit"):
                MODULE.ensure_stage_plan(
                    request, self.spec, root, "stage2"
                )
            self.assertFalse((root / "stages" / "stage2").exists())

            manifest_path = stage1 / "stage_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["git_commit"] = commit
            manifest["spec_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.UserError, "spec_sha256"):
                MODULE.ensure_stage_plan(
                    request, self.spec, root, "stage2"
                )
            self.assertFalse((root / "stages" / "stage2").exists())

            manifest["spec_sha256"] = spec_sha256
            manifest["method"] = "tent"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.UserError, "method"):
                MODULE.ensure_stage_plan(
                    request, self.spec, root, "stage2"
                )

            manifest["method"] = "fstta"
            manifest["settings"] = ["etpnav-r2r-ce"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.UserError, "settings"):
                MODULE.ensure_stage_plan(
                    request, self.spec, root, "stage2"
                )

    def test_frozen_selection_rejects_identity_and_setting_conflicts(self):
        settings = ["duet-r2r", "etpnav-r2r-ce"]
        commit = "a" * 40
        spec_sha256 = MODULE.sha256(MODULE.SPEC_PATH)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(MODULE, "git", return_value=commit):
            root = Path(directory)
            write_frozen_artifacts(
                root, "tent", settings, commit="b" * 40,
                spec_sha256=spec_sha256,
            )
            with self.assertRaisesRegex(MODULE.UserError, "git_commit"):
                MODULE._load_frozen_selection(
                    root, "tent", settings, self.spec
                )

            write_frozen_artifacts(
                root, "tent", settings, commit=commit,
                spec_sha256="0" * 64,
            )
            with self.assertRaisesRegex(MODULE.UserError, "spec_sha256"):
                MODULE._load_frozen_selection(
                    root, "tent", settings, self.spec
                )

            write_frozen_artifacts(
                root, "tent", settings, commit=commit,
                spec_sha256=spec_sha256,
            )
            with self.assertRaisesRegex(MODULE.UserError, "setting set mismatch"):
                MODULE._load_frozen_selection(
                    root, "tent", settings[:1], self.spec
                )

    def test_subset_resume_cannot_overwrite_full_search_artifacts(self):
        full_settings = ["duet-r2r", "etpnav-r2r-ce"]
        commit = "a" * 40
        spec_sha256 = MODULE.sha256(MODULE.SPEC_PATH)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(MODULE, "git", return_value=commit):
            root = Path(directory)
            write_stage_manifest(
                root, "final", "tent", full_settings,
                commit=commit, spec_sha256=spec_sha256,
            )
            protected = {}
            for name in (
                "FINAL_SELECTION.json", "FROZEN_HPARAMETERS.json",
                "ORDER_ROBUSTNESS.json",
            ):
                path = root / name
                path.write_bytes((name + "\n").encode("utf-8"))
                protected[path] = path.read_bytes()
            request = args(
                method="tent", settings=full_settings[:1], stage="final",
                resume=True, episodes=-1,
            )
            with self.assertRaisesRegex(MODULE.UserError, "settings"):
                MODULE.ensure_stage_plan(
                    request, self.spec, root, "final"
                )
            self.assertEqual(
                {path: path.read_bytes() for path in protected}, protected
            )

    def test_strict_full_stages_reject_256_episode_override(self):
        for stage in ("final_controls", "final", "orders"):
            with self.subTest(stage=stage), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                MODULE.parse_args([
                    "tent", "--batch-id", "batch", "--stage", stage,
                    "--episodes", "256",
                ])
            candidates = {
                "duet-r2r": [MODULE._candidate({"lr": 1e-6})]
            }
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory, \
                    self.assertRaisesRegex(MODULE.UserError, "requires protocol"):
                MODULE.build_jobs(
                    args(
                        method="tent", settings=["duet-r2r"], stage=stage,
                        episodes=256,
                    ),
                    self.spec,
                    Path(directory),
                    stage=stage,
                    candidates_by_setting=candidates,
                )

    def test_prerequisite_rejects_batch_episode_and_job_plan_mismatch(self):
        settings = ["duet-r2r"]
        commit = "a" * 40
        spec_sha256 = MODULE.sha256(MODULE.SPEC_PATH)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(MODULE, "git", return_value=commit):
            root = Path(directory)
            write_stage_manifest(
                root, "controls", "fstta", settings, commit=commit,
                spec_sha256=spec_sha256, batch_id="wrong-batch",
            )
            write_stage_manifest(
                root, "stage1", "fstta", settings, commit=commit,
                spec_sha256=spec_sha256, batch_id="unit-test",
            )
            request = args(
                method="fstta", settings=settings, stage="stage2",
                resume=False, batch_id="unit-test",
            )
            with self.assertRaisesRegex(MODULE.UserError, "batch_id"):
                MODULE.ensure_stage_plan(request, self.spec, root, "stage2")

            write_stage_manifest(
                root, "controls", "fstta", settings, commit=commit,
                spec_sha256=spec_sha256, batch_id="unit-test",
            )
            write_stage_manifest(
                root, "stage1", "fstta", settings, commit=commit,
                spec_sha256=spec_sha256, batch_id="unit-test", episodes=128,
            )
            with self.assertRaisesRegex(MODULE.UserError, "episodes"):
                MODULE.ensure_stage_plan(request, self.spec, root, "stage2")

            stage1 = write_stage_manifest(
                root, "stage1", "fstta", settings, commit=commit,
                spec_sha256=spec_sha256, batch_id="unit-test",
            )
            job_path = stage1 / "jobs" / "0" / "job.json"
            job = json.loads(job_path.read_text(encoding="utf-8"))
            job["episodes"] = 128
            job_path.write_text(json.dumps(job), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.UserError, "job .*episodes"):
                MODULE.ensure_stage_plan(request, self.spec, root, "stage2")


if __name__ == "__main__":
    unittest.main()
