"""Standard-library scheduler tests; no simulator, dataset, Torch or GPU needed."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "avn_reval_search_under_test", ROOT / "avn/scripts/run_avn_reval_search.py")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def sample_spec():
    cells = []
    for model in runner.MODELS:
        for method in runner.SEARCH_METHODS:
            for setting in runner.SETTINGS:
                candidates = []
                for index in range(12):
                    point = ({"lr": "1e-8", "gamma": 0.99, "p": 0.05, "alpha": -0.2}
                             if method == "feedtta" else
                             {"lr_query": "1e-8", "lr_self": "1e-9", "query_threshold": 0.5,
                              "mix_lambda": 0.5, "self_loss_weight": 0.1})
                    candidates.append(dict(point, id="candidate-{:02d}".format(index)))
                cells.append({"model": model, "method": method, "source_setting": setting,
                              "candidates": candidates})
    revaluation = []
    for model in runner.MODELS:
        for method in (("source",) if model == "smt_audio" else ("source", "tent", "fstta", "eam")):
            for setting in runner.SETTINGS:
                point = {"source": {}, "tent": {"lr": "1e-6", "scope": "last_k_ln", "update_interval": 1},
                         "fstta": {"fast_lr": "1e-6", "fast_window": 3, "slow_lr": "1e-4", "slow_window": 4},
                         "eam": {"lr": "1e-5", "update_interval": 1}}[method]
                revaluation.append({"model": model, "method": method, "source_setting": setting, "point": point})
    return {"schema": runner.SCHEMA, "seed": 0, "audio_seed": 0, "episodes": 2000,
            "scheduler": {"rounds": 4, "candidates_per_cell_per_round": 3,
                          "lanes": [{"model": model, "method": method, "gpu": index}
                                    for index, (model, method) in enumerate(
                                        (m, t) for m in runner.MODELS for t in runner.SEARCH_METHODS)]},
            "protocol": {"overrides_by_model": {"enmus": {
                "EVAL.PROTOCOL_PROFILE": "enmus_clavn_aligned_v1",
                "TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_MODE": "episode_seeded_v1",
                "TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_SEED": 0}}},
            "search": {"cells": cells}, "revaluation": revaluation}


class FakeProcess:
    """Finite deterministic workload: exit failures still let peer jobs finish."""
    serial = 10000000

    def __init__(self, exit_code=0):
        FakeProcess.serial += 1
        self.pid = FakeProcess.serial
        self.remaining = 2
        self.exit_code, self.returncode = exit_code, None

    def poll(self):
        self.remaining -= 1
        if self.remaining <= 0:
            self.returncode = self.exit_code
        return self.returncode

    def wait(self):
        self.returncode = self.exit_code
        return self.returncode


class MockCampaign(runner.Campaign):
    """Replace only simulator launch/certification, preserving scheduler state."""
    failures = set()
    launched = []
    max_parallel = {}

    def check_runtime(self):
        pass

    def launch(self, job):
        source = self.source_for(job)
        number, directory = runner.next_attempt_dir(self.log_directory / job.key)
        (directory / "console.log").write_text("mock workload\n")
        attempt = {"number": number, "directory": str(directory), "status": "running",
                   "run_tag": runner.attempt_tag(self.batch_id, job, number),
                   "host": runner.socket.gethostname(), "pid": None, "birth": None}
        self.state["jobs"].setdefault(job.key, {"attempts": []})["attempts"].append(attempt)
        process = FakeProcess(1 if job.key in self.failures else 0)
        self.workers.append(runner.Worker(process, io.StringIO(), job, attempt, source))
        self.launched.append((job, number))
        self.max_parallel[job.gpu] = max(self.max_parallel.get(job.gpu, 0),
                                         sum(w.job.gpu == job.gpu for w in self.workers))
        self.save()
        return True

    def certify(self, job, attempt, source):
        directory = Path(attempt["directory"])
        artifacts = {}
        for key in ("manifest", "stats", "diagnostics"):
            path = directory / (key + ".json")
            runner.write_json(path, {"job": job.key, "artifact": key})
            artifacts[key] = str(path)
            artifacts[key + "_sha256"] = runner.sha256_file(path)
        result = {"job": runner.asdict(job), "batch_id": self.batch_id, "certified": True,
                  "identity_sha256": runner.content_digest(self.identity), "run_tag": attempt["run_tag"],
                  "metrics": {"spl": 0.3, "success": 0.5, "softspl": 0.4},
                  "diagnostic_summary": {"relative_param_drift": 0.001, "query_rate": 0.25},
                  "audio_evidence": {}, "matched_source": None if source is None else {
                      "manifest": source["manifest"], "manifest_sha256": source["manifest_sha256"]}, **artifacts}
        path = directory / "result.json"
        runner.write_json(path, result)
        attempt.update({"status": "completed", "result": str(path)})
        self.results[job.key] = result
        return result


def reload_fake(manifest, *args, **kwargs):
    return runner.read_json(Path(manifest).parent / "result.json")


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.spec = sample_spec()

    def test_exact_96_jobs_four_barriers_and_six_per_gpu(self):
        jobs = runner.build_jobs(self.spec, "search")
        self.assertEqual(len(jobs), 96)
        self.assertEqual(len({job.key for job in jobs}), 96)
        for round_index in range(1, 5):
            subset = [job for job in jobs if job.round == round_index]
            self.assertEqual(len(subset), 24)
            for gpu in map(str, range(4)):
                lane = [job for job in subset if job.gpu == gpu]
                self.assertEqual(len(lane), 6)
                self.assertEqual(sum(job.source_setting == "single_source" for job in lane), 3)
                self.assertEqual(sum(job.source_setting == "multi_source" for job in lane), 3)
                self.assertEqual(len({(job.model, job.method) for job in lane}), 1)

    def test_smoke_has_eight_cells_and_four_source_controls(self):
        jobs = runner.build_jobs(self.spec, "smoke")
        self.assertEqual(len(jobs), 12)
        self.assertEqual(sum(job.method == "source" for job in jobs), 4)
        self.assertTrue(all(job.episodes == 20 for job in jobs))
        self.assertEqual(len(runner.build_jobs(self.spec, "reval")), 10)

    def test_frozen_audio_override_only_applies_to_enmus(self):
        jobs = runner.build_jobs(self.spec, "search")
        for job in jobs:
            overrides = runner.expected_overrides(self.spec, job)
            self.assertEqual(overrides["EVAL.ACTION_SELECTION"], "sample")
            self.assertEqual(overrides["NUM_PROCESSES"], "1")
            self.assertEqual("EVAL.PROTOCOL_PROFILE" in overrides, job.model == "enmus")
            command = runner.job_command(self.spec, job)
            self.assertEqual(Path(command[1]).name, "eval_" + job.model + ".sh")

    def test_invalid_grid_and_protocol_are_rejected_before_launch(self):
        mutations = [lambda s: s["search"]["cells"][0]["candidates"].pop(),
                     lambda s: s["search"]["cells"][0]["candidates"][0].update(lr=0),
                     lambda s: s["search"]["cells"][0]["candidates"][0].update(overrides={"EVAL.ACTION_SELECTION": "argmax"}),
                     lambda s: s["scheduler"]["lanes"][1].update(gpu=0)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                spec = copy.deepcopy(self.spec)
                mutate(spec)
                with self.assertRaises(runner.CampaignError):
                    runner.validate_spec(spec)

    def test_dry_run_never_preflights_or_writes_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            runner.write_json(path, self.spec)
            with mock.patch.object(runner, "preflight", side_effect=AssertionError("preflight called")), \
                    mock.patch.object(runner, "atomic_write", side_effect=AssertionError("write called")), \
                    mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                status = runner.main(["--spec", str(path), "--stage", "search", "--dry-run",
                                      "--smt-python", "/missing/smt/python", "--enmus-python", "/missing/enmus/python"])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.getvalue())["job_count"], 96)

    def test_separate_interpreter_parents_are_preserved(self):
        original = {"PATH": "/bin:/usr/bin"}
        smt = runner.interpreter_environment("/envs/smt/bin/python", original)
        enmus = runner.interpreter_environment("/envs/enmus/bin/python3", original)
        self.assertTrue(smt["PATH"].startswith("/envs/smt/bin:"))
        self.assertTrue(enmus["PATH"].startswith("/envs/enmus/bin:"))
        self.assertEqual(original["PATH"], "/bin:/usr/bin")


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.spec, self.identity = sample_spec(), {"runtime_sha256": "a" * 64}
        MockCampaign.failures, MockCampaign.launched, MockCampaign.max_parallel = set(), [], {}
        self.validator = mock.patch.object(runner, "validate_run", side_effect=reload_fake)
        self.validator.start()
        self.stdout = mock.patch("sys.stdout", new_callable=io.StringIO)
        self.stdout.start()

    def tearDown(self):
        self.stdout.stop()
        self.validator.stop()
        self.temporary.cleanup()

    def make_campaign(self, **kwargs):
        return MockCampaign(self.spec, self.identity, "mock-batch", root=self.root, **kwargs)

    def test_all_stage_obeys_full_barrier_order_and_caps(self):
        campaign = self.make_campaign()
        self.assertEqual(campaign.run("all", poll_seconds=0), 0)
        self.assertEqual(len(MockCampaign.launched), 118)
        starts = [entry["label"] for entry in campaign.state["events"] if entry["event"] == "barrier_started"]
        self.assertEqual(starts, ["smoke-source", "smoke", "reval-source", "reval",
                                  "search-round-1", "search-round-2", "search-round-3", "search-round-4"])
        self.assertEqual(MockCampaign.max_parallel, {str(i): 6 for i in range(4)})
        summary = runner.read_json(campaign.directory / "selected_configs.json")
        self.assertTrue(all(cell["complete"] for cell in summary["cells"]))

    def test_search_refuses_missing_same_batch_source(self):
        campaign = self.make_campaign()
        with self.assertRaisesRegex(runner.CampaignError, "four completed certified Source"):
            campaign.run("search", poll_seconds=0)
        self.assertEqual(MockCampaign.launched, [])

    def test_failure_finishes_peers_blocks_next_round_and_retry_is_new_attempt(self):
        failure = runner.build_jobs(self.spec, "search")[5]
        MockCampaign.failures = {failure.key}
        campaign = self.make_campaign()
        self.assertEqual(campaign.run("all", poll_seconds=0), 1)
        search_starts = [job for job, _ in MockCampaign.launched if job.stage == "search"]
        self.assertEqual(len(search_starts), 24)
        self.assertEqual({job.round for job in search_starts}, {1})
        self.assertEqual(sum(job.key in campaign.results for job in search_starts), 23)
        prior = Path(campaign.state["jobs"][failure.key]["attempts"][0]["directory"])
        old_log = (prior / "console.log").read_bytes()

        resumed = self.make_campaign(resume=True)
        before = len(MockCampaign.launched)
        self.assertEqual(resumed.run("search", poll_seconds=0), 1)
        self.assertEqual(len(MockCampaign.launched), before)

        MockCampaign.failures = set()
        retried = self.make_campaign(resume=True, retry_failed=True)
        self.assertEqual(retried.run("search", poll_seconds=0), 0)
        self.assertEqual(len(MockCampaign.launched) - before, 73)
        attempts = retried.state["jobs"][failure.key]["attempts"]
        self.assertEqual([attempt["status"] for attempt in attempts], ["failed", "completed"])
        self.assertNotEqual(attempts[0]["run_tag"], attempts[1]["run_tag"])
        self.assertEqual((prior / "console.log").read_bytes(), old_log)

    def test_resume_rejects_identity_changes_and_reuses_valid_work(self):
        campaign = self.make_campaign()
        self.assertEqual(campaign.run("smoke", poll_seconds=0), 0)
        for key in ("runtime_sha256", "spec_sha256", "checkpoints", "streams"):
            changed = dict(self.identity, **{key: "changed"})
            with self.subTest(key=key), self.assertRaisesRegex(runner.CampaignError, "resume identity changed"):
                MockCampaign(self.spec, changed, "mock-batch", resume=True, root=self.root)
        before = len(MockCampaign.launched)
        resumed = self.make_campaign(resume=True)
        self.assertEqual(resumed.run("smoke", poll_seconds=0), 0)
        self.assertEqual(len(MockCampaign.launched), before)
        self.assertTrue(self.validator.is_local)

    def test_resume_detects_changed_artifact(self):
        campaign = self.make_campaign()
        self.assertEqual(campaign.run("smoke", poll_seconds=0), 0)
        candidate = next(job for job in campaign.jobs if job.stage == "smoke" and job.method == "feedtta")
        Path(campaign.results[candidate.key]["stats"]).write_text("tampered")
        resumed = self.make_campaign(resume=True)
        self.assertEqual(resumed.run("smoke", poll_seconds=0), 1)
        self.assertEqual(resumed.state["jobs"][candidate.key]["attempts"][-1]["status"], "invalidated")

    def test_retry_all_recovers_invalid_source_before_dependent_cached_results(self):
        campaign = self.make_campaign()
        self.assertEqual(campaign.run("all", poll_seconds=0), 0)
        source = next(job for job in campaign.jobs if job.stage == "reval" and job.method == "source")
        Path(campaign.results[source.key]["stats"]).write_text("tampered")
        resumed = self.make_campaign(resume=True, retry_failed=True)
        before = len(MockCampaign.launched)
        self.assertEqual(resumed.run("all", poll_seconds=0), 0)
        restarted = [job for job, _ in MockCampaign.launched[before:]]
        self.assertEqual(restarted[0].key, source.key)
        self.assertEqual(len(restarted), 25)

    def test_bad_result_shape_is_a_job_failure_and_peers_complete(self):
        campaign = self.make_campaign()
        self.assertEqual(campaign.run("reval", poll_seconds=0), 0)
        original = campaign.certify
        failed = runner.build_jobs(self.spec, "search")[0]
        def certify(job, attempt, source):
            if job.key == failed.key:
                raise TypeError("bad diagnostic shape")
            return original(job, attempt, source)
        with mock.patch.object(campaign, "certify", side_effect=certify):
            self.assertEqual(campaign.run("search", poll_seconds=0), 1)
        starts = [job for job, _ in MockCampaign.launched if job.stage == "search"]
        self.assertEqual(len(starts), 24)
        self.assertEqual(sum(job.key in campaign.results for job in starts), 23)

    def test_batch_lock_is_exclusive_and_released(self):
        path = self.root / "lock"
        with runner.BatchLock(path):
            with self.assertRaises(runner.CampaignError):
                with runner.BatchLock(path):
                    pass
        with runner.BatchLock(path):
            self.assertEqual(runner.read_json(path)["pid"], runner.os.getpid())

    def test_interrupt_signals_only_current_worker_groups(self):
        campaign = self.make_campaign()
        job = next(job for job in campaign.jobs if job.method == "source")
        campaign.launch(job)
        worker = campaign.workers[0]
        worker.process.remaining = 1
        worker.process.poll = mock.Mock(side_effect=[None, 0, 0])
        with mock.patch.object(runner.os, "killpg") as killpg, \
                mock.patch.object(runner, "process_group_alive", return_value=False):
            campaign.terminate_workers()
        killpg.assert_called_once_with(worker.process.pid, signal.SIGTERM)
        self.assertEqual(worker.attempt["status"], "interrupted")
        self.assertEqual(campaign.workers, [])

    def test_cleanup_kills_living_group_when_shell_leader_has_exited(self):
        campaign = self.make_campaign()
        job = next(job for job in campaign.jobs if job.method == "source")
        campaign.launch(job)
        worker = campaign.workers[0]
        worker.process.returncode = 0
        worker.process.poll = mock.Mock(return_value=0)
        with mock.patch.object(runner.os, "killpg") as killpg, \
                mock.patch.object(runner, "process_group_alive", return_value=True), \
                mock.patch.object(runner.time, "monotonic", side_effect=[0, 11]):
            campaign.terminate_workers()
        self.assertEqual(killpg.call_args_list, [mock.call(worker.process.pid, signal.SIGTERM),
                                               mock.call(worker.process.pid, signal.SIGKILL)])


class EvidenceTest(unittest.TestCase):
    def test_source_requires_matching_real_state_hashes(self):
        job = runner.build_jobs(sample_spec(), "reval")[0]
        diagnostics = {"episodes": 2000, "action_steps": 4000, "task_action_selection": "sample",
                       "updates": 0, "slow_updates": 0, "adapted_parameter_names": [],
                       "adapted_parameter_count": 0, "relative_param_drift": 0,
                       "source_policy_frozen": True, "source_model_state_sha256": "a" * 64,
                       "final_model_state_sha256": "a" * 64}
        runner.validate_diagnostics(job, diagnostics, runner.expected_overrides(sample_spec(), job))
        for key, value in (("source_policy_frozen", False), ("final_model_state_sha256", "b" * 64)):
            with self.subTest(key=key), self.assertRaisesRegex(runner.CampaignError, "Source model state"):
                runner.validate_diagnostics(job, dict(diagnostics, **{key: value}), runner.expected_overrides(sample_spec(), job))

    def test_lower_source_score_is_still_ranked(self):
        def result(identifier, spl, success, softspl, drift):
            return {"job": {"candidate_id": identifier}, "metrics": {"spl": spl, "success": success, "softspl": softspl},
                    "diagnostic_summary": {"relative_param_drift": drift}}
        records = [result("d", 0.1, 0.2, 0.3, 0.0), result("b", 0.2, 0.2, 0.3, 0.0),
                   result("a", 0.2, 0.2, 0.3, 0.0), result("c", 0.2, 0.2, 0.3, 0.1)]
        self.assertEqual([item["job"]["candidate_id"] for item in runner.rank_results(records)], ["a", "b", "c", "d"])

    def test_enmus_counts_are_not_smt_constants(self):
        spec = sample_spec()
        job = next(j for j in runner.build_jobs(spec, "search") if j.model == "enmus" and j.method == "feedtta")
        diagnostics = {"episodes": 2000, "action_steps": 4000, "updates": 2000, "task_action_selection": "sample",
                       "adapted_parameter_names": ["net.smt_state_encoder.enmus_norm.weight", "action_distribution.linear.weight"],
                       "adapted_parameter_count": 3638884, "relative_param_drift": 0.0,
                       "mean_entropy": 1.0, "last_entropy": 1.0, "current_lr": 1e-8,
                       "param_scope": "module_prefixes", "trainable_prefixes": ["net.smt_state_encoder", "action_distribution"],
                       "feedback_type": "binary_episode_success", "feedback_episodes": 2000,
                       "policy_gradient_action": "task_runner_executed_action", "sgr_mode": "paper_main",
                       "action_selection_protocol": "sample_from_policy", "gamma": 0.99,
                       "reversal_probability": 0.05, "reversal_scale": -0.2}
        runner.validate_diagnostics(job, diagnostics, runner.expected_overrides(spec, job))
        diagnostics["adapted_parameter_names"].append("critic.weight")
        with self.assertRaisesRegex(runner.CampaignError, "critic"):
            runner.validate_diagnostics(job, diagnostics, runner.expected_overrides(spec, job))

    def test_source_must_match_batch_and_complete_stream(self):
        job = runner.build_jobs(sample_spec(), "search")[0]
        identity = {"runtime": "same"}
        result = {"certified": True, "batch_id": "current", "identity_sha256": runner.content_digest(identity),
                  "job": {"stage": "reval", "model": job.model, "method": "source", "source_setting": job.source_setting,
                          "episodes": 2000}}
        runner.require_source(result, job, identity, "current")
        for key, value in (("batch_id", "historical"), ("identity_sha256", "other")):
            with self.subTest(key=key), self.assertRaises(runner.CampaignError):
                runner.require_source(dict(result, **{key: value}), job, identity, "current")
        result["job"]["episodes"] = 20
        with self.assertRaises(runner.CampaignError):
            runner.require_source(result, job, identity, "current")

    def test_audio_must_match_actual_source_schedule_for_every_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tb = root / "raw/model/tb"
            tb.mkdir(parents=True)
            job = runner.Job("smoke", "enmus", "source", "single_source", "control", {}, "2", 0, 1)
            sys.path.insert(0, str(ROOT / "avn"))
            from navtta_avn.enmus_eval_protocol import EXPECTED, build_eval_protocol
            from navtta_avn.audio_schedule import schedule_seed
            resolved = {"SEED": 0, "TEST_EPISODE_COUNT": 1}
            for key, value in dict(EXPECTED, **{
                    "EVAL.PROTOCOL_PROFILE": "enmus_clavn_aligned_v1",
                    "TASK_CONFIG.DATASET.DATA_PATH": "single_source/val.json.gz",
                    "TASK_CONFIG.DATASET.TTA_EPISODE_SEED": 0, "TASK_CONFIG.SEED": 0,
                    "TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_SEED": 0,
                    "TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND": False}).items():
                target = resolved
                parts = key.split(".")
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = copy.deepcopy(value)
            protocol = build_eval_protocol(resolved)
            protocol["observed_action_count"] = 4
            protocol["checkpoint_load"] = {"strict": True, "checkpoint_sha256": "b" * 64,
                                            "missing_keys": [], "unexpected_keys": [], "state_dict_tensors": 10}
            digest = protocol["semantic_digest"]
            profile = "enmus_clavn_aligned_v1"
            role = {"sha256": "a" * 64, "length": 501, "active_steps": 100,
                    "rng_seed": schedule_seed(0, "single_source", "scene", "1", "target")}
            schedule = runner.content_digest({"target": role["sha256"]})
            audio = {"schema": "navtta.avn.audio_schedule.v1", "profile": profile, "source_setting": "single_source",
                     "mode": "episode_seeded_v1", "schedule_seed": 0, "semantic_digest": digest, "episode_count": 1,
                     "complete": True, "episode_order_sha256": runner.content_digest(["scene/1"]),
                     "episode_order": ["scene/1"], "episodes": {"scene/1": {"scene_id": "scene", "episode_id": "1",
                                                                         "roles": {"target": role}, "schedule_sha256": schedule}}}
            runner.write_json(tb / "eval_protocol_0.json", protocol)
            runner.write_json(tb / "audio_schedule_0.json", audio)
            source_path = root / "source_audio.json"
            runner.write_json(source_path, audio)
            source = {"audio_evidence": {"audio_path": str(source_path)}}
            overrides = sample_spec()["protocol"]["overrides_by_model"]["enmus"]
            stream = {"episode_order": ["scene/1"]}
            runner.validate_audio_evidence(root, job, stream, overrides, source)
            changed = copy.deepcopy(audio)
            changed["episodes"]["scene/1"]["schedule_sha256"] = "f" * 64
            runner.write_json(source_path, changed)
            with self.assertRaisesRegex(runner.CampaignError, "differs from matched Source"):
                runner.validate_audio_evidence(root, job, stream, overrides, source)


if __name__ == "__main__":
    unittest.main()
