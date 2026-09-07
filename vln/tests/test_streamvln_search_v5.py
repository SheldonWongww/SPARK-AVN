"""Failure-oriented tests for StreamVLN search provenance and scheduling.

The four-episode fixtures keep these tests independent of Habitat, checkpoints,
and GPUs. They exercise the same completion/selection contracts as a full run.
"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_streamvln_search_v5 as runner  # noqa: E402


class _Campaign:
    def __init__(self, root, monkeypatch):
        self.root = root
        self.monkeypatch = monkeypatch
        self.expected = {"val_unseen": 4, "val_seen": 2}
        for key, value in {
            "REPO_ROOT": root,
            "RESULTS": root / "vln/results/tuning/v5",
            "ORDER_ROOT": root / "vln/manifests/order",
            "SPEC": root / "vln/experiments/search.json",
            "LOCK_ROOT": root / "vln/results/tuning/locks",
            "EXPECTED": self.expected,
        }.items():
            monkeypatch.setattr(runner, key, value)
        for split, count in self.expected.items():
            self.write(runner.ORDER_ROOT / (split + ".json"), {
                "episodes": [
                    {"scene_id": "scene_" + str(i // 2), "episode_id": str(10 + i)}
                    for i in range(count)
                ],
            })
        parameters = {
            "fstta": {"lr_fast": 1e-6, "lr_slow": 3e-6, "m": 3, "n": 4,
                      "norm_scope": "last_ln", "optimizer": "AdamW"},
            "eam": {"lr": 1e-6, "confidence_scale": 0.4,
                    "batch_size": 2, "memory_size": 8, "update_interval": 1},
            "feedtta": {"lr": 1e-6, "alpha": -0.2, "p": 0.05,
                        "gamma": 0.7, "optimizer_eps": 1e-5},
            "atena": {"lr_query": 1e-6, "lr_self": 1e-7,
                      "query_threshold": 0.3, "mix_lambda": 0.25,
                      "update_scope": "replay_reachable_high_level_navigation"},
        }
        self.spec = {
            "schema": "navtta.streamvln_search.v5", "seed": 0,
            "expected_episodes": self.expected,
            "selection": {"primary_metric": "SPL", "secondary_metric": "SR"},
            "methods": {
                method: [
                    {"candidate_id": "candidate_" + str(i), "parameters": {
                        **deepcopy(parameters[method]),
                        ("lr_fast" if method == "fstta" else
                         "lr_query" if method == "atena" else "lr"): 1e-6 * i,
                        "streamvln_readout_protocol": "native_residual",
                    }}
                    for i in range(1, 5)
                ] for method in runner.METHODS
            },
            "frozen_tent": {
                "run_tag": "streamvln-vu-compact-tent-01-v4",
                "parameters": {
                    "lr": 3e-8, "optimizer": "Adam", "norm_scope": "last_ln",
                    "update_interval": 1, "streamvln_readout_protocol": "legacy_v4",
                },
            },
            "legacy_reuse": {},
        }
        for method in ("source", "tent"):
            path = root / "legacy" / method / "result.json"
            self.result(path, "val_unseen")
            self.spec["legacy_reuse"][method] = {
                "result_path": str(path.relative_to(root)), "sha256": runner.sha256(path),
            }
        self.write(runner.SPEC, self.spec)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def result(self, path, split, values=None):
        episodes = runner.read_json(runner.ORDER_ROOT / (split + ".json"))["episodes"]
        values = values or [(1.0, 0.5)] * len(episodes)
        rows = [{**episode, "success": success, "spl": spl}
                for episode, (success, spl) in zip(episodes, values)]
        rows.append({"length": len(rows),
                     "sucs_all": sum(r["success"] for r in rows) / len(rows),
                     "spls_all": sum(r["spl"] for r in rows) / len(rows)})
        self.result_rows(path, rows)
        return rows

    @staticmethod
    def result_rows(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def job(self, method="fstta", candidate="01", split="val_unseen"):
        parameters = {} if method == "source" else (
            self.spec["frozen_tent"]["parameters"] if method == "tent" else
            self.spec["methods"][method][0]["parameters"]
        )
        return runner.make_job(method, split, candidate, deepcopy(parameters))

    def outputs(self, job, values=None):
        self.result(job.output / "result.json", job.split, values)
        if job.method != "source":
            self.write(job.output / "tta_diagnostics.json", {
                "schema": "navtta.streamvln_tta_diagnostics.v2",
                "method": job.method,
                "readout_protocol": runner.read_json(job.config)["parameters"][
                    "streamvln_readout_protocol"],
                "episodes": self.expected[job.split],
            })

    def manifest(self, job, status="running"):
        document = {
            "run_id": job.manifest.parent.name, "task": "vln",
            "benchmark": "r2r_vlnce_v1_3_streamvln", "model": "streamvln",
            "method": job.method, "run_tag": job.tag,
            "source_setting": runner.SETTING + ":" + job.split + ":v1.3",
            "seed": 0, "git_commit": "a" * 40,
            "config": str(job.config),
            "status": status, "exit_code": 0 if status == "completed" else None,
        }
        document["immutable_identity_sha256"] = runner.immutable_identity_sha256(document)
        self.write(job.manifest, document)
        return document

    def seal(self, job):
        document = self.manifest(job, "completed")
        artifact_paths = {"job_config": job.config, "result": job.output / "result.json"}
        if job.method != "source":
            artifact_paths["tta_diagnostics"] = job.output / "tta_diagnostics.json"
        document["result_artifacts"] = [
            {"name": name, "path": str(path), "sha256": runner.sha256(path)}
            for name, path in artifact_paths.items()
        ]
        self.write(job.manifest, document)
        self.write(job.output / "completion.json", {
            "config_sha256": runner.sha256(job.config),
            "manifest_sha256": runner.sha256(job.manifest),
            "metrics": runner.validate_outputs(job),
        })

    def complete(self, job, values=None):
        self.outputs(job, values)
        self.seal(job)
        assert runner.completed(job)

    def simulate_evaluators(self, jobs, on_launch=None):
        by_tag = {job.tag: job for job in jobs}
        launches = []
        self.monkeypatch.setattr(runner, "assert_gpu_idle", lambda gpu: None)
        self.monkeypatch.setattr(runner, "create_manifest",
                                lambda job, gpu, assets: self.manifest(job))

        def evaluate(cmd, **kwargs):
            job = by_tag[cmd[cmd.index("--run-tag") + 1]]
            gpu = int(cmd[4])
            # run_source_eval.sh rejects any nonempty output directory before
            # evaluation; a launcher.log placed there used to break every job.
            assert not job.output.exists() or not any(job.output.iterdir())
            assert len(kwargs["pass_fds"]) == 2
            assert {os.fstat(fd).st_ino for fd in kwargs["pass_fds"]} == {
                (runner.LOCK_ROOT / ("gpu-{}.lock".format(gpu))).stat().st_ino,
                (runner.LOCK_ROOT / (job.tag + ".lock")).stat().st_ino,
            }
            launches.append((gpu, job.tag))
            if on_launch:
                on_launch(gpu, job)
            self.outputs(job)
            return 0

        def finish(job, status):
            assert status == 0
            self.seal(job)

        self.monkeypatch.setattr(runner.subprocess, "call", evaluate)
        self.monkeypatch.setattr(runner, "finalize", finish)
        return launches


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    return _Campaign(tmp_path, monkeypatch)


def test_search_only_schedules_four_by_four(campaign):
    jobs = runner.search_jobs(runner.load_spec())
    assert len(jobs) == len({job.tag for job in jobs}) == 16
    assert {job.method for job in jobs} == set(runner.METHODS)
    assert all(job.split == "val_unseen" for job in jobs)
    for method in runner.METHODS:
        assert sum(job.method == method for job in jobs) == 4
    assert not any(job.method in {"source", "tent", "idea"} for job in jobs)


@pytest.mark.parametrize("mutation", ["extra_method", "three_candidates", "repeated_id",
                                       "repeated_parameters", "wrong_ranking", "legacy_readout"])
def test_spec_rejects_silent_expansion_or_mixed_runtime(campaign, mutation):
    spec = deepcopy(campaign.spec)
    if mutation == "extra_method":
        spec["methods"]["tent"] = spec["methods"]["fstta"]
    elif mutation == "three_candidates":
        spec["methods"]["eam"].pop()
    elif mutation == "repeated_id":
        spec["methods"]["atena"][1]["candidate_id"] = "candidate_1"
    elif mutation == "repeated_parameters":
        spec["methods"]["eam"][1]["parameters"] = deepcopy(spec["methods"]["eam"][0]["parameters"])
    elif mutation == "wrong_ranking":
        spec["selection"] = {"primary_metric": "SR", "secondary_metric": "SPL"}
    else:
        spec["methods"]["feedtta"][0]["parameters"]["streamvln_readout_protocol"] = "legacy_v4"
    campaign.write(runner.SPEC, spec)
    with pytest.raises(ValueError):
        runner.load_spec()


@pytest.mark.parametrize("mutation", ["partial", "missing_aggregate", "duplicate", "wrong_order",
                                       "forged_aggregate", "nan", "out_of_range"])
def test_result_rejects_false_full_split_claims(campaign, mutation):
    path = campaign.root / "result.json"
    rows = campaign.result(path, "val_unseen")
    if mutation == "partial":
        rows.pop(2)
    elif mutation == "missing_aggregate":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif mutation == "wrong_order":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "forged_aggregate":
        rows[-1]["spls_all"] += 0.01
    elif mutation == "nan":
        rows[0]["spl"] = float("nan")
    else:
        rows[0]["success"] = 2.0
    campaign.result_rows(path, rows)
    with pytest.raises(ValueError):
        runner.parse_result(path, "val_unseen")


def test_result_accepts_rounded_float32_aggregate_but_counts_every_episode(campaign):
    path = campaign.root / "result.json"
    rows = campaign.result(path, "val_unseen", [(1, 0.1), (1, 0.2), (0, 0), (1, 0.3)])
    rows[-1]["spls_all"] += 2e-8
    campaign.result_rows(path, rows)
    assert runner.parse_result(path, "val_unseen")["episodes"] == 4


@pytest.mark.parametrize("mutation", ["no_completion", "config", "diagnostics", "result",
                                       "manifest_identity", "manifest_bytes", "artifact", "metrics"])
def test_completion_rejects_stale_or_modified_artifacts(campaign, mutation):
    job = campaign.job()
    campaign.complete(job)
    completion_path = job.output / "completion.json"
    completion = runner.read_json(completion_path)
    manifest = runner.read_json(job.manifest)
    if mutation == "no_completion":
        completion_path.unlink()
    elif mutation == "config":
        config = runner.read_json(job.config)
        config["parameters"]["lr_fast"] *= 10
        campaign.write(job.config, config)
    elif mutation == "diagnostics":
        diag = runner.read_json(job.output / "tta_diagnostics.json")
        diag["episodes"] -= 1
        campaign.write(job.output / "tta_diagnostics.json", diag)
    elif mutation == "result":
        with (job.output / "result.json").open("a") as stream:
            stream.write("{}\n")
    elif mutation == "manifest_identity":
        manifest["seed"] = 999
        campaign.write(job.manifest, manifest)
        completion["manifest_sha256"] = runner.sha256(job.manifest)
        campaign.write(completion_path, completion)
    elif mutation == "manifest_bytes":
        manifest["unexpected_field"] = "changed after completion"
        campaign.write(job.manifest, manifest)
    elif mutation == "artifact":
        manifest["result_artifacts"][0]["sha256"] = "0" * 64
        campaign.write(job.manifest, manifest)
        completion["manifest_sha256"] = runner.sha256(job.manifest)
        campaign.write(completion_path, completion)
    else:
        completion["metrics"]["SR"] = 1.0
        campaign.write(completion_path, completion)
    assert not runner.completed(job)


def test_self_consistent_manifest_from_another_run_is_not_current_job(campaign):
    job = campaign.job()
    campaign.complete(job)
    manifest = runner.read_json(job.manifest)
    manifest["run_tag"] = "different-campaign-run"
    manifest["run_id"] = "different-campaign-run-id"
    manifest["immutable_identity_sha256"] = runner.immutable_identity_sha256(manifest)
    campaign.write(job.manifest, manifest)
    completion = runner.read_json(job.output / "completion.json")
    completion["manifest_sha256"] = runner.sha256(job.manifest)
    campaign.write(job.output / "completion.json", completion)
    assert not runner.completed(job)


def test_completed_run_from_another_commit_cannot_enter_current_campaign(campaign):
    job = campaign.job()
    campaign.complete(job)
    campaign.write(runner.RESULTS / "campaign.json", {"git_commit": "b" * 40})
    assert not runner.completed(job)


def test_legacy_import_requires_archive_digest_even_if_metrics_match(campaign):
    bindings = runner.bind_legacy(campaign.spec)
    assert set(bindings) == {"source", "tent"}
    path = campaign.root / campaign.spec["legacy_reuse"]["source"]["result_path"]
    with path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="differs"):
        runner.bind_legacy(campaign.spec)


def _ranked_campaign(campaign):
    jobs = runner.search_jobs(campaign.spec)
    # 03 beats 02 by SPL, beats 01 by SR at tied SPL, and beats 04 by
    # deterministic candidate order. A highest-SR-only search would choose 02.
    candidates = [
        [(1, 0.75), (1, 0.75), (0, 0), (0, 0)],
        [(1, 0.25)] * 4,
        [(1, 0.5), (1, 0.5), (1, 0.5), (0, 0)],
        [(1, 0.5), (1, 0.5), (1, 0.5), (0, 0)],
    ]
    for index, job in enumerate(jobs):
        campaign.complete(job, candidates[index % 4])
    return jobs


def test_selection_requires_every_candidate_to_finish(campaign):
    jobs = _ranked_campaign(campaign)
    (jobs[-1].output / "completion.json").unlink()
    with pytest.raises(ValueError, match="16"):
        runner.select_winners(jobs)
    assert not (runner.RESULTS / "selection.json").exists()


def test_duplicate_completed_job_cannot_stand_in_for_missing_candidates(campaign):
    jobs = _ranked_campaign(campaign)
    duplicates = [jobs[index] for index in (0, 4, 8, 12) for _ in range(4)]
    with pytest.raises(ValueError):
        runner.select_winners(duplicates)


def test_selection_uses_spl_then_sr_then_stable_index_and_freezes_seen(campaign):
    jobs = _ranked_campaign(campaign)
    selection = runner.select_winners(list(reversed(jobs)))
    assert selection["selection_split"] == "val_unseen"
    assert all(record["run_tag"].endswith("-03-v5") for record in selection["records"].values())
    frozen_digest = runner.sha256(runner.RESULTS / "selection.json")
    seen = runner.seen_jobs(campaign.spec, selection)
    assert len(seen) == 6
    assert {job.method for job in seen} == {"source", "tent", *runner.METHODS}
    assert all(job.split == "val_seen" for job in seen)
    for job in seen:
        config = runner.read_json(job.config)
        if job.method in runner.METHODS:
            assert config["parameters"] == selection["records"][job.method]["parameters"]
            assert config["selection"]["selection_split"] == "val_unseen"
            assert config["selection"]["selection_sha256"] == frozen_digest
        elif job.method == "tent":
            assert config["parameters"] == campaign.spec["frozen_tent"]["parameters"]
            assert config["parameters"]["streamvln_readout_protocol"] == "legacy_v4"
        else:
            assert config["parameters"] == {}
        campaign.complete(job, [(0, 0), (0, 0)])
    assert runner.select_winners(jobs) == selection
    assert runner.sha256(runner.RESULTS / "selection.json") == frozen_digest
    with pytest.raises(ValueError):
        runner.select_winners([seen[2]] * 16)


def test_frozen_config_is_never_silently_replaced(campaign):
    job = campaign.job()
    old = job.config.read_bytes()
    config = runner.read_json(job.config)
    config["parameters"]["lr_fast"] *= 10
    with pytest.raises(ValueError, match="frozen"):
        runner.write_once(job.config, config)
    assert job.config.read_bytes() == old


def test_single_gpu_runs_serially_and_respects_launcher_empty_output_guard(campaign):
    jobs = runner.search_jobs(campaign.spec)[:4]
    launches = campaign.simulate_evaluators(jobs)
    assert runner.run_jobs(jobs, [2]) == []
    assert launches == [(2, job.tag) for job in jobs]
    assert all(runner.completed(job) for job in jobs)
    assert not any((job.output / "launcher.log").exists() for job in jobs)


def test_four_gpu_workers_never_overlap_two_evaluations_on_one_gpu(campaign):
    jobs = runner.search_jobs(campaign.spec)[:8]
    barrier = threading.Barrier(4)
    mutex = threading.Lock()
    active = set()
    observed = []

    def synchronize(gpu, job):
        with mutex:
            assert gpu not in active
            active.add(gpu)
            observed.append(len(active))
        barrier.wait(timeout=5)
        with mutex:
            active.remove(gpu)

    launches = campaign.simulate_evaluators(jobs, synchronize)
    assert runner.run_jobs(jobs, [0, 1, 2, 3]) == []
    assert len(launches) == 8
    assert max(observed) == 4
    assert all(sum(gpu == expected for gpu, _ in launches) == 2 for expected in range(4))


@contextmanager
def _other_process_holds_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen([
        sys.executable, "-u", "-c",
        "import fcntl,sys; f=open(sys.argv[1],'a+'); "
        "fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); "
        "print('locked',flush=True); sys.stdin.readline()", str(path),
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        yield
        child.communicate("release\n", timeout=5)
        assert child.returncode == 0
    finally:
        if child.poll() is None:
            child.terminate()
            child.communicate(timeout=5)


def test_cross_process_gpu_lease_blocks_second_campaign_and_releases(campaign):
    with _other_process_holds_lock(runner.LOCK_ROOT / "gpu-2.lock"):
        with pytest.raises(RuntimeError, match="reserved"):
            with runner.gpu_lease(2):
                pytest.fail("a second process acquired an occupied GPU lock")
    with runner.gpu_lease(2):
        pass


def test_cross_process_job_lease_prevents_restart_on_another_gpu(campaign, monkeypatch):
    job = campaign.job()
    job.output.mkdir(parents=True)
    (job.output / "result.json").write_bytes(b"partial evidence\n")
    campaign.manifest(job)
    old_manifest = job.manifest.read_bytes()
    monkeypatch.setattr(runner, "assert_gpu_idle", lambda gpu: pytest.fail("queried GPU before job lock"))
    monkeypatch.setattr(runner, "create_manifest", lambda *a: pytest.fail("overwrote occupied job manifest"))
    monkeypatch.setattr(runner.subprocess, "call", lambda *a, **kw: pytest.fail("relaunched occupied job"))
    with _other_process_holds_lock(runner.LOCK_ROOT / (job.tag + ".lock")):
        assert runner.run_queue(3, [job], None, retry_incomplete=True) == [job.tag]
        assert job.manifest.read_bytes() == old_manifest
        assert (job.output / "result.json").read_bytes() == b"partial evidence\n"
        assert not (runner.RESULTS / "incomplete").exists()
    with runner.job_lease(job):
        pass


@pytest.mark.parametrize("gpus", ["0,0", "-1", "", "0,no_gpu"])
def test_cli_rejects_bad_gpu_lists_before_loading_models(campaign, monkeypatch, gpus):
    monkeypatch.setattr(sys, "argv", ["runner", "--gpus", gpus, "--dry-run"])
    called = []
    monkeypatch.setattr(runner, "load_spec", lambda: called.append(True))
    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        assert runner.main() == 1
    assert called == []


def test_busy_gpu_is_not_reused(campaign, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "43210\n")
    with pytest.raises(RuntimeError, match="compute processes"):
        runner.assert_gpu_idle(0)


def test_partial_stream_requires_explicit_restart_and_keeps_original_evidence(campaign):
    job = campaign.job()
    job.output.mkdir(parents=True)
    old = b'{"episode_id": 10}\n'
    (job.output / "result.json").write_bytes(old)
    campaign.manifest(job)
    launches = campaign.simulate_evaluators([job])
    assert runner.run_jobs([job], [0]) == [job.tag]
    assert launches == []
    assert (job.output / "result.json").read_bytes() == old
    assert not (runner.RESULTS / "incomplete").exists()


def test_retry_archives_partial_state_and_starts_from_episode_one(campaign):
    job = campaign.job()
    job.output.mkdir(parents=True)
    old = b'{"episode_id": 10, "partial": true}\n'
    (job.output / "result.json").write_bytes(old)
    old_manifest = campaign.manifest(job)
    campaign.simulate_evaluators([job])
    assert runner.run_jobs([job], [0], retry_incomplete=True) == []
    archives = list((runner.RESULTS / "incomplete").iterdir())
    assert len(archives) == 1
    archived = archives[0]
    assert (archived / "output/result.json").read_bytes() == old
    assert runner.read_json(archived / "run_manifest/manifest.json") == old_manifest
    assert (archived / "job_config.json").read_bytes() == job.config.read_bytes()
    assert runner.completed(job)
    rows = [json.loads(line) for line in (job.output / "result.json").read_text().splitlines()]
    assert rows[0]["episode_id"] == "10"
    assert len(rows) == 5


def test_authenticated_complete_stream_is_skipped_without_gpu_or_process(campaign, monkeypatch):
    job = campaign.job()
    campaign.complete(job)
    monkeypatch.setattr(runner, "assert_gpu_idle", lambda gpu: pytest.fail("skip checked GPU"))
    monkeypatch.setattr(runner.subprocess, "call", lambda *a, **k: pytest.fail("skip launched process"))
    assert runner.run_jobs([job], [0]) == []
