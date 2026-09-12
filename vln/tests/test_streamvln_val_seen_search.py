"""CPU contracts for the four dedicated-GPU, three-candidate seen campaign."""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_streamvln_val_seen_search as runner

REAL_SPEC = runner.SPEC


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    spec = runner.read_json(REAL_SPEC)
    for name, value in {
        "REPO_ROOT": tmp_path,
        "RESULTS": tmp_path / "vln/results/tuning" / spec["experiment_id"],
        "SPEC": tmp_path / "vln/experiments/search.json",
        "ORDER_ROOT": tmp_path / "vln/manifests/order",
        "LOCK_ROOT": tmp_path / "vln/results/tuning/.streamvln_gpu_locks",
        "EXPECTED": {"val_seen": 4, "val_unseen": 6},
        "STOP_REQUESTED": threading.Event(),
        "ACTIVE_PROCESSES": set(),
    }.items():
        monkeypatch.setattr(runner, name, value)
    spec["expected_episodes"] = runner.EXPECTED
    audit = tmp_path / spec["historical_audit"]["path"]
    runner.atomic_json(audit, {"source_val_seen": {
        "metrics": {"SR": 50.0, "SPL": 40.0}, "formal_result_eligible": False}})
    spec["historical_audit"]["sha256"] = runner.sha256(audit)
    runner.atomic_json(runner.SPEC, spec)
    for split, count in runner.EXPECTED.items():
        runner.atomic_json(runner.ORDER_ROOT / (split + ".json"), {
            "episodes": [{"scene_id": "scene", "episode_id": str(i)} for i in range(count)]})
    return spec


def outputs(job, values=None):
    episodes = runner.read_json(runner.ORDER_ROOT / "val_seen.json")["episodes"][:job.episodes]
    values = values or [(1, .5)] * len(episodes)
    rows = [{**e, "success": sr, "spl": spl} for e, (sr, spl) in zip(episodes, values)]
    rows.append({"length": len(episodes), "sucs_all": sum(v[0] for v in values) / len(episodes),
                 "spls_all": sum(v[1] for v in values) / len(episodes)})
    job.output.mkdir(parents=True, exist_ok=True)
    (job.output / "result.json").write_text("".join(json.dumps(r) + "\n" for r in rows))
    runner.atomic_json(job.output / "tta_diagnostics.json", {
        "schema": "navtta.streamvln_tta_diagnostics.v2", "episodes": len(episodes),
        "method": job.method, "readout_protocol": "native_residual", "updates": len(episodes)})


def manifest(job, *_):
    value = {
        "run_id": job.manifest.parent.name, "task": "vln", "model": "streamvln",
        "benchmark": "r2r_vlnce_v1_3_streamvln", "method": job.method, "run_tag": job.tag,
        "source_setting": runner.SETTING + ":val_seen:v1.3", "seed": 0,
        "git_commit": "a" * 40, "config": str(job.config), "status": "running",
    }
    value["immutable_identity_sha256"] = runner.immutable_identity_sha256(value)
    runner.atomic_json(job.manifest, value)


def seal(job, status=0):
    value = runner.read_json(job.manifest)
    value.update(status="completed" if status == 0 else "failed", exit_code=status)
    value["result_artifacts"] = [
        {"name": name, "path": str(path), "sha256": runner.sha256(path)}
        for name, path in [("job_config", job.config), ("result", job.output / "result.json"),
                           ("tta_diagnostics", job.output / "tta_diagnostics.json")]]
    runner.atomic_json(job.manifest, value)
    if status == 0:
        runner.atomic_json(job.output / "completion.json", {
            "config_sha256": runner.sha256(job.config), "manifest_sha256": runner.sha256(job.manifest),
            "metrics": runner.validate_outputs(job)})


def complete(job, values=None):
    outputs(job, values)
    manifest(job)
    seal(job)
    assert runner.completed(job)


def test_exact_budget_fixed_mapping_and_supervision(campaign):
    jobs = runner.search_jobs(runner.load_spec())
    assert len(set(jobs)) == 12
    queues = runner.gpu_queues(jobs, [7, 2, 5, 1])
    for method, (gpu, queue) in zip(runner.METHODS, queues.items()):
        assert len(queue) == 3
        for index, job in enumerate(queue, 1):
            assert job.method == method and job.split == "val_seen" and not job.is_smoke
            config = runner.read_json(job.config)
            assert config["candidate_index"] == index
            assert config["stage"] == "full_val_seen_search"
            assert config["supervision"] == ("binary_episode_success" if method in ("feedtta", "atena") else "none")
            assert runner.command(job, gpu)[4] == str(gpu)
    assert {j.method for j in jobs} == {"fstta", "eam", "feedtta", "atena"}


@pytest.mark.parametrize("mutation", ["extra", "two", "duplicate", "sample", "episodic", "readout", "nan", "split", "audit"])
def test_invalid_spec_rejected(campaign, mutation):
    spec = deepcopy(campaign)
    if mutation == "extra":
        spec["methods"]["tent"] = spec["methods"]["eam"]
    elif mutation == "two":
        spec["methods"]["fstta"].pop()
    elif mutation == "duplicate":
        spec["methods"]["eam"][1] = deepcopy(spec["methods"]["eam"][0])
    elif mutation == "split":
        spec["split"] = "val_unseen"
    elif mutation == "audit":
        spec["historical_audit"]["sha256"] = "0" * 64
    else:
        key, value = {"sample": ("action_selection", "sample"), "episodic": ("episodic", True),
                      "readout": ("streamvln_readout_protocol", "legacy_v4"), "nan": ("lr", float("nan"))}[mutation]
        spec["methods"]["eam"][0]["parameters"][key] = value
    # Bypass the writer's own NaN rejection to exercise the reader contract.
    runner.SPEC.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        runner.load_spec()


@pytest.mark.parametrize("gpus", ["0,1,2", "0,1,2,2", "0,1,2,-1", "", "0,1,2,x", "0,1,2,3,4"])
def test_cli_rejects_wrong_gpu_budget_before_creating_jobs(monkeypatch, gpus):
    monkeypatch.setattr(sys, "argv", ["runner", "--dry-run", "--gpus", gpus])
    monkeypatch.setattr(runner, "load_spec", lambda: pytest.fail("loaded spec before GPU validation"))
    assert runner.main() == 1


def test_all_candidates_reach_actual_streamvln_argument_parser(campaign):
    pytest.importorskip("torch")
    sys.path.insert(0, str(SCRIPTS.parents[1] / "core"))
    sys.path.insert(0, str(SCRIPTS.parent))
    sys.path.insert(0, str(SCRIPTS.parent / "baselines/streamvln/streamvln"))
    from navtta_vln.discrete_tta import add_discrete_tta_args
    from streamvln_tta import add_streamvln_tta_args
    parser = argparse.ArgumentParser()
    add_discrete_tta_args(parser, feedtta_scope_profiles=("configured_prefixes",))
    add_streamvln_tta_args(parser)
    for job in runner.search_jobs(campaign):
        _, translated = runner.translate(runner.SETTING, job.config, str(job.output / "tta_diagnostics.json"))
        args = parser.parse_args(translated)
        assert args.tta_method == job.method
        assert args.tta_streamvln_readout_protocol == "native_residual"
        assert args.tta_action_selection == "argmax"


def test_four_workers_keep_method_affinity_and_restart_partial_stream(campaign, monkeypatch):
    jobs = runner.search_jobs(campaign)
    first = jobs[0]
    first.output.mkdir(parents=True)
    (first.output / "result.json").write_text("partial previous attempt\n")
    manifest(first)
    barrier = threading.Barrier(4)
    launches = []
    by_tag = {job.tag: job for job in jobs}
    monkeypatch.setattr(runner, "assert_gpu_idle", lambda gpu: None)
    monkeypatch.setattr(runner, "create_manifest", manifest)
    monkeypatch.setattr(runner, "finalize", seal)

    def evaluate(cmd, **kwargs):
        job = by_tag[cmd[cmd.index("--run-tag") + 1]]
        gpu = int(cmd[4])
        assert not job.output.exists() or not any(job.output.iterdir())
        assert kwargs["start_new_session"] and len(kwargs["pass_fds"]) == 2
        assert job.method == runner.METHODS[gpu]
        launches.append((gpu, job.tag))

        class Process:
            def wait(self):
                barrier.wait(timeout=10)
                outputs(job)
                return 0
        return Process()

    monkeypatch.setattr(runner.subprocess, "Popen", evaluate)
    assert runner.run_jobs(jobs, [0, 1, 2, 3], retry_incomplete=True) == []
    assert len(launches) == 12 and all(runner.completed(j) for j in jobs)
    for gpu in range(4):
        assert [tag for g, tag in launches if g == gpu] == [j.tag for j in jobs[gpu * 3:gpu * 3 + 3]]
    archived = next((runner.RESULTS / "incomplete").glob("*/output/result.json"))
    assert archived.read_text() == "partial previous attempt\n"
    # Authentication is sufficient to skip; no GPU inspection or relaunch.
    monkeypatch.setattr(runner, "assert_gpu_idle", lambda gpu: pytest.fail("inspected completed GPU"))
    assert runner.run_jobs(jobs, [0, 1, 2, 3]) == []
    assert len(launches) == 12


@pytest.mark.parametrize("mutation", ["partial", "order", "aggregate", "artifact", "commit", "config"])
def test_corrupt_or_partial_runs_cannot_be_selected(campaign, mutation):
    jobs = runner.search_jobs(campaign)
    for job in jobs:
        complete(job)
    job = jobs[-1]
    if mutation in ("partial", "order", "aggregate"):
        path = job.output / "result.json"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == "partial":
            rows = rows[:-2]
        elif mutation == "order":
            rows[0], rows[1] = rows[1], rows[0]
        else:
            rows[-1]["spls_all"] = .9
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif mutation == "artifact":
        with (job.output / "tta_diagnostics.json").open("a") as stream:
            stream.write("\n")
    elif mutation == "commit":
        runner.atomic_json(runner.RESULTS / "campaign.json", {"git_commit": "b" * 40})
    else:
        config = runner.read_json(job.config)
        config["parameters"]["lr_query"] *= 10
        runner.atomic_json(job.config, config)
    assert not runner.completed(job)
    with pytest.raises(ValueError):
        runner.select_winners(jobs, campaign)
    assert not (runner.RESULTS / "selection.json").exists()


def test_ranking_then_stable_tie_and_legacy_report(campaign):
    jobs = runner.search_jobs(campaign)
    for index, job in enumerate(jobs):
        # 01 has higher SR but lower SPL. 02/03 tie, so 02 wins.
        values = [(1, .25)] * 4 if index % 3 == 0 else [(1, .75)] * 2 + [(0, 0)] * 2
        complete(job, values)
    selection = runner.select_winners(list(reversed(jobs)), campaign)
    assert selection["selection_split"] == "val_seen"
    assert all(r["run_tag"].endswith("-02-search-v1") for r in selection["records"].values())
    assert runner.select_winners(jobs, campaign) == selection
    runner.report(jobs, campaign, selection)
    report = runner.read_json(runner.RESULTS / "report.json")
    assert report["result_role"] == "hyperparameter_development"
    assert not report["legacy_source_reference"]["formal_result_eligible"]
    assert len(report["records"]) == 12
    with pytest.raises(ValueError):
        runner.select_winners(jobs[:-1] + [jobs[0]], campaign)
    with pytest.raises(ValueError):
        runner.select_winners([replace(j, split="val_unseen") for j in jobs], campaign)


def test_smoke_and_full_jobs_pass_actual_launcher_output_guards(campaign):
    wrapper = runner.RUNNER.read_text()
    wrapper = wrapper[:wrapper.index('MATTERSIM_ROOT="${DATA_ROOT}/simulators/Matterport3DSimulator"')]
    env_root = runner.REPO_ROOT / "environment"
    wrapper = wrapper.replace("REPO_ROOT=/data1/wxy/code/NavTTA", "REPO_ROOT=" + str(runner.REPO_ROOT))
    wrapper = wrapper.replace("VLN_ROOT=/data1/wxy/exp_data/NavTTA/vln", "VLN_ROOT=" + str(env_root))
    if sys.platform == "darwin":
        # Server uses GNU realpath -m; macOS has the BSD utility. Substitute
        # only that path-normalization primitive, preserving launcher guards.
        wrapper = ("realpath() { " + sys.executable +
                   " -c 'import os,sys; print(os.path.realpath(sys.argv[-1]))' \"$@\"; }\n" + wrapper)
    path = runner.REPO_ROOT / "wrapper.sh"
    path.write_text(wrapper + '\nprintf "resolved=%s\\n" "${RESULT_ROOT}"\n')
    interpreter = env_root / "envs/streamvln/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    for rel in ["scripts/tta_config_cli.py", "navtta_vln/reverie_llm_feedback.py"]:
        dest = runner.REPO_ROOT / "vln" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SCRIPTS.parent / rel, dest)
    probes = runner.smoke_jobs(campaign)
    assert len(probes) == 4 and all(j.episodes == 2 for j in probes)
    with pytest.raises(ValueError):
        runner.select_winners(probes, campaign)
    for job in probes + runner.search_jobs(campaign):
        cmd = runner.command(job, 0)
        cmd[1] = str(path)
        result = subprocess.run(cmd + ["--dry-run"], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "resolved=" + str(job.output)
