#!/usr/bin/env python3
"""Search three val_seen configurations per method, on four dedicated GPUs.

FSTTA, EAM, FeedTTA and ATENA each own one GPU and run serially from fresh
checkpoints. This standalone campaign preserves the v6 launch, restart and
provenance contracts without changing the previous campaign or model code.
"""

import argparse
import concurrent.futures
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "tools"))
from run_manifest_identity import immutable_identity_sha256
from run_streamvln_val_unseen_search import atomic_json
from tta_config_cli import translate

SPEC = REPO_ROOT / "vln/experiments/streamvln_val_seen_search_v1.json"
RESULTS = REPO_ROOT / "vln/results/tuning/streamvln_val_seen_search_v1"
ORDER_ROOT = REPO_ROOT / "vln/manifests/episode_order/r2r_vlnce_v1_3"
ASSETS = REPO_ROOT / "vln/manifests/assets/eval_assets.json"
RUNNER = SCRIPT_DIR / "run_source_eval.sh"
METHODS = ("fstta", "eam", "feedtta", "atena")
EXPECTED = {"val_unseen": 1839, "val_seen": 778}
SETTING = "streamvln-r2r-ce"
LOCK_ROOT = REPO_ROOT / "vln/results/tuning/.streamvln_gpu_locks"
SERVER_VLN_ROOT = Path("/data1/wxy/exp_data/NavTTA/vln")
ACTIVE_PROCESSES = set()
PROCESS_LOCK = threading.Lock()
STOP_REQUESTED = threading.Event()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path, document):
    if path.exists():
        if read_json(path) != document:
            raise ValueError("refusing to replace a different frozen file: {}".format(path))
    else:
        atomic_json(path, document)


def commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
    ).strip()


def assert_committed_runtime():
    # Ignore unrelated research notes/results while ensuring that the recorded
    # commit actually describes every tracked input to this evaluation.
    paths = ["core", "vln/baselines/streamvln", "vln/navtta_vln", "vln/scripts",
             "tools", "vln/manifests/assets", "vln/manifests/environments",
             "vln/manifests/episode_order/r2r_vlnce_v1_3",
             "vln/experiments/streamvln_val_seen_search_v1.json",
             "vln/results/legacy/streamvln_v6_seen_search_audit"]
    status = subprocess.call(["git", "diff", "--quiet", "HEAD", "--"] + paths,
                             cwd=str(REPO_ROOT))
    if status:
        raise ValueError("evaluation code/config differs from HEAD; commit it before starting this versioned campaign")


@dataclass(frozen=True)
class Job:
    tag: str
    method: str
    split: str
    config: Path
    output: Path
    episodes: int
    is_smoke: bool = False

    @property
    def manifest(self):
        return REPO_ROOT / "vln/results/runs" / (
            self.tag + "-" + SETTING + "-" + self.split + "-v1.3"
        ) / "manifest.json"


def parse_result(path, split, episodes=None):
    """Require full coverage, canonical order, and an honest terminal aggregate."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    expected = read_json(ORDER_ROOT / (split + ".json"))["episodes"]
    count = EXPECTED[split] if episodes is None else episodes
    if len(expected) != EXPECTED[split] or not 0 < count <= len(expected):
        raise ValueError("invalid canonical episode manifest or requested count")
    expected = expected[:count]
    if len(rows) != count + 1 or rows[-1].get("length") != count:
        raise ValueError("incomplete result: {} (expected {} episodes + aggregate)".format(path, count))
    actual_order = [(str(r.get("scene_id")), str(r.get("episode_id"))) for r in rows[:-1]]
    expected_order = [(r["scene_id"], str(r["episode_id"])) for r in expected]
    if actual_order != expected_order or len(set(actual_order)) != count:
        raise ValueError("episode coverage/order mismatch: " + str(path))
    for field, aggregate in (("success", "sucs_all"), ("spl", "spls_all")):
        values = [float(r[field]) for r in rows[:-1]]
        final = float(rows[-1][aggregate])
        if (not all(math.isfinite(v) and 0 <= v <= 1 for v in values)
                or not math.isfinite(final)
                or abs(sum(values) / count - final) > 1e-6):
            raise ValueError("invalid {} aggregate: {}".format(field, path))
    return {"episodes": count, "SR": 100 * rows[-1]["sucs_all"],
            "SPL": 100 * rows[-1]["spls_all"]}


def finite_diagnostics(value):
    if isinstance(value, dict):
        return all(finite_diagnostics(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_diagnostics(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def validate_outputs(job):
    metrics = parse_result(job.output / "result.json", job.split, job.episodes)
    if job.method != "source":
        diag = read_json(job.output / "tta_diagnostics.json")
        protocol = read_json(job.config)["parameters"]["streamvln_readout_protocol"]
        if (diag.get("schema") != "navtta.streamvln_tta_diagnostics.v2"
                or diag.get("method") != job.method
                or diag.get("readout_protocol") != protocol
                or diag.get("episodes") != job.episodes
                or not finite_diagnostics(diag)):
            raise ValueError("incomplete/mismatched TTA diagnostics: " + job.tag)
    return metrics


def completed(job):
    """A progress line alone is never evidence that a TTA stream completed."""
    try:
        metrics = validate_outputs(job)
        record = read_json(job.output / "completion.json")
        manifest = read_json(job.manifest)
        if (manifest.get("status") != "completed" or manifest.get("exit_code") != 0
                or manifest.get("immutable_identity_sha256") != immutable_identity_sha256(manifest)
                or record.get("config_sha256") != sha256(job.config)
                or record.get("manifest_sha256") != sha256(job.manifest)
                or record.get("metrics") != metrics):
            return False
        identity = {"method": job.method, "run_tag": job.tag,
                    "source_setting": SETTING + ":" + job.split + ":v1.3",
                    "config": str(job.config), "seed": 0, "task": "vln",
                    "model": "streamvln", "benchmark": "r2r_vlnce_v1_3_streamvln"}
        if any(manifest.get(key) != value for key, value in identity.items()):
            return False
        campaign = RESULTS / "campaign.json"
        if campaign.is_file() and manifest.get("git_commit") != read_json(campaign)["git_commit"]:
            return False
        order_path = ORDER_ROOT / (job.split + ".json")
        if read_json(job.config).get("episode_order_manifest_sha256") != sha256(order_path):
            return False
        if read_json(job.config).get("seed") != 0:
            return False
        required = {"result", "job_config"}
        if job.method != "source":
            required.add("tta_diagnostics")
        artifacts = {a["name"]: a for a in manifest.get("result_artifacts", [])}
        if not required <= set(artifacts):
            return False
        paths = {"result": job.output / "result.json", "job_config": job.config,
                 "tta_diagnostics": job.output / "tta_diagnostics.json"}
        return all(sha256(paths[name]) == artifacts[name]["sha256"] for name in required)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def command(job, gpu):
    cmd = ["bash", str(RUNNER), SETTING, job.split, str(gpu), "--run-tag", job.tag]
    if job.method != "source":
        cmd += ["--tta-config", str(job.config)]
        if not job.is_smoke:
            cmd += ["--result-root", str(job.output)]
    if job.is_smoke:
        cmd += ["--smoke-episodes", str(job.episodes)]
    return cmd


def verify_assets(splits):
    """Hash the actual checkpoint shards once per invocation, not only the index."""
    assets = read_json(ASSETS)
    records = []
    for item in assets["assets"]:
        name = item["id"]
        if not (name.startswith("streamvln_") or name in {
                "r2r_vlnce_v1_3_" + split for split in splits}):
            continue
        raw_path = item["path"]
        if raw_path.startswith("/root/autodl-tmp/cache/"):
            path = SERVER_VLN_ROOT / "cache" / raw_path.split("/cache/", 1)[1]
        else:
            path = REPO_ROOT / raw_path
        if not path.is_file() or path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
            raise ValueError("asset differs from the pinned manifest: " + str(path))
        records.append({"id": name, "path": str(path), "sha256": item["sha256"], "size": item["size"]})
    provenance = {
        "schema": "navtta.streamvln_verified_assets.v1",
        "asset_manifest_sha256": sha256(ASSETS), "records": records,
    }
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    path = REPO_ROOT / "vln/manifests/generated" / ("streamvln-seen-v1-assets-" + fingerprint[:16] + ".json")
    write_once(path, provenance)
    return path


def create_manifest(job, gpu, asset_record):
    assets = {a["id"]: a for a in read_json(ASSETS)["assets"]}
    cmd = [sys.executable, str(REPO_ROOT / "tools/create_run_manifest.py"),
           "--output", str(job.manifest), "--run-id", job.manifest.parent.name,
           "--task", "vln", "--benchmark", "r2r_vlnce_v1_3_streamvln",
           "--model", "streamvln", "--method", job.method, "--run-tag", job.tag,
           "--source-setting", SETTING + ":" + job.split + ":v1.3", "--seed", "0",
           "--config", str(job.config),
           "--checkpoint", str(REPO_ROOT / assets["streamvln_index"]["path"]),
           "--aux-checkpoint", "verified_model_bundle=" + str(asset_record),
           "--dataset", str(REPO_ROOT / assets["r2r_vlnce_v1_3_" + job.split]["path"]),
           "--dataset-version", "r2r_vlnce_v1_3_streamvln",
           "--asset-manifest", str(ASSETS),
           "--environment-manifest", str(REPO_ROOT / "vln/manifests/environments/eval_environments.json"),
           "--episode-order-manifest", str(ORDER_ROOT / (job.split + ".json")),
           "--extra"] + command(job, gpu)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=env, stdout=subprocess.DEVNULL)


def finalize(job, status):
    cmd = [sys.executable, str(REPO_ROOT / "tools/finalize_run_manifest.py"),
           "--manifest", str(job.manifest), "--exit-code", str(status),
           "--artifact", "job_config=" + str(job.config)]
    for name, path in (("result", job.output / "result.json"),
                       ("tta_diagnostics", job.output / "tta_diagnostics.json")):
        if path.is_file():
            cmd += ["--artifact", name + "=" + str(path)]
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL)
    if status == 0:
        atomic_json(job.output / "completion.json", {
            "config_sha256": sha256(job.config), "manifest_sha256": sha256(job.manifest),
            "metrics": validate_outputs(job),
        })


def assert_gpu_idle(gpu):
    output = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True, stderr=subprocess.STDOUT).strip()
    if output:
        raise RuntimeError("GPU {} already has compute processes: {}".format(gpu, output))


@contextmanager
def exclusive_lease(path, description):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(description + " is reserved by another StreamVLN worker")
        try:
            yield stream.fileno()
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def gpu_lease(gpu):
    return exclusive_lease(LOCK_ROOT / ("gpu-{}.lock".format(gpu)), "GPU {}".format(gpu))


def job_lease(job):
    return exclusive_lease(LOCK_ROOT / (job.tag + ".lock"), "job " + job.tag)


def archive_incomplete(job):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = RESULTS / "incomplete" / (job.tag + "-" + stamp)
    root.mkdir(parents=True)
    if job.output.exists():
        shutil.move(str(job.output), str(root / "output"))
    if job.manifest.exists():
        shutil.move(str(job.manifest.parent), str(root / "run_manifest"))
    previous_log = RESULTS / "launcher_logs" / (job.tag + ".log")
    if previous_log.exists():
        shutil.move(str(previous_log), str(root / "launcher.log"))
    shutil.copy2(job.config, root / "job_config.json")
    print("archived incomplete stream -> {}".format(root), flush=True)


def run_one_job(gpu, job, asset_record, gpu_lock, retry_incomplete):
    # Acquire before touching the manifest, including when another invocation
    # assigns this same candidate to a different physical GPU.
    with job_lease(job) as job_lock:
        if STOP_REQUESTED.is_set():
            raise RuntimeError("campaign interrupted before launch")
        if completed(job):
            print("skip complete " + job.tag, flush=True)
            return
        # Also catches other evaluator processes, which do not acquire our locks.
        assert_gpu_idle(gpu)
        occupied = job.manifest.exists() or (job.output.exists() and any(job.output.iterdir()))
        if occupied:
            if not retry_incomplete:
                raise ValueError("incomplete output; rerun with --retry-incomplete to archive and restart: " + job.tag)
            archive_incomplete(job)
        create_manifest(job, gpu, asset_record)
        cmd = command(job, gpu)
        print("launch GPU={} {}".format(gpu, job.tag), flush=True)
        # run_source_eval.sh requires an empty result directory. Keep launcher
        # output separate from the evaluator-owned directory.
        launch_log = RESULTS / "launcher_logs" / (job.tag + ".log")
        launch_log.parent.mkdir(parents=True, exist_ok=True)
        with launch_log.open("w", encoding="utf-8") as log:
            # Keep torchrun away from the SSH terminal's process group. A
            # detached scheduler still needs nohup/tmux; its own TERM/INT is
            # forwarded only to the evaluator groups this campaign created.
            with PROCESS_LOCK:
                if STOP_REQUESTED.is_set():
                    raise RuntimeError("campaign interrupted before launch")
                process = subprocess.Popen(
                    cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT,
                    pass_fds=(gpu_lock, job_lock), start_new_session=True,
                )
                ACTIVE_PROCESSES.add(process)
            try:
                status = process.wait()
            finally:
                with PROCESS_LOCK:
                    ACTIVE_PROCESSES.discard(process)
        if status == 0:
            try:
                validate_outputs(job)
            except (OSError, ValueError, KeyError) as error:
                print("invalid output {}: {}".format(job.tag, error), flush=True)
                status = 1
        finalize(job, status)
        if status:
            raise RuntimeError("exit={}; inspect {}".format(status, launch_log))
        print("done {} {}".format(job.tag, validate_outputs(job)), flush=True)


def run_queue(gpu, jobs, asset_record, retry_incomplete=False):
    failures = []
    with gpu_lease(gpu) as lease:
        for job in jobs:
            if STOP_REQUESTED.is_set():
                failures.extend(item.tag for item in jobs[jobs.index(job):])
                break
            try:
                run_one_job(gpu, job, asset_record, lease, retry_incomplete)
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                failures.append(job.tag)
                print("FAIL {}: {}".format(job.tag, error), file=sys.stderr, flush=True)
    return failures


def stop_active(signum, _frame):
    STOP_REQUESTED.set()
    with PROCESS_LOCK:
        processes = list(ACTIVE_PROCESSES)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    print("received signal {}; stopping owned evaluator groups".format(signum), file=sys.stderr, flush=True)


def load_spec():
    spec = read_json(SPEC)
    if (spec.get("schema") != "navtta.streamvln_seen_search.v1"
            or spec.get("experiment_id") != RESULTS.name
            or spec.get("split") != "val_seen"
            or spec.get("result_role") != "hyperparameter_development"
            or spec.get("seed") != 0 or spec.get("expected_episodes") != EXPECTED):
        raise ValueError("seen search requires the canonical seed, split, episode count and campaign")
    if spec.get("selection") != {"primary_metric": "SPL", "secondary_metric": "SR"}:
        raise ValueError("selection is fixed to SPL, then SR")
    if (set(spec["methods"]) != set(METHODS)
            or spec.get("method_gpu_order") != list(METHODS)):
        raise ValueError("exactly FSTTA/EAM/FeedTTA/ATENA, in the fixed GPU order, are required")
    for method in METHODS:
        candidates = spec["methods"][method]
        if len(candidates) != 3 or len({c["candidate_id"] for c in candidates}) != 3:
            raise ValueError("exactly three distinct candidates required: " + method)
        if len({json.dumps(c["parameters"], sort_keys=True) for c in candidates}) != 3:
            raise ValueError("duplicate effective parameters: " + method)
        for candidate in candidates:
            params = candidate["parameters"]
            if (params.get("streamvln_readout_protocol") != "native_residual"
                    or params.get("action_selection") != "argmax"
                    or params.get("episodic") is not False
                    or not finite_diagnostics(params)):
                raise ValueError("seen search requires finite, native-residual, continual argmax parameters")
            if method == "fstta" and (params.get("use_slow") is not True
                                     or params.get("reset_var_hist_each_episode") is not False):
                raise ValueError("FSTTA requires both branches and stream-lifetime variance history")
    audit = spec["historical_audit"]
    if sha256(REPO_ROOT / audit["path"]) != audit["sha256"]:
        raise ValueError("historical design audit changed")
    return spec


def make_job(method, index, candidate, smoke=False):
    tag = "streamvln-vs-{}-{}-search-v1".format(method, "smoke" if smoke else "{:02d}".format(index))
    episodes = 2 if smoke else EXPECTED["val_seen"]
    config = RESULTS / "configs" / (tag + ".json")
    document = {
        "schema": "navtta.vln_tta_job.v1", "namespace": "tuning",
        "stage": "smoke" if smoke else "full_val_seen_search",
        "result_role": "smoke" if smoke else "hyperparameter_development",
        "setting": SETTING, "split": "val_seen", "episodes": episodes, "seed": 0,
        "episode_order_manifest_sha256": sha256(ORDER_ROOT / "val_seen.json"),
        "method": method, "candidate_id": candidate["candidate_id"],
        "candidate_index": index, "parameters": candidate["parameters"],
        "search_spec_sha256": sha256(SPEC),
        "adaptation_scope": "first_action_token_final_rmsnorm",
        "supervision": "binary_episode_success" if method in ("feedtta", "atena") else "none",
    }
    write_once(config, document)
    output = (REPO_ROOT / "vln/results/smoke" / tag / SETTING / "val_seen" if smoke
              else RESULTS / "jobs" / tag / "val_seen")
    translate(SETTING, config, output / "tta_diagnostics.json")
    return Job(tag, method, "val_seen", config, output, episodes, smoke)


def search_jobs(spec):
    return [make_job(method, index, candidate)
            for method in METHODS for index, candidate in enumerate(spec["methods"][method], 1)]


def smoke_jobs(spec):
    return [make_job(method, 1, spec["methods"][method][0], smoke=True) for method in METHODS]


def gpu_queues(jobs, gpus):
    if len(gpus) != 4 or len(set(gpus)) != 4 or any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError("provide exactly four distinct nonnegative physical GPU indices")
    if any(job.method not in METHODS or job.split != "val_seen" for job in jobs):
        raise ValueError("only the four methods on val_seen can enter this campaign")
    # Never round-robin candidates: a method owns the same card for all three.
    return {gpu: [job for job in jobs if job.method == method]
            for method, gpu in zip(METHODS, gpus)}


def run_jobs(jobs, gpus, asset_record=None, dry_run=False, retry_incomplete=False):
    queues = gpu_queues(jobs, gpus)
    print("jobs={} mapping={} concurrency=1 evaluation/GPU".format(
        len(jobs), dict(zip(METHODS, gpus))), flush=True)
    if dry_run:
        for gpu, queue in queues.items():
            for job in queue:
                print("GPU={} {}".format(gpu, shlex.join(command(job, gpu))))
        return []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_queue, gpu, queue, asset_record, retry_incomplete)
                   for gpu, queue in queues.items() if queue]
        for future in futures:
            failures.extend(future.result())
    return failures


def select_winners(jobs, spec):
    expected = search_jobs(spec)
    if (len(jobs) != 12 or set(jobs) != set(expected)
            or any(not completed(job) for job in jobs)):
        raise ValueError("selection requires all 12 authenticated, complete val_seen candidates")
    records = {}
    for method in METHODS:
        candidates = [job for job in jobs if job.method == method]
        ranked = sorted(candidates, key=lambda job: (
            -validate_outputs(job)["SPL"], -validate_outputs(job)["SR"], job.tag))
        winner = ranked[0]
        config = read_json(winner.config)
        records[method] = {
            "run_tag": winner.tag, "candidate_id": config["candidate_id"],
            "parameters": config["parameters"], "supervision": config["supervision"],
            "metrics": validate_outputs(winner),
            "result_sha256": sha256(winner.output / "result.json"),
            "manifest_sha256": sha256(winner.manifest),
        }
    selection = {
        "schema": "navtta.streamvln_seen_selection.v1", "selection_split": "val_seen",
        "result_role": "hyperparameter_development",
        "ranking": ["SPL_desc", "SR_desc", "candidate_index_asc"],
        "spec_sha256": sha256(SPEC), "records": records,
    }
    write_once(RESULTS / "selection.json", selection)
    return selection


def report(jobs, spec, selection=None):
    reference = read_json(REPO_ROOT / spec["historical_audit"]["path"])["source_val_seen"]
    records = []
    lines = ["# StreamVLN val_seen search v1", "",
             "Development results; selection uses full val_seen SPL, then SR. "
             "FeedTTA/ATENA consume binary episode feedback.", "",
             "Source reference is legacy: only the v6 terminal launcher aggregate is available. "
             "Deltas are historical comparisons, not a new authenticated Source control.", "",
             "| Method | Candidate | Complete | Episodes | SR | SPL | ΔSR (legacy) | ΔSPL (legacy) |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for job in jobs:
        config = read_json(job.config)
        row = {"run_tag": job.tag, "method": job.method, "split": job.split,
               "candidate_id": config["candidate_id"], "parameters": config["parameters"],
               "supervision": config["supervision"], "smoke": job.is_smoke,
               "expected_episodes": job.episodes, "complete": completed(job)}
        if row["complete"]:
            row.update(validate_outputs(job))
            row["result_sha256"] = sha256(job.output / "result.json")
            row["manifest_sha256"] = sha256(job.manifest)
            diag = read_json(job.output / "tta_diagnostics.json")
            row["diagnostics"] = {key: diag.get(key) for key in (
                "readout_protocol", "relative_param_drift", "updates", "action_comparisons",
                "policy_parameter_drift", "query_rate", "source_gate_rate", "aux_gate_rate",
                "slow_updates", "slow_skipped_updates", "feedback_episodes")}
            if not job.is_smoke:
                row["delta_vs_legacy_source_pp"] = {
                    key: row[key] - reference["metrics"][key] for key in ("SR", "SPL")}
        records.append(row)
        delta = row.get("delta_vs_legacy_source_pp", {})
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            job.method + ("†" if config["supervision"] != "none" else ""),
            config["candidate_id"], row["complete"], job.episodes,
            *["{:.4f}".format(row[k]) if k in row else "—" for k in ("SR", "SPL")],
            *["{:+.4f}".format(delta[k]) if k in delta else "—" for k in ("SR", "SPL")]))
        print("{} {}".format(job.tag, "SR={:.4f} SPL={:.4f}".format(row["SR"], row["SPL"])
                            if row["complete"] else "incomplete"), flush=True)
    if selection:
        lines += ["", "Selected candidates (development only):", ""]
        lines += ["- {}: {}".format(m, selection["records"][m]["candidate_id"]) for m in METHODS]
    atomic_json(RESULTS / "report.json", {
        "schema": "navtta.streamvln_seen_report.v1", "result_role": "hyperparameter_development",
        "records": records, "legacy_source_reference": reference, "selection": selection})
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "search", "all", "report"), default="search")
    parser.add_argument("--gpus", default="0,1,2,3", help="physical indices for FSTTA,EAM,FeedTTA,ATENA in order")
    parser.add_argument("--dry-run", action="store_true", help="validate all configs and print commands without GPUs/assets")
    parser.add_argument("--retry-incomplete", action="store_true", help="archive partial streams and restart from episode one")
    args = parser.parse_args()
    try:
        gpus = [int(g) for g in args.gpus.split(",")]
        gpu_queues([], gpus)
        spec = load_spec()
        jobs = search_jobs(spec)
        if args.stage == "report":
            selection = select_winners(jobs, spec) if all(completed(job) for job in jobs) else None
            report(jobs, spec, selection)
            return 0
        if args.dry_run:
            if args.stage in ("smoke", "all"):
                run_jobs(smoke_jobs(spec), gpus, dry_run=True)
            if args.stage in ("search", "all"):
                run_jobs(jobs, gpus, dry_run=True)
            return 0
        assert_committed_runtime()
        write_once(RESULTS / "campaign.json", {
            "git_commit": commit(), "spec_sha256": sha256(SPEC), "seed": 0,
            "split": "val_seen", "result_role": "hyperparameter_development",
            "method_gpus": dict(zip(METHODS, gpus)),
        })
        print("Verifying pinned StreamVLN checkpoint shards and val_seen dataset hashes...", flush=True)
        assets = verify_assets(["val_seen"])
        signal.signal(signal.SIGTERM, stop_active)
        signal.signal(signal.SIGINT, stop_active)
        if args.stage in ("smoke", "all"):
            probes = smoke_jobs(spec)
            failed = run_jobs(probes, gpus, assets, retry_incomplete=args.retry_incomplete)
            if failed or args.stage == "smoke":
                report(probes, spec)
                return 1 if failed else 0
        failed = run_jobs(jobs, gpus, assets, retry_incomplete=args.retry_incomplete)
        selection = None if failed else select_winners(jobs, spec)
        report(jobs, spec, selection)
        if failed:
            print("failed jobs: " + ", ".join(failed), file=sys.stderr)
        return 1 if failed else 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
