#!/usr/bin/env python3
"""Sixteen StreamVLN val_unseen candidates, then frozen val_seen evaluation.

Each GPU owns one serial worker and a process-lifetime lock. Source val_unseen
and Tent selection are authenticated v4 imports, never search jobs. Incomplete
TTA streams must restart from their first episode; optimizer state is not saved.
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
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "tools"))
from run_manifest_identity import immutable_identity_sha256
from run_streamvln_val_unseen_search import atomic_json
from tta_config_cli import translate

SPEC = REPO_ROOT / "vln/experiments/streamvln_val_unseen_search_v5.json"
RESULTS = REPO_ROOT / "vln/results/tuning/streamvln_val_unseen_search_v5"
ORDER_ROOT = REPO_ROOT / "vln/manifests/episode_order/r2r_vlnce_v1_3"
ASSETS = REPO_ROOT / "vln/manifests/assets/eval_assets.json"
RUNNER = SCRIPT_DIR / "run_source_eval.sh"
METHODS = ("fstta", "eam", "feedtta", "atena")
EXPECTED = {"val_unseen": 1839, "val_seen": 778}
SETTING = "streamvln-r2r-ce"
LOCK_ROOT = REPO_ROOT / "vln/results/tuning/.streamvln_gpu_locks"
SERVER_VLN_ROOT = Path("/data1/wxy/exp_data/NavTTA/vln")


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
             "vln/experiments/streamvln_val_unseen_search_v5.json"]
    status = subprocess.call(["git", "diff", "--quiet", "HEAD", "--"] + paths,
                             cwd=str(REPO_ROOT))
    if status:
        raise ValueError("evaluation code/config differs from HEAD; commit it before starting this versioned campaign")


def load_spec():
    spec = read_json(SPEC)
    if spec.get("schema") != "navtta.streamvln_search.v5":
        raise ValueError("invalid v5 search spec")
    if spec.get("seed") != 0 or spec.get("expected_episodes") != EXPECTED:
        raise ValueError("search must use the canonical seed and complete splits")
    if spec.get("selection") != {"primary_metric": "SPL", "secondary_metric": "SR"}:
        raise ValueError("v5 selection is fixed to SPL, then SR")
    if set(spec["methods"]) != set(METHODS):
        raise ValueError("only FSTTA/EAM/FeedTTA/ATENA may be searched")
    for method in METHODS:
        candidates = spec["methods"][method]
        if len(candidates) != 4 or len({c["candidate_id"] for c in candidates}) != 4:
            raise ValueError("exactly four distinct candidates required: " + method)
        if len({json.dumps(c["parameters"], sort_keys=True) for c in candidates}) != 4:
            raise ValueError("duplicate effective parameters: " + method)
        if any(c["parameters"].get("streamvln_readout_protocol") != "native_residual"
               for c in candidates):
            raise ValueError("new search requires native_residual readout")
    if spec["frozen_tent"]["parameters"].get("streamvln_readout_protocol") != "legacy_v4":
        raise ValueError("Tent must retain its selected v4 protocol")
    return spec


@dataclass(frozen=True)
class Job:
    tag: str
    method: str
    split: str
    config: Path
    output: Path

    @property
    def manifest(self):
        return REPO_ROOT / "vln/results/runs" / (
            self.tag + "-" + SETTING + "-" + self.split + "-v1.3"
        ) / "manifest.json"


def make_job(method, split, candidate_id, parameters, selection=None):
    short_split = "vu" if split == "val_unseen" else "vs"
    tag = "streamvln-{}-{}-{}-v5".format(short_split, method, candidate_id)
    config = RESULTS / "configs" / (tag + ".json")
    document = {
        "schema": "navtta.vln_tta_job.v1", "namespace": "tuning",
        "stage": "full_val_unseen_search" if split == "val_unseen" else "frozen_val_seen_eval",
        "setting": SETTING, "split": split, "episodes": EXPECTED[split],
        "method": method, "candidate_id": candidate_id,
        "parameters": parameters, "search_spec_sha256": sha256(SPEC),
        "adaptation_scope": "first_action_token_final_rmsnorm" if method != "source" else "none",
        "supervision": "binary_episode_success" if method in ("feedtta", "atena") else "none",
    }
    if selection is not None:
        document["selection"] = selection
    write_once(config, document)
    output = RESULTS / "jobs" / tag / split
    if method == "source":
        # The existing launcher puts unadapted Source under its Source namespace.
        output = REPO_ROOT / "vln/results/source" / tag / SETTING / split
    translate(SETTING, config, output / "tta_diagnostics.json")
    return Job(tag, method, split, config, output)


def search_jobs(spec):
    return [make_job(method, "val_unseen", "{:02d}".format(index), c["parameters"])
            for method in METHODS for index, c in enumerate(spec["methods"][method], 1)]


def parse_result(path, split):
    """Require full coverage, canonical order, and an honest terminal aggregate."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    expected = read_json(ORDER_ROOT / (split + ".json"))["episodes"]
    count = EXPECTED[split]
    if len(expected) != count or len(rows) != count + 1 or rows[-1].get("length") != count:
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


def bind_legacy(spec, source_result=None, tent_result=None):
    imports = {}
    for method, override in (("source", source_result), ("tent", tent_result)):
        reference = spec["legacy_reuse"][method]
        path = Path(override) if override else REPO_ROOT / reference["result_path"]
        if sha256(path) != reference["sha256"]:
            raise ValueError("{} v4 result differs from the supplied archive: {}".format(method, path))
        imports[method] = {
            "path": str(path.resolve()), "sha256": reference["sha256"],
            "metrics": parse_result(path, "val_unseen"),
            "provenance_status": "legacy_archive_without_complete_run_manifest",
        }
    write_once(RESULTS / "legacy_reuse.json", imports)
    return imports


def validate_outputs(job):
    metrics = parse_result(job.output / "result.json", job.split)
    if job.method != "source":
        diag = read_json(job.output / "tta_diagnostics.json")
        protocol = read_json(job.config)["parameters"]["streamvln_readout_protocol"]
        if (diag.get("schema") != "navtta.streamvln_tta_diagnostics.v2"
                or diag.get("method") != job.method
                or diag.get("readout_protocol") != protocol
                or diag.get("episodes") != EXPECTED[job.split]):
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
        cmd += ["--tta-config", str(job.config), "--result-root", str(job.output)]
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
    path = REPO_ROOT / "vln/manifests/generated" / ("streamvln-v5-assets-" + fingerprint[:16] + ".json")
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
            raise RuntimeError(description + " is reserved by another StreamVLN v5 worker")
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
        if completed(job):
            print("skip complete " + job.tag, flush=True)
            return
        # Also catches pre-v5 processes, which do not acquire our locks.
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
            status = subprocess.call(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT,
                                     pass_fds=(gpu_lock, job_lock))
        if status == 0:
            try:
                validate_outputs(job)
            except (OSError, ValueError, KeyError) as error:
                print("invalid output {}: {}".format(job.tag, error), flush=True)
                status = 1
        finalize(job, status)
        if status:
            raise RuntimeError("exit={}; inspect {}".format(status, launch_log))
        print("done {} {}".format(job.tag, parse_result(job.output / "result.json", job.split)), flush=True)


def run_queue(gpu, jobs, asset_record, retry_incomplete=False):
    failures = []
    with gpu_lease(gpu) as lease:
        for job in jobs:
            try:
                run_one_job(gpu, job, asset_record, lease, retry_incomplete)
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                failures.append(job.tag)
                print("FAIL {}: {}".format(job.tag, error), file=sys.stderr, flush=True)
    return failures


def run_jobs(jobs, gpus, asset_record=None, dry_run=False, retry_incomplete=False):
    queues = {gpu: jobs[index::len(gpus)] for index, gpu in enumerate(gpus)}
    print("jobs={} GPUs={} concurrency=1 evaluation/GPU".format(len(jobs), gpus), flush=True)
    if dry_run:
        for gpu, queue in queues.items():
            for job in queue:
                print("GPU={} {}".format(gpu, shlex.join(command(job, gpu))))
        return []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(run_queue, gpu, queue, asset_record, retry_incomplete)
                   for gpu, queue in queues.items() if queue]
        for future in futures:
            failures.extend(future.result())
    return failures


def select_winners(jobs):
    if len(jobs) != 16 or any(not completed(job) for job in jobs):
        raise ValueError("selection requires all 16 authenticated, complete val_unseen results")
    if any(len({getattr(job, field) for job in jobs}) != 16 for field in ("tag", "config", "output")):
        raise ValueError("selection requires 16 distinct jobs, configs, and output directories")
    records = {}
    for method in METHODS:
        candidates = [job for job in jobs if job.method == method and job.split == "val_unseen"]
        if len(candidates) != 4:
            raise ValueError("selection requires four candidates per method")
        ranked = sorted(candidates, key=lambda job: (
            -parse_result(job.output / "result.json", job.split)["SPL"],
            -parse_result(job.output / "result.json", job.split)["SR"], job.tag,
        ))
        winner = ranked[0]
        records[method] = {
            "run_tag": winner.tag, "parameters": read_json(winner.config)["parameters"],
            "metrics": parse_result(winner.output / "result.json", "val_unseen"),
            "result_sha256": sha256(winner.output / "result.json"),
            "manifest_sha256": sha256(winner.manifest),
        }
    selection = {"schema": "navtta.streamvln_selection.v5", "selection_split": "val_unseen",
                 "ranking": ["SPL_desc", "SR_desc", "candidate_index_asc"],
                 "spec_sha256": sha256(SPEC), "records": records}
    write_once(RESULTS / "selection.json", selection)
    return selection


def seen_jobs(spec, selection):
    binding = {"selection_split": "val_unseen", "selection_sha256": sha256(RESULTS / "selection.json")}
    jobs = [make_job("source", "val_seen", "frozen", {}, {"source": "unchanged_checkpoint"}),
            make_job("tent", "val_seen", "frozen", spec["frozen_tent"]["parameters"],
                     {"v4_run_tag": spec["frozen_tent"]["run_tag"], "selection_split": "val_unseen"})]
    jobs += [make_job(method, "val_seen", "frozen", selection["records"][method]["parameters"], binding)
             for method in METHODS]
    return jobs


def report(jobs):
    records = []
    identical = {}
    baselines = {}
    legacy_path = RESULTS / "legacy_reuse.json"
    legacy = read_json(legacy_path) if legacy_path.exists() else {}
    if legacy:
        baselines["val_unseen"] = legacy["source"]["metrics"]
    for job in jobs:
        if job.method == "source" and completed(job):
            baselines[job.split] = parse_result(job.output / "result.json", job.split)
    for job in jobs:
        row = {"run_tag": job.tag, "method": job.method, "split": job.split,
               "parameters": read_json(job.config)["parameters"], "complete": completed(job)}
        if row["complete"]:
            row.update(parse_result(job.output / "result.json", job.split))
            digest = sha256(job.output / "result.json")
            row["result_sha256"] = digest
            identical.setdefault(digest, []).append(job.tag)
            if job.split in baselines:
                row["delta_vs_source_pp"] = {key: row[key] - baselines[job.split][key] for key in ("SR", "SPL")}
            if job.method != "source":
                diag = read_json(job.output / "tta_diagnostics.json")
                for key in ("readout_protocol", "relative_param_drift", "updates",
                            "action_comparisons", "parameter_changes", "query_rate",
                            "policy_parameter_drift", "source_gate_rate", "aux_gate_rate",
                            "slow_updates", "slow_skipped_updates", "feedback_episodes"):
                    row[key] = diag.get(key)
        records.append(row)
    atomic_json(RESULTS / "report.json", {
        "records": records, "legacy_reuse": legacy,
        "identical_complete_result_groups": [tags for tags in identical.values() if len(tags) > 1],
        "selection": read_json(RESULTS / "selection.json") if (RESULTS / "selection.json").exists() else None,
    })
    lines = ["# StreamVLN v5", "", "val_unseen selects parameters; val_seen evaluates frozen choices.", "",
             "| Run | Split | SR | SPL | ΔSR | ΔSPL |", "|---|---|---:|---:|---:|---:|"]
    if legacy:
        for method in ("source", "tent"):
            metrics = legacy[method]["metrics"]
            lines.append("| {} (v4 legacy reuse) | val_unseen | {:.4f} | {:.4f} | — | — |".format(
                method, metrics["SR"], metrics["SPL"]))
    for row in records:
        print("{} {}".format(row["run_tag"],
              "SR={:.4f} SPL={:.4f}".format(row["SR"], row["SPL"]) if row["complete"] else "incomplete"))
        if row["complete"]:
            delta = row.get("delta_vs_source_pp")
            lines.append("| {} | {} | {:.4f} | {:.4f} | {} | {} |".format(
                row["run_tag"], row["split"], row["SR"], row["SPL"],
                "{:+.4f}".format(delta["SR"]) if delta else "—",
                "{:+.4f}".format(delta["SPL"]) if delta else "—"))
        else:
            lines.append("| {} | {} | incomplete | — | — | — |".format(row["run_tag"], row["split"]))
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("search", "seen", "all", "report"), default="search")
    parser.add_argument("--gpus", default="0,1,2,3", help="physical GPU indices, one serial worker each")
    parser.add_argument("--dry-run", action="store_true", help="translate configs/print commands without models or assets")
    parser.add_argument("--retry-incomplete", action="store_true", help="archive partial streams and restart from episode one")
    parser.add_argument("--source-result", type=Path, help="relocated v4 Source result.json (must match archive SHA256)")
    parser.add_argument("--tent-result", type=Path, help="relocated v4 Tent-01 result.json (must match archive SHA256)")
    args = parser.parse_args()
    try:
        gpus = [int(g) for g in args.gpus.split(",")]
        if not gpus or min(gpus) < 0 or len(gpus) != len(set(gpus)):
            raise ValueError("invalid/duplicate GPUs")
        spec = load_spec()
        jobs = search_jobs(spec)
        if args.stage == "report":
            if (RESULTS / "selection.json").exists():
                selection = select_winners(jobs)
                jobs += seen_jobs(spec, selection)
            report(jobs)
            return 0
        if args.dry_run:
            if args.stage in ("search", "all"):
                run_jobs(jobs, gpus, dry_run=True)
            print("After 16 complete val_unseen runs: freeze winners by SPL, then SR; val_seen never selects parameters.")
            if args.stage in ("seen", "all"):
                if all(completed(job) for job in jobs):
                    run_jobs(seen_jobs(spec, select_winners(jobs)), gpus, dry_run=True)
                else:
                    print("val_seen: Source + v4 Tent-01 + four frozen winners (6 jobs; winners pending).")
            return 0
        assert_committed_runtime()
        bind_legacy(spec, args.source_result, args.tent_result)
        # Pin one code/config version for this campaign. A changed implementation
        # needs a new version/output directory, not a silently mixed ranking.
        write_once(RESULTS / "campaign.json", {"git_commit": commit(), "spec_sha256": sha256(SPEC), "seed": 0})
        if args.stage == "seen":
            selection = select_winners(jobs)
        splits = ["val_unseen", "val_seen"] if args.stage == "all" else ["val_seen" if args.stage == "seen" else "val_unseen"]
        print("Verifying pinned StreamVLN checkpoint shards and dataset hashes once...", flush=True)
        assets = verify_assets(splits)
        if args.stage in ("search", "all"):
            failed = run_jobs(jobs, gpus, assets, retry_incomplete=args.retry_incomplete)
            report(jobs)
            if failed:
                print("failed jobs: " + ", ".join(failed), file=sys.stderr)
                return 1
            selection = select_winners(jobs)
        if args.stage in ("seen", "all"):
            frozen = seen_jobs(spec, selection)
            failed = run_jobs(frozen, gpus, assets, retry_incomplete=args.retry_incomplete)
            report(jobs + frozen)
            if failed:
                return 1
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
