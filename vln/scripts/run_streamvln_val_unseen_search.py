#!/usr/bin/env python3
"""Run the 41-job StreamVLN val-unseen Source/TTA campaign on GPUs 2 and 3."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
RESULTS = REPO_ROOT / "vln/results/tuning/streamvln_val_unseen_search_v1"
TRANSLATOR = REPO_ROOT / "vln/scripts/tta_config_cli.py"
CREATE_MANIFEST = REPO_ROOT / "tools/create_run_manifest.py"
FINALIZE_MANIFEST = REPO_ROOT / "tools/finalize_run_manifest.py"
EXPECTED_EPISODES = 1839


SEARCH = {
    "tent": [
        {"lr": lr, "norm_scope": "last_ln", "update_interval": interval,
         "optimizer": "Adam", "max_grad_norm": 1.0}
        for lr in (1e-8, 3e-8, 1e-7, 3e-7)
        for interval in (1, 4)
    ],
    "fstta": [
        {"lr_fast": lr, "lr_slow": slow, "norm_scope": "last_ln",
         "m": 3, "n": 4, "q": 0.1, "rho": 0.95, "tau": 0.7,
         "a": 0.9, "b": 1.1, "fast_grad_mode": "concordant",
         "use_fast_lr_scaler": True, "use_slow": True,
         "reset_optimizer_each_episode": True,
         "reset_var_hist_each_episode": False,
         "reset_slow_optimizer_each_window": False,
         "slow_optimizer": "AdamW", "weight_decay": 0.0,
         "max_grad_norm": 1.0}
        for lr in (1e-8, 3e-8, 1e-7, 3e-7)
        for slow in (1e-8, 1e-7)
    ],
    "eam": [
        {"lr": lr, "confidence_scale": confidence,
         "memory_size": 32, "batch_size": 8, "update_interval": 1,
         "optimizer": "Adam", "weight_decay": 0.0,
         "max_grad_norm": 1.0}
        for lr in (1e-8, 3e-8, 1e-7, 3e-7)
        for confidence in (0.2, 0.4)
    ],
    "feedtta": [
        {"lr": lr, "p": probability, "alpha": -0.2,
         "sgr_seed": 0, "sgr_mode": "paper_main", "gamma": 0.99,
         "normalize_gradient": False,
         "action_selection": "argmax", "optimizer": "Adam",
         "optimizer_eps": 1e-5, "weight_decay": 0.0,
         "max_grad_norm": 1.0}
        for lr in (1e-9, 3e-9, 1e-8, 3e-8)
        for probability in (0.0, 0.05)
    ],
    "atena": [
        {"lr_query": lr, "lr_self": lr / 10.0,
         "mix_lambda": mix, "query_threshold": 0.1,
         "self_loss_weight": 0.1, "weight_decay": 0.01,
         "max_grad_norm": 1.0,
         "update_scope": "replay_reachable_high_level_navigation"}
        for lr in (1e-8, 3e-8, 1e-7, 3e-7)
        for mix in (0.3, 0.5)
    ],
}


def atomic_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def complete(path):
    result = path / "result.json"
    if not result.is_file():
        return False
    try:
        lines = [line for line in result.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        return bool(lines) and json.loads(lines[-1]).get("length") == EXPECTED_EPISODES
    except (OSError, ValueError, AttributeError):
        return False


def build_jobs(methods):
    jobs = []
    if "source" in methods:
        tag = "streamvln-vu-source-v1"
        output = REPO_ROOT / "vln/results/source" / tag / "streamvln-r2r-ce/val_unseen"
        jobs.append((tag, "source", None, output))
    for method in ("tent", "fstta", "eam", "feedtta", "atena"):
        if method not in methods:
            continue
        for index, parameters in enumerate(SEARCH[method], 1):
            tag = "streamvln-vu-{}-{:02d}-v1".format(method, index)
            config = RESULTS / "configs" / (tag + ".json")
            output = RESULTS / "jobs" / tag / "val_unseen"
            atomic_json(config, {
                "schema": "navtta.vln_tta_job.v1",
                "namespace": "tuning",
                "stage": "full_val_unseen_search",
                "setting": "streamvln-r2r-ce",
                "split": "val_unseen",
                "episodes": EXPECTED_EPISODES,
                "method": method,
                "search_method": method,
                "parameters": parameters,
            })
            jobs.append((tag, method, config, output))
    return jobs


def command(job, gpu, dry_run=False):
    tag, method, config, output = job
    cmd = [
        str(RUNNER), "streamvln-r2r-ce", "val_unseen", str(gpu),
        "--run-tag", tag,
    ]
    if method != "source":
        cmd.extend(["--tta-config", str(config), "--result-root", str(output)])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def create_manifest(job, gpu, cmd):
    tag, method, config, _ = job
    manifest = (
        REPO_ROOT / "vln/results/runs" /
        (tag + "-streamvln-r2r-ce-val_unseen-v1.3") / "manifest.json"
    )
    baseline_config = REPO_ROOT / "vln/baselines/streamvln/config/vln_r2r.yaml"
    checkpoint = (
        REPO_ROOT / "vln/checkpoints/streamvln" /
        "StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3" /
        "model.safetensors.index.json"
    )
    dataset = REPO_ROOT / "vln/data/datasets/r2r/val_unseen/val_unseen.json.gz"
    order = REPO_ROOT / "vln/manifests/episode_order/r2r_vlnce_v1_3/val_unseen.json"
    create = [
        sys.executable, str(CREATE_MANIFEST), "--output", str(manifest),
        "--run-id", manifest.parent.name, "--task", "vln",
        "--benchmark", "r2r_vlnce_v1_3_streamvln",
        "--model", "streamvln", "--method", method,
        "--run-tag", tag, "--source-setting",
        "streamvln-r2r-ce:val_unseen:v1.3", "--seed", "0",
        "--config", str(config or baseline_config),
        "--checkpoint", str(checkpoint), "--dataset", str(dataset),
        "--dataset-version", "r2r_vlnce_v1_3_streamvln",
        "--asset-manifest", str(REPO_ROOT / "vln/manifests/assets/eval_assets.json"),
        "--environment-manifest", str(
            REPO_ROOT / "vln/manifests/environments/eval_environments.json"
        ),
        "--episode-order-manifest", str(order), "--extra",
    ] + list(cmd)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.check_call(create, cwd=str(REPO_ROOT), env=environment)
    return manifest


def finalize_manifest(manifest, status, job):
    _, method, config, output = job
    command = [
        sys.executable, str(FINALIZE_MANIFEST), "--manifest", str(manifest),
        "--exit-code", str(status),
    ]
    result = output / "result.json"
    if result.is_file():
        command.extend(["--artifact", "result={}".format(result)])
    diagnostics = output / "tta_diagnostics.json"
    if diagnostics.is_file():
        command.extend(["--artifact", "tta_diagnostics={}".format(diagnostics)])
    if method != "source" and config is not None and config.is_file():
        command.extend(["--artifact", "job_config={}".format(config)])
    subprocess.check_call(command, cwd=str(REPO_ROOT))


def run_queue(gpu, jobs, dry_run):
    failures = []
    for job in jobs:
        tag, _, config, output = job
        if complete(output):
            print("skip complete {} gpu={}".format(tag, gpu), flush=True)
            continue
        if output.exists() and any(output.iterdir()) and not dry_run:
            print("FAIL nonempty incomplete output: {}".format(output), flush=True)
            failures.append(tag)
            break
        cmd = command(job, gpu, dry_run=dry_run)
        print("launch gpu={} {}".format(gpu, " ".join(cmd)), flush=True)
        if dry_run:
            if config is not None:
                status = subprocess.call([
                    sys.executable, str(TRANSLATOR),
                    "--setting", "streamvln-r2r-ce",
                    "--config", str(config),
                    "--diagnostics", str(output / "tta_diagnostics.json"),
                ], cwd=str(REPO_ROOT))
            else:
                status = 0
        else:
            try:
                manifest = create_manifest(job, gpu, cmd)
                status = subprocess.call(cmd, cwd=str(REPO_ROOT))
                if status == 0 and not complete(output):
                    status = 1
                finalize_manifest(manifest, status, job)
            except subprocess.CalledProcessError as error:
                status = int(error.returncode or 1)
        if status != 0:
            print("FAIL {} exit={}".format(tag, status), flush=True)
            failures.append(tag)
            if not dry_run:
                break
        elif not dry_run and not complete(output):
            print("FAIL {} missing complete aggregate".format(tag), flush=True)
            failures.append(tag)
        else:
            print("done {}".format(tag), flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument(
        "--methods", default="source,tent,fstta,eam,feedtta,atena",
        help="comma-separated subset",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if gpus != [2, 3]:
        raise SystemExit("this campaign is fixed to --gpus 2,3")
    methods = [value.strip().lower() for value in args.methods.split(",")]
    allowed = {"source", "tent", "fstta", "eam", "feedtta", "atena"}
    if not methods or len(methods) != len(set(methods)) or not set(methods) <= allowed:
        raise SystemExit("invalid or duplicate --methods")
    jobs = build_jobs(methods)
    queues = {gpu: [] for gpu in gpus}
    for index, job in enumerate(jobs):
        queues[gpus[index % len(gpus)]].append(job)
    print("jobs={} gpu2={} gpu3={} concurrency=1/card".format(
        len(jobs), len(queues[2]), len(queues[3])
    ), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_queue, gpu, queues[gpu], args.dry_run)
            for gpu in gpus
        ]
        failures = []
        for future in futures:
            failures.extend(future.result())
    if failures:
        print("failed jobs: {}".format(", ".join(failures)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
