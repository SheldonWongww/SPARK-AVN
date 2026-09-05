#!/usr/bin/env python3
"""Run the 16-job compact StreamVLN val-unseen campaign on GPUs 0, 1, 2, 3.

The campaign contains one fresh Source evaluation plus three candidates for
each of Tent, FSTTA, EAM, FeedTTA, and ATENA.  Jobs are distributed evenly, so
every GPU runs four jobs serially.  A failed job is recorded but does not
prevent the remaining jobs assigned to that GPU from starting.
"""

import argparse
import concurrent.futures
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_streamvln_val_unseen_search as shared


REPO_ROOT = shared.REPO_ROOT
RUNNER = shared.RUNNER
TRANSLATOR = shared.TRANSLATOR
RESULTS = REPO_ROOT / "vln/results/tuning/streamvln_val_unseen_compact_v4"
EXPECTED_EPISODES = shared.EXPECTED_EPISODES
TTA_METHOD_ORDER = ("tent", "fstta", "eam", "feedtta", "atena")
METHOD_ORDER = ("source",) + TTA_METHOD_ORDER


# The StreamVLN adapter updates only a float32 copy of Qwen's final RMSNorm
# action readout.  Learning rates are therefore bracketed conservatively even
# when the source profile came from a smaller VLN policy.
SEARCH = {
    "tent": [
        (
            "low_lr_bracket",
            {
                "lr": 3e-8,
                "norm_scope": "last_ln",
                "update_interval": 1,
                "optimizer": "Adam",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "action_selection": "argmax",
            },
        ),
        (
            "middle_lr_bracket",
            {
                "lr": 1e-7,
                "norm_scope": "last_ln",
                "update_interval": 1,
                "optimizer": "Adam",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "action_selection": "argmax",
            },
        ),
        (
            "r2r_ce_low_lr_transfer",
            {
                "lr": 3e-7,
                "norm_scope": "last_ln",
                "update_interval": 1,
                "optimizer": "AdamW",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "action_selection": "argmax",
            },
        ),
    ],
    "fstta": [
        (
            "conservative_m3_n4",
            {
                "lr_fast": 3e-8,
                "lr_slow": 1e-7,
                "norm_scope": "last_ln",
                "m": 3,
                "n": 4,
                "q": 0.1,
                "rho": 0.95,
                "tau": 0.7,
                "a": 0.9,
                "b": 1.1,
                "fast_grad_mode": "concordant",
                "use_fast_lr_scaler": True,
                "use_slow": True,
                "reset_optimizer_each_episode": True,
                "reset_var_hist_each_episode": False,
                "reset_slow_optimizer_each_window": False,
                "optimizer": "AdamW",
                "slow_optimizer": "AdamW",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "action_selection": "argmax",
            },
        ),
        (
            "middle_m3_n4",
            {
                "lr_fast": 1e-7,
                "lr_slow": 3e-7,
                "norm_scope": "last_ln",
                "m": 3,
                "n": 4,
                "q": 0.1,
                "rho": 0.95,
                "tau": 0.7,
                "a": 0.9,
                "b": 1.1,
                "fast_grad_mode": "concordant",
                "use_fast_lr_scaler": True,
                "use_slow": True,
                "reset_optimizer_each_episode": True,
                "reset_var_hist_each_episode": False,
                "reset_slow_optimizer_each_window": False,
                "optimizer": "AdamW",
                "slow_optimizer": "AdamW",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "action_selection": "argmax",
            },
        ),
        (
            "r2r_ce_m4_n16_transfer",
            {
                "lr_fast": 3e-7,
                "lr_slow": 1e-6,
                "norm_scope": "last_ln",
                "m": 4,
                "n": 16,
                "q": 0.1,
                "rho": 0.95,
                "tau": 0.7,
                "a": 0.9,
                "b": 1.1,
                "fast_grad_mode": "concordant",
                "use_fast_lr_scaler": True,
                "use_slow": True,
                "reset_optimizer_each_episode": True,
                "reset_var_hist_each_episode": False,
                "reset_slow_optimizer_each_window": False,
                "optimizer": "AdamW",
                "slow_optimizer": "AdamW",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "action_selection": "argmax",
            },
        ),
    ],
    "eam": [
        (
            "bevbert_r2r_ce_transfer",
            {
                "lr": 5e-8,
                "memory_size": 64,
                "batch_size": 16,
                "confidence_scale": 0.5,
                "update_interval": 4,
                "optimizer": "Adam",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "action_selection": "argmax",
            },
        ),
        (
            "reverie_transfer",
            {
                "lr": 2e-7,
                "memory_size": 32,
                "batch_size": 8,
                "confidence_scale": 0.4,
                "update_interval": 1,
                "optimizer": "Adam",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "action_selection": "argmax",
            },
        ),
        (
            "etpnav_r2r_ce_transfer",
            {
                "lr": 6e-7,
                "memory_size": 32,
                "batch_size": 8,
                "confidence_scale": 0.4,
                "update_interval": 1,
                "optimizer": "Adam",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "action_selection": "argmax",
            },
        ),
    ],
    "feedtta": [
        (
            "hamt_r2r_shape_low_lr",
            {
                "lr": 3e-9,
                "p": 0.05,
                "alpha": 0.1,
                "gamma": 0.7,
                "sgr_seed": 0,
                "sgr_mode": "paper_main",
                "scope_profile": "paper_full",
                "normalize_gradient": False,
                "action_selection": "argmax",
                "optimizer": "Adam",
                "optimizer_eps": 1e-5,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            },
        ),
        (
            "goat_r2r_shape_middle_lr",
            {
                "lr": 1e-8,
                "p": 0.05,
                "alpha": 0.1,
                "gamma": 0.95,
                "sgr_seed": 0,
                "sgr_mode": "paper_main",
                "scope_profile": "paper_full",
                "normalize_gradient": False,
                "action_selection": "argmax",
                "optimizer": "Adam",
                "optimizer_eps": 1e-5,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            },
        ),
        (
            "reverie_shape_high_lr",
            {
                "lr": 3e-8,
                "p": 0.05,
                "alpha": -0.2,
                "gamma": 0.7,
                "sgr_seed": 0,
                "sgr_mode": "paper_main",
                "scope_profile": "paper_full",
                "normalize_gradient": False,
                "action_selection": "argmax",
                "optimizer": "Adam",
                "optimizer_eps": 1e-5,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            },
        ),
    ],
    "atena": [
        (
            "hamt_r2r_shape_low_lr",
            {
                "lr_query": 3e-8,
                "lr_self": 3e-9,
                "mix_lambda": 0.1,
                "query_threshold": 0.15,
                "self_loss_weight": 0.1,
                "optimizer": "AdamW",
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "action_selection": "argmax",
                "update_scope": "replay_reachable_high_level_navigation",
            },
        ),
        (
            "etpnav_r2r_ce_shape_middle_lr",
            {
                "lr_query": 1e-7,
                "lr_self": 1e-8,
                "mix_lambda": 0.5,
                "query_threshold": 0.3,
                "self_loss_weight": 0.1,
                "optimizer": "AdamW",
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "action_selection": "argmax",
                "update_scope": "replay_reachable_high_level_navigation",
            },
        ),
        (
            "bevbert_r2r_ce_shape_high_lr",
            {
                "lr_query": 3e-7,
                "lr_self": 3e-8,
                "mix_lambda": 0.25,
                "query_threshold": 0.1,
                "self_loss_weight": 0.1,
                "optimizer": "AdamW",
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "action_selection": "argmax",
                "update_scope": "replay_reachable_high_level_navigation",
            },
        ),
    ],
}


def build_jobs(methods):
    jobs = []
    if "source" in methods:
        tag = "streamvln-vu-compact-source-v4"
        output = (
            REPO_ROOT / "vln/results/source" / tag
            / "streamvln-r2r-ce/val_unseen"
        )
        jobs.append((tag, "source", None, output))

    for method in TTA_METHOD_ORDER:
        if method not in methods:
            continue
        candidates = SEARCH[method]
        if len(candidates) != 3:
            raise RuntimeError("{} must have exactly three candidates".format(method))
        for index, (profile, parameters) in enumerate(candidates, 1):
            tag = "streamvln-vu-compact-{}-{:02d}-v4".format(method, index)
            config = RESULTS / "configs" / (tag + ".json")
            output = RESULTS / "jobs" / tag / "val_unseen"
            shared.atomic_json(config, {
                "schema": "navtta.vln_tta_job.v1",
                "namespace": "tuning",
                "stage": "compact_val_unseen_search",
                "setting": "streamvln-r2r-ce",
                "split": "val_unseen",
                "episodes": EXPECTED_EPISODES,
                "method": method,
                "search_method": method,
                "candidate_id": profile,
                "selection_basis": (
                    "R2R/REVERIE transfer; EAM also uses ETPNav/BEVBert "
                    "R2R-CE targeted-gap evidence"
                ),
                "parameters": parameters,
            })
            jobs.append((tag, method, config, output))
    return jobs


def run_queue(gpu, jobs, dry_run):
    failures = []
    for job in jobs:
        tag, _, config, output = job
        if shared.complete(output):
            print("skip complete {} gpu={}".format(tag, gpu), flush=True)
            continue
        if output.exists() and any(output.iterdir()) and not dry_run:
            print("FAIL nonempty incomplete output: {}".format(output), flush=True)
            failures.append(tag)
            # Jobs are independent; do not strand the rest of this GPU queue.
            continue

        cmd = shared.command(job, gpu, dry_run=dry_run)
        print("launch gpu={} {}".format(gpu, " ".join(cmd)), flush=True)
        if dry_run:
            if config is None:
                status = 0
            else:
                status = subprocess.call([
                    sys.executable,
                    str(TRANSLATOR),
                    "--setting", "streamvln-r2r-ce",
                    "--config", str(config),
                    "--diagnostics", str(output / "tta_diagnostics.json"),
                ], cwd=str(REPO_ROOT))
        else:
            try:
                manifest = shared.create_manifest(job, gpu, cmd)
                status = subprocess.call(cmd, cwd=str(REPO_ROOT))
                if status == 0 and not shared.complete(output):
                    status = 1
                shared.finalize_manifest(manifest, status, job)
            except subprocess.CalledProcessError as error:
                status = int(error.returncode or 1)

        if status != 0:
            print("FAIL {} exit={}".format(tag, status), flush=True)
            failures.append(tag)
            # Continue so one bad method/configuration does not stop a card.
            continue
        print("done {}".format(tag), flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--methods",
        default=",".join(METHOD_ORDER),
        help="comma-separated subset; IDEA is intentionally absent",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        gpus = [int(value.strip()) for value in args.gpus.split(",")]
    except ValueError as error:
        raise SystemExit("--gpus must be a comma-separated integer list") from error
    if gpus != [0, 1, 2, 3]:
        raise SystemExit("this campaign is fixed to --gpus 0,1,2,3")

    methods = [value.strip().lower() for value in args.methods.split(",")]
    allowed = set(METHOD_ORDER)
    if not methods or len(methods) != len(set(methods)) or not set(methods) <= allowed:
        raise SystemExit("invalid or duplicate --methods")

    jobs = build_jobs(methods)
    queues = {gpu: [] for gpu in gpus}
    for index, job in enumerate(jobs):
        queues[gpus[index % len(gpus)]].append(job)

    counts = " ".join(
        "gpu{}={}".format(gpu, len(queues[gpu])) for gpu in gpus
    )
    print(
        "jobs={} {} concurrency=1/card".format(
            len(jobs), counts
        ),
        flush=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
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
