#!/usr/bin/env python3
"""Stage 2: sweep the official FeedTTA SGR grid and mechanism controls."""

import argparse
import math
from pathlib import Path
import re

from feedtta_grid_common import (
    ExperimentSpec,
    MODELS,
    SearchPoint,
    add_common_arguments,
    cli_main,
    finalize_common_args,
)


STAGE1_LRS = ("1e-8", "3e-8", "1e-7", "3e-7", "1e-6", "5e-6")
STAGE1_GAMMAS = ("0.90", "0.95", "0.99", "1.0")
OFFICIAL_P = ("0.01", "0.05", "0.1", "0.2", "0.3")
OFFICIAL_ALPHA = ("-0.01", "-0.025", "-0.05", "-0.075", "-0.1", "-0.2", "-0.3")
CONTROLS = (
    ("no_sgr", "0.0", "-0.2"),
    ("gradient_dropout", "0.05", "0.0"),
    ("gradient_scaling_0p05", "0.05", "0.05"),
    ("gradient_scaling_0p1", "0.05", "0.1"),
)


def batch_id(value):
    if len(value) > 64 or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None:
        raise argparse.ArgumentTypeError("invalid Stage-1 batch id")
    return value


def canonical_choice(allowed, label):
    def parse(value):
        try:
            numeric = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("{} must be numeric".format(label)) from error
        if not math.isfinite(numeric):
            raise argparse.ArgumentTypeError("{} must be finite".format(label))
        for candidate in allowed:
            if math.isclose(numeric, float(candidate), rel_tol=1e-12, abs_tol=0.0):
                return candidate
        raise argparse.ArgumentTypeError(
            "{} must be selected from {}".format(label, ",".join(allowed))
        )
    return parse


def model_points(lr, gamma):
    official = tuple(
        SearchPoint(
            variant="official_sgr",
            lr=lr,
            gamma=gamma,
            p=p,
            alpha=alpha,
            smoke_anchor=(p == "0.05" and alpha == "-0.2"),
        )
        for p in OFFICIAL_P
        for alpha in OFFICIAL_ALPHA
    )
    controls = tuple(
        SearchPoint(
            variant=variant,
            lr=lr,
            gamma=gamma,
            p=p,
            alpha=alpha,
        )
        for variant, p, alpha in CONTROLS
    )
    points = official + controls
    if len(official) != 35 or len(points) != 39:
        raise RuntimeError("FeedTTA Stage 2 must contain 35+4 points/model")
    return points


def build_spec(args) -> ExperimentSpec:
    selected = {
        "smt_audio": {
            "lr": args.smt_audio_lr,
            "gamma": args.smt_audio_gamma,
        },
        "enmus": {
            "lr": args.enmus_lr,
            "gamma": args.enmus_gamma,
        },
    }
    points_by_model = {
        model: model_points(values["lr"], values["gamma"])
        for model, values in selected.items()
    }
    return ExperimentSpec(
        stage="stage2",
        experiment="feedtta_avn_sgr_grid_v1",
        log_dir_name="feedtta_stage2",
        launcher_path=Path(__file__).resolve(),
        experiment_spec_path=(
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "feedtta_stage2.yaml"
        ),
        points_by_model=points_by_model,
        grid_metadata={
            "selected_stage1_intensity": selected,
            "official_p": list(OFFICIAL_P),
            "official_alpha": list(OFFICIAL_ALPHA),
            "controls": [
                {"variant": variant, "p": p, "alpha": alpha}
                for variant, p, alpha in CONTROLS
            ],
            "official_jobs_per_model": 35,
            "control_jobs_per_model": 4,
            "jobs_per_model": 39,
            "total_jobs": 78,
            "selection_rule": (
                "maximize SPL subject to SR >= matched Source; use SR then "
                "lower drift as tie breakers"
            ),
        },
        prerequisite_batch_id=args.stage1_batch_id,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run 78 FeedTTA Stage-2 jobs: the official 5x7 SGR grid plus "
            "four controls for each AVN model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Use each model's reviewed Stage-1 winner, for example:\n"
            "  python3 avn/scripts/run_feedtta_stage2.py "
            "--stage1-batch-id feedtta-stage1-v1-seed0 "
            "--smt-audio-lr 1e-7 --smt-audio-gamma 0.99 "
            "--enmus-lr 3e-8 --enmus-gamma 0.95 "
            "--gpus 0,1,2,3 --jobs-per-gpu 2 "
            "--batch-id feedtta-stage2-v1-seed0"
        ),
    )
    parser.add_argument(
        "--stage1-batch-id",
        required=True,
        type=batch_id,
        help="completed 48-job Stage-1 batch used to select LR/gamma",
    )
    parser.add_argument(
        "--smt-audio-lr",
        required=True,
        type=canonical_choice(STAGE1_LRS, "SMT+Audio LR"),
        help="reviewed SMT+Audio Stage-1 learning rate",
    )
    parser.add_argument(
        "--smt-audio-gamma",
        required=True,
        type=canonical_choice(STAGE1_GAMMAS, "SMT+Audio gamma"),
        help="reviewed SMT+Audio Stage-1 discount factor",
    )
    parser.add_argument(
        "--enmus-lr",
        required=True,
        type=canonical_choice(STAGE1_LRS, "ENMuS LR"),
        help="reviewed ENMuS Stage-1 learning rate",
    )
    parser.add_argument(
        "--enmus-gamma",
        required=True,
        type=canonical_choice(STAGE1_GAMMAS, "ENMuS gamma"),
        help="reviewed ENMuS Stage-1 discount factor",
    )
    add_common_arguments(parser, "feedtta-stage2-v1-seed0")
    return finalize_common_args(parser, parser.parse_args())


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(cli_main(build_spec(arguments), arguments))
