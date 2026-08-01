#!/usr/bin/env python3
"""Stage 1: jointly calibrate FeedTTA learning rate and discount factor."""

import argparse
from pathlib import Path

from feedtta_grid_common import (
    ExperimentSpec,
    MODELS,
    SearchPoint,
    add_common_arguments,
    cli_main,
    finalize_common_args,
)


LRS = ("1e-8", "3e-8", "1e-7", "3e-7", "1e-6", "5e-6")
GAMMAS = ("0.90", "0.95", "0.99", "1.0")
FIXED_P = "0.05"
FIXED_ALPHA = "-0.2"


def build_spec() -> ExperimentSpec:
    points = tuple(
        SearchPoint(
            variant="intensity",
            lr=lr,
            gamma=gamma,
            p=FIXED_P,
            alpha=FIXED_ALPHA,
            smoke_anchor=(lr == "5e-6" and gamma == "0.99"),
        )
        for lr in LRS
        for gamma in GAMMAS
    )
    if len(points) != 24:
        raise RuntimeError("FeedTTA Stage 1 must contain exactly 24 points/model")
    return ExperimentSpec(
        stage="stage1",
        experiment="feedtta_avn_intensity_grid_v1",
        log_dir_name="feedtta_stage1",
        launcher_path=Path(__file__).resolve(),
        experiment_spec_path=(
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "feedtta_stage1.yaml"
        ),
        points_by_model={model: points for model in MODELS},
        grid_metadata={
            "learning_rates": list(LRS),
            "gammas": list(GAMMAS),
            "fixed_p": FIXED_P,
            "fixed_alpha": FIXED_ALPHA,
            "jobs_per_model": len(points),
            "total_jobs": len(points) * len(MODELS),
            "selection_rule": (
                "maximize SPL subject to SR >= matched Source; use SR then "
                "lower drift as tie breakers"
            ),
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run 48 FeedTTA Stage-1 jobs: two AVN models x six learning "
            "rates x four discount factors."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Full run:\n"
            "  python3 avn/scripts/run_feedtta_stage1.py --gpus 0,1,2,3 "
            "--jobs-per-gpu 2 --batch-id feedtta-stage1-v1-seed0\n\n"
            "Two-job pathway smoke:\n"
            "  python3 avn/scripts/run_feedtta_stage1.py --gpus 0,1,2,3 "
            "--jobs-per-gpu 1 --smoke --episodes 2 --allow-dirty "
            "--batch-id feedtta-stage1-smoke"
        ),
    )
    add_common_arguments(parser, "feedtta-stage1-v1-seed0")
    return finalize_common_args(parser, parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(cli_main(build_spec(), parse_args()))
