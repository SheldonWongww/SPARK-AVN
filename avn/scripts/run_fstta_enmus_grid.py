#!/usr/bin/env python3
"""Run the 48-job ENMuS single-source FSTTA calibration grid.

This is a model-level development experiment.  It samples a fixed 20 x 100
stream from the 20 x 900 target-like ``tta_test/.../train`` pool and must not
be reported as the final ``val`` result.  Every job keeps the complete FSTTA
method enabled; there are no FAST-only controls in this grid.
"""

from pathlib import Path
import sys

import run_fstta_explorations as runner


SUITE = "enmus_intensity"


def build_grid():
    """Return 4 x 2 x 3 x 2 complete-FSTTA configurations."""
    return [
        runner.base_job(
            SUITE,
            fast_lr=fast_lr,
            fast_window=fast_window,
            slow_lr=slow_lr,
            slow_window=slow_window,
            q="0.1",
            use_slow=True,
            fast_grad_mode="concordant",
            use_fast_lr_scaler=True,
            slow_optimizer="AdamW",
            slow_momentum="0.0",
            reset_slow_optimizer_each_window=False,
        )
        for fast_lr in ("1e-8", "3e-8", "1e-7", "3e-7")
        for fast_window in (16, 32)
        for slow_lr in ("1e-5", "3e-5", "1e-4")
        for slow_window in (32, 64)
    ]


def build_plan(selected_suite, batch_id, gpus):
    """Assign every hyperparameter value across all four physical GPUs."""
    if selected_suite not in (SUITE, "all"):
        raise RuntimeError("unsupported ENMuS grid suite: {}".format(selected_suite))
    if len(gpus) != 4:
        raise RuntimeError("the ENMuS FSTTA grid requires exactly four GPU ids")

    fast_lr_index = {
        value: index
        for index, value in enumerate(("1e-8", "3e-8", "1e-7", "3e-7"))
    }
    fast_window_index = {16: 0, 32: 1}
    slow_lr_index = {
        value: index
        for index, value in enumerate(("1e-5", "3e-5", "1e-4"))
    }
    slow_window_index = {32: 0, 64: 1}

    plan = []
    for job_id, config in enumerate(build_grid()):
        non_fast_index = (
            (
                fast_window_index[config.fast_window] * len(slow_lr_index)
                + slow_lr_index[config.slow_lr]
            )
            * len(slow_window_index)
            + slow_window_index[config.slow_window]
        )
        gpu_index = (
            fast_lr_index[config.fast_lr] + non_fast_index
        ) % len(gpus)
        plan.append(
            runner.PlannedJob(
                job_id=job_id,
                suite_job_id=job_id,
                run_tag=runner.job_tag(batch_id, job_id, config),
                gpu=gpus[gpu_index],
                config=config,
            )
        )

    if len(plan) != 48 or len({job.run_tag for job in plan}) != 48:
        raise RuntimeError("internal ENMuS FSTTA plan is invalid")
    if any(sum(job.gpu == gpu for job in plan) != 12 for gpu in gpus):
        raise RuntimeError("ENMuS FSTTA jobs are not balanced across GPUs")
    for fast_lr in fast_lr_index:
        for slow_lr in slow_lr_index:
            for gpu in gpus:
                count = sum(
                    job.config.fast_lr == fast_lr
                    and job.config.slow_lr == slow_lr
                    and job.gpu == gpu
                    for job in plan
                )
                if count != 1:
                    raise RuntimeError(
                        "ENMuS LR treatments are confounded with physical GPUs"
                    )
    return plan


def configure_runner():
    root = runner.REPO_ROOT
    enmus_root = root / "avn" / "baselines" / "enmus"
    auxiliary_root = (
        enmus_root
        / "data"
        / "pretrained_weights"
        / "semantic_audionav"
        / "enmus"
    )

    runner.MODEL = "enmus"
    runner.SOURCE_SETTING = "single_source"
    runner.EVAL_SPLIT = "train"
    runner.EXPLICIT_EVAL_SPLIT = True
    runner.EXPERIMENT_TITLE = "AVN ENMuS FSTTA model-level calibration grid"
    runner.BATCH_ID_PREFIX = "fstta-enmus-grid"
    runner.FIXED_SUITE = SUITE
    runner.ALLOW_DIRTY_OPTION = False
    runner.REQUIRED_GPU_COUNT = 4
    runner.ACTION_SELECTION = None
    runner.RUNNER = root / "avn" / "scripts" / "eval_enmus.sh"
    runner.PROVENANCE_SOURCE_FILES = (
        Path(__file__).resolve(),
        Path(runner.__file__).resolve(),
        runner.RUNNER.resolve(),
        runner.FINGERPRINT_TOOL.resolve(),
        runner.MANIFEST_VALIDATOR.resolve(),
    )
    runner.CHECKPOINT = (
        root
        / "avn"
        / "checkpoints"
        / "source"
        / "enmus"
        / "single_source_best_val.pth"
    )
    runner.DATASET = (
        root
        / "avn"
        / "data"
        / "datasets"
        / "tta_test"
        / "single_source"
        / "mp3d"
        / "v1"
        / "train"
        / "train.json.gz"
    )
    runner.LOG_BASE = root / "avn" / "results" / "logs" / "fstta_enmus_grid"
    runner.AUXILIARY_CHECKPOINTS = {
        "audio_encoder": auxiliary_root / "audio_encoder_best_val.pth",
        "visual_encoder": auxiliary_root / "visual_encoder_best_val.pth",
        "seld_encoder": auxiliary_root / "seld_crnn_best_val.h5",
    }
    runner.RUNNER_ENV = {"NAVTTA_EVAL_SPLIT": "train"}

    runner.SUITE_ORDER = (SUITE,)
    runner.EXPECTED_SUITE_COUNTS = {SUITE: 48}
    runner.EXPECTED_ALL_COUNT = 48
    runner.SUITE_SLUGS = {SUITE: "ei"}
    runner.SUITE_BUILDERS = {SUITE: build_grid}
    runner.build_plan = build_plan


def main():
    if "--allow-dirty" in sys.argv[1:]:
        raise RuntimeError(
            "ENMuS calibration runs require a clean tracked worktree"
        )
    configure_runner()
    configs = build_grid()
    combinations = {
        (
            config.fast_lr,
            config.fast_window,
            config.slow_lr,
            config.slow_window,
        )
        for config in configs
    }
    if (
        len(configs) != 48
        or len(combinations) != 48
        or any(not config.use_slow for config in configs)
    ):
        raise RuntimeError("internal ENMuS FSTTA grid definition is invalid")
    return runner.main(sys.argv[1:])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
