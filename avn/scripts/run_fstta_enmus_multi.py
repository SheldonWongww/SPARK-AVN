#!/usr/bin/env python3
"""Revalidate frozen ENMuS FSTTA job 0 on the canonical multi-source stream.

This is a one-job main-table runner, not a hyperparameter search.  It reuses
the provenance-aware FSTTA scheduler while pinning the model checkpoint,
dataset index, seed-0 2,000-episode stream, and complete FAST/SLOW method
configuration selected by the ENMuS single-source calibration grid.
"""

from pathlib import Path
import sys

import run_fstta_explorations as runner


SUITE = "enmus_multi_frozen"


def build_frozen_job():
    """Return the single frozen ENMuS job-0 configuration."""
    return [
        runner.base_job(
            SUITE,
            fast_lr="1e-8",
            fast_window=16,
            slow_lr="1e-5",
            slow_window=32,
            q="0.1",
            use_slow=True,
            fast_grad_mode="concordant",
            use_fast_lr_scaler=True,
            slow_optimizer="AdamW",
            slow_momentum="0.0",
            reset_slow_optimizer_each_window=False,
        )
    ]


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
    runner.SOURCE_SETTING = "multi_source"
    runner.EVAL_SPLIT = "val"
    runner.EXPLICIT_EVAL_SPLIT = True
    runner.RESULT_ROLE = "frozen_revalidation"
    runner.EXPERIMENT_TITLE = "AVN ENMuS FSTTA frozen multi-source revalidation"
    runner.BATCH_ID_PREFIX = "fstta-enmus-multi"
    runner.FIXED_SUITE = SUITE
    runner.ALLOW_DIRTY_OPTION = False
    runner.REQUIRED_GPU_COUNT = 1
    runner.DEFAULT_GPUS = "0"
    runner.DEFAULT_JOBS_PER_GPU = 1
    # Freeze the shared FSTTA controls as well as the job-specific grid axes,
    # so later changes to the exploratory runner cannot silently alter this
    # main-table revalidation protocol.
    runner.NORM_SCOPE = "last_k_ln"
    runner.LAST_K_LN = 4
    runner.EPISODIC = False
    runner.STEPS = 1
    runner.RESET_BN_STATS = True
    runner.RHO = "0.95"
    runner.TAU = "0.7"
    runner.A = "0.9"
    runner.B = "1.1"
    runner.FAST_OPTIMIZER = "AdamW"
    runner.BETA1 = "0.9"
    runner.BETA2 = "0.99"
    runner.WEIGHT_DECAY = "0.0"
    runner.MAX_GRAD_NORM = "1.0"
    runner.RESET_FAST_OPTIMIZER_EACH_EPISODE = True
    runner.EIGEN_EPS = "1e-6"
    # ENMuS uses its native sampled-action evaluation path.
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
        / "multi_source_best_val.pth"
    )
    runner.DATASET = (
        root
        / "avn"
        / "data"
        / "datasets"
        / "tta_test"
        / "multi_source"
        / "mp3d"
        / "v1"
        / "val"
        / "val.json.gz"
    )
    runner.LOG_BASE = (
        root / "avn" / "results" / "logs" / "fstta_enmus_multi"
    )
    runner.AUXILIARY_CHECKPOINTS = {
        "audio_encoder": auxiliary_root / "audio_encoder_best_val.pth",
        "visual_encoder": auxiliary_root / "visual_encoder_best_val.pth",
        "seld_encoder": auxiliary_root / "seld_crnn_best_val.h5",
    }
    runner.EXPECTED_AUXILIARY_CHECKPOINT_SHA256 = {
        "audio_encoder": (
            "0939d0546f47b291eac35f6570c3fdb97c46dd45e0730d5e559f5696d4233b47"
        ),
        "visual_encoder": (
            "daa326623772a5fb2db3abd74413ea0bbc822f2dc66029db5a25828df95a69b5"
        ),
        "seld_encoder": (
            "7d36aa62eb6fa8043bebb8ebb8bb11a972f5fec9e039ae7260400dbdc38646db"
        ),
    }
    runner.RUNNER_ENV = {"NAVTTA_EVAL_SPLIT": "val"}
    runner.EXPECTED_CHECKPOINT_SHA256 = (
        "3b1ccc9421b8fd6b9bad8a165528fa2323161d8b3c74642be13b2a59a5ae0464"
    )
    runner.EXPECTED_DATASET_INDEX_SHA256 = (
        "45d8dbdea540e78b01b252a3958afc4657745374d45185100d731df6a6cb849d"
    )
    runner.EXPECTED_STREAM_EPISODES = 2000
    runner.EXPECTED_STREAM_ORDER_SHA256 = (
        "cc2f1ce8319fae6a1313750c2b1235ac39985e8d2fe6270a70fda7b12d2a6525"
    )
    runner.EXPECTED_STREAM_CONTENT_SHA256 = (
        "deab5e0c91abeb999563927b6c80c05bc6dfcbdd94b455b815bd303386918f2d"
    )

    runner.SUITE_ORDER = (SUITE,)
    runner.EXPECTED_SUITE_COUNTS = {SUITE: 1}
    runner.EXPECTED_ALL_COUNT = 1
    runner.SUITE_SLUGS = {SUITE: "emf"}
    runner.SUITE_BUILDERS = {SUITE: build_frozen_job}


def main():
    configure_runner()
    jobs = build_frozen_job()
    if len(jobs) != 1:
        raise RuntimeError("internal frozen ENMuS plan is not a single job")
    args = runner.parse_args(sys.argv[1:])
    if args.episodes != runner.EXPECTED_STREAM_EPISODES:
        raise RuntimeError(
            "frozen ENMuS revalidation requires exactly 2000 episodes"
        )
    return runner.main(sys.argv[1:])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
