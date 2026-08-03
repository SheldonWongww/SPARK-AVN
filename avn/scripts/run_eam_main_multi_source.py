#!/usr/bin/env python3
"""Run the two frozen EAM configurations on canonical multi-source AVN.

This is a main-table revalidation, not a hyperparameter search.  Each model
uses the configuration selected on the canonical single-source development
stream, while the checkpoint, dataset, seed, episode count, action protocol,
and episode order match the existing multi-source Source/Tent/FSTTA runs.
"""

from pathlib import Path
import sys

import run_eam_boundary_grid as runner


FROZEN_POINTS = {
    # Single-source EAM boundary-grid job 5.
    "smt_audio": (("1e-8", 128),),
    # Single-source EAM boundary-grid job 11.
    "enmus": (("3e-9", 128),),
}


def configure_runner() -> None:
    """Point the shared provenance-aware EAM scheduler at multi-source val."""
    root = runner.REPO_ROOT

    runner.SOURCE_SETTING = "multi_source"
    runner.EVAL_SPLIT = "val"
    runner.RESULT_ROLE = "frozen_revalidation"
    runner.EXPERIMENT_NAME = "eam_frozen_multi_source_main_v1"
    runner.PROTOCOL = "source_tent_fstta_aligned_multi_source_val_seed0"
    runner.EXPERIMENT_TITLE = "AVN EAM frozen multi-source main-table runs"
    runner.BATCH_ID_PREFIX = "eam-main-multi-v1-seed0"
    runner.RUN_TAG_PREFIX = "eammain"
    runner.GLOBAL_LOCK_DESCRIPTION = "EAM frozen multi-source scheduler"
    runner.LAUNCHER_DISPLAY = "avn/scripts/run_eam_main_multi_source.py"
    runner.REQUIRED_GPU_COUNT = 2
    runner.DEFAULT_GPUS = "0,1"
    runner.LOG_BASE = root / "avn" / "results" / "logs" / "eam_main"

    runner.LRS = ("1e-8", "3e-9")
    runner.UPDATE_INTERVALS = (128,)
    runner.POINTS_BY_MODEL = FROZEN_POINTS

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
    runner.MODEL_CONFIGS = {
        "smt_audio": (
            "ss_baselines/savi/config/tta_avn/multi_source/"
            "smt_audio_tta_test.yaml"
        ),
        "enmus": "sen_baselines/enmus/config/multi_source/enmus_tta_test.yaml",
    }
    runner.MODEL_CHECKPOINTS = {
        "smt_audio": (
            root
            / "avn"
            / "checkpoints"
            / "source"
            / "smt_audio"
            / "multi_best_val.pth"
        ),
        "enmus": (
            root
            / "avn"
            / "checkpoints"
            / "source"
            / "enmus"
            / "multi_source_best_val.pth"
        ),
    }
    runner.MODEL_CHECKPOINT_SHA256 = {
        "smt_audio": (
            "c5c039a35da13a58a8f771738208603c93d3727dbebbea0a6dbfcccd16bdddd8"
        ),
        "enmus": (
            "3b1ccc9421b8fd6b9bad8a165528fa2323161d8b3c74642be13b2a59a5ae0464"
        ),
    }
    runner.CANONICAL_DATASET_INDEX_SHA256 = (
        "45d8dbdea540e78b01b252a3958afc4657745374d45185100d731df6a6cb849d"
    )
    runner.CANONICAL_STREAM_ORDER_SHA256 = (
        "cc2f1ce8319fae6a1313750c2b1235ac39985e8d2fe6270a70fda7b12d2a6525"
    )
    runner.CANONICAL_STREAM_CONTENT_SHA256 = (
        "deab5e0c91abeb999563927b6c80c05bc6dfcbdd94b455b815bd303386918f2d"
    )

    # Include both the thin experiment definition and the shared scheduler in
    # the tracked-code/provenance gate used by formal runs.
    runner.PROVENANCE_SOURCE_FILES = (
        Path(__file__).resolve(),
        Path(runner.__file__).resolve(),
        runner.FINGERPRINT_TOOL.resolve(),
        runner.MANIFEST_VALIDATOR.resolve(),
        *(runner.MODEL_RUNNERS[model].resolve() for model in runner.MODELS),
    )


def validate_frozen_definition() -> None:
    expected = {
        "smt_audio": ("1e-8", 128),
        "enmus": ("3e-9", 128),
    }
    actual = {
        model: points[0]
        for model, points in runner.POINTS_BY_MODEL.items()
        if len(points) == 1
    }
    if actual != expected or set(runner.POINTS_BY_MODEL) != set(runner.MODELS):
        raise runner.UserError("invalid frozen EAM multi-source definition")


def main(argv=None) -> int:
    configure_runner()
    validate_frozen_definition()
    return runner.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except runner.UserError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
