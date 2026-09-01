#!/usr/bin/env python3
"""Run the blank-row AVN search for one navigation model.

Usage::

    python3 avn/scripts/run_avn_blank_val_search.py smt_audio [runner options]
    python3 avn/scripts/run_avn_blank_val_search.py enmus [runner options]

The implementation reuses the audited AVN scheduler.  Each invocation owns
three method lanes (FeedTTA, ATENA, IDEA), and each lane completes
single-source before starting multi-source.
"""

from pathlib import Path
import sys

import run_smt_audio_val_search as runner


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("smt_audio", "enmus")
METHODS = ("feedtta", "atena", "idea")


def configure(model: str) -> None:
    runner.MODEL = model
    runner.METHODS = METHODS
    runner.SPEC_SCHEMA = "navtta.avn.blank_val_search.v1"
    runner.BATCH_SCHEMA = "navtta.avn.blank_val_search.batch.v1"
    runner.CAMPAIGN_TITLE = "{} AVN blank-row val search".format(model)
    runner.BATCH_ID_PREFIX = "avn-{}-blank-val-v1".format(model.replace("_", "-"))
    runner.RUN_TAG_PREFIX = "avnb1-{}".format(
        "smt" if model == "smt_audio" else "enmus"
    )
    runner.DEFAULT_SPEC = ROOT / "avn/experiments/{}_blank_val_search_v1.json".format(model)
    runner.LOG_ROOT = ROOT / "avn/results/logs/blank_val_search" / model

    runner.DATASETS = {
        setting: ROOT / (
            "avn/data/datasets/tta_test/{}/mp3d/v1/val/val.json.gz".format(
                setting
            )
        )
        for setting in runner.SOURCE_SETTINGS
    }
    runner.SOURCE_DATASETS = {
        setting: ROOT / (
            "avn/data/datasets/train/{}/mp3d/v1/train/train.json.gz".format(
                setting
            )
        )
        for setting in runner.SOURCE_SETTINGS
    }

    common_runtime = (
        Path(__file__).resolve(),
        Path(runner.__file__).resolve(),
        runner.FINGERPRINT_TOOL,
        runner.MANIFEST_VALIDATOR,
        ROOT / "avn/scripts/build_idea_source_manifest.py",
        ROOT / "avn/navtta_avn/idea_source.py",
        ROOT / "core/navtta_core/tta/tta_core.py",
        ROOT / "core/navtta_core/tta/idea.py",
        ROOT / "core/navtta_core/tta/fusion.py",
    )

    if model == "smt_audio":
        runner.RUNNER = ROOT / "avn/scripts/eval_smt_audio.sh"
        runner.BASELINE_ROOT = ROOT / "avn/baselines/smt_audio"
        runner.MODEL_CONFIG = (
            "ss_baselines/savi/config/tta_avn/{}/smt_audio_tta_test.yaml"
        )
        runner.CHECKPOINTS = {
            "single_source": ROOT / "avn/checkpoints/source/smt_audio/single_best_val.pth",
            "multi_source": ROOT / "avn/checkpoints/source/smt_audio/multi_best_val.pth",
        }
        runner.IDEA_COLLECTOR = ROOT / "avn/scripts/collect_smt_audio_idea_source_stats.sh"
        runner.IDEA_MANIFESTS = {
            setting: ROOT / (
                "avn/manifests/idea_source/smt_audio_{}_sample_seed0.json".format(
                    setting
                )
            )
            for setting in runner.SOURCE_SETTINGS
        }
        runner.IDEA_STATS = {
            setting: ROOT / (
                "avn/results/idea_source_statistics/"
                "smt_audio_{}_sample_seed0.json".format(setting)
            )
            for setting in runner.SOURCE_SETTINGS
        }
        runner.REQUIRE_PINNED_IDEA_MANIFESTS = True
        runner.AUXILIARY_CHECKPOINTS = ()
        runner.RUNTIME_FILES = common_runtime + (
            runner.RUNNER,
            runner.IDEA_COLLECTOR,
            ROOT / "avn/baselines/smt_audio/ss_baselines/savi/ppo/ppo_trainer.py",
            ROOT / "avn/baselines/smt_audio/ss_baselines/savi/config/default.py",
            ROOT / "avn/baselines/smt_audio/ss_baselines/savi/config/tta_avn/single_source/smt_audio_tta_test.yaml",
            ROOT / "avn/baselines/smt_audio/ss_baselines/savi/config/tta_avn/multi_source/smt_audio_tta_test.yaml",
            ROOT / "avn/baselines/smt_audio/ss_baselines/savi/config/tta_avn/single_source/smt_audio_idea_source_stats.yaml",
            ROOT / "avn/baselines/smt_audio/ss_baselines/savi/config/tta_avn/multi_source/smt_audio_idea_source_stats.yaml",
            *runner.IDEA_MANIFESTS.values(),
        )
        return

    runner.RUNNER = ROOT / "avn/scripts/eval_enmus.sh"
    runner.BASELINE_ROOT = ROOT / "avn/baselines/enmus"
    runner.MODEL_CONFIG = "sen_baselines/enmus/config/{}/enmus_tta_test.yaml"
    runner.CHECKPOINTS = {
        "single_source": ROOT / "avn/checkpoints/source/enmus/single_source_best_val.pth",
        "multi_source": ROOT / "avn/checkpoints/source/enmus/multi_source_best_val.pth",
    }
    runner.IDEA_COLLECTOR = ROOT / "avn/scripts/collect_enmus_idea_source_stats.sh"
    runner.IDEA_MANIFESTS = {
        setting: ROOT / (
            "avn/manifests/idea_source/enmus_{}_sample_seed0.json".format(setting)
        )
        for setting in runner.SOURCE_SETTINGS
    }
    runner.IDEA_STATS = {
        setting: ROOT / (
            "avn/results/idea_source_statistics/"
            "enmus_{}_sample_seed0.json".format(setting)
        )
        for setting in runner.SOURCE_SETTINGS
    }
    auxiliary_root = (
        ROOT / "avn/baselines/enmus/data/pretrained_weights/semantic_audionav/enmus"
    )
    runner.AUXILIARY_CHECKPOINTS = (
        auxiliary_root / "audio_encoder_best_val.pth",
        auxiliary_root / "visual_encoder_best_val.pth",
        auxiliary_root / "seld_crnn_best_val.h5",
    )
    runner.REQUIRE_PINNED_IDEA_MANIFESTS = False
    runner.RUNTIME_FILES = common_runtime + (
        runner.RUNNER,
        runner.IDEA_COLLECTOR,
        ROOT / "avn/baselines/enmus/sen_baselines/enmus/ddppo/ddppo_enmus_trainer.py",
        ROOT / "avn/baselines/enmus/sen_baselines/enmus/config/default.py",
        ROOT / "avn/baselines/enmus/sen_baselines/enmus/config/single_source/enmus_tta_test.yaml",
        ROOT / "avn/baselines/enmus/sen_baselines/enmus/config/multi_source/enmus_tta_test.yaml",
        ROOT / "avn/baselines/enmus/sen_baselines/enmus/config/single_source/enmus_idea_source_stats.yaml",
        ROOT / "avn/baselines/enmus/sen_baselines/enmus/config/multi_source/enmus_idea_source_stats.yaml",
    )


def main(argv=None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] not in MODELS:
        print(
            "usage: run_avn_blank_val_search.py <smt_audio|enmus> [options]",
            file=sys.stderr,
        )
        return 2
    configure(values.pop(0))
    return runner.main(values)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except runner.UserError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
