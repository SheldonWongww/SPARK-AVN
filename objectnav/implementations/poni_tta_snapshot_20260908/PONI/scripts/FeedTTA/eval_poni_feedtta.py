#!/usr/bin/env python3
"""Run PONI with FeedTTA and terminal success feedback."""

import argparse
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PONI_ROOT = Path(os.environ.get("PONI_ROOT", SCRIPT_DIR.parents[1])).resolve()
NAVTTA_ROOT = Path(os.environ.get("NAVTTA_ROOT", PONI_ROOT.parent / "NavTTA")).resolve()
for root in (SCRIPT_DIR, PONI_ROOT / "scripts", PONI_ROOT, PONI_ROOT / "dependencies/astar_pycpp", PONI_ROOT / "hlab", NAVTTA_ROOT / "core"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from feedtta_rednet import FeedTTASettings, SemanticPredRedNetFeedTTA, configure_feedtta  # noqa: E402
import utils.rednet_semantic_prediction as rednet_module  # noqa: E402

rednet_module.SemanticPredRedNet = SemanticPredRedNetFeedTTA
from feedtta_global_agent import FeedTTAGlobalAgent  # noqa: E402
import global_agent as global_agent_module  # noqa: E402

global_agent_module.GlobalAgent = FeedTTAGlobalAgent
import transfer_evaluator  # noqa: E402
from tta_feedback_evaluator import install_feedback_env_hook  # noqa: E402

install_feedback_env_hook(transfer_evaluator)
from eval_poni import run_exp  # noqa: E402


def boolean(value):
    value = value.lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def prefixes(value):
    output = tuple(item.strip() for item in value.split(",") if item.strip())
    if not output:
        raise argparse.ArgumentTypeError("at least one module prefix is required")
    return output


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--run-type", choices=["eval"], default="eval")
    p.add_argument("--exp-config", required=True)
    p.add_argument("--feedtta-lr", type=float, default=5e-6)
    p.add_argument("--feedtta-p", type=float, default=0.05)
    p.add_argument("--feedtta-alpha", type=float, default=-0.2)
    p.add_argument("--feedtta-sgr-seed", type=int, default=0)
    p.add_argument("--feedtta-gamma", type=float, default=0.99)
    p.add_argument("--feedtta-normalize-gradient", type=boolean, default=False)
    p.add_argument("--feedtta-trainable-prefixes", type=prefixes, default=FeedTTASettings.trainable_prefixes)
    p.add_argument("--feedtta-optimizer", choices=["Adam", "AdamW", "SGD"], default="Adam")
    p.add_argument("--feedtta-momentum", type=float, default=0.9)
    p.add_argument("--feedtta-beta1", type=float, default=0.9)
    p.add_argument("--feedtta-beta2", type=float, default=0.999)
    p.add_argument("--feedtta-weight-decay", type=float, default=0.0)
    p.add_argument("--feedtta-eps", type=float, default=1e-5)
    p.add_argument("--feedtta-max-grad-norm", type=float, default=0.0)
    p.add_argument("--feedtta-diagnostics", default="")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p


def main():
    args = parser().parse_args()
    configure_feedtta(
        lr=args.feedtta_lr,
        reversal_probability=args.feedtta_p,
        reversal_scale=args.feedtta_alpha,
        sgr_seed=args.feedtta_sgr_seed,
        gamma=args.feedtta_gamma,
        normalize_gradient=args.feedtta_normalize_gradient,
        trainable_prefixes=args.feedtta_trainable_prefixes,
        optimizer=args.feedtta_optimizer,
        momentum=args.feedtta_momentum,
        beta1=args.feedtta_beta1,
        beta2=args.feedtta_beta2,
        weight_decay=args.feedtta_weight_decay,
        optimizer_eps=args.feedtta_eps,
        max_grad_norm=args.feedtta_max_grad_norm,
        diagnostics_path=args.feedtta_diagnostics,
    )
    run_exp(args.exp_config, args.run_type, args.opts)


if __name__ == "__main__":
    main()

