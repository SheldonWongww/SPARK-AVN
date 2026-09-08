#!/usr/bin/env python3
"""Run PONI evaluation with NavTTA Tent injected into RedNet."""

import argparse
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PONI_ROOT = Path(os.environ.get("PONI_ROOT", SCRIPT_DIR.parents[1])).resolve()
NAVTTA_ROOT = Path(
    os.environ.get("NAVTTA_ROOT", PONI_ROOT.parent / "NavTTA")
).resolve()
HLAB_DIR = PONI_ROOT / "hlab"

for import_root in (
    SCRIPT_DIR,
    PONI_ROOT,
    PONI_ROOT / "dependencies" / "astar_pycpp",
    HLAB_DIR,
    NAVTTA_ROOT / "core",
):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

try:
    from tent_rednet import SemanticPredRedNetTent, configure_tent
    import utils.rednet_semantic_prediction as rednet_module
except ImportError as error:
    raise SystemExit(
        "Could not import PONI/NavTTA dependencies. Set PONI_ROOT and "
        "NAVTTA_ROOT to the two repository roots. Original error: {}".format(error)
    )

# GlobalAgent uses ``from utils.rednet_semantic_prediction import ...``.  Patch
# the source module before importing PONI's evaluator, leaving its files intact.
rednet_module.SemanticPredRedNet = SemanticPredRedNetTent

# Keep NavTTA's episode hooks accurate (including episodic reset and per-episode
# update budgets) without editing PONI's GlobalAgent or TransferEvaluator.
from tent_global_agent import TentGlobalAgent  # noqa: E402
import global_agent as global_agent_module  # noqa: E402

global_agent_module.GlobalAgent = TentGlobalAgent

from eval_poni import run_exp  # noqa: E402  (must follow the injection above)


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("expected true or false, got {!r}".format(value))


def parse_prefixes(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate PONI with continual Tent adaptation on RedNet."
    )
    parser.add_argument("--run-type", choices=["eval"], default="eval")
    parser.add_argument("--exp-config", required=True)

    # Defaults mirror NavTTA's TentAdapter/build_adapter.  RedNet uses BN, so
    # norm_scope defaults to 'bn' instead of NavTTA's transformer-oriented LN.
    parser.add_argument("--tta-lr", type=float, default=1e-6)
    parser.add_argument("--tta-steps", type=int, default=1)
    parser.add_argument("--tta-episodic", type=parse_bool, default=False)
    parser.add_argument("--tta-reset-bn-stats", type=parse_bool, default=True)
    parser.add_argument("--tta-norm-scope", choices=["bn", "all"], default="bn")
    parser.add_argument("--tta-norm-prefixes", type=parse_prefixes, default=())
    parser.add_argument(
        "--tta-entropy-mode",
        choices=["all", "valid_depth", "foreground_valid_depth"],
        default="all",
    )
    parser.add_argument("--tta-min-adaptation-pixels", type=int, default=128)
    parser.add_argument(
        "--tta-optimizer", choices=["Adam", "AdamW", "SGD"], default="Adam"
    )
    parser.add_argument("--tta-momentum", type=float, default=0.9)
    parser.add_argument("--tta-beta1", type=float, default=0.9)
    parser.add_argument("--tta-beta2", type=float, default=0.999)
    parser.add_argument("--tta-weight-decay", type=float, default=0.0)
    parser.add_argument("--tta-update-interval", type=int, default=1)
    parser.add_argument("--tta-max-updates-per-episode", type=int, default=-1)
    parser.add_argument("--tta-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--tta-diagnostics", default=None)
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="PONI config overrides, e.g. TEST_EPISODE_COUNT 1",
    )
    return parser


def main():
    args = build_parser().parse_args()
    configure_tent(
        lr=args.tta_lr,
        steps=args.tta_steps,
        episodic=args.tta_episodic,
        reset_bn_stats=args.tta_reset_bn_stats,
        norm_scope=args.tta_norm_scope,
        norm_prefixes=args.tta_norm_prefixes,
        entropy_mode=args.tta_entropy_mode,
        min_adaptation_pixels=args.tta_min_adaptation_pixels,
        optimizer=args.tta_optimizer,
        momentum=args.tta_momentum,
        beta1=args.tta_beta1,
        beta2=args.tta_beta2,
        weight_decay=args.tta_weight_decay,
        update_interval=args.tta_update_interval,
        max_updates_per_episode=args.tta_max_updates_per_episode,
        max_grad_norm=args.tta_max_grad_norm,
        diagnostics_path=args.tta_diagnostics,
    )
    run_exp(
        exp_config=args.exp_config,
        run_type=args.run_type,
        opts=args.opts,
    )


if __name__ == "__main__":
    main()
