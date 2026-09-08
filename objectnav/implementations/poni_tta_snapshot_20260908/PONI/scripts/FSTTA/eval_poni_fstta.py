#!/usr/bin/env python3
"""Run PONI evaluation with NavTTA FSTTA injected into RedNet."""

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
    from fstta_rednet import SemanticPredRedNetFSTTA, configure_fstta
    import utils.rednet_semantic_prediction as rednet_module
except ImportError as error:
    raise SystemExit(
        "Could not import PONI/NavTTA dependencies. Set PONI_ROOT and "
        "NAVTTA_ROOT to the repository roots. Original error: {}".format(error)
    )

rednet_module.SemanticPredRedNet = SemanticPredRedNetFSTTA

from fstta_global_agent import FSTTAGlobalAgent  # noqa: E402
import global_agent as global_agent_module  # noqa: E402

global_agent_module.GlobalAgent = FSTTAGlobalAgent

from eval_poni import run_exp  # noqa: E402


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("expected true or false, got {!r}".format(value))


def optional_string(value):
    value = str(value).strip()
    return value or None


def optional_float(value):
    value = float(value)
    return None if value < 0.0 else value


def parse_prefixes(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate PONI with FSTTA adaptation on RedNet."
    )
    parser.add_argument("--run-type", choices=["eval"], default="eval")
    parser.add_argument("--exp-config", required=True)
    parser.add_argument("--fstta-lr-fast", type=float, default=1e-6)
    parser.add_argument("--fstta-lr-slow", type=float, default=1e-4)
    parser.add_argument("--fstta-m", type=int, default=3)
    parser.add_argument("--fstta-n", type=int, default=4)
    parser.add_argument("--fstta-q", type=float, default=0.1)
    parser.add_argument("--fstta-rho", type=float, default=0.95)
    parser.add_argument("--fstta-tau", type=float, default=0.7)
    parser.add_argument("--fstta-a", type=float, default=0.9)
    parser.add_argument("--fstta-b", type=float, default=1.1)
    parser.add_argument("--fstta-steps", type=int, default=1)
    parser.add_argument("--fstta-episodic", type=parse_bool, default=False)
    parser.add_argument("--fstta-use-slow", type=parse_bool, default=True)
    parser.add_argument("--fstta-reset-bn-stats", type=parse_bool, default=True)
    parser.add_argument("--fstta-norm-scope", choices=["bn", "all"], default="bn")
    parser.add_argument("--fstta-norm-prefixes", type=parse_prefixes, default=())
    parser.add_argument(
        "--fstta-entropy-mode",
        choices=["all", "valid_depth", "foreground_valid_depth"],
        default="all",
    )
    parser.add_argument("--fstta-min-adaptation-pixels", type=int, default=128)
    parser.add_argument(
        "--fstta-optimizer", choices=["Adam", "AdamW", "SGD"], default="AdamW"
    )
    parser.add_argument("--fstta-momentum", type=float, default=0.9)
    parser.add_argument("--fstta-beta1", type=float, default=0.9)
    parser.add_argument("--fstta-beta2", type=float, default=0.99)
    parser.add_argument("--fstta-weight-decay", type=float, default=0.0)
    parser.add_argument("--fstta-max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--fstta-reset-optimizer-each-episode", type=parse_bool, default=True
    )
    parser.add_argument("--fstta-eigen-eps", type=float, default=1e-6)
    parser.add_argument(
        "--fstta-fast-grad-mode",
        choices=["concordant", "mean", "last"],
        default="concordant",
    )
    parser.add_argument("--fstta-use-fast-lr-scaler", type=parse_bool, default=True)
    parser.add_argument("--fstta-slow-optimizer", type=optional_string, default=None)
    parser.add_argument("--fstta-slow-momentum", type=optional_float, default=None)
    parser.add_argument(
        "--fstta-reset-slow-optimizer-each-window",
        type=parse_bool,
        default=False,
    )
    parser.add_argument("--fstta-diagnostics", default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser


def main():
    args = build_parser().parse_args()
    configure_fstta(
        lr_fast=args.fstta_lr_fast,
        lr_slow=args.fstta_lr_slow,
        fast_window=args.fstta_m,
        slow_window=args.fstta_n,
        q=args.fstta_q,
        rho=args.fstta_rho,
        tau=args.fstta_tau,
        scale_min=args.fstta_a,
        scale_max=args.fstta_b,
        steps=args.fstta_steps,
        episodic=args.fstta_episodic,
        use_slow=args.fstta_use_slow,
        reset_bn_stats=args.fstta_reset_bn_stats,
        norm_scope=args.fstta_norm_scope,
        norm_prefixes=args.fstta_norm_prefixes,
        entropy_mode=args.fstta_entropy_mode,
        min_adaptation_pixels=args.fstta_min_adaptation_pixels,
        optimizer=args.fstta_optimizer,
        momentum=args.fstta_momentum,
        beta1=args.fstta_beta1,
        beta2=args.fstta_beta2,
        weight_decay=args.fstta_weight_decay,
        max_grad_norm=args.fstta_max_grad_norm,
        reset_optimizer_each_episode=args.fstta_reset_optimizer_each_episode,
        eigen_eps=args.fstta_eigen_eps,
        fast_grad_mode=args.fstta_fast_grad_mode,
        use_fast_lr_scaler=args.fstta_use_fast_lr_scaler,
        slow_optimizer=args.fstta_slow_optimizer,
        slow_momentum=args.fstta_slow_momentum,
        reset_slow_optimizer_each_window=(
            args.fstta_reset_slow_optimizer_each_window
        ),
        diagnostics_path=args.fstta_diagnostics,
    )
    run_exp(args.exp_config, args.run_type, args.opts)


if __name__ == "__main__":
    main()
