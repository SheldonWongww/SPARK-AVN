#!/usr/bin/env python3
"""Run PONI with ATENA and lazily queried terminal feedback."""

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

from atena_rednet import ATENASettings, SemanticPredRedNetATENA, configure_atena  # noqa: E402
import utils.rednet_semantic_prediction as rednet_module  # noqa: E402

rednet_module.SemanticPredRedNet = SemanticPredRedNetATENA
from atena_global_agent import ATENAGlobalAgent  # noqa: E402
import global_agent as global_agent_module  # noqa: E402

global_agent_module.GlobalAgent = ATENAGlobalAgent
import transfer_evaluator  # noqa: E402
from tta_feedback_evaluator import install_feedback_env_hook  # noqa: E402

install_feedback_env_hook(transfer_evaluator)
from eval_poni import run_exp  # noqa: E402


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--run-type", choices=["eval"], default="eval")
    p.add_argument("--exp-config", required=True)
    p.add_argument("--atena-lr-query", type=float, default=1e-6)
    p.add_argument("--atena-lr-self", type=float, default=1e-7)
    p.add_argument("--atena-mix-lambda", type=float, default=0.5)
    p.add_argument("--atena-query-threshold", type=float, default=0.1)
    p.add_argument("--atena-self-loss-weight", type=float, default=0.1)
    p.add_argument("--atena-param-scope", choices=["all", "bn"], default="all")
    p.add_argument("--atena-optimizer", choices=["Adam", "AdamW", "SGD"], default="AdamW")
    p.add_argument("--atena-momentum", type=float, default=0.9)
    p.add_argument("--atena-beta1", type=float, default=0.9)
    p.add_argument("--atena-beta2", type=float, default=0.999)
    p.add_argument("--atena-weight-decay", type=float, default=0.01)
    p.add_argument("--atena-max-grad-norm", type=float, default=0.0)
    p.add_argument("--atena-diagnostics", default="")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p


def main():
    args = parser().parse_args()
    configure_atena(
        lr_query=args.atena_lr_query,
        lr_self=args.atena_lr_self,
        mix_lambda=args.atena_mix_lambda,
        query_threshold=args.atena_query_threshold,
        self_loss_weight=args.atena_self_loss_weight,
        param_scope=args.atena_param_scope,
        optimizer=args.atena_optimizer,
        momentum=args.atena_momentum,
        beta1=args.atena_beta1,
        beta2=args.atena_beta2,
        weight_decay=args.atena_weight_decay,
        max_grad_norm=args.atena_max_grad_norm,
        diagnostics_path=args.atena_diagnostics,
    )
    run_exp(args.exp_config, args.run_type, args.opts)


if __name__ == "__main__":
    main()

