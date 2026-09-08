#!/usr/bin/env python3
"""Run PONI with EAM injected into RedNet."""

import argparse
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PONI_ROOT = Path(os.environ.get("PONI_ROOT", SCRIPT_DIR.parents[1])).resolve()
NAVTTA_ROOT = Path(os.environ.get("NAVTTA_ROOT", PONI_ROOT.parent / "NavTTA")).resolve()
for root in (SCRIPT_DIR, PONI_ROOT, PONI_ROOT / "dependencies/astar_pycpp", PONI_ROOT / "hlab", NAVTTA_ROOT / "core"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from eam_rednet import SemanticPredRedNetEAM, configure_eam  # noqa: E402
import utils.rednet_semantic_prediction as rednet_module  # noqa: E402

rednet_module.SemanticPredRedNet = SemanticPredRedNetEAM
from eam_global_agent import EAMGlobalAgent  # noqa: E402
import global_agent as global_agent_module  # noqa: E402

global_agent_module.GlobalAgent = EAMGlobalAgent
from eval_poni import run_exp  # noqa: E402


def prefixes(value):
    output = tuple(item.strip() for item in value.split(",") if item.strip())
    if not output:
        raise argparse.ArgumentTypeError("at least one module prefix is required")
    return output


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--run-type", choices=["eval"], default="eval")
    p.add_argument("--exp-config", required=True)
    p.add_argument("--eam-lr", type=float, default=1e-5)
    p.add_argument("--eam-confidence-scale", type=float, default=0.4)
    p.add_argument("--eam-aux-weight", type=float, default=0.5)
    p.add_argument("--eam-memory-size", type=int, default=32)
    p.add_argument("--eam-batch-size", type=int, default=8)
    p.add_argument("--eam-update-interval", type=int, default=1)
    p.add_argument("--eam-inference-interval", type=int, default=1)
    p.add_argument("--eam-max-reliable-pixels", type=int, default=-1)
    p.add_argument(
        "--eam-deployment-scope",
        choices=["selected", "all_valid"],
        default="selected",
    )
    p.add_argument(
        "--eam-pixel-selection",
        choices=["low_entropy", "mixed_disagreement"],
        default="low_entropy",
    )
    p.add_argument("--eam-trainable-prefixes", type=prefixes, default=EAMSettings.trainable_prefixes)
    p.add_argument("--eam-optimizer", choices=["Adam", "AdamW", "SGD"], default="Adam")
    p.add_argument("--eam-momentum", type=float, default=0.9)
    p.add_argument("--eam-beta1", type=float, default=0.9)
    p.add_argument("--eam-beta2", type=float, default=0.999)
    p.add_argument("--eam-weight-decay", type=float, default=0.0)
    p.add_argument("--eam-max-grad-norm", type=float, default=0.0)
    p.add_argument("--eam-diagnostics", default="")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p


# Imported after parser declaration to keep the defaults in one place.
from eam_rednet import EAMSettings  # noqa: E402


def main():
    args = parser().parse_args()
    configure_eam(
        lr=args.eam_lr,
        confidence_scale=args.eam_confidence_scale,
        aux_weight=args.eam_aux_weight,
        memory_size=args.eam_memory_size,
        batch_size=args.eam_batch_size,
        update_interval=args.eam_update_interval,
        inference_interval=args.eam_inference_interval,
        max_reliable_pixels=args.eam_max_reliable_pixels,
        deployment_scope=args.eam_deployment_scope,
        pixel_selection=args.eam_pixel_selection,
        trainable_prefixes=args.eam_trainable_prefixes,
        optimizer=args.eam_optimizer,
        momentum=args.eam_momentum,
        beta1=args.eam_beta1,
        beta2=args.eam_beta2,
        weight_decay=args.eam_weight_decay,
        max_grad_norm=args.eam_max_grad_norm,
        diagnostics_path=args.eam_diagnostics,
    )
    run_exp(args.exp_config, args.run_type, args.opts)


if __name__ == "__main__":
    main()
