"""Tent adaptation wrapper for PONI's RedNet semantic predictor.

This module intentionally lives outside ``hlab`` so the original PONI source
tree remains unchanged.  The evaluation entry point injects
``SemanticPredRedNetTent`` before PONI imports ``GlobalAgent``.
"""

import atexit
from dataclasses import asdict, dataclass, replace
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from navtta_core.tta import TentAdapter
from utils.rednet_semantic_prediction import SemanticPredRedNet
from scripts.tta_rednet_utils import (
    MaskDiagnostics,
    adaptation_model,
    select_entropy_logits,
)


@dataclass(frozen=True)
class TentSettings:
    """NavTTA Tent defaults, specialized to RedNet's BatchNorm layers."""

    lr: float = 1e-6
    steps: int = 1
    episodic: bool = False
    reset_bn_stats: bool = True
    norm_scope: str = "bn"
    norm_prefixes: Tuple[str, ...] = ()
    entropy_mode: str = "all"
    min_adaptation_pixels: int = 128
    optimizer: str = "Adam"
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    update_interval: int = 1
    max_updates_per_episode: int = -1
    max_grad_norm: float = 1.0
    diagnostics_path: Optional[str] = None


_SETTINGS = TentSettings()


def configure_tent(**overrides) -> TentSettings:
    """Set process-local Tent options before PONI constructs GlobalAgent."""

    global _SETTINGS
    _SETTINGS = replace(TentSettings(), **overrides)
    return _SETTINGS


class SemanticPredRedNetTent(SemanticPredRedNet):
    """PONI RedNet predictor with continual Tent entropy minimization.

    RedNet returns semantic logits as ``B x C x H x W``.  NavTTA's Tent
    adapter consumes categorical logits with classes in the last dimension,
    so pixels are flattened to ``(B*H*W) x C`` for the entropy objective.
    The semantic map used at the current navigation step comes from the same
    pre-update logits; the update affects subsequent observations.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        settings = _SETTINGS
        if settings.norm_scope not in ("bn", "all"):
            raise ValueError(
                "RedNet Tent supports norm_scope='bn' or 'all'; got {!r}".format(
                    settings.norm_scope
                )
            )

        self.tent_settings = settings
        self.mask_diagnostics = MaskDiagnostics()
        selected_model = adaptation_model(self.model, settings.norm_prefixes)
        self.tent = TentAdapter(
            selected_model,
            lr=settings.lr,
            steps=settings.steps,
            episodic=settings.episodic,
            reset_bn_stats=settings.reset_bn_stats,
            scope=settings.norm_scope,
            optimizer_name=settings.optimizer,
            momentum=settings.momentum,
            beta1=settings.beta1,
            beta2=settings.beta2,
            weight_decay=settings.weight_decay,
            update_interval=settings.update_interval,
            max_updates_per_episode=settings.max_updates_per_episode,
            max_grad_norm=settings.max_grad_norm,
        )
        self._diagnostics_written = False
        if settings.diagnostics_path:
            atexit.register(self.write_tent_diagnostics)

        logging.info(
            "[Tent/RedNet] initialized with %d tensors; lr=%g, optimizer=%s, "
            "scope=%s, episodic=%s",
            len(self.tent.params),
            settings.lr,
            settings.optimizer,
            settings.norm_scope,
            settings.episodic,
        )

    def get_predictions(self, batched_rgb, batched_depth):
        """Predict semantics and perform one Tent update for this observation."""

        raw_depth = batched_depth
        # TransferEvaluator calls GlobalAgent under torch.no_grad().  Tent must
        # explicitly re-enable autograd only around RedNet and its loss.
        with torch.enable_grad():
            normalized_depth = self.normalize_depth(batched_depth)
            normalized_rgb = self.normalize_rgb(batched_rgb)
            logits = self.model(normalized_rgb, normalized_depth)
            pixel_logits, selected_count, total_count = select_entropy_logits(
                logits,
                raw_depth,
                mode=self.tent_settings.entropy_mode,
                min_pixels=self.tent_settings.min_adaptation_pixels,
            )
            if pixel_logits is not None:
                self.tent.adapt(pixel_logits)
            self.mask_diagnostics.record(
                selected_count, total_count, pixel_logits is not None
            )

        with torch.no_grad():
            probabilities = F.softmax(logits.detach(), dim=1)
            semantic_inputs = self.process_predictions(probabilities, raw_depth)
        return semantic_inputs

    def episode_start(self):
        """Forward the navigation episode boundary to NavTTA."""

        self.tent.episode_start()

    def episode_end(self, episode_stats=None):
        """Forward the completed navigation episode to NavTTA."""

        self.tent.episode_end(episode_stats)

    def write_tent_diagnostics(self):
        """Persist adapter settings and final diagnostics at normal shutdown."""

        path_value = self.tent_settings.diagnostics_path
        if not path_value or self._diagnostics_written:
            return
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": "tent",
            "adapted_component": "rednet_semantic_segmentation",
            "settings": asdict(self.tent_settings),
            "selected_parameters": list(self.tent.names),
            "diagnostics": self.tent.diagnostics(),
            "mask_diagnostics": self.mask_diagnostics.as_dict(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
        self._diagnostics_written = True
