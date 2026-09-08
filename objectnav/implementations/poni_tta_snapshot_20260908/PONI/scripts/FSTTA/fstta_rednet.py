"""FSTTA adaptation wrapper for PONI's RedNet semantic predictor."""

import atexit
from dataclasses import asdict, dataclass, replace
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from navtta_core.tta import FSTTAAdapter
from utils.rednet_semantic_prediction import SemanticPredRedNet
from scripts.tta_rednet_utils import (
    MaskDiagnostics,
    adaptation_model,
    select_entropy_logits,
)


@dataclass(frozen=True)
class FSTTASettings:
    """NavTTA FSTTA defaults, specialized to RedNet BatchNorm."""

    lr_fast: float = 1e-6
    lr_slow: float = 1e-4
    fast_window: int = 3
    slow_window: int = 4
    q: float = 0.1
    rho: float = 0.95
    tau: float = 0.7
    scale_min: float = 0.9
    scale_max: float = 1.1
    steps: int = 1
    episodic: bool = False
    use_slow: bool = True
    reset_bn_stats: bool = True
    norm_scope: str = "bn"
    norm_prefixes: Tuple[str, ...] = ()
    entropy_mode: str = "all"
    min_adaptation_pixels: int = 128
    optimizer: str = "AdamW"
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    reset_optimizer_each_episode: bool = True
    eigen_eps: float = 1e-6
    fast_grad_mode: str = "concordant"
    use_fast_lr_scaler: bool = True
    slow_optimizer: Optional[str] = None
    slow_momentum: Optional[float] = None
    reset_slow_optimizer_each_window: bool = False
    diagnostics_path: Optional[str] = None


_SETTINGS = FSTTASettings()


def configure_fstta(**overrides) -> FSTTASettings:
    """Set process-local FSTTA options before PONI builds GlobalAgent."""

    global _SETTINGS
    _SETTINGS = replace(FSTTASettings(), **overrides)
    return _SETTINGS


class SemanticPredRedNetFSTTA(SemanticPredRedNet):
    """PONI RedNet predictor with NavTTA fast-slow adaptation."""

    def __init__(self, cfg):
        super().__init__(cfg)
        settings = _SETTINGS
        if settings.norm_scope not in ("bn", "all"):
            raise ValueError(
                "RedNet FSTTA supports norm_scope='bn' or 'all'; got {!r}".format(
                    settings.norm_scope
                )
            )
        self.fstta_settings = settings
        self.mask_diagnostics = MaskDiagnostics()
        selected_model = adaptation_model(self.model, settings.norm_prefixes)
        self.fstta = FSTTAAdapter(
            selected_model,
            lr_fast=settings.lr_fast,
            lr_slow=settings.lr_slow,
            M=settings.fast_window,
            N=settings.slow_window,
            q=settings.q,
            rho=settings.rho,
            tau=settings.tau,
            a=settings.scale_min,
            b=settings.scale_max,
            steps=settings.steps,
            episodic=settings.episodic,
            use_slow=settings.use_slow,
            reset_bn_stats=settings.reset_bn_stats,
            scope=settings.norm_scope,
            optimizer_name=settings.optimizer,
            momentum=settings.momentum,
            beta1=settings.beta1,
            beta2=settings.beta2,
            weight_decay=settings.weight_decay,
            max_grad_norm=settings.max_grad_norm,
            reset_optimizer_each_episode=settings.reset_optimizer_each_episode,
            eigen_eps=settings.eigen_eps,
            fast_grad_mode=settings.fast_grad_mode,
            use_fast_lr_scaler=settings.use_fast_lr_scaler,
            slow_optimizer_name=settings.slow_optimizer,
            slow_momentum=settings.slow_momentum,
            reset_slow_optimizer_each_window=(
                settings.reset_slow_optimizer_each_window
            ),
        )
        self._diagnostics_written = False
        if settings.diagnostics_path:
            atexit.register(self.write_fstta_diagnostics)
        logging.info(
            "[FSTTA/RedNet] initialized with %d tensors; fast_lr=%g, "
            "slow_lr=%g, M=%d, N=%d, optimizer=%s, scope=%s",
            len(self.fstta.params),
            settings.lr_fast,
            settings.lr_slow,
            settings.fast_window,
            settings.slow_window,
            settings.optimizer,
            settings.norm_scope,
        )

    def get_predictions(self, batched_rgb, batched_depth):
        """Predict semantics and feed this observation to FSTTA."""

        raw_depth = batched_depth
        with torch.enable_grad():
            normalized_depth = self.normalize_depth(batched_depth)
            normalized_rgb = self.normalize_rgb(batched_rgb)
            logits = self.model(normalized_rgb, normalized_depth)
            pixel_logits, selected_count, total_count = select_entropy_logits(
                logits,
                raw_depth,
                mode=self.fstta_settings.entropy_mode,
                min_pixels=self.fstta_settings.min_adaptation_pixels,
            )
            if pixel_logits is not None:
                self.fstta.adapt(pixel_logits)
            self.mask_diagnostics.record(
                selected_count, total_count, pixel_logits is not None
            )

        with torch.no_grad():
            probabilities = F.softmax(logits.detach(), dim=1)
            semantic_inputs = self.process_predictions(probabilities, raw_depth)
        return semantic_inputs

    def episode_start(self):
        self.fstta.episode_start()

    def episode_end(self, episode_stats=None):
        self.fstta.episode_end(episode_stats)

    def write_fstta_diagnostics(self):
        """Persist settings and final fast/slow diagnostics."""

        path_value = self.fstta_settings.diagnostics_path
        if not path_value or self._diagnostics_written:
            return
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": "fstta",
            "adapted_component": "rednet_semantic_segmentation",
            "settings": asdict(self.fstta_settings),
            "selected_parameters": list(self.fstta.names),
            "diagnostics": self.fstta.diagnostics(),
            "mask_diagnostics": self.mask_diagnostics.as_dict(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
        self._diagnostics_written = True
