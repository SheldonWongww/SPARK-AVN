"""Episode-feedback FeedTTA adaptation for PONI RedNet."""

import atexit
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Tuple

import torch

from navtta_core.tta import FEEDTTAAdapter
from utils.rednet_semantic_prediction import SemanticPredRedNet


@dataclass(frozen=True)
class FeedTTASettings:
    lr: float = 5e-6
    reversal_probability: float = 0.05
    reversal_scale: float = -0.2
    sgr_seed: int = 0
    gamma: float = 0.99
    normalize_gradient: bool = False
    trainable_prefixes: Tuple[str, ...] = (
        "deconv4", "agant0", "final_conv", "final_deconv_custom"
    )
    optimizer: str = "Adam"
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    optimizer_eps: float = 1e-5
    max_grad_norm: float = 0.0
    diagnostics_path: str = ""


_SETTINGS = FeedTTASettings()


def configure_feedtta(**overrides):
    global _SETTINGS
    _SETTINGS = replace(FeedTTASettings(), **overrides)
    return _SETTINGS


class SemanticPredRedNetFeedTTA(SemanticPredRedNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        settings = _SETTINGS
        self.feedtta_settings = settings
        self.feedtta = FEEDTTAAdapter(
            self.model,
            lr=settings.lr,
            reversal_probability=settings.reversal_probability,
            reversal_scale=settings.reversal_scale,
            sgr_seed=settings.sgr_seed,
            gamma=settings.gamma,
            normalize_gradient=settings.normalize_gradient,
            episodic=False,
            scope="module_prefixes",
            trainable_prefixes=settings.trainable_prefixes,
            optimizer_name=settings.optimizer,
            momentum=settings.momentum,
            beta1=settings.beta1,
            beta2=settings.beta2,
            weight_decay=settings.weight_decay,
            optimizer_eps=settings.optimizer_eps,
            max_grad_norm=settings.max_grad_norm,
            action_selection_protocol="semantic_pixel_argmax",
        )
        self._diagnostics_written = False
        if settings.diagnostics_path:
            atexit.register(self.write_feedtta_diagnostics)

    def get_predictions(self, batched_rgb, batched_depth):
        raw_depth = batched_depth
        with torch.enable_grad():
            normalized_depth = self.normalize_depth(batched_depth)
            normalized_rgb = self.normalize_rgb(batched_rgb)
            logits_map = self.model(normalized_rgb, normalized_depth)
            logits = logits_map.permute(0, 2, 3, 1).reshape(
                -1, logits_map.shape[1]
            )
            semantic_action = logits.detach().argmax(dim=-1).to(torch.uint8)
            self.feedtta.adapt(logits, action=semantic_action)
        probabilities = logits.detach().softmax(dim=-1).view(
            logits_map.shape[0], logits_map.shape[2], logits_map.shape[3], -1
        ).permute(0, 3, 1, 2)
        with torch.no_grad():
            return self.process_predictions(probabilities, raw_depth)

    def episode_start(self):
        self.feedtta.episode_start()

    def episode_end(self, episode_stats):
        self.feedtta.episode_end(episode_stats)

    def write_feedtta_diagnostics(self):
        path_value = self.feedtta_settings.diagnostics_path
        if not path_value or self._diagnostics_written:
            return
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": "feedtta",
            "adapted_component": "rednet_late_decoder",
            "semantic_decision": "per_pixel_argmax",
            "feedback": "binary_navigation_success",
            "settings": asdict(self.feedtta_settings),
            "selected_parameters": list(self.feedtta.names),
            "diagnostics": self.feedtta.diagnostics(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        self._diagnostics_written = True
