"""Active episode-level ATENA adaptation for PONI RedNet."""

import atexit
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from navtta_core.tta import ATENAAdapter
from utils.rednet_semantic_prediction import SemanticPredRedNet


@dataclass(frozen=True)
class ATENASettings:
    lr_query: float = 1e-6
    lr_self: float = 1e-7
    mix_lambda: float = 0.5
    query_threshold: float = 0.1
    self_loss_weight: float = 0.1
    param_scope: str = "all"
    optimizer: str = "AdamW"
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01
    max_grad_norm: float = 0.0
    diagnostics_path: str = ""


_SETTINGS = ATENASettings()


def configure_atena(**overrides):
    global _SETTINGS
    _SETTINGS = replace(ATENASettings(), **overrides)
    return _SETTINGS


def _forward_rednet_features(model, inputs):
    fuses = model.forward_downsample(inputs["rgb"], inputs["depth"])
    logits_map = model.forward_upsample(*fuses)
    features = F.adaptive_avg_pool2d(fuses[-1], 1).flatten(1)
    logits = logits_map.permute(0, 2, 3, 1).reshape(-1, logits_map.shape[1])
    return features, logits


class SemanticPredRedNetATENA(SemanticPredRedNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        settings = _SETTINGS
        self.atena_settings = settings
        self.atena = ATENAAdapter(
            self.model,
            lr_query=settings.lr_query,
            lr_self=settings.lr_self,
            mix_lambda=settings.mix_lambda,
            query_threshold=settings.query_threshold,
            self_loss_weight=settings.self_loss_weight,
            episodic=False,
            scope=settings.param_scope,
            optimizer_name=settings.optimizer,
            momentum=settings.momentum,
            beta1=settings.beta1,
            beta2=settings.beta2,
            weight_decay=settings.weight_decay,
            max_grad_norm=settings.max_grad_norm,
            forward_policy=_forward_rednet_features,
        )
        self._diagnostics_written = False
        if settings.diagnostics_path:
            atexit.register(self.write_atena_diagnostics)

    def get_predictions(self, batched_rgb, batched_depth):
        raw_depth = batched_depth
        with torch.no_grad():
            normalized_depth = self.normalize_depth(batched_depth)
            normalized_rgb = self.normalize_rgb(batched_rgb)
            policy_inputs = {
                "rgb": normalized_rgb.detach(),
                "depth": normalized_depth.detach(),
            }
            features, logits = _forward_rednet_features(
                self.model, policy_inputs
            )
            # RedNet has 29 classes; uint8 reduces per-step CPU trajectory
            # storage 8x, while ATENA converts it back to long for one-hot use.
            semantic_action = logits.argmax(dim=-1).to(torch.uint8)
            self.atena.adapt(
                logits,
                action=semantic_action,
                features=features,
                policy_inputs=policy_inputs,
            )
        height, width = raw_depth.shape[-2:]
        probabilities = logits.softmax(dim=-1).view(
            raw_depth.shape[0], height, width, -1
        ).permute(0, 3, 1, 2)
        return self.process_predictions(probabilities, raw_depth)

    def episode_start(self):
        self.atena.episode_start()

    def episode_end(self, episode_stats):
        self.atena.episode_end(episode_stats)

    def write_atena_diagnostics(self):
        path_value = self.atena_settings.diagnostics_path
        if not path_value or self._diagnostics_written:
            return
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": "atena",
            "adapted_component": "rednet_full_policy",
            "semantic_decision": "per_pixel_argmax",
            "episode_feature": "global_average_pooled_fuse4",
            "feedback": "active_binary_success_or_self_prediction",
            "settings": asdict(self.atena_settings),
            "selected_parameters": list(self.atena.names),
            "diagnostics": self.atena.diagnostics(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        self._diagnostics_written = True
