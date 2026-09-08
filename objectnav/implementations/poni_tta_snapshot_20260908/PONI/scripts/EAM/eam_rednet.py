"""EAM source/auxiliary adaptation for PONI RedNet."""

import atexit
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Tuple

import torch

from navtta_core.tta import EAMAdapter
from utils.rednet_semantic_prediction import SemanticPredRedNet


@dataclass(frozen=True)
class EAMSettings:
    lr: float = 1e-5
    confidence_scale: float = 0.4
    aux_weight: float = 0.5
    memory_size: int = 32
    batch_size: int = 8
    update_interval: int = 1
    inference_interval: int = 1
    max_reliable_pixels: int = -1
    deployment_scope: str = "selected"
    pixel_selection: str = "low_entropy"
    trainable_prefixes: Tuple[str, ...] = (
        "deconv4", "agant0", "final_conv", "final_deconv_custom"
    )
    optimizer: str = "Adam"
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    max_grad_norm: float = 0.0
    diagnostics_path: str = ""


_SETTINGS = EAMSettings()


def configure_eam(**overrides):
    global _SETTINGS
    _SETTINGS = replace(EAMSettings(), **overrides)
    return _SETTINGS


def _forward_rednet(model, inputs):
    logits = model(inputs["rgb"], inputs["depth"])
    logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    indices = inputs.get("pixel_indices")
    return logits if indices is None else logits.index_select(0, indices.long())


def _training_pixel_indices(
    logits_map,
    raw_depth,
    limit,
    strategy="low_entropy",
    auxiliary_logits=None,
):
    total = logits_map.shape[0] * logits_map.shape[2] * logits_map.shape[3]
    if int(limit) < 0 or int(limit) >= total:
        return None
    flat = logits_map.permute(0, 2, 3, 1).reshape(-1, logits_map.shape[1])
    depth = raw_depth[:, 0]
    valid = (depth >= 1.0) & (depth <= 5.0)
    source_classes = logits_map.argmax(dim=1)
    foreground = source_classes != 0
    auxiliary_flat = auxiliary_logits
    if auxiliary_flat is not None:
        auxiliary_classes = auxiliary_flat.argmax(dim=-1).view_as(source_classes)
        foreground = foreground | (auxiliary_classes != 0)
    candidates = (valid & foreground).reshape(-1).nonzero().view(-1)
    if candidates.numel() == 0:
        candidates = valid.reshape(-1).nonzero().view(-1)
    if candidates.numel() == 0:
        candidates = torch.arange(total, device=flat.device)
    count = min(int(limit), int(candidates.numel()))
    entropy = -(
        flat.softmax(dim=-1) * flat.log_softmax(dim=-1)
    ).sum(dim=-1)
    if strategy == "low_entropy":
        reliable = torch.topk(
            entropy.index_select(0, candidates), count, largest=False
        ).indices
        return candidates.index_select(0, reliable)
    if strategy != "mixed_disagreement":
        raise ValueError("Unknown EAM pixel selection {!r}".format(strategy))
    if auxiliary_flat is None:
        raise ValueError(
            "mixed_disagreement selection requires full auxiliary logits"
        )

    low_count = max(1, count // 2)
    low_local = torch.topk(
        entropy.index_select(0, candidates), low_count, largest=False
    ).indices
    low_indices = candidates.index_select(0, low_local)
    informative_count = count - low_count
    if informative_count == 0:
        return low_indices

    available = torch.ones(total, dtype=torch.bool, device=flat.device)
    available[low_indices] = False
    informative_candidates = candidates[available.index_select(0, candidates)]
    source_probs = flat.softmax(dim=-1)
    auxiliary_probs = auxiliary_flat.softmax(dim=-1)
    disagreement = (
        flat.argmax(dim=-1) != auxiliary_flat.argmax(dim=-1)
    ).to(source_probs.dtype)
    probability_gap = (source_probs - auxiliary_probs).abs().sum(dim=-1)
    # A class disagreement always outranks a same-class probability shift.
    priority = 3.0 * disagreement + probability_gap
    chosen_local = torch.topk(
        priority.index_select(0, informative_candidates),
        min(informative_count, int(informative_candidates.numel())),
        largest=True,
    ).indices
    informative = informative_candidates.index_select(0, chosen_local)
    return torch.cat((low_indices, informative), dim=0)


class SemanticPredRedNetEAM(SemanticPredRedNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        settings = _SETTINGS
        if settings.update_interval < 1:
            raise ValueError("EAM update_interval must be positive")
        if settings.inference_interval < 1:
            raise ValueError("EAM inference_interval must be positive")
        if settings.max_reliable_pixels == 0:
            raise ValueError("EAM max_reliable_pixels must be -1 or positive")
        if settings.deployment_scope not in ("selected", "all_valid"):
            raise ValueError("EAM deployment_scope must be selected or all_valid")
        if settings.pixel_selection not in ("low_entropy", "mixed_disagreement"):
            raise ValueError(
                "EAM pixel_selection must be low_entropy or mixed_disagreement"
            )
        if (
            settings.pixel_selection == "mixed_disagreement"
            and settings.deployment_scope != "all_valid"
        ):
            raise ValueError(
                "mixed_disagreement requires deployment_scope=all_valid"
            )
        self.eam_settings = settings
        self.eam = EAMAdapter(
            self.model,
            lr=settings.lr,
            confidence_scale=settings.confidence_scale,
            aux_weight=settings.aux_weight,
            memory_size=settings.memory_size,
            batch_size=settings.batch_size,
            # Navigation-step subsampling below owns the public interval. Every
            # sampled step is a native EAM update opportunity.
            update_interval=1,
            param_scope="module_prefixes",
            trainable_prefixes=settings.trainable_prefixes,
            optimizer_name=settings.optimizer,
            momentum=settings.momentum,
            beta1=settings.beta1,
            beta2=settings.beta2,
            weight_decay=settings.weight_decay,
            max_grad_norm=settings.max_grad_norm,
            episodic=False,
            forward_policy=_forward_rednet,
        )
        self._diagnostics_written = False
        self.navigation_steps = 0
        self.sampled_steps = 0
        self.deployment_steps = 0
        self.source_only_steps = 0
        self.total_valid_step_pixels = 0
        self.deployment_pixels = 0
        self.deployment_aux_gate_pixels = 0
        self.argmax_flip_pixels = 0
        self.confidence_crossing_pixels = 0
        self.semantic_map_changed_pixels = 0
        if settings.diagnostics_path:
            atexit.register(self.write_eam_diagnostics)

    def get_predictions(self, batched_rgb, batched_depth):
        self.navigation_steps += 1
        raw_depth = batched_depth
        with torch.no_grad():
            normalized_depth = self.normalize_depth(batched_depth)
            normalized_rgb = self.normalize_rgb(batched_rgb)
        with torch.no_grad():
            source_map = self.model(normalized_rgb, normalized_depth)
            source_all = source_map.permute(0, 2, 3, 1).reshape(
                -1, source_map.shape[1]
            )
        valid_depth = (raw_depth[:, 0] >= 1.0) & (raw_depth[:, 0] <= 5.0)
        valid_flat = valid_depth.reshape(-1)
        self.total_valid_step_pixels += int(valid_flat.sum().item())

        should_deploy = (
            (self.navigation_steps - 1) % self.eam_settings.inference_interval == 0
        )
        should_update = (
            (self.navigation_steps - 1) % self.eam_settings.update_interval == 0
        )

        output_logits = source_all.detach().clone()
        deployment_auxiliary_all = None
        if should_deploy:
            self.deployment_steps += 1
            if self.eam_settings.deployment_scope == "all_valid":
                deployment_indices = None
            else:
                deployment_indices = _training_pixel_indices(
                    source_map,
                    raw_depth,
                    self.eam_settings.max_reliable_pixels,
                )
            deployment_inputs = {
                "rgb": normalized_rgb.detach(),
                "depth": normalized_depth.detach(),
                "pixel_indices": deployment_indices,
            }
            deployment_source = (
                source_all if deployment_indices is None
                else source_all.index_select(0, deployment_indices.long())
            )
            combined, use_aux, deployment_auxiliary = (
                self.eam.combine_for_inference(
                    deployment_source,
                    policy_inputs=deployment_inputs,
                )
            )
            if deployment_indices is None:
                output_logits = combined.detach()
                deployment_auxiliary_all = deployment_auxiliary.detach()
                comparison_mask = valid_flat
                source_compared = source_all[comparison_mask]
                output_compared = output_logits[comparison_mask]
                gate_compared = use_aux[comparison_mask]
            else:
                output_logits.index_copy_(
                    0, deployment_indices.long(), combined.detach()
                )
                selected_valid = valid_flat.index_select(
                    0, deployment_indices.long()
                )
                source_compared = deployment_source[selected_valid]
                output_compared = combined.detach()[selected_valid]
                gate_compared = use_aux[selected_valid]
            self.deployment_pixels += int(source_compared.shape[0])
            self.deployment_aux_gate_pixels += int(gate_compared.sum().item())
            self.argmax_flip_pixels += int(
                (
                    source_compared.argmax(dim=-1)
                    != output_compared.argmax(dim=-1)
                ).sum().item()
            )
            source_confident = source_compared.softmax(dim=-1).amax(dim=-1) >= 0.9
            output_confident = output_compared.softmax(dim=-1).amax(dim=-1) >= 0.9
            self.confidence_crossing_pixels += int(
                (source_confident != output_confident).sum().item()
            )
        else:
            self.source_only_steps += 1

        if should_update:
            self.sampled_steps += 1
            indices = _training_pixel_indices(
                source_map,
                raw_depth,
                self.eam_settings.max_reliable_pixels,
                strategy=self.eam_settings.pixel_selection,
                auxiliary_logits=deployment_auxiliary_all,
            )
            policy_inputs = {
                "rgb": normalized_rgb.detach(),
                "depth": normalized_depth.detach(),
                "pixel_indices": indices,
            }
            self.eam.before_inference(policy_inputs=policy_inputs)
            source_logits = (
                source_all if indices is None
                else source_all.index_select(0, indices.long())
            )
            with torch.enable_grad():
                training_combined = self.eam.prepare_action(
                    source_logits, policy_inputs=policy_inputs
                )
                semantic_action = (
                    training_combined.detach().argmax(dim=-1).to(torch.uint8)
                )
                self.eam.adapt(training_combined, action=semantic_action)

        source_probabilities = source_all.softmax(dim=-1).view(
            source_map.shape[0], source_map.shape[2], source_map.shape[3], -1
        ).permute(0, 3, 1, 2)
        probabilities = output_logits.softmax(dim=-1).view(
            source_map.shape[0], source_map.shape[2], source_map.shape[3], -1
        ).permute(0, 3, 1, 2)
        with torch.no_grad():
            source_semantic = self.process_predictions(
                source_probabilities, raw_depth
            )
            output_semantic = self.process_predictions(probabilities, raw_depth)
            changed = (source_semantic != output_semantic).any(dim=1) & valid_depth
            self.semantic_map_changed_pixels += int(changed.sum().item())
            return output_semantic

    def episode_start(self):
        self.eam.episode_start()

    def episode_end(self, episode_stats=None):
        self.eam.episode_end(episode_stats)

    def write_eam_diagnostics(self):
        path_value = self.eam_settings.diagnostics_path
        if not path_value or self._diagnostics_written:
            return
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": "eam",
            "adapted_component": "rednet_late_decoder_auxiliary_branch",
            "semantic_decision": "per_pixel_class",
            "settings": asdict(self.eam_settings),
            "selected_parameters": list(self.eam.names),
            "diagnostics": self.eam.diagnostics(),
            "navigation_step_diagnostics": {
                "navigation_steps": self.navigation_steps,
                "sampled_eam_steps": self.sampled_steps,
                "deployment_steps": self.deployment_steps,
                "source_only_steps": self.source_only_steps,
                "inference_interval": self.eam_settings.inference_interval,
                "update_interval": self.eam_settings.update_interval,
                "deployment_scope": self.eam_settings.deployment_scope,
                "pixel_selection": self.eam_settings.pixel_selection,
                "total_valid_step_pixels": self.total_valid_step_pixels,
                "deployment_pixels": self.deployment_pixels,
                "deployment_aux_gate_pixels": self.deployment_aux_gate_pixels,
                "argmax_flip_pixels": self.argmax_flip_pixels,
                "confidence_crossing_pixels": self.confidence_crossing_pixels,
                "semantic_map_changed_pixels": self.semantic_map_changed_pixels,
                "argmax_flip_rate": self.argmax_flip_pixels
                / max(1, self.total_valid_step_pixels),
                "confidence_crossing_rate": self.confidence_crossing_pixels
                / max(1, self.total_valid_step_pixels),
                "semantic_map_change_rate": self.semantic_map_changed_pixels
                / max(1, self.total_valid_step_pixels),
                "deployment_aux_gate_rate": self.deployment_aux_gate_pixels
                / max(1, self.deployment_pixels),
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        self._diagnostics_written = True
