#!/usr/bin/env python3
"""Test-time adaptation for sequential audio-visual navigation policies.

The adapters deliberately expose episode hooks.  FSTTA's fast statistics are
computed inside one episode, while its slow update is performed after N whole
episodes (not after N optimizer steps).  Evaluation code must therefore use a
single environment and call ``episode_start`` / ``episode_end``.
"""
from copy import deepcopy
import hashlib
import logging
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


BN_LAYERS = (nn.BatchNorm1d, nn.BatchNorm2d)


def softmax_entropy(logits):
    """Per-sample Shannon entropy for categorical logits."""
    return -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)


def _make_optimizer(
    params,
    name,
    lr,
    momentum=0.9,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0,
    eps=1e-8,
):
    name = str(name).lower()
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            eps=eps,
        )
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            eps=eps,
        )
    raise ValueError("Unsupported TTA optimizer: {}".format(name))


def configure_tta_model(
    model,
    reset_bn_stats=True,
    scope="last_k_ln",
    last_k=4,
):
    """Freeze ``model`` and enable only the requested norm affine parameters.

    Supported scopes:
      * ``first_ln``: first LayerNorm module
      * ``last_ln``: last LayerNorm module
      * ``last_k_ln``: last K LayerNorm modules (recommended for AVN/FSTTA)
      * ``ln``: all LayerNorm modules
      * ``gn``: all GroupNorm modules
      * ``bn``: all BatchNorm modules
      * ``all``: LayerNorm + GroupNorm + BatchNorm

    Unlike the previous implementation, ``ln`` never includes GroupNorm.
    """
    model.eval()
    model.requires_grad_(False)
    scope = str(scope).lower()
    valid_scopes = (
        "first_ln", "last_ln", "last_k_ln", "ln", "gn", "bn", "all"
    )
    if scope not in valid_scopes:
        raise ValueError(
            "Unknown TTA norm scope {!r}; expected one of {}".format(
                scope, valid_scopes
            )
        )

    modules = list(model.named_modules())
    if scope in ("first_ln", "last_ln", "last_k_ln"):
        candidates = [(name, module) for name, module in modules
                      if isinstance(module, nn.LayerNorm)]
        if not candidates:
            raise ValueError(
                "No LayerNorm modules found for TTA scope={!r}".format(scope)
            )
        if scope == "first_ln":
            selected = candidates[:1]
        elif scope == "last_ln":
            selected = candidates[-1:]
        else:
            last_k = int(last_k)
            if last_k <= 0:
                raise ValueError("TTA.LAST_K_LN must be positive")
            if len(candidates) < last_k:
                raise ValueError(
                    "Requested last {} LayerNorm modules, but model only has {}"
                    .format(last_k, len(candidates))
                )
            selected = candidates[-last_k:]
    else:
        selected = []
        for name, module in modules:
            is_selected = (
                (scope in ("ln", "all") and isinstance(module, nn.LayerNorm))
                or (scope in ("gn", "all") and isinstance(module, nn.GroupNorm))
                or (scope in ("bn", "all") and isinstance(module, BN_LAYERS))
            )
            if is_selected:
                selected.append((name, module))

    params, names = [], []
    for module_name, module in selected:
        if isinstance(module, BN_LAYERS) and reset_bn_stats:
            module.train()
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        for param_name, param in module.named_parameters(recurse=False):
            if param_name in ("weight", "bias"):
                param.requires_grad_(True)
                params.append(param)
                names.append("{}.{}".format(module_name, param_name))

    if not params:
        raise ValueError(
            "No normalization affine parameters found for TTA scope={!r}".format(
                scope
            )
        )
    logging.info(
        "[TTA] adapting %d tensors (%d scalars): %s",
        len(params),
        sum(param.numel() for param in params),
        ", ".join(names),
    )
    return params, names


def configure_module_prefixes(model, prefixes):
    """Freeze ``model`` and enable parameters under explicit module prefixes."""
    prefixes = tuple(str(prefix).rstrip(".") for prefix in prefixes)
    if not prefixes:
        raise ValueError("At least one trainable module prefix is required")

    model.eval()
    model.requires_grad_(False)
    params, names = [], []
    for name, param in model.named_parameters():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            param.requires_grad_(True)
            params.append(param)
            names.append(name)
    if not params:
        raise ValueError(
            "No parameters matched trainable module prefixes: {}".format(prefixes)
        )
    logging.info(
        "[TTA] adapting %d tensors (%d scalars) under prefixes %s",
        len(params),
        sum(param.numel() for param in params),
        ", ".join(prefixes),
    )
    return params, names


def _flatten_params(params):
    return torch.cat([param.detach().reshape(-1) for param in params], dim=0)


def _flatten_grads(params):
    chunks = []
    for param in params:
        if param.grad is None:
            chunks.append(torch.zeros_like(param).reshape(-1))
        else:
            chunks.append(param.grad.detach().reshape(-1))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _copy_flat_to_params(flat, params):
    offset = 0
    for param in params:
        count = param.numel()
        param.copy_(flat[offset:offset + count].view_as(param))
        offset += count
    if offset != flat.numel():
        raise ValueError("Flat parameter vector has an unexpected size")


def _copy_flat_to_grads(flat, params):
    offset = 0
    for param in params:
        count = param.numel()
        grad = flat[offset:offset + count].view_as(param)
        if param.grad is None:
            param.grad = grad.clone()
        else:
            param.grad.copy_(grad)
        offset += count


def _relative_drift(current, reference):
    denom = reference.norm().clamp_min(1e-12)
    return ((current - reference).norm() / denom).item()


def _parameter_state_sha256(params, names):
    """Hash the exact selected parameter state without serialization metadata."""
    digest = hashlib.sha256()
    for name, param in zip(names, params):
        value = param.detach().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        # Viewing as bytes also supports dtypes (for example bfloat16) that
        # NumPy cannot always expose directly.
        raw = value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes(
            order="C"
        )
        digest.update(raw)
    return digest.hexdigest()


def module_state_sha256(module):
    """Hash every registered parameter and buffer, including non-persistent.

    ``state_dict`` intentionally omits buffers registered with
    ``persistent=False``.  Torch 1.9 also lacks the ``remove_duplicate``
    argument on ``named_parameters`` and ``named_buffers``, so the audit walks
    every named submodule and reads its registration tables directly.  This
    retains registered ``None`` values, tied aliases, shared-module aliases,
    and non-persistent buffers.  The digest includes state kind,
    fully-qualified name, dtype, shape, and exact bytes, so a frozen parameter,
    running statistic, or transient runtime buffer cannot change unnoticed.
    """
    digest = hashlib.sha256()
    entries = []
    seen = set()
    for module_name, submodule in module.named_modules(remove_duplicate=False):
        prefix = module_name + "." if module_name else ""
        for kind, registrations in (
            ("parameter", submodule._parameters),
            ("buffer", submodule._buffers),
        ):
            for local_name, value in registrations.items():
                name = prefix + local_name
                key = (kind, name)
                if key in seen:
                    raise RuntimeError(
                        "duplicate named {} entry {!r}".format(kind, name)
                    )
                seen.add(key)
                entries.append((kind, name, value))

    for kind, name, value in sorted(entries, key=lambda item: item[:2]):
        digest.update(kind.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        if value is None:
            digest.update(b"none\0")
            continue
        if not torch.is_tensor(value):
            raise TypeError(
                "state_dict value {!r} is not a Tensor or None".format(name)
            )
        tensor = value.detach().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes(
                order="C"
            )
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _tree_map(value, tensor_fn):
    """Apply ``tensor_fn`` to tensors in a nested policy-input structure."""
    if torch.is_tensor(value):
        return tensor_fn(value)
    if isinstance(value, dict):
        return {key: _tree_map(item, tensor_fn) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_map(item, tensor_fn) for item in value)
    if isinstance(value, list):
        return [_tree_map(item, tensor_fn) for item in value]
    return value


def _tree_to_cpu(value):
    return _tree_map(value, lambda tensor: tensor.detach().cpu().clone())


def _tree_to_device(value, device):
    return _tree_map(value, lambda tensor: tensor.to(device, non_blocking=True))


def _tree_tensor_bytes(value):
    """Return the tensor payload size of a nested policy-input snapshot."""
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tree_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tree_tensor_bytes(item) for item in value)
    return 0


def _episode_success(episode_stats):
    if callable(episode_stats):
        # Active-feedback methods may defer oracle access until after deciding
        # whether an episode is queried. The provider must return the same
        # minimal stats mapping accepted by the eager path.
        episode_stats = episode_stats()
    if episode_stats is None or "success" not in episode_stats:
        raise ValueError(
            "This TTA method requires episode_stats['success']; do not use "
            "distance_to_goal or another evaluation oracle as a substitute."
        )
    return bool(float(episode_stats["success"]) >= 0.5)


def _set_flat_grad(flat, params, max_grad_norm):
    """Install a flat gradient and optionally clip it; nonpositive disables clipping."""
    _copy_flat_to_grads(flat, params)
    limit = float(max_grad_norm)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        params, limit if limit > 0 else math.inf
    )
    return float(grad_norm.item())


def _small_row_svd(matrix):
    """Compute the FSTTA low-rank SVD on CPU and return it to ``matrix``.

    FSTTA decomposes an ``M x D`` matrix with a very small ``M``.  CUDA 11.1
    builds used by the pinned VLN-CE environment can fail while creating a
    cuSOLVER handle even for this tiny decomposition.  Moving the detached
    matrix to CPU is cheap for the selected LayerNorm parameters, avoids that
    runtime-specific failure, and gives every baseline the same LAPACK path.
    """
    work = matrix.detach().to(device="cpu", dtype=torch.float32)
    _, singular_values, vh = torch.linalg.svd(work, full_matrices=False)
    return (
        singular_values.to(device=matrix.device, dtype=matrix.dtype),
        vh.to(device=matrix.device, dtype=matrix.dtype),
    )


def _concordant_grad_and_trace(grad_list, eigen_eps=1e-6):
    """Low-rank GDA with the length calibration from FSTTA Eq. (5).

    The released full-matrix implementation clamps *every* covariance
    eigenvalue before applying inverse-variance weighting.  In the usual
    ``D >> M`` setting, most of those eigenvalues belong to the covariance
    nullspace.  A truncated SVD therefore cannot simply sum the returned
    nonzero eigendirections: doing so silently discards the mean-gradient
    component in that nullspace.  The residual below is exactly that omitted
    component, weighted by the same eigenvalue floor as a full ``D x D``
    eigendecomposition.
    """
    eigen_eps = float(eigen_eps)
    if not math.isfinite(eigen_eps) or eigen_eps <= 0.0:
        raise ValueError("FSTTA.EIGEN_EPS must be finite and positive")
    gradients = torch.stack(grad_list, dim=0)
    mean_grad = gradients.mean(dim=0)
    if len(grad_list) < 2:
        return mean_grad, torch.zeros((), device=mean_grad.device)

    centered = gradients - mean_grad.unsqueeze(0)
    # centered is M x D and M is small, so this avoids a D x D covariance.
    singular_values, vh = _small_row_svd(centered)
    eigenvalues = singular_values.square() / float(len(grad_list) - 1)
    sigma = eigenvalues.sum()
    if eigenvalues.numel() == 0:
        return mean_grad, sigma

    # Eigenvalues at or below the floor all receive 1 / eigen_eps.  This set
    # includes both numerically tiny returned directions and the unreturned
    # D-rank orthogonal complement of the small-row SVD.
    valid = eigenvalues > eigen_eps
    if bool(valid.any()):
        basis = vh[valid]                          # [rank, D]
        values = eigenvalues[valid]               # [rank]
        projections = basis @ mean_grad           # [rank]
        covariance_component = (
            projections.unsqueeze(1) * basis
        ).sum(dim=0)
        nullspace_component = mean_grad - covariance_component
        concordant = nullspace_component / eigen_eps
        concordant = concordant + (
            (projections / values).unsqueeze(1) * basis
        ).sum(dim=0)
    else:
        concordant = mean_grad / eigen_eps

    # Eq. (5): keep the new direction, restore the mean-gradient magnitude.
    concordant_norm = concordant.norm()
    mean_norm = mean_grad.norm()
    if concordant_norm <= 1e-12 or mean_norm <= 1e-12:
        return mean_grad, sigma
    concordant = concordant * (mean_norm / concordant_norm)
    return concordant, sigma


class _AdapterDiagnostics:
    # EAM keeps the deployed/source branch frozen and computes gradients only
    # through its auxiliary copy. Other adapters differentiate source logits.
    requires_source_grad = True

    def _init_diagnostics(self):
        self.action_steps = 0
        self.update_count = 0
        self.loss_sum = 0.0
        self.last_loss = 0.0
        self.last_grad_norm = 0.0
        self.current_lr = self.base_lr
        self.parameter_write_attempts = 0
        self.suppressed_parameter_write_attempts = 0
        self.optimizer_step_attempts = 0
        self.suppressed_optimizer_step_attempts = 0
        if not hasattr(self, "zero_update_audit"):
            self.zero_update_audit = False
            self.audit_parameter_state_before_sha256 = None
            self.audit_parameter_state_after_sha256 = None
            self.audit_model_states_before_sha256 = None
            self.audit_model_states_after_sha256 = None
            self.audit_expected_episodes = None
        elif self.zero_update_audit:
            self.audit_parameter_state_after_sha256 = None
            self.audit_model_states_after_sha256 = None

    def _audit_deployed_modules(self):
        """Return named deployed modules whose complete state must not change."""
        return (("primary_model", self.model),)

    def _capture_audit_model_states(self):
        modules = self._audit_deployed_modules()
        names = [name for name, _ in modules]
        if len(names) != len(set(names)) or "primary_model" not in names:
            raise RuntimeError(
                "audit deployed modules require unique names and primary_model"
            )
        return {
            name: module_state_sha256(module)
            for name, module in modules
        }

    def enable_zero_update_audit(self, expected_episodes=None):
        """Exercise adaptation while suppressing every parameter write.

        This is deliberately enabled only after an adapter has been fully
        constructed.  It is an evidence mode, not a learning-rate trick: the
        native loss, backward, replay, gating, feedback, and update scheduling
        paths still run, but writes at optimizer/copy/restore boundaries are
        counted and suppressed.
        """
        if self.action_steps or self.update_count:
            raise RuntimeError("zero-update audit must be enabled before adaptation")
        self.zero_update_audit = True
        self.parameter_write_attempts = 0
        self.suppressed_parameter_write_attempts = 0
        self.optimizer_step_attempts = 0
        self.suppressed_optimizer_step_attempts = 0
        if expected_episodes is not None and int(expected_episodes) <= 0:
            raise ValueError("audit expected episode count must be positive")
        self.audit_expected_episodes = (
            None if expected_episodes is None else int(expected_episodes)
        )
        self.audit_parameter_state_before_sha256 = _parameter_state_sha256(
            self.params, self.names
        )
        self.audit_parameter_state_after_sha256 = None
        self.audit_model_states_before_sha256 = (
            self._capture_audit_model_states()
        )
        self.audit_model_states_after_sha256 = None
        return self

    def _optimizer_step(self, optimizer):
        self.parameter_write_attempts += 1
        self.optimizer_step_attempts += 1
        if self.zero_update_audit:
            self.suppressed_parameter_write_attempts += 1
            self.suppressed_optimizer_step_attempts += 1
            return False
        optimizer.step()
        return True

    def _parameter_write(self, callback):
        """Run one non-optimizer parameter write unless audit suppresses it."""
        self.parameter_write_attempts += 1
        if self.zero_update_audit:
            self.suppressed_parameter_write_attempts += 1
            return False
        with torch.no_grad():
            callback()
        return True

    def prepare_action(self, source_logits, **kwargs):
        """Return logits used to sample the environment action."""
        return source_logits

    def before_inference(self, **kwargs):
        """Optional hook that runs before the policy forward for one step."""
        return None

    def select_action(self, distribution):
        """Select the environment action under the adapter's protocol."""
        return distribution.sample()

    def _record_loss(self, loss):
        value = float(loss.detach().item())
        self.action_steps += 1
        self.loss_sum += value
        self.last_loss = value

    def diagnostics(self):
        current = _flatten_params(self.params)
        output = {
            "action_steps": int(self.action_steps),
            "updates": int(self.update_count),
            "episodes": int(getattr(self, "episode_count", 0)),
            "slow_updates": int(getattr(self, "slow_update_count", 0)),
            "mean_entropy": self.loss_sum / max(1, self.action_steps),
            "last_entropy": self.last_loss,
            "last_grad_norm": self.last_grad_norm,
            "current_lr": self.current_lr,
            "max_grad_norm": (
                float(self.max_grad_norm)
                if hasattr(self, "max_grad_norm") else None
            ),
            "relative_param_drift": _relative_drift(current, self._source_flat),
            "adapted_parameter_names": list(self.names),
        }
        if self.zero_update_audit:
            final_evidence_due = (
                self.audit_expected_episodes is None
                or int(getattr(self, "episode_count", 0))
                >= self.audit_expected_episodes
            )
            if (
                final_evidence_due
                and self.audit_parameter_state_after_sha256 is None
            ):
                self.audit_parameter_state_after_sha256 = (
                    _parameter_state_sha256(self.params, self.names)
                )
                self.audit_model_states_after_sha256 = (
                    self._capture_audit_model_states()
                )
            state_after = self.audit_parameter_state_after_sha256
            model_states_before = self.audit_model_states_before_sha256
            model_states_after = self.audit_model_states_after_sha256
            primary_before = (
                None if model_states_before is None else
                model_states_before["primary_model"]
            )
            primary_after = (
                None if model_states_after is None else
                model_states_after["primary_model"]
            )
            output.update({
                "audit_mode": "zero_update_adapter_parity",
                "parameter_writes_suppressed": True,
                "parameter_write_attempts": int(
                    self.parameter_write_attempts
                ),
                "suppressed_parameter_write_attempts": int(
                    self.suppressed_parameter_write_attempts
                ),
                "optimizer_step_attempts": int(
                    self.optimizer_step_attempts
                ),
                "suppressed_optimizer_step_attempts": int(
                    self.suppressed_optimizer_step_attempts
                ),
                "parameter_state_before_sha256": (
                    self.audit_parameter_state_before_sha256
                ),
                "parameter_state_after_sha256": state_after,
                "parameter_state_hash_match": (
                    None if state_after is None else
                    state_after == self.audit_parameter_state_before_sha256
                ),
                "model_state_before_sha256": primary_before,
                "model_state_after_sha256": primary_after,
                "model_state_hash_match": (
                    None if primary_after is None else
                    primary_after == primary_before
                ),
                "deployed_model_states_before_sha256": model_states_before,
                "deployed_model_states_after_sha256": model_states_after,
                "deployed_model_states_hash_match": (
                    None if model_states_after is None else
                    model_states_after == model_states_before
                ),
                "audit_expected_episodes": self.audit_expected_episodes,
            })
        return output


class TentAdapter(_AdapterDiagnostics):
    """Tent entropy minimization on a strictly selected norm parameter set."""

    def __init__(
        self,
        model,
        lr=1e-6,
        steps=1,
        episodic=False,
        reset_bn_stats=True,
        scope="last_k_ln",
        last_k=4,
        optimizer_name="Adam",
        momentum=0.9,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        update_interval=1,
        max_updates_per_episode=-1,
        max_grad_norm=1.0,
    ):
        if int(steps) != 1:
            raise ValueError(
                "TTA.STEPS != 1 requires a new policy forward after every update; "
                "the navigation adapter intentionally supports STEPS=1 only."
            )
        update_interval = int(update_interval)
        if update_interval < 1:
            raise ValueError("Tent UPDATE_INTERVAL must be positive")
        self.model = model
        self.episodic = bool(episodic)
        self.base_lr = float(lr)
        self.update_interval = update_interval
        self.canonical_update_interval = self.update_interval == 1
        if not self.canonical_update_interval:
            logging.warning(
                "Tent UPDATE_INTERVAL=%d is a non-canonical schedule ablation; "
                "canonical Tent updates after every policy forward",
                self.update_interval,
            )
        self.max_updates_per_episode = int(max_updates_per_episode)
        if self.max_updates_per_episode < -1:
            raise ValueError(
                "TTA.MAX_UPDATES_PER_EPISODE must be -1 (unlimited) or "
                "a nonnegative integer"
            )
        self.max_grad_norm = float(max_grad_norm)
        self.params, self.names = configure_tta_model(
            model, reset_bn_stats, scope, last_k
        )
        self._optimizer_args = dict(
            name=optimizer_name,
            lr=self.base_lr,
            momentum=momentum,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
        )
        self.optimizer = _make_optimizer(self.params, **self._optimizer_args)
        self._model_state = deepcopy(self.model.state_dict())
        self._optim_state = deepcopy(self.optimizer.state_dict())
        self._source_flat = _flatten_params(self.params).clone()
        self.episode_count = 0
        self.episode_update_count = 0
        self.skipped_updates_by_budget = 0
        self._init_diagnostics()

    @torch.enable_grad()
    def adapt(self, logits, **kwargs):
        loss = softmax_entropy(logits).mean()
        scheduled_update = (
            (self.action_steps + 1) % self.update_interval == 0
        )
        within_episode_budget = (
            self.max_updates_per_episode < 0
            or self.episode_update_count < self.max_updates_per_episode
        )
        should_update = scheduled_update and within_episode_budget
        self._record_loss(loss)
        if should_update:
            # The per-episode budget counts scheduled native optimizer
            # attempts.  In zero-write audit mode the write is suppressed, but
            # the same later decisions must still be skipped by the budget.
            self.episode_update_count += 1
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.params,
                self.max_grad_norm if self.max_grad_norm > 0 else math.inf,
            )
            self.last_grad_norm = float(grad_norm.item())
            wrote_parameters = self._optimizer_step(self.optimizer)
            self.optimizer.zero_grad(set_to_none=True)
            if wrote_parameters:
                self.update_count += 1
        elif scheduled_update:
            self.skipped_updates_by_budget += 1
        return loss.detach()

    def reset(self):
        self._parameter_write(
            lambda: self.model.load_state_dict(self._model_state, strict=True)
        )
        self.optimizer.load_state_dict(self._optim_state)

    def episode_start(self):
        if self.episodic:
            self.reset()
        self.episode_update_count = 0

    def episode_end(self, episode_stats=None):
        self.episode_count += 1

    def diagnostics(self):
        output = super().diagnostics()
        output.update({
            "episode_updates": self.episode_update_count,
            "update_interval": self.update_interval,
            "update_interval_unit": "policy_forward",
            "canonical_update_interval": self.canonical_update_interval,
            "tent_schedule": (
                "canonical_per_forward"
                if self.canonical_update_interval
                else "noncanonical_interval_ablation"
            ),
            "max_updates_per_episode": self.max_updates_per_episode,
            "skipped_updates_by_budget": self.skipped_updates_by_budget,
        })
        return output


class FSTTAAdapter(_AdapterDiagnostics):
    """Fast-Slow TTA with action-level GDA and episode-level PDA."""

    def __init__(
        self,
        model,
        lr_fast=1e-5,
        lr_slow=1e-4,
        M=3,
        N=4,
        q=0.1,
        rho=0.95,
        tau=0.7,
        a=0.9,
        b=1.1,
        steps=1,
        episodic=False,
        use_slow=True,
        reset_bn_stats=True,
        scope="last_k_ln",
        last_k=4,
        optimizer_name="AdamW",
        momentum=0.9,
        beta1=0.9,
        beta2=0.99,
        weight_decay=0.0,
        max_grad_norm=1.0,
        reset_optimizer_each_episode=True,
        eigen_eps=1e-6,
        fast_grad_mode="concordant",
        use_fast_lr_scaler=True,
        slow_optimizer_name=None,
        slow_momentum=None,
        reset_slow_optimizer_each_window=False,
        reset_var_hist_each_episode=False,
    ):
        if int(steps) != 1:
            raise ValueError("FSTTA supports one policy forward/update per action")
        if int(M) < 1:
            raise ValueError("FSTTA.M must be positive")
        if int(N) < 1:
            raise ValueError("FSTTA.N must be positive")
        if not 0.0 < float(q) < 1.0:
            raise ValueError("FSTTA.Q must be in (0, 1)")
        if not 0.0 <= float(rho) <= 1.0:
            raise ValueError("FSTTA.RHO must be in [0, 1]")
        numeric_values = {
            "LR": float(lr_fast),
            "LR_SLOW": float(lr_slow),
            "TAU": float(tau),
            "A": float(a),
            "B": float(b),
            "MAX_GRAD_NORM": float(max_grad_norm),
            "EIGEN_EPS": float(eigen_eps),
        }
        if any(not math.isfinite(value) for value in numeric_values.values()):
            raise ValueError("FSTTA numeric hyperparameters must be finite")
        if float(lr_fast) <= 0.0 or float(lr_slow) <= 0.0:
            raise ValueError("FSTTA learning rates must be positive")
        if float(a) <= 0.0 or float(b) < float(a):
            raise ValueError("FSTTA LR scale bounds require 0 < A <= B")
        if float(max_grad_norm) < 0.0:
            raise ValueError("FSTTA.MAX_GRAD_NORM must be nonnegative")
        if float(eigen_eps) <= 0.0:
            raise ValueError("FSTTA.EIGEN_EPS must be positive")
        if bool(episodic) and bool(use_slow):
            raise ValueError(
                "FSTTA slow adaptation requires TTA.EPISODIC=False so its "
                "trajectory and slow optimizer can persist across episodes"
            )
        fast_grad_mode = str(fast_grad_mode).lower()
        valid_fast_grad_modes = ("concordant", "mean", "last")
        if fast_grad_mode not in valid_fast_grad_modes:
            raise ValueError(
                "Unknown FSTTA.FAST_GRAD_MODE {!r}; expected one of {}".format(
                    fast_grad_mode, valid_fast_grad_modes
                )
            )
        self.model = model
        self.M = int(M)
        self.N = int(N)
        self.q = float(q)
        self.rho = float(rho)
        self.tau = float(tau)
        self.a = float(a)
        self.b = float(b)
        self.base_lr = float(lr_fast)
        self.lr_slow = float(lr_slow)
        self.episodic = bool(episodic)
        self.use_slow = bool(use_slow)
        self.fast_grad_mode = fast_grad_mode
        self.use_fast_lr_scaler = bool(use_fast_lr_scaler)
        self.slow_momentum = float(
            momentum if slow_momentum is None else slow_momentum
        )
        self.reset_slow_optimizer_each_window = bool(
            reset_slow_optimizer_each_window
        )
        self.max_grad_norm = float(max_grad_norm)
        self.reset_optimizer_each_episode = bool(reset_optimizer_each_episode)
        # Paper Eq. (6) maintains historical variance over the complete test
        # stream.  The released code rebuilds FAST per rollout and therefore
        # resets this value; ``True`` is retained as that code-path ablation.
        self.reset_var_hist_each_episode = bool(reset_var_hist_each_episode)
        self.eigen_eps = float(eigen_eps)

        self.params, self.names = configure_tta_model(
            model, reset_bn_stats, scope, last_k
        )
        self._optimizer_args = dict(
            name=optimizer_name,
            lr=self.base_lr,
            momentum=momentum,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
        )
        self.optimizer = _make_optimizer(self.params, **self._optimizer_args)
        self._model_state = deepcopy(self.model.state_dict())
        self._optim_state = deepcopy(self.optimizer.state_dict())
        self._source_flat = _flatten_params(self.params).clone()

        # Keep an independent optimizer for the SLOW parameter anchor. The
        # default follows the released FSTTA implementation's persistent
        # AdamW, while explicit ablations can reset its moments per window or
        # use the paper equation's plain SGD step.
        # A single flat Parameter is element-wise equivalent to optimizing the
        # selected normalization tensors with one shared optimizer group, and
        # avoids retaining a second full policy on the GPU.
        self.slow_anchor = nn.Parameter(
            self._source_flat.detach().clone(), requires_grad=True
        )
        if slow_optimizer_name is None:
            slow_optimizer_name = optimizer_name
        self._slow_optimizer_args = dict(
            name=slow_optimizer_name,
            lr=self.lr_slow,
            momentum=self.slow_momentum,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
        )
        self.slow_optimizer = _make_optimizer(
            [self.slow_anchor], **self._slow_optimizer_args
        )
        self._slow_optim_state = deepcopy(self.slow_optimizer.state_dict())

        self.grad_buffer = []
        self.var_hist = None
        self.episode_count = 0
        self.slow_update_count = 0
        self.slow_trajectory = []
        self.audit_shadow_fast = None
        self.audit_shadow_fast_optimizer = None
        self.audit_shadow_slow_anchor = None
        self.audit_shadow_slow_optimizer = None
        self._init_diagnostics()
        self._init_fstta_diagnostics()

    def _init_fstta_diagnostics(self):
        self.slow_attempt_count = 0
        self.slow_skip_count = 0
        self.discarded_fast_gradients = 0
        self.last_sigma = 0.0
        self.lr_scale_count = 0
        self.lr_scale_sum = 0.0
        self.lr_scale_min = math.inf
        self.lr_scale_max = -math.inf
        self.lr_scale_lower_hits = 0
        self.lr_scale_upper_hits = 0
        self.last_slow_reference_norm = 0.0
        self.last_slow_grad_norm = 0.0
        self.last_slow_step_norm = 0.0
        self.last_slow_snap_norm = 0.0
        self.slow_optimizer_reset_count = 0
        self.fast_optimizer_attempt_count = 0
        self.fast_optimizer_suppressed_count = 0
        self.slow_optimizer_attempt_count = 0
        self.slow_optimizer_suppressed_count = 0
        self.completed_slow_window_count = 0

    def enable_zero_update_audit(self, expected_episodes=None):
        super().enable_zero_update_audit(expected_episodes)
        self.audit_slow_anchor_before_sha256 = _parameter_state_sha256(
            [self.slow_anchor], ["slow_anchor"]
        )
        self.audit_slow_anchor_after_sha256 = None
        # The deployed policy and slow anchor remain immutable, but the audit
        # must still exercise the native FAST->SLOW schedule.  These flat
        # shadows receive the same clipped gradients and optimizer semantics;
        # they are evidence-only state and are never used for environment
        # actions or exposed as deployed model parameters.
        self.audit_shadow_fast = nn.Parameter(
            self._source_flat.detach().clone(), requires_grad=True
        )
        self.audit_shadow_fast_optimizer = _make_optimizer(
            [self.audit_shadow_fast], **self._optimizer_args
        )
        self.audit_shadow_slow_anchor = nn.Parameter(
            self._source_flat.detach().clone(), requires_grad=True
        )
        self.audit_shadow_slow_optimizer = _make_optimizer(
            [self.audit_shadow_slow_anchor], **self._slow_optimizer_args
        )
        return self

    def _record_lr_scale(self, scale, sigma):
        scale = float(scale)
        self.current_lr = scale * self.base_lr
        self.last_sigma = float(sigma.detach().item())
        self.lr_scale_count += 1
        self.lr_scale_sum += scale
        self.lr_scale_min = min(self.lr_scale_min, scale)
        self.lr_scale_max = max(self.lr_scale_max, scale)
        tolerance = 1e-12
        if abs(scale - self.a) <= tolerance:
            self.lr_scale_lower_hits += 1
        if abs(scale - self.b) <= tolerance:
            self.lr_scale_upper_hits += 1

    def _clear_optimizer_state(self):
        self.optimizer.state.clear()
        if self.audit_shadow_fast_optimizer is not None:
            self.audit_shadow_fast_optimizer.state.clear()
        self._reset_fast_lr()

    def _reset_fast_lr(self):
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr
        self.current_lr = self.base_lr

    def _fast_grad_and_trace(self, grad_list):
        if self.fast_grad_mode == "concordant":
            return _concordant_grad_and_trace(grad_list, self.eigen_eps)

        gradients = torch.stack(grad_list, dim=0)
        mean_grad = gradients.mean(dim=0)
        if len(grad_list) < 2:
            sigma = mean_grad.new_zeros(())
        else:
            centered = gradients - mean_grad.unsqueeze(0)
            sigma = centered.square().sum() / float(len(grad_list) - 1)
        fast_grad = mean_grad if self.fast_grad_mode == "mean" else gradients[-1]
        return fast_grad, sigma

    @torch.no_grad()
    def _snap_fast_to_slow_anchor(self, fast_before_snap=None):
        if self.zero_update_audit:
            if fast_before_snap is None:
                fast_before_snap = self.audit_shadow_fast.detach().clone()
            anchor_for_norm = self.audit_shadow_slow_anchor.detach()
        else:
            if fast_before_snap is None:
                fast_before_snap = _flatten_params(self.params)
            anchor_for_norm = self.slow_anchor.detach()
        self.last_slow_snap_norm = float(
            (fast_before_snap - anchor_for_norm).norm().item()
        )
        self._parameter_write(
            lambda: _copy_flat_to_params(self.slow_anchor.detach(), self.params)
        )
        if self.zero_update_audit:
            self.audit_shadow_fast.copy_(anchor_for_norm)
        # Fast optimizer moments describe the pre-snap trajectory and must not
        # be applied to the newly deployed slow anchor. Slow moments persist.
        self._clear_optimizer_state()

    @torch.enable_grad()
    def adapt(self, logits, **kwargs):
        loss = softmax_entropy(logits).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.grad_buffer.append(_flatten_grads(self.params).clone())
        self._record_loss(loss)

        if len(self.grad_buffer) == self.M:
            fast_grad, sigma = self._fast_grad_and_trace(self.grad_buffer)
            if self.var_hist is None:
                scale = min(self.b, max(self.a, 1.0 + self.tau))
                self.var_hist = sigma.detach()
            else:
                delta = torch.abs(sigma.detach() - self.var_hist)
                scale = torch.clamp(1.0 + self.tau - delta, self.a, self.b).item()
                self.var_hist = (
                    self.rho * self.var_hist
                    + (1.0 - self.rho) * sigma.detach()
                )
            if not self.use_fast_lr_scaler:
                scale = 1.0
            self._record_lr_scale(scale, sigma)
            for group in self.optimizer.param_groups:
                group["lr"] = self.current_lr

            self.optimizer.zero_grad(set_to_none=True)
            _copy_flat_to_grads(fast_grad, self.params)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.params,
                self.max_grad_norm if self.max_grad_norm > 0 else math.inf,
            )
            self.last_grad_norm = float(grad_norm.item())
            clipped_fast_grad = _flatten_grads(self.params).clone()
            self.fast_optimizer_attempt_count += 1
            wrote_parameters = self._optimizer_step(self.optimizer)
            if self.zero_update_audit:
                self.fast_optimizer_suppressed_count += 1
                for group in self.audit_shadow_fast_optimizer.param_groups:
                    group["lr"] = self.current_lr
                self.audit_shadow_fast_optimizer.zero_grad(set_to_none=True)
                self.audit_shadow_fast.grad = clipped_fast_grad
                self.audit_shadow_fast_optimizer.step()
                self.audit_shadow_fast_optimizer.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)
            self.grad_buffer = []
            if wrote_parameters:
                self.update_count += 1
        else:
            self.optimizer.zero_grad(set_to_none=True)
        return loss.detach()

    @torch.no_grad()
    def _slow_step(self):
        self.slow_attempt_count += 1
        self.completed_slow_window_count += 1
        if self.reset_slow_optimizer_each_window:
            # A trajectory window is an independent SLOW optimization
            # problem in this ablation. Clear moments even when its geometry
            # later proves degenerate and no parameter step can be taken.
            self.slow_optimizer.state.clear()
            if self.audit_shadow_slow_optimizer is not None:
                self.audit_shadow_slow_optimizer.state.clear()
            self.slow_optimizer_reset_count += 1
        self.last_slow_reference_norm = 0.0
        self.last_slow_grad_norm = 0.0
        self.last_slow_step_norm = 0.0
        self.last_slow_snap_norm = 0.0
        anchor = (
            self.audit_shadow_slow_anchor.detach()
            if self.zero_update_audit else self.slow_anchor.detach()
        )
        fast_before_snap = (
            self.audit_shadow_fast.detach().clone()
            if self.zero_update_audit else _flatten_params(self.params).clone()
        )
        states = [anchor.clone()] + self.slow_trajectory
        matrix = torch.stack(states, dim=0)
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        singular_values, vh = _small_row_svd(centered)
        eigenvalues = singular_values.square() / float(max(1, self.N))
        if eigenvalues.numel() == 0 or eigenvalues.norm() <= 1e-12:
            logging.warning("[FSTTA] slow update skipped: degenerate trajectory")
            self.slow_skip_count += 1
            self.slow_trajectory = []
            self._snap_fast_to_slow_anchor(fast_before_snap)
            return

        # Eq. (8): recent episode states receive larger weights (q < 1).
        weights = torch.tensor(
            [self.q ** (self.N - index) for index in range(1, self.N + 1)],
            device=matrix.device,
            dtype=matrix.dtype,
        )
        weights = weights / weights.sum()
        deviations = torch.stack(
            [anchor - state for state in self.slow_trajectory], dim=0
        )
        reference = (weights.unsqueeze(1) * deviations).sum(dim=0)
        reference_norm = reference.norm()
        self.last_slow_reference_norm = float(reference_norm.item())
        if reference_norm <= 1e-12:
            logging.warning("[FSTTA] slow update skipped: zero reference direction")
            self.slow_skip_count += 1
            self.slow_trajectory = []
            self._snap_fast_to_slow_anchor(fast_before_snap)
            return

        valid = eigenvalues > eigenvalues.max().clamp_min(1e-12) * self.eigen_eps
        values = eigenvalues[valid]
        basis = vh[valid]
        coefficients = values / values.norm().clamp_min(1e-12)
        signs = torch.sign(basis @ reference)
        signs[signs == 0] = 1
        slow_grad = (
            coefficients.unsqueeze(1)
            * signs.unsqueeze(1)
            * basis
        ).sum(dim=0)
        slow_grad = slow_grad * reference_norm
        self.last_slow_grad_norm = float(slow_grad.norm().item())

        anchor_before = self.slow_anchor.detach().clone()
        self.slow_optimizer.zero_grad(set_to_none=True)
        self.slow_anchor.grad = slow_grad.detach().clone()
        self.slow_optimizer_attempt_count += 1
        wrote_parameters = self._optimizer_step(self.slow_optimizer)
        if self.zero_update_audit:
            self.slow_optimizer_suppressed_count += 1
            shadow_anchor_before = self.audit_shadow_slow_anchor.detach().clone()
            self.audit_shadow_slow_optimizer.zero_grad(set_to_none=True)
            self.audit_shadow_slow_anchor.grad = slow_grad.detach().clone()
            self.audit_shadow_slow_optimizer.step()
            self.audit_shadow_slow_optimizer.zero_grad(set_to_none=True)
            anchor_after = self.audit_shadow_slow_anchor.detach()
            step_reference = shadow_anchor_before
        else:
            anchor_after = self.slow_anchor.detach()
            step_reference = anchor_before
        self.slow_optimizer.zero_grad(set_to_none=True)
        self.last_slow_step_norm = float(
            (anchor_after - step_reference).norm().item()
        )

        self.slow_trajectory = []
        if wrote_parameters:
            self.slow_update_count += 1
        self._snap_fast_to_slow_anchor(fast_before_snap)

    def reset(self):
        self._parameter_write(
            lambda: self.model.load_state_dict(self._model_state, strict=True)
        )
        self.optimizer.load_state_dict(self._optim_state)
        self._parameter_write(
            lambda: self.slow_anchor.copy_(_flatten_params(self.params))
        )
        self.slow_optimizer.load_state_dict(
            deepcopy(self._slow_optim_state)
        )
        self.slow_optimizer.zero_grad(set_to_none=True)
        self.grad_buffer = []
        self.var_hist = None
        self.episode_count = 0
        self.slow_update_count = 0
        self.slow_trajectory = []
        if self.zero_update_audit:
            with torch.no_grad():
                self.audit_shadow_fast.copy_(self._source_flat)
                self.audit_shadow_slow_anchor.copy_(self._source_flat)
            self.audit_shadow_fast_optimizer.state.clear()
            self.audit_shadow_slow_optimizer.state.clear()
        self._init_diagnostics()
        self._init_fstta_diagnostics()

    def episode_start(self):
        if self.episodic:
            self.reset()
        self.grad_buffer = []
        # The paper-level default retains Eq. (6)'s history across episodes.
        # Resetting here reproduces the released per-rollout construction only.
        if self.reset_var_hist_each_episode:
            self.var_hist = None
        if self.reset_optimizer_each_episode:
            self._clear_optimizer_state()
        else:
            self._reset_fast_lr()

    def episode_end(self, episode_stats=None):
        self.episode_count += 1
        # Never combine gradients from two unrelated trajectories.
        self.discarded_fast_gradients += len(self.grad_buffer)
        self.grad_buffer = []
        if self.use_slow:
            episode_state = (
                self.audit_shadow_fast.detach().clone()
                if self.zero_update_audit else _flatten_params(self.params).clone()
            )
            self.slow_trajectory.append(episode_state)
            if len(self.slow_trajectory) == self.N:
                self._slow_step()

    def diagnostics(self):
        output = super().diagnostics()
        scale_count = max(1, self.lr_scale_count)
        current = _flatten_params(self.params)
        anchor = self.slow_anchor.detach()
        output.update({
            "fast_lr": self.base_lr,
            "fast_window": self.M,
            "fast_grad_mode": self.fast_grad_mode,
            "gda_inverse_variance": (
                "full_covariance_with_clamped_nullspace"
            ),
            "gda_eigenvalue_floor": self.eigen_eps,
            "gda_preserves_covariance_nullspace": True,
            "use_fast_lr_scaler": self.use_fast_lr_scaler,
            "use_slow": self.use_slow,
            "slow_window": self.N,
            "slow_lr": self.lr_slow,
            "q": self.q,
            "fast_optimizer": self.optimizer.__class__.__name__,
            "slow_optimizer": self.slow_optimizer.__class__.__name__,
            "slow_momentum": self.slow_momentum,
            "reset_slow_optimizer_each_window": (
                self.reset_slow_optimizer_each_window
            ),
            "slow_optimizer_resets": self.slow_optimizer_reset_count,
            "slow_attempts": self.slow_attempt_count,
            "completed_slow_windows": self.completed_slow_window_count,
            "fast_optimizer_attempts": self.fast_optimizer_attempt_count,
            "fast_optimizer_attempts_suppressed": (
                self.fast_optimizer_suppressed_count
            ),
            "slow_optimizer_attempts": self.slow_optimizer_attempt_count,
            "slow_optimizer_attempts_suppressed": (
                self.slow_optimizer_suppressed_count
            ),
            "slow_skipped_updates": self.slow_skip_count,
            "slow_pending_episodes": len(self.slow_trajectory),
            "discarded_fast_gradients": self.discarded_fast_gradients,
            "last_sigma": self.last_sigma,
            "variance_history_lifetime": (
                "episode"
                if self.episodic or self.reset_var_hist_each_episode
                else "test_stream"
            ),
            "reset_var_hist_each_episode": self.reset_var_hist_each_episode,
            "variance_history_initialized": self.var_hist is not None,
            "lr_scale_mean": self.lr_scale_sum / scale_count,
            "lr_scale_min": (
                self.lr_scale_min if self.lr_scale_count else 0.0
            ),
            "lr_scale_max": (
                self.lr_scale_max if self.lr_scale_count else 0.0
            ),
            "lr_scale_lower_hits": self.lr_scale_lower_hits,
            "lr_scale_upper_hits": self.lr_scale_upper_hits,
            "last_slow_reference_norm": self.last_slow_reference_norm,
            "last_slow_grad_norm": self.last_slow_grad_norm,
            "last_slow_step_norm": self.last_slow_step_norm,
            "last_slow_snap_norm": self.last_slow_snap_norm,
            "relative_slow_anchor_drift": _relative_drift(
                anchor, self._source_flat
            ),
            "relative_fast_to_slow_anchor": _relative_drift(
                current, anchor
            ),
        })
        if self.zero_update_audit:
            if (
                output["parameter_state_after_sha256"] is not None
                and self.audit_slow_anchor_after_sha256 is None
            ):
                self.audit_slow_anchor_after_sha256 = _parameter_state_sha256(
                    [self.slow_anchor], ["slow_anchor"]
                )
            output.update({
                "slow_anchor_state_before_sha256": (
                    self.audit_slow_anchor_before_sha256
                ),
                "slow_anchor_state_after_sha256": (
                    self.audit_slow_anchor_after_sha256
                ),
                "slow_anchor_state_hash_match": (
                    None if self.audit_slow_anchor_after_sha256 is None else
                    self.audit_slow_anchor_after_sha256
                    == self.audit_slow_anchor_before_sha256
                ),
            })
        return output


class EAMAdapter(_AdapterDiagnostics):
    """Algorithm-3-aligned, step-level Elastic Adaptation Model.

    Each online action step is one sample ``x``.  Following the paper's
    pseudocode literally, ``x`` is first offered to the reservoir and replay is
    then sampled from that *updated* reservoir.  While ``|M| < K``, both
    branches still infer the current sample for the online action, but no replay
    batch is formed and no model update is allowed.  Once ``|M| >= K``,
    ``K - 1`` entries are sampled from the whole updated reservoir and the
    current sample is appended explicitly.  Consequently, a retained current
    sample may occur twice in ``B``; Algorithm 3 contains no exclusion step.

    The final decision for the current action is fixed before the optimizer
    update, so adaptation affects only later actions.

    The source branch stays frozen.  The auxiliary branch is initialized from
    the source policy and only explicitly selected post-encoder modules are
    adapted.  Complete policy inputs and the executed action are snapshotted on
    CPU, including the external-memory state, so historical steps can be
    replayed without another environment interaction.
    """

    requires_source_grad = False

    def __init__(
        self,
        model,
        lr=1e-5,
        confidence_scale=0.4,
        memory_size=32,
        batch_size=8,
        update_interval=1,
        param_scope="module_prefixes",
        trainable_prefixes=(),
        optimizer_name="Adam",
        momentum=0.9,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        max_grad_norm=0.0,
        episodic=False,
        forward_policy=None,
    ):
        if bool(episodic):
            raise ValueError(
                "Continual step-level EAM requires TTA.EPISODIC=False"
            )
        self.param_scope = str(param_scope).lower()
        if self.param_scope != "module_prefixes":
            raise ValueError(
                "Step-replay EAM requires PARAM_SCOPE="
                "'module_prefixes' so frozen feature producers are selected "
                "explicitly; got {!r}".format(self.param_scope)
            )
        self.trainable_prefixes = tuple(str(item) for item in trainable_prefixes)
        if forward_policy is not None and not callable(forward_policy):
            raise TypeError("EAM forward_policy must be callable")
        self.forward_policy_callback = forward_policy
        self.memory_size = int(memory_size)
        self.batch_size = int(batch_size)
        self.update_interval = int(update_interval)
        if self.memory_size < 1:
            raise ValueError("EAM.MEMORY_SIZE must be positive")
        if self.batch_size < 1:
            raise ValueError("EAM.BATCH_SIZE must be positive")
        if self.memory_size < self.batch_size:
            raise ValueError(
                "EAM.MEMORY_SIZE must be greater than or equal to "
                "EAM.BATCH_SIZE so the replay warm-up can complete"
            )
        if self.update_interval < 1:
            raise ValueError("EAM.UPDATE_INTERVAL must be positive")
        numeric_values = {
            "LR": float(lr),
            "CONFIDENCE_SCALE": float(confidence_scale),
            "MAX_GRAD_NORM": float(max_grad_norm),
        }
        if any(not math.isfinite(value) for value in numeric_values.values()):
            raise ValueError("EAM numeric hyperparameters must be finite")
        if float(lr) <= 0.0:
            raise ValueError("EAM.LR must be positive")
        if float(confidence_scale) < 0.0:
            raise ValueError("EAM.CONFIDENCE_SCALE must be nonnegative")
        if float(max_grad_norm) < 0.0:
            raise ValueError("EAM.MAX_GRAD_NORM must be nonnegative")

        self.source_model = model
        self.source_model.eval()
        self.source_model.requires_grad_(False)
        self.aux_model = deepcopy(model)
        self.aux_model.eval()
        self.params, self.names = configure_module_prefixes(
            self.aux_model, self.trainable_prefixes
        )
        if not self.params:
            raise ValueError("EAM did not select any auxiliary parameters")

        self.base_lr = float(lr)
        self.max_grad_norm = float(max_grad_norm)
        self.confidence_scale = float(confidence_scale)
        self.episodic = False
        self.optimizer = _make_optimizer(
            self.params,
            optimizer_name,
            self.base_lr,
            momentum=momentum,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
        )
        # EAM already keeps source + auxiliary policies on the GPU. Keep the
        # explicit/manual reset snapshot on CPU to avoid a third GPU-sized
        # model copy.
        self._model_state = {
            key: value.detach().cpu().clone()
            for key, value in self.aux_model.state_dict().items()
        }
        self._optim_state = deepcopy(self.optimizer.state_dict())
        self._source_flat = _flatten_params(self.params).clone()
        self.replay = []
        self.seen_samples = 0
        self.accepted_samples = 0
        self.aux_used_steps = 0
        self.source_confident_steps = 0
        self.episode_count = 0
        self._pending_replay = None
        self._cached_current = None
        self.update_attempt_count = 0
        self.no_reliable_update_count = 0
        self.replayed_step_count = 0
        self.replay_aux_used_steps = 0
        self.current_replay_duplicates = 0
        self.warmup_no_update_steps = 0
        self.valid_action_metadata_steps = 0
        self.valid_action_fallback_steps = 0
        self.masked_invalid_action_slots = 0
        # Deprecated compatibility counter.  It counts warm-up action steps,
        # not training batches or optimizer updates.
        self.current_only_batches = 0
        self.train_loss_sum = 0.0
        self.last_train_loss = 0.0
        self._init_diagnostics()

    @property
    def device(self):
        return self.params[0].device

    def _audit_deployed_modules(self):
        # Both policies participate in deployed decisions: source provides the
        # confidence/pseudo-label branch and auxiliary provides the adaptive
        # branch.  A write to either must fail parity validation.
        return (
            ("primary_model", self.source_model),
            ("auxiliary_model", self.aux_model),
        )

    @staticmethod
    def _default_forward_policy(model, policy_inputs):
        features, _, _ = model.net(
            policy_inputs["observations"],
            policy_inputs["rnn_hidden_states"],
            policy_inputs["prev_actions"],
            policy_inputs["masks"],
            policy_inputs.get("ext_memory"),
            policy_inputs.get("ext_memory_masks"),
        )
        return features, model.action_distribution(features).logits

    def _forward_policy(self, model, policy_inputs):
        callback = self.forward_policy_callback
        output = (
            self._default_forward_policy(model, policy_inputs)
            if callback is None
            else callback(model, policy_inputs)
        )
        if isinstance(output, tuple) and len(output) == 2:
            _, logits = output
        else:
            logits = output
        if not torch.is_tensor(logits) or logits.ndim < 2:
            raise ValueError(
                "EAM forward_policy must return logits or (features, logits)"
            )
        return logits

    def _forward_aux(self, policy_inputs):
        return self._forward_policy(self.aux_model, policy_inputs)

    def _forward_source(self, policy_inputs):
        # Algorithm 3 runs both branches over B.  Historical source decisions
        # are therefore recomputed instead of being read from a logits cache.
        with torch.no_grad():
            return self._forward_policy(self.source_model, policy_inputs)

    @staticmethod
    def _extract_valid_action_spec(
        policy_inputs,
        valid_action_count=None,
        valid_action_mask=None,
    ):
        """Return detached action-space metadata supplied by task glue.

        The explicit keyword API is preferred.  Mirroring the same keys inside
        ``policy_inputs`` is supported so replay callbacks can keep all
        per-sample state in one snapshot.
        """
        if isinstance(policy_inputs, dict):
            if valid_action_count is None:
                valid_action_count = policy_inputs.get("valid_action_count")
            if valid_action_mask is None:
                valid_action_mask = policy_inputs.get("valid_action_mask")
        if valid_action_count is not None and valid_action_mask is not None:
            raise ValueError(
                "EAM accepts either valid_action_count or valid_action_mask, "
                "not both"
            )
        return (
            _tree_to_cpu(valid_action_count),
            _tree_to_cpu(valid_action_mask),
        )

    @staticmethod
    def _valid_action_mask(
        logits,
        valid_action_count=None,
        valid_action_mask=None,
    ):
        """Normalize and validate per-sample valid-action metadata."""
        if valid_action_count is not None and valid_action_mask is not None:
            raise ValueError(
                "EAM accepts either valid_action_count or valid_action_mask, "
                "not both"
            )
        action_dim = int(logits.shape[-1])
        sample_shape = tuple(logits.shape[:-1])
        if valid_action_count is not None:
            if isinstance(valid_action_count, bool):
                raise ValueError("EAM valid_action_count must contain integers")
            counts = torch.as_tensor(
                valid_action_count, device=logits.device
            )
            if counts.dtype == torch.bool or not bool(torch.isfinite(
                counts.to(torch.float64)
            ).all()):
                raise ValueError(
                    "EAM valid_action_count must contain finite integers"
                )
            rounded = counts.to(torch.float64).round()
            if not bool((counts.to(torch.float64) == rounded).all()):
                raise ValueError("EAM valid_action_count must contain integers")
            expected_samples = math.prod(sample_shape)
            if counts.numel() == 1:
                counts = rounded.to(torch.long).expand(sample_shape)
            elif counts.numel() == expected_samples:
                counts = rounded.to(torch.long).reshape(sample_shape)
            else:
                raise ValueError(
                    "EAM valid_action_count shape {} does not match logits "
                    "sample shape {}".format(tuple(counts.shape), sample_shape)
                )
            if bool(((counts < 1) | (counts > action_dim)).any()):
                raise ValueError(
                    "EAM valid_action_count values must be in [1, {}]".format(
                        action_dim
                    )
                )
            indices = torch.arange(action_dim, device=logits.device)
            mask = indices < counts.unsqueeze(-1)
        elif valid_action_mask is not None:
            mask = torch.as_tensor(valid_action_mask, device=logits.device)
            if mask.dtype != torch.bool:
                if not bool(((mask == 0) | (mask == 1)).all()):
                    raise ValueError(
                        "EAM valid_action_mask must be boolean or binary"
                    )
                mask = mask.bool()
            try:
                mask = torch.broadcast_to(mask, logits.shape)
            except RuntimeError as error:
                raise ValueError(
                    "EAM valid_action_mask shape {} is not broadcastable to "
                    "logits shape {}".format(
                        tuple(mask.shape), tuple(logits.shape)
                    )
                ) from error
            if not bool(mask.any(dim=-1).all()):
                raise ValueError(
                    "EAM valid_action_mask must retain at least one action "
                    "per sample"
                )
        else:
            mask = torch.ones_like(logits, dtype=torch.bool)

        if not bool(torch.isfinite(logits.masked_select(mask)).all()):
            raise ValueError(
                "EAM logits contain a non-finite value in a valid action slot; "
                "pass the per-sample valid_action_count or valid_action_mask "
                "for padded decisions"
            )
        return mask

    @staticmethod
    def _masked_log_probs(logits, valid_action_mask):
        floor = torch.finfo(logits.dtype).min
        return logits.masked_fill(~valid_action_mask, floor).log_softmax(dim=-1)

    @classmethod
    def _masked_entropy(cls, logits, valid_action_mask):
        log_probs = cls._masked_log_probs(logits, valid_action_mask)
        probs = log_probs.exp()
        valid_log_probs = log_probs.masked_fill(~valid_action_mask, 0.0)
        return -(probs * valid_log_probs).sum(dim=-1)

    def _threshold(self, valid_action_count):
        if torch.is_tensor(valid_action_count):
            counts = valid_action_count.to(dtype=torch.float32).clamp_min(2.0)
            return self.confidence_scale * counts.log()
        return self.confidence_scale * math.log(
            max(2, int(valid_action_count))
        )

    def _combine(
        self,
        source_logits,
        aux_logits,
        valid_action_count=None,
        valid_action_mask=None,
    ):
        if source_logits.shape != aux_logits.shape:
            raise ValueError(
                "EAM source and auxiliary logits must have identical shapes"
            )
        valid_mask = self._valid_action_mask(
            source_logits,
            valid_action_count=valid_action_count,
            valid_action_mask=valid_action_mask,
        )
        # Validate the auxiliary branch independently; invalid padded slots are
        # intentionally ignored even when they contain arbitrary sentinels.
        if not bool(torch.isfinite(aux_logits.masked_select(valid_mask)).all()):
            raise ValueError(
                "EAM auxiliary logits contain a non-finite value in a valid "
                "action slot"
            )
        valid_counts = valid_mask.sum(dim=-1)
        threshold = self._threshold(valid_counts)
        aux_entropy = self._masked_entropy(aux_logits, valid_mask)
        use_aux = aux_entropy < threshold
        source_probs = self._masked_log_probs(
            source_logits, valid_mask
        ).exp()
        aux_probs = self._masked_log_probs(aux_logits, valid_mask).exp()
        combined_probs = (
            source_probs
            + use_aux.to(aux_probs.dtype).unsqueeze(-1) * aux_probs
        )
        combined_probs = combined_probs / combined_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        # Return log-probabilities so the trainer can reconstruct its native
        # categorical distribution without changing action tensor shapes.
        combined_log_probs = combined_probs.clamp_min(1e-8).log()
        combined_log_probs = combined_log_probs.masked_fill(~valid_mask, -math.inf)
        return combined_log_probs, use_aux

    def before_inference(
        self,
        policy_inputs=None,
        valid_action_count=None,
        valid_action_mask=None,
        **kwargs
    ):
        """Perform Algorithm 3's buffer update and replay before inference."""
        if policy_inputs is None:
            raise ValueError("EAM requires policy_inputs before inference")
        if self._pending_replay is not None or self._cached_current is not None:
            raise RuntimeError(
                "EAM.before_inference called twice without completing the step"
            )

        valid_action_count, valid_action_mask = self._extract_valid_action_spec(
            policy_inputs,
            valid_action_count=valid_action_count,
            valid_action_mask=valid_action_mask,
        )
        current_entry = self._reservoir_add(
            policy_inputs,
            valid_action_count=valid_action_count,
            valid_action_mask=valid_action_mask,
        )
        if len(self.replay) < self.batch_size:
            # Algorithm 3 performs inference but does not form B or update the
            # auxiliary model until the updated reservoir reaches K entries.
            replay_entries = None
            self.warmup_no_update_steps += 1
            self.current_only_batches += 1
        else:
            replay_entries = (
                random.sample(self.replay, self.batch_size - 1)
                if self.batch_size > 1 else []
            )
        self._pending_replay = (
            current_entry,
            replay_entries,
            valid_action_count,
            valid_action_mask,
        )

    @torch.enable_grad()
    def prepare_action(
        self,
        source_logits,
        policy_inputs=None,
        valid_action_count=None,
        valid_action_mask=None,
        **kwargs
    ):
        if policy_inputs is None:
            raise ValueError("EAM requires policy_inputs for its auxiliary branch")
        if self._cached_current is not None:
            raise RuntimeError(
                "EAM.prepare_action called twice without the intervening adapt"
            )

        # Algorithm 3, lines 1--18: update M first, then form B from the
        # updated M.  Do not filter the just-inserted entry from this draw.
        source_current = source_logits.detach()
        if self._pending_replay is None:
            raise RuntimeError(
                "EAM.before_inference must run before the source policy forward"
            )
        (
            current_entry,
            replay_entries,
            pending_valid_action_count,
            pending_valid_action_mask,
        ) = self._pending_replay
        self._pending_replay = None
        supplied_count, supplied_mask = self._extract_valid_action_spec(
            policy_inputs,
            valid_action_count=valid_action_count,
            valid_action_mask=valid_action_mask,
        )
        if supplied_count is not None or supplied_mask is not None:
            if (
                pending_valid_action_count is not None
                or pending_valid_action_mask is not None
            ):
                pending_mask = self._valid_action_mask(
                    source_current,
                    valid_action_count=pending_valid_action_count,
                    valid_action_mask=pending_valid_action_mask,
                )
                supplied_valid_mask = self._valid_action_mask(
                    source_current,
                    valid_action_count=supplied_count,
                    valid_action_mask=supplied_mask,
                )
                if not torch.equal(pending_mask, supplied_valid_mask):
                    raise ValueError(
                        "EAM valid-action metadata changed between "
                        "before_inference and prepare_action"
                    )
            pending_valid_action_count = supplied_count
            pending_valid_action_mask = supplied_mask
            if current_entry is not None:
                current_entry["valid_action_count"] = _tree_to_cpu(
                    supplied_count
                )
                current_entry["valid_action_mask"] = _tree_to_cpu(supplied_mask)

        # Both branches always infer x so the online action remains available
        # during replay warm-up.  Algorithm 3 forms the training batch B only
        # after the updated reservoir contains at least K entries.
        aux_current = self._forward_aux(policy_inputs)
        replay_ready = replay_entries is not None
        if replay_ready:
            # Variable-length navigation memories make a physical tensor batch
            # impractical, so old entries are forwarded separately.  This is
            # numerically the same mini-batch in eval mode.
            source_batch, aux_batch, valid_mask_batch = [], [], []
            for entry in replay_entries:
                if entry is current_entry:
                    # B_h may contain x because sampling uses the updated M.
                    # Reuse the deterministic forward while retaining x twice.
                    source_replay = source_current
                    aux_replay = aux_current
                    replay_count = pending_valid_action_count
                    replay_mask = pending_valid_action_mask
                    self.current_replay_duplicates += 1
                else:
                    replay_inputs = _tree_to_device(
                        entry["policy_inputs"], self.device
                    )
                    source_replay = self._forward_source(replay_inputs)
                    aux_replay = self._forward_aux(replay_inputs)
                    replay_count = entry.get("valid_action_count")
                    replay_mask = entry.get("valid_action_mask")
                source_batch.append(source_replay)
                aux_batch.append(aux_replay)
                valid_mask_batch.append(self._valid_action_mask(
                    source_replay,
                    valid_action_count=replay_count,
                    valid_action_mask=replay_mask,
                ))

            # The explicit current x is the last element of B = B_h union x.
            source_batch.append(source_current)
            aux_batch.append(aux_current)
            current_valid_mask = self._valid_action_mask(
                source_current,
                valid_action_count=pending_valid_action_count,
                valid_action_mask=pending_valid_action_mask,
            )
            valid_mask_batch.append(current_valid_mask)

            # Candidate navigation sets are variable-length in VLN.  Keep B as
            # a logical list and evaluate each row in its own action space.
            combined_batch, use_aux_batch = [], []
            for source_decision, auxiliary_decision, decision_valid_mask in zip(
                source_batch, aux_batch, valid_mask_batch
            ):
                combined_decision, use_aux = self._combine(
                    source_decision,
                    auxiliary_decision,
                    valid_action_mask=decision_valid_mask,
                )
                combined_batch.append(combined_decision)
                use_aux_batch.append(use_aux)
            combined_current = combined_batch[-1]
            current_use_aux = use_aux_batch[-1]
        else:
            # Warm-up is inference-only: do not represent x as a one-item
            # replay batch, because adapt() must not treat it as trainable B.
            current_valid_mask = self._valid_action_mask(
                source_current,
                valid_action_count=pending_valid_action_count,
                valid_action_mask=pending_valid_action_mask,
            )
            combined_current, current_use_aux = self._combine(
                source_current,
                aux_current,
                valid_action_mask=current_valid_mask,
            )
            source_batch = None
            aux_batch = None
            combined_batch = None
            use_aux_batch = None
            valid_mask_batch = None

        self.aux_used_steps += int(current_use_aux.sum().item())
        current_valid_counts = current_valid_mask.sum(dim=-1)
        threshold = self._threshold(current_valid_counts)
        source_reliable = (
            self._masked_entropy(source_current, current_valid_mask) < threshold
        )
        self.source_confident_steps += int(source_reliable.sum().item())
        if (
            pending_valid_action_count is not None
            or pending_valid_action_mask is not None
        ):
            self.valid_action_metadata_steps += 1
        else:
            self.valid_action_fallback_steps += 1
        self.masked_invalid_action_slots += int(
            (~current_valid_mask).sum().item()
        )
        if current_entry is not None:
            # The paper describes memory units as observations plus action
            # decisions.  These snapshots are diagnostic/reconstructive only;
            # replay still recomputes both branches as Algorithm 3 specifies.
            current_entry["source_decision"] = _tree_to_cpu(source_current)
            current_entry["auxiliary_decision"] = _tree_to_cpu(aux_current)
            current_entry["final_decision"] = _tree_to_cpu(combined_current)

        self._cached_current = {
            "entry": current_entry,
            "replay_ready": replay_ready,
            "source_batch": source_batch,
            "aux_batch": aux_batch,
            "combined_batch": combined_batch,
            "use_aux": use_aux_batch,
            "valid_action_masks": valid_mask_batch,
        }
        self._record_loss(
            self._masked_entropy(combined_current, current_valid_mask).mean()
        )
        return combined_current

    def _reservoir_add(
        self,
        policy_inputs,
        action=None,
        valid_action_count=None,
        valid_action_mask=None,
    ):
        entry = {
            "policy_inputs": _tree_to_cpu(policy_inputs),
            "valid_action_count": _tree_to_cpu(valid_action_count),
            "valid_action_mask": _tree_to_cpu(valid_action_mask),
            "source_decision": None,
            "auxiliary_decision": None,
            "final_decision": None,
            "action": (
                None if action is None else action.detach().cpu().clone()
            ),
        }
        self.seen_samples += 1
        if len(self.replay) < self.memory_size:
            self.replay.append(entry)
            return entry
        index = random.randrange(self.seen_samples)
        if index < self.memory_size:
            self.replay[index] = entry
            return entry
        return None

    @torch.enable_grad()
    def adapt(self, logits, action=None, **kwargs):
        if self._cached_current is None:
            raise RuntimeError("EAM.prepare_action must run before EAM.adapt")
        cached = self._cached_current
        current_entry = cached["entry"]
        if current_entry is not None:
            current_entry["action"] = (
                None if action is None else _tree_to_cpu(action)
            )
        can_update = (
            cached["replay_ready"]
            and self.action_steps % self.update_interval == 0
        )

        if can_update:
            source_batch = cached["source_batch"]
            aux_batch = cached["aux_batch"]
            combined_batch = cached["combined_batch"]
            use_aux_batch = cached["use_aux"]
            valid_mask_batch = cached["valid_action_masks"]
            self.update_attempt_count += 1
            self.replayed_step_count += sum(
                int(decision.shape[0]) for decision in source_batch
            )
            self.replay_aux_used_steps += sum(
                int(use_aux.sum().item()) for use_aux in use_aux_batch
            )
            reliable_losses = []
            reliable_count = 0
            for (
                source_decision,
                auxiliary_decision,
                combined_decision,
                decision_valid_mask,
            ) in zip(
                source_batch, aux_batch, combined_batch, valid_mask_batch
            ):
                threshold = self._threshold(decision_valid_mask.sum(dim=-1))
                reliable = (
                    self._masked_entropy(
                        source_decision, decision_valid_mask
                    ) < threshold
                )
                if not bool(reliable.any()):
                    continue
                pseudo = combined_decision.detach().argmax(dim=-1)
                masked_auxiliary = auxiliary_decision.masked_fill(
                    ~decision_valid_mask,
                    torch.finfo(auxiliary_decision.dtype).min,
                )
                flat_reliable = reliable.reshape(-1)
                reliable_losses.append(F.cross_entropy(
                    masked_auxiliary.reshape(
                        -1, masked_auxiliary.shape[-1]
                    )[flat_reliable],
                    pseudo.reshape(-1)[flat_reliable],
                    reduction="sum",
                ))
                reliable_count += int(reliable.sum().item())
            if reliable_count:
                loss = torch.stack(reliable_losses).sum() / float(reliable_count)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.params,
                    self.max_grad_norm if self.max_grad_norm > 0 else math.inf,
                )
                self.last_grad_norm = float(grad_norm.item())
                wrote_parameters = self._optimizer_step(self.optimizer)
                self.optimizer.zero_grad(set_to_none=True)
                loss_value = float(loss.detach().item())
                if wrote_parameters:
                    self.update_count += 1
                self.accepted_samples += reliable_count
                self.train_loss_sum += loss_value
                self.last_train_loss = loss_value
            else:
                self.no_reliable_update_count += 1

        self._cached_current = None
        return None

    def reset(self):
        self._parameter_write(
            lambda: self.aux_model.load_state_dict(
                self._model_state, strict=True
            )
        )
        self.optimizer.load_state_dict(self._optim_state)
        self.replay = []
        self.seen_samples = 0
        self.accepted_samples = 0
        self.aux_used_steps = 0
        self.source_confident_steps = 0
        self.episode_count = 0
        self._pending_replay = None
        self._cached_current = None
        self.update_attempt_count = 0
        self.no_reliable_update_count = 0
        self.replayed_step_count = 0
        self.replay_aux_used_steps = 0
        self.current_replay_duplicates = 0
        self.warmup_no_update_steps = 0
        self.valid_action_metadata_steps = 0
        self.valid_action_fallback_steps = 0
        self.masked_invalid_action_slots = 0
        self.current_only_batches = 0
        self.train_loss_sum = 0.0
        self.last_train_loss = 0.0
        self._init_diagnostics()

    def episode_start(self):
        if self._pending_replay is not None or self._cached_current is not None:
            raise RuntimeError(
                "EAM episode_start called before the previous step adapted"
            )

    def episode_end(self, episode_stats=None, **kwargs):
        if self._pending_replay is not None or self._cached_current is not None:
            raise RuntimeError("EAM episode ended before its final step adapted")
        self.episode_count += 1

    def diagnostics(self):
        output = super().diagnostics()
        output.update({
            "replay_size": len(self.replay),
            "replay_unit": "action_step",
            "replay_storage": (
                "full_policy_input_action_and_decision_snapshot"
            ),
            "replay_sampling": "updated_reservoir_including_current",
            "short_buffer_behavior": "warmup_no_update",
            "update_timing": "after_preupdate_action_selection",
            "seen_samples": self.seen_samples,
            "seen_steps": self.seen_samples,
            "accepted_samples": self.accepted_samples,
            "aux_used_steps": self.aux_used_steps,
            "source_confident_steps": self.source_confident_steps,
            "source_gate_rate": (
                self.source_confident_steps / max(1, self.action_steps)
            ),
            "aux_gate_rate": self.aux_used_steps / max(1, self.action_steps),
            "update_attempts": self.update_attempt_count,
            "updates_skipped_no_reliable": self.no_reliable_update_count,
            "replayed_steps": self.replayed_step_count,
            "replay_aux_used_steps": self.replay_aux_used_steps,
            "current_replay_duplicates": self.current_replay_duplicates,
            "warmup_no_update_steps": self.warmup_no_update_steps,
            # Deprecated alias retained for old result readers.  Despite its
            # historical name, these are inference-only warm-up steps.
            "current_only_batches": self.current_only_batches,
            "current_only_batches_semantics": (
                "legacy_alias_of_warmup_no_update_steps"
            ),
            "mean_train_loss": (
                self.train_loss_sum / max(1, self.update_count)
            ),
            "last_train_loss": self.last_train_loss,
            "confidence_scale": self.confidence_scale,
            "optimizer": self.optimizer.__class__.__name__,
            "adapted_parameter_count": sum(
                parameter.numel() for parameter in self.params
            ),
            "param_scope": self.param_scope,
            "trainable_prefixes": list(self.trainable_prefixes),
            "memory_size_steps": self.memory_size,
            "batch_size_steps": self.batch_size,
            "update_interval": self.update_interval,
            "update_interval_unit": "action_step",
            "loss_reduction": "mean_over_reliable_replay_steps",
            "reliability_action_space": (
                "per_sample_valid_action_count_or_mask"
            ),
            "valid_action_api": (
                "valid_action_count_or_valid_action_mask"
            ),
            "valid_action_metadata_steps": self.valid_action_metadata_steps,
            "valid_action_fallback_steps": self.valid_action_fallback_steps,
            "masked_invalid_action_slots": self.masked_invalid_action_slots,
        })
        if self.zero_update_audit:
            before = self.audit_model_states_before_sha256 or {}
            after = self.audit_model_states_after_sha256 or {}
            output.update({
                "auxiliary_model_state_before_sha256": before.get(
                    "auxiliary_model"
                ),
                "auxiliary_model_state_after_sha256": after.get(
                    "auxiliary_model"
                ),
                "auxiliary_model_state_hash_match": (
                    None if not after else
                    after.get("auxiliary_model")
                    == before.get("auxiliary_model")
                ),
            })
        return output


class FEEDTTAAdapter(_AdapterDiagnostics):
    """Episode-feedback policy-gradient adaptation with SGR.

    Modality encoders remain frozen.  The AVN mapping of the paper's
    "cross-modal encoder and subsequent modules" is expressed explicitly by
    ``trainable_prefixes`` so that task-specific module names do not leak into
    the shared implementation.  Per-action score gradients are folded into a
    single discounted accumulator, which is exactly equivalent to Eq. (3) but
    keeps memory O(number of adapted parameters), independent of trajectory
    length.  The paper writes trajectories as ``tau ~ pi`` but does not state
    the action selector used in its released experiments.  The adapter accepts
    the action actually executed by the task runner so task glue can preserve
    either an on-policy sample or a target-native deterministic policy and
    report that protocol explicitly.
    """

    def __init__(
        self,
        model,
        lr=5e-6,
        reversal_probability=0.05,
        reversal_scale=-0.2,
        sgr_seed=0,
        gamma=0.99,
        normalize_gradient=False,
        episodic=False,
        reset_bn_stats=False,
        scope="module_prefixes",
        last_k=4,
        trainable_prefixes=(),
        optimizer_name="Adam",
        momentum=0.9,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        optimizer_eps=1e-5,
        max_grad_norm=0.0,
        action_selection_protocol="sample_from_policy",
        sgr_mode="paper_main",
    ):
        self.model = model
        self.base_lr = float(lr)
        self.reversal_probability = float(reversal_probability)
        self.reversal_scale = float(reversal_scale)
        self.sgr_seed = int(sgr_seed)
        self.sgr_mode = str(sgr_mode).lower()
        valid_sgr_modes = ("paper_main", "appendix_b1")
        if self.sgr_mode not in valid_sgr_modes:
            raise ValueError(
                "Unknown FEEDTTA.SGR_MODE {!r}; expected one of {}".format(
                    sgr_mode, valid_sgr_modes
                )
            )
        self.gamma = float(gamma)
        self.normalize_gradient = bool(normalize_gradient)
        self.action_selection_protocol = str(action_selection_protocol)
        if not self.action_selection_protocol:
            raise ValueError(
                "FEEDTTA action_selection_protocol must be nonempty"
            )
        self.episodic = bool(episodic)
        self.max_grad_norm = float(max_grad_norm)
        if self.episodic:
            raise ValueError(
                "FEEDTTA updates only at episode end; EPISODIC=True would "
                "reset the update before it can affect the next episode"
            )
        if not 0.0 <= self.reversal_probability < 1.0:
            raise ValueError("FEEDTTA reversal probability must be in [0, 1)")
        if not math.isfinite(self.reversal_scale):
            raise ValueError("FEEDTTA reversal scale must be finite")
        if not math.isfinite(self.base_lr) or self.base_lr <= 0.0:
            raise ValueError("FEEDTTA learning rate must be finite and positive")
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("FEEDTTA gamma must be finite and in [0, 1]")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm < 0.0:
            raise ValueError(
                "FEEDTTA max grad norm must be finite and nonnegative"
            )
        denominator = (
            self.reversal_scale * self.reversal_probability
            + (1.0 - self.reversal_probability)
        )
        if denominator <= 0.0:
            raise ValueError("FEEDTTA SGR denominator must be positive")
        self._sgr_denominator = denominator
        if self.reversal_probability == 0.0:
            self.regularizer_variant = "none"
        elif self.reversal_scale < 0.0:
            self.regularizer_variant = "stochastic_gradient_reversion"
        elif self.reversal_scale == 0.0:
            self.regularizer_variant = "gradient_dropout"
        else:
            self.regularizer_variant = "gradient_scaling"
            logging.warning(
                "FEEDTTA ALPHA=%s is nonnegative, so the selected gradient "
                "coordinates are not reversed (variant=%s)",
                self.reversal_scale,
                self.regularizer_variant,
            )

        self.param_scope = str(scope).lower()
        self.trainable_prefixes = tuple(trainable_prefixes)
        if self.param_scope == "module_prefixes":
            self.params, self.names = configure_module_prefixes(
                model, self.trainable_prefixes
            )
        else:
            # Retained for explicit parameter-efficient ablations only.
            self.params, self.names = configure_tta_model(
                model, reset_bn_stats, self.param_scope, last_k
            )
        self.optimizer = _make_optimizer(
            self.params,
            optimizer_name,
            self.base_lr,
            momentum=momentum,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=optimizer_eps,
        )
        self._model_state = deepcopy(self.model.state_dict())
        self._optim_state = deepcopy(self.optimizer.state_dict())
        self._source_flat = _flatten_params(self.params).clone()
        self._trajectory_gradient = None
        self._trajectory_discount_mass = 0.0
        self._trajectory_step_count = 0
        self.episode_count = 0
        self.successful_episodes = 0
        self.failed_episodes = 0
        self.total_trajectory_steps = 0
        self.max_trajectory_steps = 0
        self.action_nll_sum = 0.0
        self.last_action_nll = 0.0
        self.sgr_selected_dimensions = 0
        self.sgr_sampled_dimensions = 0
        self.last_sgr_selected_dimensions = 0
        self.last_sgr_selected_fraction = 0.0
        self._sgr_generators = {}
        self.feedback_episode_count = 0
        self.episode_end_optimizer_attempt_count = 0
        self._init_diagnostics()

    def _clear_trajectory(self):
        self._trajectory_gradient = None
        self._trajectory_discount_mass = 0.0
        self._trajectory_step_count = 0

    def _accumulate_step_gradient(self, step_gradient):
        """Fold one score gradient into the exact Eq. (3) discounted sum."""
        if self._trajectory_gradient is None:
            self._trajectory_gradient = step_gradient.clone()
            self._trajectory_discount_mass = 1.0
        else:
            self._trajectory_gradient.mul_(self.gamma).add_(step_gradient)
            self._trajectory_discount_mass = (
                self.gamma * self._trajectory_discount_mass + 1.0
            )
        self._trajectory_step_count += 1

    @torch.enable_grad()
    def adapt(self, logits, action=None, **kwargs):
        if action is None:
            raise ValueError("FEEDTTA requires the executed action")
        action = action.long().view(-1, 1)
        nll = -logits.log_softmax(dim=-1).gather(1, action).mean()
        self.optimizer.zero_grad(set_to_none=True)
        nll.backward()
        self._accumulate_step_gradient(_flatten_grads(self.params))
        self.optimizer.zero_grad(set_to_none=True)
        self._record_loss(softmax_entropy(logits).mean())
        nll_value = float(nll.detach().item())
        self.action_nll_sum += nll_value
        self.last_action_nll = nll_value
        return nll.detach()

    def _aggregate_trajectory(self):
        if self._trajectory_gradient is None or self._trajectory_step_count == 0:
            raise RuntimeError("FEEDTTA has no trajectory gradient to aggregate")
        gradient = self._trajectory_gradient
        if self.normalize_gradient:
            gradient = gradient / max(self._trajectory_discount_mass, 1e-12)
        return gradient

    def _sgr_generator(self, device):
        key = (device.type, device.index)
        generator = self._sgr_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.sgr_seed)
            self._sgr_generators[key] = generator
        return generator

    def _sample_sgr_mask(self, grad):
        random_values = torch.rand(
            grad.shape,
            dtype=grad.dtype,
            device=grad.device,
            generator=self._sgr_generator(grad.device),
        )
        return random_values < self.reversal_probability

    def _apply_sgr(self, grad):
        if self.reversal_probability == 0.0:
            self.last_sgr_selected_dimensions = 0
            self.last_sgr_selected_fraction = 0.0
            return grad
        selected = self._sample_sgr_mask(grad)
        selected_count = int(selected.sum().item())
        dimension_count = int(grad.numel())
        self.last_sgr_selected_dimensions = selected_count
        self.last_sgr_selected_fraction = selected_count / max(1, dimension_count)
        self.sgr_selected_dimensions += selected_count
        self.sgr_sampled_dimensions += dimension_count
        transformed = torch.where(
            selected,
            self.reversal_scale * grad,
            grad,
        )
        if self.sgr_mode == "appendix_b1":
            # Appendix B.1 derives expectation preservation by normalizing the
            # complete masked gradient, including selected coordinates.
            return transformed / self._sgr_denominator
        # Canonical paper-main mode follows Eq. (5), where only unselected
        # coordinates receive the denominator.  It is deliberately explicit
        # because this formula is not the unbiased appendix transformation.
        return torch.where(
            selected,
            transformed,
            transformed / self._sgr_denominator,
        )

    def reset(self):
        self._parameter_write(
            lambda: self.model.load_state_dict(self._model_state, strict=True)
        )
        self.optimizer.load_state_dict(self._optim_state)
        self._clear_trajectory()
        self.episode_count = 0
        self.successful_episodes = 0
        self.failed_episodes = 0
        self.total_trajectory_steps = 0
        self.max_trajectory_steps = 0
        self.action_nll_sum = 0.0
        self.last_action_nll = 0.0
        self.sgr_selected_dimensions = 0
        self.sgr_sampled_dimensions = 0
        self.last_sgr_selected_dimensions = 0
        self.last_sgr_selected_fraction = 0.0
        self._sgr_generators = {}
        self.feedback_episode_count = 0
        self.episode_end_optimizer_attempt_count = 0
        self._init_diagnostics()

    def episode_start(self):
        if self._trajectory_gradient is not None or self._trajectory_step_count:
            raise RuntimeError(
                "FEEDTTA episode_start called before the previous episode ended"
            )

    def episode_end(self, episode_stats=None):
        success = _episode_success(episode_stats)
        self.episode_count += 1
        self.feedback_episode_count += 1
        self.successful_episodes += int(success)
        self.failed_episodes += int(not success)
        if self._trajectory_gradient is None:
            return
        # Optimizer performs descent on NLL. Success reinforces the executed
        # trajectory; failure applies the opposite direction. The task glue
        # records whether that trajectory was sampled or selected by argmax.
        grad = self._aggregate_trajectory()
        grad = grad if success else -grad
        grad = self._apply_sgr(grad)
        trajectory_steps = self._trajectory_step_count
        self.optimizer.zero_grad(set_to_none=True)
        self.last_grad_norm = _set_flat_grad(
            grad, self.params, self.max_grad_norm
        )
        self.episode_end_optimizer_attempt_count += 1
        wrote_parameters = self._optimizer_step(self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)
        if wrote_parameters:
            self.update_count += 1
        self.total_trajectory_steps += trajectory_steps
        self.max_trajectory_steps = max(
            self.max_trajectory_steps, trajectory_steps
        )
        self._clear_trajectory()

    def diagnostics(self):
        output = super().diagnostics()
        output.update({
            "successful_feedback_episodes": self.successful_episodes,
            "failed_feedback_episodes": self.failed_episodes,
            "feedback_type": "binary_episode_success",
            "feedback_values": "+1_success_-1_failure",
            "action_selection_protocol": self.action_selection_protocol,
            "update_timing": "once_after_episode_feedback",
            "reversal_probability": self.reversal_probability,
            "reversal_scale": self.reversal_scale,
            "sgr_seed": self.sgr_seed,
            "sgr_rng": "dedicated_torch_generator",
            "sgr_denominator": self._sgr_denominator,
            "sgr_mode": self.sgr_mode,
            "sgr_rule": (
                "main_text_eq5_unselected_coordinates_scaled"
                if self.sgr_mode == "paper_main"
                else "appendix_b1_all_coordinates_scaled"
            ),
            "sgr_expectation_preserving": self.sgr_mode == "appendix_b1",
            "regularizer_variant": self.regularizer_variant,
            "last_sgr_selected_dimensions": self.last_sgr_selected_dimensions,
            "last_sgr_selected_fraction": self.last_sgr_selected_fraction,
            "mean_sgr_selected_fraction": (
                self.sgr_selected_dimensions
                / max(1, self.sgr_sampled_dimensions)
            ),
            "param_scope": self.param_scope,
            "trainable_prefixes": list(self.trainable_prefixes),
            "adapted_parameter_count": sum(
                parameter.numel() for parameter in self.params
            ),
            "optimizer": self.optimizer.__class__.__name__,
            "gamma": self.gamma,
            "normalize_gradient": self.normalize_gradient,
            "episodic": self.episodic,
            "trajectory_gradient_storage": "online_discounted_accumulator",
            "gradient_accumulator_elements": int(self._source_flat.numel()),
            "total_trajectory_steps": self.total_trajectory_steps,
            "policy_gradient_steps": self.total_trajectory_steps,
            "feedback_episodes": self.feedback_episode_count,
            "episode_end_optimizer_attempts": (
                self.episode_end_optimizer_attempt_count
            ),
            "max_trajectory_steps": self.max_trajectory_steps,
            "mean_trajectory_steps": (
                self.total_trajectory_steps / max(1, self.update_count)
            ),
            "current_trajectory_steps": self._trajectory_step_count,
            "mean_action_nll": self.action_nll_sum / max(1, self.action_steps),
            "last_action_nll": self.last_action_nll,
        })
        return output


class ATENAAdapter(_AdapterDiagnostics):
    """Official-code-aligned active episodic mixture-entropy optimization.

    The official DUET/ETPNav implementations retain the full episode graph and
    apply one update at episode end.  AVN episodes can contain hundreds of raw
    RGB/depth/audio forwards, so retaining that graph is not practical.  This
    adapter stores detached policy inputs on CPU and replays one action step at
    a time after the binary episode outcome is known.  The accumulated gradient
    is exactly the gradient of the official joint mean loss in eval mode, while
    peak GPU activation memory is independent of episode length.
    """

    # The online action pass only records detached inputs/features.  Gradients
    # are reconstructed at episode end after the success/failure sign is known.
    requires_source_grad = False

    def __init__(
        self,
        model,
        lr_query=1e-6,
        lr_self=1e-7,
        mix_lambda=0.5,
        query_threshold=0.1,
        self_loss_weight=0.1,
        episodic=False,
        reset_bn_stats=False,
        scope="all",
        last_k=4,
        optimizer_name="AdamW",
        momentum=0.9,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.01,
        max_grad_norm=0.0,
        forward_policy=None,
    ):
        numeric_values = {
            "LR_QUERY": float(lr_query),
            "LR_SELF": float(lr_self),
            "MIX_LAMBDA": float(mix_lambda),
            "QUERY_THRESHOLD": float(query_threshold),
            "SELF_LOSS_WEIGHT": float(self_loss_weight),
            "MAX_GRAD_NORM": float(max_grad_norm),
        }
        if any(not math.isfinite(value) for value in numeric_values.values()):
            raise ValueError("ATENA hyperparameters must be finite")
        if bool(episodic):
            raise ValueError(
                "ATENA requires TTA.EPISODIC=False; resetting at the next "
                "episode would discard every episode-end update"
            )
        if float(lr_query) <= 0.0 or float(lr_self) <= 0.0:
            raise ValueError("ATENA learning rates must be positive")
        if not 0.0 <= float(mix_lambda) <= 1.0:
            raise ValueError("ATENA mixture lambda must be in [0, 1]")
        if float(query_threshold) < 0.0:
            raise ValueError("ATENA query threshold must be nonnegative")
        if float(self_loss_weight) < 0.0:
            raise ValueError("ATENA self-loss weight must be nonnegative")
        if float(max_grad_norm) < 0.0:
            raise ValueError("ATENA max grad norm must be nonnegative")
        self.model = model
        self.base_lr = float(lr_query)
        self.lr_query = float(lr_query)
        self.lr_self = float(lr_self)
        self.mix_lambda = float(mix_lambda)
        self.query_threshold = float(query_threshold)
        self.self_loss_weight = float(self_loss_weight)
        self.episodic = False
        self.max_grad_norm = float(max_grad_norm)
        self.param_scope = str(scope).lower()
        if forward_policy is not None and not callable(forward_policy):
            raise TypeError("ATENA forward_policy must be callable")
        self.forward_policy_callback = forward_policy

        named_parameters = list(self.model.named_parameters())
        noncritic_parameters = [
            (name, param) for name, param in named_parameters
            if not name.startswith("critic.")
        ]
        critic_parameters = [
            (name, param) for name, param in named_parameters
            if name.startswith("critic.")
        ]
        self.policy_parameter_count = sum(
            param.numel() for _, param in noncritic_parameters
        )
        self.preexisting_frozen_policy_parameter_count = sum(
            param.numel() for _, param in noncritic_parameters
            if not param.requires_grad
        )
        self.excluded_top_level_critic_parameter_count = sum(
            param.numel() for _, param in critic_parameters
        )

        if self.param_scope == "all":
            # Match the official implementation's optimizer over the complete
            # action policy while preserving parameters that the baseline
            # itself intentionally froze.  Habitat's actor_critic also contains
            # a value-only critic; DUET's official vln_bert optimizer has no
            # equivalent critic, and the navigation loss never uses it.
            self.model.eval()
            selected = [
                (name, param) for name, param in self.model.named_parameters()
                if param.requires_grad and not name.startswith("critic.")
            ]
            if not selected:
                raise ValueError("ATENA found no trainable policy parameters")
            self.names = [name for name, _ in selected]
            self.params = [param for _, param in selected]
            if self.preexisting_frozen_policy_parameter_count:
                logging.warning(
                    "ATENA scope='all' can only update pre-existing trainable "
                    "non-critic parameters; %d frozen policy scalars require "
                    "task-level trainability wiring before replay reachability "
                    "can be verified",
                    self.preexisting_frozen_policy_parameter_count,
                )
        else:
            # Retained only for explicitly named parameter-efficient ablations.
            self.params, self.names = configure_tta_model(
                model, reset_bn_stats, self.param_scope, last_k
            )
        self._optimizer_args = dict(
            name=optimizer_name,
            momentum=momentum,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
        )
        self.optimizer = None
        self._model_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        self._source_flat = _flatten_params(self.params).clone()

        self.self_prediction_head = None
        self._head_state = None
        self.audit_head_state_before_sha256 = None
        self.audit_head_state_after_sha256 = None
        self.trajectory = []
        self.trajectory_features = []
        self.trajectory_mixture_entropies = []
        self.trajectory_entropies = []
        self.action_dim = None
        self.episode_count = 0
        self.query_count = 0
        self.query_prediction_correct = 0
        self.self_label_count = 0
        self.self_feedback_successes = 0
        self.queried_feedback_successes = 0
        self.feedback_observed_count = 0
        self.episode_entropy_sum = 0.0
        self.query_entropy_sum = 0.0
        self.self_entropy_sum = 0.0
        self.self_loss_sum = 0.0
        self.last_self_loss = 0.0
        self.max_trajectory_steps = 0
        self.max_trajectory_storage_bytes = 0
        self.last_trajectory_storage_bytes = 0
        self.max_replay_feature_error = 0.0
        self.adaptation_time_seconds = 0.0
        self.max_episode_adaptation_seconds = 0.0
        self.query_gate_evaluation_count = 0
        self.self_prediction_evaluation_count = 0
        self.replayed_step_count = 0
        # ``scope='all'`` is only a candidate scope until a real task replay
        # callback has been differentiated.  Task models often expose modules
        # (for example pretraining/object heads) that are not on the deployed
        # navigation path.  They must never be silently placed in ATENA's
        # episode optimizer.
        self.replay_reachability_preflight_performed = False
        self.replay_reachability_validated = False
        self.replay_reachability_validation_result = "not_run"
        self.replay_reachability_candidate_names = list(self.names)
        self.replay_reachable_parameter_names = []
        self.replay_unreachable_parameter_names = []
        self.replay_determinism_validated_episodes = 0
        self._init_diagnostics()

    def _ensure_head(self, feature_dim, device, dtype):
        if self.self_prediction_head is not None:
            expected = int(self.self_prediction_head[0].in_features)
            if int(feature_dim) != expected:
                raise ValueError(
                    "ATENA policy feature dimension changed from {} to {}"
                    .format(expected, int(feature_dim))
                )
            return
        feature_dim = int(feature_dim)
        self.self_prediction_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim, eps=1e-12),
            nn.Linear(feature_dim, 1),
        ).to(device=device, dtype=dtype)
        # DUET uses the BERT module initializer for this MLP.
        for module in self.self_prediction_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self._head_state = deepcopy(self.self_prediction_head.state_dict())
        if self.zero_update_audit:
            self.audit_head_state_before_sha256 = module_state_sha256(
                self.self_prediction_head
            )
            self.audit_head_state_after_sha256 = None

    def _build_episode_optimizer(self, lr):
        # The official code constructs a fresh AdamW after each episode, so no
        # optimizer moments are shared between queried and self-labelled data.
        if (
            not self.replay_reachability_validated
            or self.names != self.replay_reachable_parameter_names
        ):
            raise RuntimeError(
                "ATENA refuses to construct an optimizer before replay "
                "reachability is validated"
            )
        parameters = self.params + list(self.self_prediction_head.parameters())
        return _make_optimizer(
            parameters,
            lr=float(lr),
            **self._optimizer_args
        )

    @property
    def device(self):
        return self.params[0].device

    @staticmethod
    def _default_forward_policy(model, policy_inputs):
        features, _, _ = model.net(
            policy_inputs["observations"],
            policy_inputs["rnn_hidden_states"],
            policy_inputs["prev_actions"],
            policy_inputs["masks"],
            policy_inputs.get("ext_memory"),
            policy_inputs.get("ext_memory_masks"),
        )
        logits = model.action_distribution(features).logits
        return features, logits

    def _forward_policy(self, model, policy_inputs):
        callback = self.forward_policy_callback
        output = (
            self._default_forward_policy(model, policy_inputs)
            if callback is None
            else callback(model, policy_inputs)
        )
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError(
                "ATENA forward_policy must return (features, logits)"
            )
        features, logits = output
        if (
            not torch.is_tensor(features)
            or not torch.is_tensor(logits)
            or features.ndim < 2
            or logits.ndim < 2
        ):
            raise ValueError(
                "ATENA forward_policy returned invalid features or logits"
            )
        return features, logits

    @torch.enable_grad()
    def _preflight_replay_reachability(self, policy_inputs):
        """Bind the optimizer scope to the real replay computation graph.

        This runs exactly once, on the first real navigation input, before an
        episode optimizer exists.  Both callback outputs are differentiated:
        logits carry ATENA's mixture-entropy loss and features carry the
        auxiliary self-prediction loss.  A non-critic candidate is retained
        only when at least one of those outputs is connected to it.
        """
        if self.replay_reachability_preflight_performed:
            if not self.replay_reachability_validated:
                raise RuntimeError("ATENA replay reachability preflight failed")
            return
        if self.optimizer is not None or self.action_steps or self.update_count:
            raise RuntimeError(
                "ATENA replay reachability must be verified before adaptation"
            )

        candidate_params = list(self.params)
        candidate_names = list(self.names)
        if not candidate_params:
            raise RuntimeError("ATENA has no replay reachability candidates")
        if any(not parameter.requires_grad for parameter in candidate_params):
            raise RuntimeError(
                "ATENA replay reachability candidates must require gradients"
            )

        self.replay_reachability_validation_result = "running"
        try:
            replay_inputs = _tree_to_device(
                policy_inputs, candidate_params[0].device
            )
            replay_features, replay_logits = self._forward_policy(
                self.model, replay_inputs
            )
            # Use both true callback outputs. Replacing masked non-finite
            # logits keeps the scalar finite without changing connectivity.
            objective = replay_features.float().sum()
            finite_logits = torch.where(
                torch.isfinite(replay_logits),
                replay_logits,
                torch.zeros_like(replay_logits),
            )
            objective = objective + finite_logits.float().sum()
            if not objective.requires_grad:
                raise RuntimeError(
                    "ATENA replay callback outputs are detached from every "
                    "candidate policy parameter"
                )
            # None, rather than numerical magnitude, is the structural
            # reachability test; locally zero gradients remain valid.
            gradients = torch.autograd.grad(
                objective,
                candidate_params,
                allow_unused=True,
            )
        except Exception:
            self.replay_reachability_preflight_performed = True
            self.replay_reachability_validation_result = "failed"
            raise
        reachable = [
            (name, parameter)
            for name, parameter, gradient in zip(
                candidate_names, candidate_params, gradients
            )
            if gradient is not None
        ]
        unreachable = [
            (name, parameter)
            for name, parameter, gradient in zip(
                candidate_names, candidate_params, gradients
            )
            if gradient is None
        ]
        self.replay_reachability_preflight_performed = True
        self.replay_reachable_parameter_names = [
            name for name, _ in reachable
        ]
        self.replay_unreachable_parameter_names = [
            name for name, _ in unreachable
        ]
        if not reachable:
            self.replay_reachability_validation_result = "failed"
            raise RuntimeError(
                "ATENA replay callback reaches no non-critic policy parameter"
            )

        # This changes autograd bookkeeping only; no parameter value and no
        # optimizer state is written during the preflight.
        for _, parameter in unreachable:
            parameter.requires_grad_(False)
            parameter.grad = None
        self.names = [name for name, _ in reachable]
        self.params = [parameter for _, parameter in reachable]
        self._source_flat = _flatten_params(self.params).clone()
        if self.zero_update_audit:
            # Audit mode is enabled at adapter construction time, before task
            # inputs exist. Rebind its parameter digest to the now-validated
            # effective optimizer scope; the complete model digest is unchanged.
            self.audit_parameter_state_before_sha256 = (
                _parameter_state_sha256(self.params, self.names)
            )
            self.audit_parameter_state_after_sha256 = None
        self.replay_reachability_validated = True
        self.replay_reachability_validation_result = "passed"

    def _mixture_entropy(self, logits, action):
        probs = logits.softmax(dim=-1)
        one_hot = F.one_hot(
            action.long().view(-1), num_classes=int(logits.shape[-1])
        ).to(probs.dtype)
        mixture = self.mix_lambda * one_hot + (1.0 - self.mix_lambda) * probs
        return -(
            mixture * mixture.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()

    @staticmethod
    def _install_accumulated_gradients(params, gradients, seen):
        for param, gradient, was_seen in zip(params, gradients, seen):
            param.grad = gradient if was_seen else None

    def select_action(self, distribution):
        # Eq. (1) defines the selected pseudo-expert action as policy argmax,
        # and the official evaluation executes that same greedy action.
        return distribution.probs.argmax(dim=-1, keepdim=True)

    @torch.no_grad()
    def adapt(
        self,
        logits,
        action=None,
        features=None,
        policy_inputs=None,
        **kwargs
    ):
        if action is None or features is None or policy_inputs is None:
            raise ValueError(
                "ATENA requires executed actions, policy features, and "
                "policy_inputs for exact episode-end gradient replay"
            )
        self._preflight_replay_reachability(policy_inputs)
        self.action_dim = int(logits.shape[-1])
        self._ensure_head(features.shape[-1], features.device, features.dtype)
        pseudo_action = logits.detach().argmax(dim=-1)
        if not torch.equal(action.detach().long().view(-1), pseudo_action.view(-1)):
            raise ValueError(
                "ATENA requires the executed action to equal the policy argmax"
            )
        mixture_entropy = self._mixture_entropy(logits, pseudo_action)
        original_entropy = softmax_entropy(logits).mean()
        snapshot = _tree_to_cpu(policy_inputs)
        self.trajectory.append({
            "policy_inputs": snapshot,
            "action": action.detach().cpu().clone(),
        })
        self.trajectory_features.append(features.detach().cpu().clone())
        self.trajectory_mixture_entropies.append(float(mixture_entropy.item()))
        self.trajectory_entropies.append(float(original_entropy.item()))
        self._record_loss(mixture_entropy)
        self.last_trajectory_storage_bytes += (
            _tree_tensor_bytes(snapshot)
            + _tree_tensor_bytes(self.trajectory_features[-1])
            + _tree_tensor_bytes(self.trajectory[-1]["action"])
        )
        self.max_trajectory_storage_bytes = max(
            self.max_trajectory_storage_bytes,
            self.last_trajectory_storage_bytes,
        )
        return mixture_entropy

    def reset(self):
        self._parameter_write(
            lambda: self.model.load_state_dict(self._model_state, strict=True)
        )
        if self.self_prediction_head is not None and self._head_state is not None:
            self._parameter_write(
                lambda: self.self_prediction_head.load_state_dict(
                    self._head_state, strict=True
                )
            )
        self.optimizer = None
        self.trajectory = []
        self.trajectory_mixture_entropies = []
        self.trajectory_features = []
        self.trajectory_entropies = []
        self.episode_count = 0
        self.query_count = 0
        self.query_prediction_correct = 0
        self.self_label_count = 0
        self.self_feedback_successes = 0
        self.queried_feedback_successes = 0
        self.feedback_observed_count = 0
        self.episode_entropy_sum = 0.0
        self.query_entropy_sum = 0.0
        self.self_entropy_sum = 0.0
        self.self_loss_sum = 0.0
        self.last_self_loss = 0.0
        self.max_trajectory_steps = 0
        self.max_trajectory_storage_bytes = 0
        self.last_trajectory_storage_bytes = 0
        self.max_replay_feature_error = 0.0
        self.adaptation_time_seconds = 0.0
        self.max_episode_adaptation_seconds = 0.0
        self.query_gate_evaluation_count = 0
        self.self_prediction_evaluation_count = 0
        self.replayed_step_count = 0
        self.replay_determinism_validated_episodes = 0
        self._init_diagnostics()

    def episode_start(self):
        if self.trajectory:
            raise RuntimeError(
                "ATENA episode_start called before the previous episode ended"
            )
        self.trajectory = []
        self.trajectory_mixture_entropies = []
        self.trajectory_features = []
        self.trajectory_entropies = []
        self.last_trajectory_storage_bytes = 0

    @torch.enable_grad()
    def episode_end(self, episode_stats=None):
        import time

        self.episode_count += 1
        if not self.trajectory:
            return
        adaptation_started = time.perf_counter()

        trajectory_steps = len(self.trajectory)
        self.max_trajectory_steps = max(self.max_trajectory_steps, trajectory_steps)
        mean_feature = torch.cat(
            self.trajectory_features, dim=0
        ).to(self.device).mean(0, keepdim=True).detach().requires_grad_(True)
        prediction_logit = self.self_prediction_head(mean_feature).view(())
        self.self_prediction_evaluation_count += 1
        predicted_success = bool(torch.sigmoid(prediction_logit.detach()) > 0.5)
        average_entropy = sum(self.trajectory_entropies) / len(self.trajectory_entropies)
        self.episode_entropy_sum += average_entropy
        query = average_entropy > self.query_threshold
        self.query_gate_evaluation_count += 1
        if query:
            # Access evaluator/human feedback only for actively queried
            # episodes. Non-query episodes remain self-supervised.
            actual_success = _episode_success(episode_stats)
            feedback_success = actual_success
            self.query_count += 1
            self.feedback_observed_count += 1
            self.query_prediction_correct += int(predicted_success == actual_success)
            self.queried_feedback_successes += int(actual_success)
            self.query_entropy_sum += average_entropy
            policy_lr = self.lr_query
        else:
            feedback_success = predicted_success
            self.self_label_count += 1
            self.self_feedback_successes += int(feedback_success)
            self.self_entropy_sum += average_entropy
            policy_lr = self.lr_self

        target = torch.tensor(
            float(feedback_success), device=prediction_logit.device
        )
        self_prediction_loss = F.binary_cross_entropy_with_logits(
            prediction_logit, target
        )
        self.last_self_loss = float(self_prediction_loss.detach().item())
        self.self_loss_sum += self.last_self_loss

        self.optimizer = self._build_episode_optimizer(policy_lr)
        self.current_lr = policy_lr
        self.optimizer.zero_grad(set_to_none=True)
        head_parameters = list(self.self_prediction_head.parameters())
        weighted_self_loss = self.self_loss_weight * self_prediction_loss
        self_gradients = torch.autograd.grad(
            weighted_self_loss,
            [mean_feature] + head_parameters,
            allow_unused=True,
        )
        feature_gradient = self_gradients[0]
        if feature_gradient is None:
            feature_gradient = torch.zeros_like(mean_feature)

        accumulated = [torch.zeros_like(param) for param in self.params]
        gradient_seen = [False for _ in self.params]
        feedback_sign = 1.0 if feedback_success else -1.0
        replay_scale = 1.0 / float(trajectory_steps)

        for index, entry in enumerate(self.trajectory):
            self.replayed_step_count += 1
            replay_inputs = _tree_to_device(entry["policy_inputs"], self.device)
            replay_features, replay_logits = self._forward_policy(
                self.model, replay_inputs
            )
            replay_action = entry["action"].to(self.device)
            mixture_entropy = self._mixture_entropy(
                replay_logits, replay_action
            )
            step_objective = (
                feedback_sign * replay_scale * mixture_entropy
                + replay_scale * (
                    replay_features * feature_gradient.detach()
                ).sum()
            )
            gradients = torch.autograd.grad(
                step_objective,
                self.params,
                allow_unused=True,
            )
            for param_index, gradient in enumerate(gradients):
                if gradient is not None:
                    accumulated[param_index].add_(gradient.detach())
                    gradient_seen[param_index] = True

            # Verify deterministic replay once per episode.  A mismatch means
            # dropout/state mutation would invalidate gradient equivalence.
            if index == 0:
                reference = self.trajectory_features[0].to(self.device)
                replay_error = float(
                    (replay_features.detach() - reference).abs().max().item()
                )
                self.max_replay_feature_error = max(
                    self.max_replay_feature_error, replay_error
                )
                if replay_error > 1e-5:
                    raise RuntimeError(
                        "ATENA episode replay is not deterministic "
                        "(feature max abs error {:.6g})".format(replay_error)
                    )
                self.replay_determinism_validated_episodes += 1

        self._install_accumulated_gradients(
            self.params, accumulated, gradient_seen
        )
        for parameter, gradient in zip(head_parameters, self_gradients[1:]):
            parameter.grad = None if gradient is None else gradient.detach()
        optimized_parameters = (
            self.params + head_parameters
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            optimized_parameters,
            self.max_grad_norm if self.max_grad_norm > 0 else math.inf,
        )
        self.last_grad_norm = float(grad_norm.item())
        wrote_parameters = self._optimizer_step(self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)
        if wrote_parameters:
            self.update_count += 1
        adaptation_seconds = time.perf_counter() - adaptation_started
        self.adaptation_time_seconds += adaptation_seconds
        self.max_episode_adaptation_seconds = max(
            self.max_episode_adaptation_seconds, adaptation_seconds
        )

        self.trajectory = []
        self.trajectory_mixture_entropies = []
        self.trajectory_features = []
        self.trajectory_entropies = []

    def diagnostics(self):
        output = super().diagnostics()
        exact_replay_validated = (
            self.replay_reachability_validated
            and self.replay_determinism_validated_episodes > 0
        )
        output.update({
            "queries": self.query_count,
            "query_rate": self.query_count / max(1, self.episode_count),
            "query_prediction_accuracy": (
                self.query_prediction_correct / max(1, self.query_count)
            ),
            "self_label_episodes": self.self_label_count,
            "query_gate_evaluations": self.query_gate_evaluation_count,
            "self_prediction_evaluations": (
                self.self_prediction_evaluation_count
            ),
            "replayed_steps": self.replayed_step_count,
            "self_feedback_successes": self.self_feedback_successes,
            "queried_feedback_successes": self.queried_feedback_successes,
            "feedback_observed_episodes": self.feedback_observed_count,
            "feedback_observation_rate": (
                self.feedback_observed_count / max(1, self.episode_count)
            ),
            "mean_episode_entropy": (
                self.episode_entropy_sum / max(1, self.episode_count)
            ),
            "mean_query_episode_entropy": (
                self.query_entropy_sum / max(1, self.query_count)
            ),
            "mean_self_episode_entropy": (
                self.self_entropy_sum / max(1, self.self_label_count)
            ),
            "mean_self_prediction_loss": (
                self.self_loss_sum / max(1, self.update_count)
            ),
            "last_self_prediction_loss": self.last_self_loss,
            "mix_lambda": self.mix_lambda,
            "query_threshold": self.query_threshold,
            "lr_query": self.lr_query,
            "lr_self": self.lr_self,
            "self_loss_weight": self.self_loss_weight,
            "param_scope": self.param_scope,
            "update_scope_semantics": (
                "preexisting_trainable_noncritic_policy_parameters"
                if self.param_scope == "all"
                else "normalization_affine_ablation"
            ),
            "policy_parameter_count": self.policy_parameter_count,
            "preexisting_frozen_policy_parameter_count": (
                self.preexisting_frozen_policy_parameter_count
            ),
            "excluded_top_level_critic_parameter_count": (
                self.excluded_top_level_critic_parameter_count
            ),
            "all_noncritic_parameters_preexisting_trainable": (
                self.param_scope == "all"
                and self.preexisting_frozen_policy_parameter_count == 0
            ),
            "requires_task_trainability_wiring": (
                self.param_scope == "all"
                and self.preexisting_frozen_policy_parameter_count > 0
            ),
            # Core can report Python parameter flags, but only task glue knows
            # whether those tensors are the complete action-policy path used by
            # its forward callback.
            "requires_task_policy_scope_verification": (
                self.param_scope == "all"
            ),
            "replay_reachability_preflight_performed": (
                self.replay_reachability_preflight_performed
            ),
            "replay_reachability_validated": (
                self.replay_reachability_validated
            ),
            "replay_reachability_validation_result": (
                self.replay_reachability_validation_result
            ),
            "replay_reachability_candidate_parameter_count": len(
                self.replay_reachability_candidate_names
            ),
            "replay_reachability_candidate_parameter_names": list(
                self.replay_reachability_candidate_names
            ),
            "replay_reachable_parameter_count": len(
                self.replay_reachable_parameter_names
            ),
            "replay_reachable_parameter_names": list(
                self.replay_reachable_parameter_names
            ),
            "replay_unreachable_parameter_count": len(
                self.replay_unreachable_parameter_names
            ),
            "replay_unreachable_parameter_names": list(
                self.replay_unreachable_parameter_names
            ),
            "optimizer_policy_parameter_names": list(self.names),
            "optimizer_policy_scope_matches_reachable": (
                self.replay_reachability_validated
                and self.names == self.replay_reachable_parameter_names
            ),
            "replay_determinism_validated_episodes": (
                self.replay_determinism_validated_episodes
            ),
            "action_selection": "argmax",
            "feedback": "binary_episode_success_or_self_prediction",
            "adapted_parameter_count": sum(
                parameter.numel() for parameter in self.params
            ),
            "self_prediction_parameter_count": (
                sum(
                    parameter.numel()
                    for parameter in self.self_prediction_head.parameters()
                ) if self.self_prediction_head is not None else 0
            ),
            "self_prediction_feature_dim": (
                int(self.self_prediction_head[0].in_features)
                if self.self_prediction_head is not None else 0
            ),
            "auxiliary_head_constructed": (
                self.self_prediction_head is not None
            ),
            "optimizer": (
                self.optimizer.__class__.__name__
                if self.optimizer is not None else self._optimizer_args["name"]
            ),
            "retains_episode_graph": False,
            "exact_episode_replay_enabled": exact_replay_validated,
            "exact_episode_replay_validated": exact_replay_validated,
            "gradient_reconstruction": (
                "exact_step_replay_in_eval_mode"
                if exact_replay_validated else
                "pending_reachability_and_determinism_validation"
            ),
            "replay_determinism_validation": (
                "first_step_feature_max_abs_error_le_1e-5"
                if self.replay_determinism_validated_episodes > 0 else
                "pending"
            ),
            "max_trajectory_steps": self.max_trajectory_steps,
            "last_trajectory_storage_bytes": self.last_trajectory_storage_bytes,
            "max_trajectory_storage_bytes": self.max_trajectory_storage_bytes,
            "max_replay_feature_abs_error": self.max_replay_feature_error,
            "mean_episode_adaptation_seconds": (
                self.adaptation_time_seconds / max(1, self.update_count)
            ),
            "max_episode_adaptation_seconds": (
                self.max_episode_adaptation_seconds
            ),
            "cuda_peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda" else 0
            ),
        })
        if self.zero_update_audit and self.self_prediction_head is not None:
            if (
                output["parameter_state_after_sha256"] is not None
                and self.audit_head_state_after_sha256 is None
            ):
                self.audit_head_state_after_sha256 = module_state_sha256(
                    self.self_prediction_head
                )
            output.update({
                "auxiliary_head_state_before_sha256": (
                    self.audit_head_state_before_sha256
                ),
                "auxiliary_head_state_after_sha256": (
                    self.audit_head_state_after_sha256
                ),
                "auxiliary_head_state_hash_match": (
                    None if self.audit_head_state_after_sha256 is None else
                    self.audit_head_state_after_sha256
                    == self.audit_head_state_before_sha256
                ),
            })
        return output


def build_adapter(model, tta_cfg, forward_policy=None, fusion_protocol=None):
    """Build a sequential test-time adapter from a config node.

    ``forward_policy`` is an optional task adapter used by replay-based
    methods.  It must return ``(features, logits)`` from a detached policy
    input snapshot.  Omitting it preserves the Habitat actor-critic path used
    by AVN.

    ``fusion_protocol`` is an optional task adapter required by IDEA.  It is an
    :class:`navtta_core.tta.idea.IDEAFusionProtocol` describing how to inject a
    soft prompt into the model's fusion module and read per-layer statistics.
    """
    method = str(getattr(tta_cfg, "METHOD", "none")).lower()
    if method in ("none", "", "source"):
        return None

    def value(key, default):
        return getattr(tta_cfg, key, default)

    audit_zero_update = bool(value("AUDIT_ZERO_UPDATE", False))
    audit_expected_episodes = value("AUDIT_EXPECTED_EPISODES", None)

    def finalize(adapter):
        return (
            adapter.enable_zero_update_audit(audit_expected_episodes)
            if audit_zero_update else adapter
        )

    common = dict(
        steps=int(value("STEPS", 1)),
        episodic=bool(value("EPISODIC", False)),
        reset_bn_stats=bool(value("RESET_BN_STATS", True)),
        scope=str(value("NORM_SCOPE", "last_k_ln")),
        last_k=int(value("LAST_K_LN", 4)),
        optimizer_name=str(value("OPTIMIZER", "Adam")),
        momentum=float(value("MOMENTUM", 0.9)),
        beta1=float(value("BETA1", 0.9)),
        beta2=float(value("BETA2", 0.999)),
        weight_decay=float(value("WEIGHT_DECAY", 0.0)),
        max_grad_norm=float(value("MAX_GRAD_NORM", 1.0)),
    )
    lr = float(value("LR", 1e-6))
    if method == "tent":
        return finalize(TentAdapter(
            model,
            lr=lr,
            update_interval=int(value("UPDATE_INTERVAL", 1)),
            max_updates_per_episode=int(
                value("MAX_UPDATES_PER_EPISODE", -1)
            ),
            **common
        ))
    if method == "fstta":
        fstta_cfg = getattr(tta_cfg, "FSTTA", None)

        def fvalue(key, default):
            return getattr(fstta_cfg, key, default) if fstta_cfg is not None else default

        # FAST and SLOW own independent optimizers. The defaults preserve the
        # released-code-style persistent AdamW semantics for both branches;
        # explicit exploration settings may change only the SLOW optimizer.
        common["optimizer_name"] = str(fvalue("OPTIMIZER", "AdamW"))
        common["beta1"] = float(fvalue("BETA1", 0.9))
        common["beta2"] = float(fvalue("BETA2", 0.99))
        common["weight_decay"] = float(fvalue("WEIGHT_DECAY", 0.0))
        return finalize(FSTTAAdapter(
            model,
            lr_fast=lr,
            lr_slow=float(fvalue("LR_SLOW", 1e-4)),
            M=int(fvalue("M", 3)),
            N=int(fvalue("N", 4)),
            q=float(fvalue("Q", 0.1)),
            rho=float(fvalue("RHO", 0.95)),
            tau=float(fvalue("TAU", 0.7)),
            a=float(fvalue("A", 0.9)),
            b=float(fvalue("B", 1.1)),
            use_slow=bool(fvalue("USE_SLOW", True)),
            fast_grad_mode=str(fvalue("FAST_GRAD_MODE", "concordant")),
            use_fast_lr_scaler=bool(
                fvalue("USE_FAST_LR_SCALER", True)
            ),
            slow_optimizer_name=(
                str(fvalue("SLOW_OPTIMIZER", "")).strip() or None
            ),
            slow_momentum=(
                None
                if float(fvalue("SLOW_MOMENTUM", -1.0)) < 0.0
                else float(fvalue("SLOW_MOMENTUM", -1.0))
            ),
            reset_slow_optimizer_each_window=bool(
                fvalue("RESET_SLOW_OPTIMIZER_EACH_WINDOW", False)
            ),
            reset_optimizer_each_episode=bool(
                fvalue("RESET_OPTIMIZER_EACH_EPISODE", True)
            ),
            reset_var_hist_each_episode=bool(
                fvalue("RESET_VAR_HIST_EACH_EPISODE", False)
            ),
            eigen_eps=float(fvalue("EIGEN_EPS", 1e-6)),
            **common
        ))
    if method == "eam":
        eam_cfg = getattr(tta_cfg, "EAM", None)

        def evalue(key, default):
            return getattr(eam_cfg, key, default) if eam_cfg is not None else default

        return finalize(EAMAdapter(
            model,
            lr=float(evalue("LR", 1e-5)),
            confidence_scale=float(evalue("CONFIDENCE_SCALE", 0.4)),
            memory_size=int(evalue("MEMORY_SIZE", 32)),
            batch_size=int(evalue("BATCH_SIZE", 8)),
            update_interval=int(evalue("UPDATE_INTERVAL", 1)),
            param_scope=str(evalue("PARAM_SCOPE", "module_prefixes")),
            trainable_prefixes=tuple(evalue(
                "TRAINABLE_PREFIXES",
                (),
            )),
            optimizer_name=str(evalue("OPTIMIZER", "Adam")),
            momentum=float(evalue("MOMENTUM", 0.9)),
            beta1=float(evalue("BETA1", 0.9)),
            beta2=float(evalue("BETA2", 0.999)),
            weight_decay=float(evalue("WEIGHT_DECAY", 0.0)),
            max_grad_norm=float(evalue("MAX_GRAD_NORM", 0.0)),
            episodic=common["episodic"],
            forward_policy=forward_policy,
        ))
    if method == "feedtta":
        feed_cfg = getattr(tta_cfg, "FEEDTTA", None)

        def fdvalue(key, default):
            return getattr(feed_cfg, key, default) if feed_cfg is not None else default

        return finalize(FEEDTTAAdapter(
            model,
            lr=float(fdvalue("LR", 5e-6)),
            reversal_probability=float(fdvalue("P", 0.05)),
            reversal_scale=float(fdvalue("ALPHA", -0.2)),
            sgr_seed=int(fdvalue("SGR_SEED", 0)),
            sgr_mode=str(fdvalue("SGR_MODE", "paper_main")),
            gamma=float(fdvalue("GAMMA", 0.99)),
            normalize_gradient=bool(fdvalue("NORMALIZE_GRADIENT", False)),
            episodic=common["episodic"],
            reset_bn_stats=common["reset_bn_stats"],
            scope=str(fdvalue("PARAM_SCOPE", "module_prefixes")),
            last_k=common["last_k"],
            trainable_prefixes=tuple(fdvalue(
                "TRAINABLE_PREFIXES",
                (),
            )),
            optimizer_name=str(fdvalue("OPTIMIZER", "Adam")),
            momentum=common["momentum"],
            beta1=float(fdvalue("BETA1", 0.9)),
            beta2=float(fdvalue("BETA2", 0.999)),
            weight_decay=float(fdvalue("WEIGHT_DECAY", 0.0)),
            optimizer_eps=float(fdvalue("EPS", 1e-5)),
            max_grad_norm=float(fdvalue("MAX_GRAD_NORM", 0.0)),
            action_selection_protocol=str(fdvalue(
                "ACTION_SELECTION_PROTOCOL", "sample_from_policy"
            )),
        ))
    if method == "atena":
        atena_cfg = getattr(tta_cfg, "ATENA", None)

        def avalue(key, default):
            return getattr(atena_cfg, key, default) if atena_cfg is not None else default

        return finalize(ATENAAdapter(
            model,
            lr_query=float(avalue("LR_QUERY", 1e-6)),
            lr_self=float(avalue("LR_SELF", 1e-7)),
            mix_lambda=float(avalue("MIX_LAMBDA", 0.5)),
            query_threshold=float(avalue("QUERY_THRESHOLD", 0.1)),
            self_loss_weight=float(avalue("SELF_LOSS_WEIGHT", 0.1)),
            episodic=common["episodic"],
            reset_bn_stats=common["reset_bn_stats"],
            scope=str(avalue("PARAM_SCOPE", "all")),
            last_k=common["last_k"],
            optimizer_name=str(avalue("OPTIMIZER", "AdamW")),
            momentum=common["momentum"],
            beta1=float(avalue("BETA1", 0.9)),
            beta2=float(avalue("BETA2", 0.999)),
            weight_decay=float(avalue("WEIGHT_DECAY", 0.01)),
            max_grad_norm=float(avalue("MAX_GRAD_NORM", 0.0)),
            forward_policy=forward_policy,
        ))
    if method == "idea":
        from .idea import IDEAAdapter

        idea_cfg = getattr(tta_cfg, "IDEA", None)

        def ivalue(key, default):
            return getattr(idea_cfg, key, default) if idea_cfg is not None else default

        if fusion_protocol is None:
            raise ValueError(
                "IDEA requires a fusion_protocol; pass build_adapter(model, "
                "cfg, fusion_protocol=...) with the task's IDEAFusionProtocol"
            )
        return finalize(IDEAAdapter(
            model,
            fusion_protocol,
            prompt_length=int(ivalue("PROMPT_LENGTH", 4)),
            capacity=int(ivalue("K_MAX", 32)),
            lam=float(ivalue("LAMBDA", 0.4)),
            tau=float(ivalue("TAU", 0.7)),
            fisher_beta=float(ivalue("FISHER_BETA", 0.1)),
            opt_steps=int(ivalue("OPT_STEPS", 50)),
            lr=float(ivalue("LR", 3e-3)),
            optimizer_name=str(ivalue("OPTIMIZER", "AdamW")),
            momentum=float(ivalue("MOMENTUM", 0.9)),
            beta1=float(ivalue("BETA1", 0.9)),
            beta2=float(ivalue("BETA2", 0.999)),
            weight_decay=float(ivalue("WEIGHT_DECAY", 0.0)),
            use_fisher=bool(ivalue("USE_FISHER", True)),
            ridge=float(ivalue("RIDGE", 1e-4)),
            max_grad_norm=float(ivalue("MAX_GRAD_NORM", 0.0)),
            episodic=common["episodic"],
            prompt_init_std=float(ivalue("PROMPT_INIT_STD", 0.02)),
            seed=int(ivalue("SEED", 0)),
        ))
    raise ValueError("Unknown TTA method: {}".format(method))
