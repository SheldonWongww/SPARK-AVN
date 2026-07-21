#!/usr/bin/env python3
"""Test-time adaptation for sequential audio-visual navigation policies.

The adapters deliberately expose episode hooks.  FSTTA's fast statistics are
computed inside one episode, while its slow update is performed after N whole
episodes (not after N optimizer steps).  Evaluation code must therefore use a
single environment and call ``episode_start`` / ``episode_end``.
"""
from copy import deepcopy
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


def _episode_success(episode_stats):
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


def _concordant_grad_and_trace(grad_list, eigen_eps=1e-6):
    """Low-rank GDA with the length calibration from FSTTA Eq. (5)."""
    gradients = torch.stack(grad_list, dim=0)
    mean_grad = gradients.mean(dim=0)
    if len(grad_list) < 2:
        return mean_grad, torch.zeros((), device=mean_grad.device)

    centered = gradients - mean_grad.unsqueeze(0)
    # centered is M x D and M is small, so this avoids a D x D covariance.
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values.square() / float(len(grad_list) - 1)
    sigma = eigenvalues.sum()
    if eigenvalues.numel() == 0:
        return mean_grad, sigma

    threshold = eigenvalues.max().clamp_min(1e-12) * float(eigen_eps)
    valid = eigenvalues > threshold
    if not bool(valid.any()):
        return mean_grad, sigma

    basis = vh[valid]                              # [rank, D]
    values = eigenvalues[valid]                   # [rank]
    projections = basis @ mean_grad               # [rank]
    concordant = ((projections / values).unsqueeze(1) * basis).sum(dim=0)

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

    def prepare_action(self, source_logits, **kwargs):
        """Return logits used to sample the environment action."""
        return source_logits

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
        return {
            "action_steps": int(self.action_steps),
            "updates": int(self.update_count),
            "episodes": int(getattr(self, "episode_count", 0)),
            "slow_updates": int(getattr(self, "slow_update_count", 0)),
            "mean_entropy": self.loss_sum / max(1, self.action_steps),
            "last_entropy": self.last_loss,
            "last_grad_norm": self.last_grad_norm,
            "current_lr": self.current_lr,
            "relative_param_drift": _relative_drift(current, self._source_flat),
            "adapted_parameter_names": list(self.names),
        }


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
        max_grad_norm=1.0,
    ):
        if int(steps) != 1:
            raise ValueError(
                "TTA.STEPS != 1 requires a new policy forward after every update; "
                "the navigation adapter intentionally supports STEPS=1 only."
            )
        self.model = model
        self.episodic = bool(episodic)
        self.base_lr = float(lr)
        self.update_interval = max(1, int(update_interval))
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
        self._init_diagnostics()

    @torch.enable_grad()
    def adapt(self, logits, **kwargs):
        loss = softmax_entropy(logits).mean()
        should_update = ((self.action_steps + 1) % self.update_interval == 0)
        self._record_loss(loss)
        if should_update:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.params, self.max_grad_norm
            )
            self.last_grad_norm = float(grad_norm.item())
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.update_count += 1
        return loss.detach()

    def reset(self):
        self.model.load_state_dict(self._model_state, strict=True)
        self.optimizer.load_state_dict(self._optim_state)

    def episode_start(self):
        if self.episodic:
            self.reset()

    def episode_end(self, episode_stats=None):
        self.episode_count += 1


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
    ):
        if int(steps) != 1:
            raise ValueError("FSTTA supports one policy forward/update per action")
        if not 0.0 < float(q) < 1.0:
            raise ValueError("FSTTA.Q must be in (0, 1)")
        self.model = model
        self.M = max(1, int(M))
        self.N = max(1, int(N))
        self.q = float(q)
        self.rho = float(rho)
        self.tau = float(tau)
        self.a = float(a)
        self.b = float(b)
        self.base_lr = float(lr_fast)
        self.lr_slow = float(lr_slow)
        self.episodic = bool(episodic)
        self.use_slow = bool(use_slow)
        self.max_grad_norm = float(max_grad_norm)
        self.reset_optimizer_each_episode = bool(reset_optimizer_each_episode)
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

        self.grad_buffer = []
        self.var_hist = None
        self.episode_count = 0
        self.slow_update_count = 0
        self.slow_anchor = self._source_flat.clone()
        self.slow_trajectory = []
        self._init_diagnostics()

    def _clear_optimizer_state(self):
        self.optimizer.state.clear()
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr
        self.current_lr = self.base_lr

    @torch.enable_grad()
    def adapt(self, logits, **kwargs):
        loss = softmax_entropy(logits).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.grad_buffer.append(_flatten_grads(self.params).clone())
        self._record_loss(loss)

        if len(self.grad_buffer) == self.M:
            concordant, sigma = _concordant_grad_and_trace(
                self.grad_buffer, self.eigen_eps
            )
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
            self.current_lr = scale * self.base_lr
            for group in self.optimizer.param_groups:
                group["lr"] = self.current_lr

            self.optimizer.zero_grad(set_to_none=True)
            _copy_flat_to_grads(concordant, self.params)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.params, self.max_grad_norm
            )
            self.last_grad_norm = float(grad_norm.item())
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.grad_buffer = []
            self.update_count += 1
        else:
            self.optimizer.zero_grad(set_to_none=True)
        return loss.detach()

    @torch.no_grad()
    def _slow_step(self):
        states = [self.slow_anchor] + self.slow_trajectory
        matrix = torch.stack(states, dim=0)
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        eigenvalues = singular_values.square() / float(max(1, self.N))
        if eigenvalues.numel() == 0 or eigenvalues.norm() <= 1e-12:
            logging.warning("[FSTTA] slow update skipped: degenerate trajectory")
            self.slow_trajectory = []
            return

        # Eq. (8): recent episode states receive larger weights (q < 1).
        weights = torch.tensor(
            [self.q ** (self.N - index) for index in range(1, self.N + 1)],
            device=matrix.device,
            dtype=matrix.dtype,
        )
        weights = weights / weights.sum()
        deviations = torch.stack(
            [self.slow_anchor - state for state in self.slow_trajectory], dim=0
        )
        reference = (weights.unsqueeze(1) * deviations).sum(dim=0)
        reference_norm = reference.norm()
        if reference_norm <= 1e-12:
            logging.warning("[FSTTA] slow update skipped: zero reference direction")
            self.slow_trajectory = []
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

        self.slow_anchor = self.slow_anchor - self.lr_slow * slow_grad
        _copy_flat_to_params(self.slow_anchor, self.params)
        self.slow_trajectory = []
        self.slow_update_count += 1
        self._clear_optimizer_state()

    def reset(self):
        self.model.load_state_dict(self._model_state, strict=True)
        self.optimizer.load_state_dict(self._optim_state)
        self.grad_buffer = []
        self.var_hist = None
        self.episode_count = 0
        self.slow_update_count = 0
        self.slow_anchor = _flatten_params(self.params).clone()
        self.slow_trajectory = []
        self._init_diagnostics()

    def episode_start(self):
        if self.episodic:
            self.reset()
        self.grad_buffer = []
        if self.reset_optimizer_each_episode:
            self._clear_optimizer_state()

    def episode_end(self, episode_stats=None):
        self.episode_count += 1
        # Never combine gradients from two unrelated trajectories.
        self.grad_buffer = []
        if self.use_slow:
            self.slow_trajectory.append(_flatten_params(self.params).clone())
            if len(self.slow_trajectory) == self.N:
                self._slow_step()


class EAMAdapter(_AdapterDiagnostics):
    """Paper-aligned source-free Elastic Adaptation Model.

    The deployed policy is the frozen source branch. A same-architecture
    auxiliary branch is adapted from confident pseudo labels. Historical
    policy inputs are stored on CPU and replayed without environment steps.
    For memory-based AVN policies, replay uses the memory snapshot observed at
    collection time; the frozen source logits are stored with that snapshot.

    The paper updates the complete auxiliary model whenever a full replay
    mini-batch is available. ``param_scope`` and ``update_interval`` remain
    configurable only to support explicit AVN ablations; their defaults match
    the paper.
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
        param_scope="all",
        last_k=4,
        optimizer_name="Adam",
        momentum=0.9,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        max_grad_norm=0.0,
        episodic=False,
        reset_bn_stats=False,
    ):
        self.source_model = model
        self.source_model.eval()
        self.source_model.requires_grad_(False)
        self.aux_model = deepcopy(model)
        self.aux_model.eval()
        self.param_scope = str(param_scope).lower()
        self.last_k = int(last_k)

        if self.param_scope == "all":
            self.aux_model.requires_grad_(True)
            self.params = list(self.aux_model.parameters())
            self.names = [name for name, _ in self.aux_model.named_parameters()]
        elif self.param_scope == "decision_head":
            self.aux_model.requires_grad_(False)
            self.params, self.names = [], []
            for name, param in self.aux_model.action_distribution.named_parameters():
                param.requires_grad_(True)
                self.params.append(param)
                self.names.append("action_distribution.{}".format(name))
        else:
            self.params, self.names = configure_tta_model(
                self.aux_model,
                reset_bn_stats=reset_bn_stats,
                scope=self.param_scope,
                last_k=self.last_k,
            )
        if not self.params:
            raise ValueError("EAM did not select any auxiliary parameters")

        self.base_lr = float(lr)
        self.max_grad_norm = float(max_grad_norm)
        self.confidence_scale = float(confidence_scale)
        self.memory_size = max(1, int(memory_size))
        self.batch_size = max(1, int(batch_size))
        self.update_interval = max(1, int(update_interval))
        self.episodic = bool(episodic)
        self.optimizer = _make_optimizer(
            self.params,
            optimizer_name,
            self.base_lr,
            momentum=momentum,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
        )
        # EAM already keeps source + auxiliary policies on the GPU. Store the
        # optional episodic-reset snapshot on CPU to avoid a third GPU-sized
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
        self.episode_count = 0
        self._cached_current = None
        self._init_diagnostics()

    @property
    def device(self):
        return self.params[0].device

    def _forward_aux(self, policy_inputs):
        features, _, _ = self.aux_model.net(
            policy_inputs["observations"],
            policy_inputs["rnn_hidden_states"],
            policy_inputs["prev_actions"],
            policy_inputs["masks"],
            policy_inputs.get("ext_memory"),
            policy_inputs.get("ext_memory_masks"),
        )
        return self.aux_model.action_distribution(features).logits

    def _threshold(self, action_dim):
        return self.confidence_scale * math.log(max(2, int(action_dim)))

    def _combine(self, source_logits, aux_logits):
        threshold = self._threshold(source_logits.shape[-1])
        aux_entropy = softmax_entropy(aux_logits)
        use_aux = aux_entropy < threshold
        source_probs = source_logits.softmax(dim=-1)
        aux_probs = aux_logits.softmax(dim=-1)
        combined_probs = (
            source_probs
            + use_aux.to(aux_probs.dtype).unsqueeze(-1) * aux_probs
        )
        combined_probs = combined_probs / combined_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        # Return log-probabilities so the trainer can reconstruct its native
        # categorical distribution without changing action tensor shapes.
        return combined_probs.clamp_min(1e-8).log(), use_aux

    @torch.enable_grad()
    def prepare_action(self, source_logits, policy_inputs=None, **kwargs):
        if policy_inputs is None:
            raise ValueError("EAM requires policy_inputs for its auxiliary branch")
        aux_logits = self._forward_aux(policy_inputs)
        combined, use_aux = self._combine(source_logits.detach(), aux_logits)
        self.aux_used_steps += int(use_aux.sum().item())
        self._cached_current = (source_logits.detach(), aux_logits, policy_inputs)
        self._record_loss(softmax_entropy(combined).mean())
        return combined

    def _reservoir_add(self, source_logits, policy_inputs):
        entry = {
            "source_logits": source_logits.detach().cpu().clone(),
            "policy_inputs": _tree_to_cpu(policy_inputs),
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
    def adapt(self, logits, policy_inputs=None, **kwargs):
        if self._cached_current is None:
            raise RuntimeError("EAM.prepare_action must run before EAM.adapt")
        source_current, aux_current, cached_inputs = self._cached_current
        # Algorithm 3 updates the reservoir before replay. Keep the current
        # entry out of the historical draw because it is already included
        # explicitly in the mini-batch.
        current_entry = self._reservoir_add(source_current, cached_inputs)
        history_pool = [
            entry for entry in self.replay if entry is not current_entry
        ]
        needed_history = self.batch_size - 1
        can_update = (
            len(self.replay) >= self.batch_size
            and self.action_steps % self.update_interval == 0
        )

        if can_update:
            replay_entries = (
                random.sample(history_pool, needed_history)
                if needed_history else []
            )
            source_logits = [source_current]
            aux_logits = [aux_current]
            for entry in replay_entries:
                replay_inputs = _tree_to_device(entry["policy_inputs"], self.device)
                source_logits.append(entry["source_logits"].to(self.device))
                aux_logits.append(self._forward_aux(replay_inputs))

            source_batch = torch.cat(source_logits, dim=0)
            aux_batch = torch.cat(aux_logits, dim=0)
            combined, _ = self._combine(source_batch, aux_batch)
            threshold = self._threshold(source_batch.shape[-1])
            reliable = softmax_entropy(source_batch) < threshold
            if bool(reliable.any()):
                pseudo = combined.detach().argmax(dim=-1)
                loss = F.cross_entropy(aux_batch[reliable], pseudo[reliable])
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # EAM does not use gradient clipping in the paper. A positive
                # value enables it only for an explicitly requested ablation;
                # infinity reports the norm without altering the gradients.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.params,
                    self.max_grad_norm if self.max_grad_norm > 0 else math.inf,
                )
                self.last_grad_norm = float(grad_norm.item())
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.update_count += 1
                self.accepted_samples += int(reliable.sum().item())

        self._cached_current = None
        return None

    def reset(self):
        self.aux_model.load_state_dict(self._model_state, strict=True)
        self.optimizer.load_state_dict(self._optim_state)
        self.replay = []
        self.seen_samples = 0
        self.accepted_samples = 0
        self.aux_used_steps = 0
        self._cached_current = None
        self._init_diagnostics()

    def episode_start(self):
        if self.episodic:
            self.reset()

    def episode_end(self, episode_stats=None):
        self.episode_count += 1

    def diagnostics(self):
        output = super().diagnostics()
        output.update({
            "replay_size": len(self.replay),
            "seen_samples": self.seen_samples,
            "accepted_samples": self.accepted_samples,
            "aux_used_steps": self.aux_used_steps,
            "param_scope": self.param_scope,
            "update_interval": self.update_interval,
        })
        return output


class FEEDTTAAdapter(_AdapterDiagnostics):
    """Paper-aligned episode-feedback REINFORCE with SGR.

    Modality encoders remain frozen.  The AVN mapping of the paper's
    "cross-modal encoder and subsequent modules" is expressed explicitly by
    ``trainable_prefixes`` so that task-specific module names do not leak into
    the shared implementation.
    """

    def __init__(
        self,
        model,
        lr=5e-6,
        reversal_probability=0.05,
        reversal_scale=0.1,
        gamma=0.99,
        normalize_gradient=False,
        episodic=False,
        reset_bn_stats=False,
        scope="module_prefixes",
        last_k=4,
        trainable_prefixes=("net.smt_state_encoder", "action_distribution"),
        optimizer_name="Adam",
        momentum=0.9,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        optimizer_eps=1e-5,
        max_grad_norm=0.0,
    ):
        self.model = model
        self.base_lr = float(lr)
        self.reversal_probability = float(reversal_probability)
        self.reversal_scale = float(reversal_scale)
        self.gamma = float(gamma)
        self.normalize_gradient = bool(normalize_gradient)
        self.episodic = bool(episodic)
        self.max_grad_norm = float(max_grad_norm)
        if not 0.0 <= self.reversal_probability < 1.0:
            raise ValueError("FEEDTTA reversal probability must be in [0, 1)")
        denominator = (
            self.reversal_scale * self.reversal_probability
            + (1.0 - self.reversal_probability)
        )
        if denominator <= 0.0:
            raise ValueError("FEEDTTA SGR denominator must be positive")
        self._sgr_denominator = denominator

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
        self.trajectory_grads = []
        self.episode_count = 0
        self.successful_episodes = 0
        self._init_diagnostics()

    @torch.enable_grad()
    def adapt(self, logits, action=None, **kwargs):
        if action is None:
            raise ValueError("FEEDTTA requires the sampled action")
        action = action.long().view(-1, 1)
        nll = -logits.log_softmax(dim=-1).gather(1, action).mean()
        self.optimizer.zero_grad(set_to_none=True)
        nll.backward()
        self.trajectory_grads.append(_flatten_grads(self.params).clone())
        self.optimizer.zero_grad(set_to_none=True)
        self._record_loss(softmax_entropy(logits).mean())
        return nll.detach()

    def _aggregate_trajectory(self):
        count = len(self.trajectory_grads)
        weights = torch.tensor(
            [self.gamma ** (count - 1 - index) for index in range(count)],
            device=self.trajectory_grads[0].device,
            dtype=self.trajectory_grads[0].dtype,
        )
        if self.normalize_gradient:
            weights = weights / weights.sum().clamp_min(1e-12)
        return (weights.unsqueeze(1) * torch.stack(self.trajectory_grads)).sum(0)

    def _apply_sgr(self, grad):
        if self.reversal_probability == 0.0:
            return grad
        selected = torch.rand_like(grad) < self.reversal_probability
        return torch.where(
            selected,
            self.reversal_scale * grad,
            grad / self._sgr_denominator,
        )

    def reset(self):
        self.model.load_state_dict(self._model_state, strict=True)
        self.optimizer.load_state_dict(self._optim_state)
        self.trajectory_grads = []
        self.episode_count = 0
        self.successful_episodes = 0
        self._init_diagnostics()

    def episode_start(self):
        if self.episodic:
            self.reset()
        self.trajectory_grads = []

    def episode_end(self, episode_stats=None):
        success = _episode_success(episode_stats)
        self.episode_count += 1
        self.successful_episodes += int(success)
        if not self.trajectory_grads:
            return
        # Optimizer performs descent on NLL. Success reinforces the sampled
        # trajectory; failure applies the opposite direction.
        grad = self._aggregate_trajectory()
        grad = grad if success else -grad
        grad = self._apply_sgr(grad)
        self.optimizer.zero_grad(set_to_none=True)
        self.last_grad_norm = _set_flat_grad(
            grad, self.params, self.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.update_count += 1
        self.trajectory_grads = []

    def diagnostics(self):
        output = super().diagnostics()
        output.update({
            "successful_feedback_episodes": self.successful_episodes,
            "reversal_probability": self.reversal_probability,
            "reversal_scale": self.reversal_scale,
            "param_scope": self.param_scope,
            "trainable_prefixes": list(self.trainable_prefixes),
            "gamma": self.gamma,
            "normalize_gradient": self.normalize_gradient,
        })
        return output


class ATENAAdapter(_AdapterDiagnostics):
    """Paper-aligned active episodic mixture-entropy optimization.

    ATENA keeps the episode graph and jointly optimizes the signed mean mixture
    entropy and the online success-prediction loss at episode end.  This follows
    the official DUET implementation; long AVN episodes therefore have a higher
    memory cost than step-wise entropy adapters.
    """

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
    ):
        if not 0.0 <= float(mix_lambda) <= 1.0:
            raise ValueError("ATENA mixture lambda must be in [0, 1]")
        self.model = model
        self.base_lr = float(lr_query)
        self.lr_query = float(lr_query)
        self.lr_self = float(lr_self)
        self.mix_lambda = float(mix_lambda)
        self.query_threshold = float(query_threshold)
        self.self_loss_weight = float(self_loss_weight)
        self.episodic = bool(episodic)
        self.max_grad_norm = float(max_grad_norm)
        self.param_scope = str(scope).lower()

        if self.param_scope == "all":
            # Match the official implementation's optimizer over the complete
            # pretrained policy while preserving parameters that the baseline
            # itself intentionally froze.
            self.model.eval()
            selected = [
                (name, param) for name, param in self.model.named_parameters()
                if param.requires_grad
            ]
            if not selected:
                raise ValueError("ATENA found no trainable policy parameters")
            self.names = [name for name, _ in selected]
            self.params = [param for _, param in selected]
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
        self.trajectory_mixture_entropies = []
        self.trajectory_features = []
        self.trajectory_entropies = []
        self.action_dim = None
        self.episode_count = 0
        self.query_count = 0
        self.query_prediction_correct = 0
        self.self_feedback_successes = 0
        self._init_diagnostics()

    def _ensure_head(self, feature_dim, device, dtype):
        if self.self_prediction_head is not None:
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

    def _build_episode_optimizer(self, lr):
        # The official code constructs a fresh AdamW after each episode, so no
        # optimizer moments are shared between queried and self-labelled data.
        parameters = self.params + list(self.self_prediction_head.parameters())
        return _make_optimizer(
            parameters,
            lr=float(lr),
            **self._optimizer_args
        )

    def select_action(self, distribution):
        # Eq. (1) defines the selected pseudo-expert action as policy argmax,
        # and the official evaluation executes that same greedy action.
        return distribution.probs.argmax(dim=-1, keepdim=True)

    @torch.enable_grad()
    def adapt(self, logits, action=None, features=None, **kwargs):
        if action is None or features is None:
            raise ValueError("ATENA requires executed actions and policy features")
        self.action_dim = int(logits.shape[-1])
        self._ensure_head(features.shape[-1], features.device, features.dtype)
        pseudo_action = logits.detach().argmax(dim=-1)
        if not torch.equal(action.detach().long().view(-1), pseudo_action.view(-1)):
            raise ValueError(
                "ATENA requires the executed action to equal the policy argmax"
            )
        probs = logits.softmax(dim=-1)
        one_hot = F.one_hot(
            pseudo_action, num_classes=self.action_dim
        ).to(probs.dtype)
        mixture = self.mix_lambda * one_hot + (1.0 - self.mix_lambda) * probs
        mixture_entropy = -(
            mixture * mixture.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()

        # Keep both tensors attached: ATENA performs one joint episode-level
        # backward pass for L_mix + gamma * L_self.
        self.trajectory_mixture_entropies.append(mixture_entropy)
        original_entropy = softmax_entropy(logits).mean()
        self.trajectory_entropies.append(float(original_entropy.detach().item()))
        self.trajectory_features.append(features)
        self._record_loss(mixture_entropy)
        return mixture_entropy.detach()

    def reset(self):
        self.model.load_state_dict(self._model_state, strict=True)
        if self.self_prediction_head is not None and self._head_state is not None:
            self.self_prediction_head.load_state_dict(self._head_state, strict=True)
        self.optimizer = None
        self.trajectory_mixture_entropies = []
        self.trajectory_features = []
        self.trajectory_entropies = []
        self.episode_count = 0
        self.query_count = 0
        self.query_prediction_correct = 0
        self.self_feedback_successes = 0
        self._init_diagnostics()

    def episode_start(self):
        if self.episodic:
            self.reset()
        self.trajectory_mixture_entropies = []
        self.trajectory_features = []
        self.trajectory_entropies = []

    def episode_end(self, episode_stats=None):
        self.episode_count += 1
        if not self.trajectory_features:
            return

        mean_feature = torch.cat(self.trajectory_features, dim=0).mean(0, keepdim=True)
        prediction_logit = self.self_prediction_head(mean_feature).view(())
        predicted_success = bool(torch.sigmoid(prediction_logit.detach()) > 0.5)
        average_entropy = sum(self.trajectory_entropies) / len(self.trajectory_entropies)
        query = average_entropy > self.query_threshold
        if query:
            # Access evaluator/human feedback only for actively queried
            # episodes. Non-query episodes remain self-supervised.
            actual_success = _episode_success(episode_stats)
            feedback_success = actual_success
            self.query_count += 1
            self.query_prediction_correct += int(predicted_success == actual_success)
            policy_lr = self.lr_query
        else:
            feedback_success = predicted_success
            self.self_feedback_successes += int(feedback_success)
            policy_lr = self.lr_self

        mixture_loss = torch.stack(
            self.trajectory_mixture_entropies, dim=0
        ).mean()
        mixture_loss = mixture_loss if feedback_success else -mixture_loss
        target = torch.tensor(
            float(feedback_success), device=prediction_logit.device
        )
        self_prediction_loss = F.binary_cross_entropy_with_logits(
            prediction_logit, target
        )
        total_loss = mixture_loss + self.self_loss_weight * self_prediction_loss

        self.optimizer = self._build_episode_optimizer(policy_lr)
        self.current_lr = policy_lr
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimized_parameters = (
            self.params + list(self.self_prediction_head.parameters())
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            optimized_parameters,
            self.max_grad_norm if self.max_grad_norm > 0 else math.inf,
        )
        self.last_grad_norm = float(grad_norm.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.update_count += 1

        self.trajectory_mixture_entropies = []
        self.trajectory_features = []
        self.trajectory_entropies = []

    def diagnostics(self):
        output = super().diagnostics()
        output.update({
            "queries": self.query_count,
            "query_rate": self.query_count / max(1, self.episode_count),
            "query_prediction_accuracy": (
                self.query_prediction_correct / max(1, self.query_count)
            ),
            "self_feedback_successes": self.self_feedback_successes,
            "mix_lambda": self.mix_lambda,
            "query_threshold": self.query_threshold,
            "param_scope": self.param_scope,
            "action_selection": "argmax",
            "retains_episode_graph": True,
        })
        return output


def build_adapter(model, tta_cfg):
    """Build a sequential AVN test-time adapter from a config node."""
    method = str(getattr(tta_cfg, "METHOD", "none")).lower()
    if method in ("none", "", "source"):
        return None

    def value(key, default):
        return getattr(tta_cfg, key, default)

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
        return TentAdapter(
            model,
            lr=lr,
            update_interval=int(value("UPDATE_INTERVAL", 1)),
            **common
        )
    if method == "fstta":
        fstta_cfg = getattr(tta_cfg, "FSTTA", None)

        def fvalue(key, default):
            return getattr(fstta_cfg, key, default) if fstta_cfg is not None else default

        # FSTTA can use optimizer settings independently from Tent.
        common["optimizer_name"] = str(fvalue("OPTIMIZER", "AdamW"))
        common["beta1"] = float(fvalue("BETA1", 0.9))
        common["beta2"] = float(fvalue("BETA2", 0.99))
        common["weight_decay"] = float(fvalue("WEIGHT_DECAY", 0.0))
        return FSTTAAdapter(
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
            reset_optimizer_each_episode=bool(
                fvalue("RESET_OPTIMIZER_EACH_EPISODE", True)
            ),
            eigen_eps=float(fvalue("EIGEN_EPS", 1e-6)),
            **common
        )
    if method == "eam":
        eam_cfg = getattr(tta_cfg, "EAM", None)

        def evalue(key, default):
            return getattr(eam_cfg, key, default) if eam_cfg is not None else default

        return EAMAdapter(
            model,
            lr=float(evalue("LR", 1e-5)),
            confidence_scale=float(evalue("CONFIDENCE_SCALE", 0.4)),
            memory_size=int(evalue("MEMORY_SIZE", 32)),
            batch_size=int(evalue("BATCH_SIZE", 8)),
            update_interval=int(evalue("UPDATE_INTERVAL", 1)),
            param_scope=str(evalue("PARAM_SCOPE", "all")),
            last_k=int(evalue("LAST_K_LN", common["last_k"])),
            optimizer_name=str(evalue("OPTIMIZER", "Adam")),
            momentum=float(evalue("MOMENTUM", 0.9)),
            beta1=float(evalue("BETA1", 0.9)),
            beta2=float(evalue("BETA2", 0.999)),
            weight_decay=float(evalue("WEIGHT_DECAY", 0.0)),
            max_grad_norm=float(evalue("MAX_GRAD_NORM", 0.0)),
            episodic=common["episodic"],
            reset_bn_stats=common["reset_bn_stats"],
        )
    if method == "feedtta":
        feed_cfg = getattr(tta_cfg, "FEEDTTA", None)

        def fdvalue(key, default):
            return getattr(feed_cfg, key, default) if feed_cfg is not None else default

        return FEEDTTAAdapter(
            model,
            lr=float(fdvalue("LR", 5e-6)),
            reversal_probability=float(fdvalue("P", 0.05)),
            reversal_scale=float(fdvalue("ALPHA", 0.1)),
            gamma=float(fdvalue("GAMMA", 0.99)),
            normalize_gradient=bool(fdvalue("NORMALIZE_GRADIENT", False)),
            episodic=common["episodic"],
            reset_bn_stats=common["reset_bn_stats"],
            scope=str(fdvalue("PARAM_SCOPE", "module_prefixes")),
            last_k=common["last_k"],
            trainable_prefixes=tuple(fdvalue(
                "TRAINABLE_PREFIXES",
                ("net.smt_state_encoder", "action_distribution"),
            )),
            optimizer_name=str(fdvalue("OPTIMIZER", "Adam")),
            momentum=common["momentum"],
            beta1=float(fdvalue("BETA1", 0.9)),
            beta2=float(fdvalue("BETA2", 0.999)),
            weight_decay=float(fdvalue("WEIGHT_DECAY", 0.0)),
            optimizer_eps=float(fdvalue("EPS", 1e-5)),
            max_grad_norm=float(fdvalue("MAX_GRAD_NORM", 0.0)),
        )
    if method == "atena":
        atena_cfg = getattr(tta_cfg, "ATENA", None)

        def avalue(key, default):
            return getattr(atena_cfg, key, default) if atena_cfg is not None else default

        return ATENAAdapter(
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
        )
    raise ValueError("Unknown TTA method: {}".format(method))
