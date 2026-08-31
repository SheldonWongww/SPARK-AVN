#!/usr/bin/env python3
"""IDEA: Inter-Domain BridgE with Historical Assets (ICML 2026).

IDEA turns online adaptation into an *asset* problem instead of a sequence of
isolated, forgettable updates.  Each visited domain contributes a triplet
``A = {P*, Gamma, u}`` -- a Fisher-guided soft prompt ``P*``, its prompt-free
feature-statistics descriptor ``Gamma`` and an uncertainty score ``u`` -- to a
capacity-bounded library ``M``.  For every new navigation step IDEA first tries
a *training-free shortcut*: it projects the target statistics onto the convex
hull of the stored descriptors (a small QP with a KKT closed form) to synthesise
a bridge prompt.  When the bridge already covers the domain the shortcut is
used directly; otherwise IDEA warm-starts from the bridge, optimises the prompt
for a few steps with a multi-layer moment-matching loss and stores the result.

The base navigation policy is *never* modified: IDEA only optimises external
soft prompts, so it is genuinely training-free with respect to ``pi_theta``.

This module deliberately contains only the task-agnostic algorithm.  Everything
model-specific (how a prompt is injected, how fused tokens and per-layer
statistics are read) is expressed through :class:`IDEAFusionProtocol`, which a
task binds by *calling frozen sub-modules of its own model* -- no edit to the
model's ``forward`` is required.  The same adapter therefore serves the AVN
policies (SMT+Audio, ENMuS) and the VLN policies (DUET, HAMT, ...).
"""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from .tta_core import (
    _AdapterDiagnostics,
    _make_optimizer,
    _parameter_state_sha256,
    softmax_entropy,
)


# ---------------------------------------------------------------------------
# Fusion protocol: the single model-specific seam.
# ---------------------------------------------------------------------------
class IDEAFusionProtocol:
    """Model-specific contract that lets IDEA drive one navigation step.

    A task implements this by *calling* its (frozen) fusion sub-modules; it must
    not require any change to the underlying model definition.  Every method
    operates on a streaming step and pools statistics over every valid
    candidate/navigable-node token.  Offline source statistics use global
    ``sum/sumsq/count`` moments across every valid token of every step; averaging
    per-step means or standard deviations is not the same estimator and is not
    accepted.

    Shapes (``C`` = feature dim, ``L`` = prompt length, ``M`` = #align layers,
    ``A`` = action dimension, ``N`` = #candidate nodes for this step):

    * A per-layer statistic is a pair ``(mu, sigma)`` with ``mu, sigma`` each of
      shape ``[C]`` -- the mean and (biased-corrected) std of the fused tokens
      pooled over the ``N`` node dimension.
    * A prompt ``P`` has shape ``[L, C]`` and is prepended to the visual tokens
      inside the fusion transformer.
    """

    @property
    def feature_dim(self):
        """Return the fused feature dimension ``C``."""
        raise NotImplementedError

    @property
    def num_layers(self):
        """Return the number of aligned fusion layers ``M``."""
        raise NotImplementedError

    def source_statistics(self, device, dtype):
        """Return the offline source anchor ``Gamma_S`` as ``M`` ``(mu, sigma)``.

        These are pre-computed once from a small source subset (the paper uses
        128 trajectories).  The final entry (index ``-1``) also defines the
        descriptor space used for asset ``Gamma`` and the coverage gate.
        """
        raise NotImplementedError

    @property
    def source_statistics_metadata(self):
        """Describe the validated offline artifact used by this protocol.

        Evaluation protocols must return a mapping with
        ``mode='offline_artifact'``, its file ``sha256``, schema, trajectory
        count and provenance.  IDEA deliberately has no target-stream warmup
        fallback.
        """
        raise NotImplementedError

    def fused_forward(self, policy_inputs, prompt):
        """Run the fusion with an optional soft prompt.

        Args:
            policy_inputs: the immutable per-step policy snapshot.
            prompt: a ``[L, C]`` tensor to prepend to the visual tokens, or
                ``None`` for a prompt-free pass.

        Returns a pair ``(layer_stats, logits)`` where ``layer_stats`` is a list
        of ``M`` ``(mu, sigma)`` pairs and ``logits`` has shape ``[1, A]``.  When
        ``prompt`` is a tensor that ``requires_grad`` the returned statistics
        must be differentiable with respect to it (this is what the prompt
        optimiser backpropagates through).
        """
        raise NotImplementedError

    def fisher_forward(self, policy_inputs):
        """Return prompt-free per-layer features connected to the logits.

        Returns ``(layer_features, logits, valid_masks)``. ``layer_features``
        is a list of ``M`` tensors whose final dimension is ``C``;
        ``valid_masks`` contains one boolean mask per layer with exactly the
        corresponding feature tensor's leading shape.  The masks identify the
        real navigable-node tokens used by Eq. (7), excluding padding, history,
        object-only and other non-action tokens.  Each layer feature must be
        part of the autograd graph of ``logits`` so that IDEA can form
        ``grad(log pi(a), Z_l)`` for the Fisher trace (Eq. 7-8).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Pure, task-agnostic math (unit-tested in isolation).
# ---------------------------------------------------------------------------
def _stats_vector(mu, sigma):
    """Vectorise a Gaussian descriptor ``Gamma = [mu; sigma]`` into ``[2C]``."""
    return torch.cat([mu.reshape(-1), sigma.reshape(-1)], dim=0)


def _base_parameter_snapshot(model):
    """Return compact, content-addressed evidence for every base parameter.

    The snapshot retains only names, version counters, and SHA256 strings.  It
    hashes one parameter at a time through the shared Torch-1.9-compatible
    helper, so IDEA never keeps a second model-sized tensor copy resident.
    Per-parameter digests let diagnostics identify an exact changed name even
    when a write bypasses PyTorch's tensor version counter.
    """
    named = sorted(model.named_parameters(), key=lambda item: item[0])
    names = tuple(name for name, _ in named)
    if len(names) != len(set(names)):
        raise RuntimeError("IDEA base policy has duplicate parameter names")

    name_set = hashlib.sha256()
    name_set.update(b"navtta.idea.base_parameter_name_set.v1\0")
    content = hashlib.sha256()
    content.update(b"navtta.idea.base_parameter_content.v1\0")
    parameter_sha256 = {}
    versions = {}
    for name, parameter in named:
        encoded_name = name.encode("utf-8")
        name_set.update(encoded_name)
        name_set.update(b"\0")
        parameter_digest = _parameter_state_sha256(
            (parameter,), (name,)
        )
        parameter_sha256[name] = parameter_digest
        versions[name] = int(parameter._version)
        content.update(encoded_name)
        content.update(b"\0")
        content.update(parameter_digest.encode("ascii"))
        content.update(b"\0")
    return {
        "names": names,
        "name_set_sha256": name_set.hexdigest(),
        "content_sha256": content.hexdigest(),
        "parameter_sha256": parameter_sha256,
        "versions": versions,
    }


def _gaussian_w2(mu_a, sigma_a, mu_b, sigma_b):
    """2-Wasserstein distance between diagonal Gaussians.

    For diagonal covariances the Bures term reduces to ``||sigma_a - sigma_b||``,
    so ``W = sqrt(||mu_a - mu_b||^2 + ||sigma_a - sigma_b||^2)`` (Eq. 10/11).
    """
    mean_gap = (mu_a - mu_b).pow(2).sum()
    std_gap = (sigma_a - sigma_b).pow(2).sum()
    return torch.sqrt((mean_gap + std_gap).clamp_min(0.0))


def _alignment_discrepancy(source_stats, target_stats, weights):
    """Weighted multi-layer moment-matching loss (Eq. 4-5).

    ``d^(l) = ||mu_S - mu_t||_2 + ||sigma_S - sigma_t||_2`` and the objective is
    ``sum_l alpha_l d^(l)``.  ``weights`` is the Fisher-guided ``alpha`` vector.
    """
    total = None
    for weight, (mu_s, sigma_s), (mu_t, sigma_t) in zip(
        weights, source_stats, target_stats
    ):
        mean_term = (mu_s - mu_t).norm()
        std_term = (sigma_s - sigma_t).norm()
        layer = weight * (mean_term + std_term)
        total = layer if total is None else total + layer
    if total is None:
        raise ValueError("IDEA alignment received no layers")
    return total


def solve_bridge_weights(gammas, target, uncertainties, lam, ridge=1e-4):
    """Closed-form convex-hull projection weights (Eq. 12).

    Solves ``min_w ||A w - b||^2 + lam w^T U w  s.t. 1^T w = 1, w >= 0`` where
    ``A = [Gamma_1, ..., Gamma_K] in R^{2C x K}`` (so ``A^T A = gammas gammas^T``),
    ``b = Gamma_t`` and ``U = diag(u)``.  The equality-constrained problem has
    the KKT closed form ``w = H^{-1}(g - nu 1)`` with ``H = A^T A + lam U``,
    ``g = A^T b`` and ``nu = (1^T H^{-1} g - 1) / (1^T H^{-1} 1)``.  The rare
    non-negativity violation is handled by clamp-and-renormalise, which keeps the
    bridge inside the simplex while preserving the closed-form efficiency.

    Args:
        gammas: ``[K, 2C]`` stacked asset descriptors.
        target: ``[2C]`` target-domain descriptor.
        uncertainties: ``[K]`` per-asset uncertainty scores ``u``.
        lam: regularisation strength ``lambda``.
        ridge: small diagonal jitter that keeps ``H`` positive definite.
    """
    num_assets = gammas.shape[0]
    device, dtype = gammas.device, gammas.dtype
    if num_assets == 0:
        raise ValueError("IDEA cannot build a bridge from an empty library")
    if num_assets == 1:
        return torch.ones(1, device=device, dtype=dtype)

    hessian = gammas @ gammas.t()                       # A^T A, [K, K]
    hessian = hessian + lam * torch.diag(uncertainties)
    hessian = hessian + ridge * torch.eye(
        num_assets, device=device, dtype=dtype
    )
    projection = gammas @ target                        # g = A^T b, [K]
    ones = torch.ones(num_assets, device=device, dtype=dtype)

    # Solve H [x_g | x_1] = [g | 1] once instead of forming H^{-1} explicitly.
    rhs = torch.stack([projection, ones], dim=1)
    solved = torch.linalg.solve(hessian, rhs)
    hinv_g, hinv_one = solved[:, 0], solved[:, 1]
    nu = (ones @ hinv_g - 1.0) / (ones @ hinv_one).clamp_min(1e-12)
    weights = hinv_g - nu * hinv_one

    # Simplex projection: clamp negatives, renormalise to sum one.
    weights = weights.clamp_min(0.0)
    total = weights.sum()
    if total <= 1e-12:
        return ones / num_assets
    return weights / total


# ---------------------------------------------------------------------------
# Asset library.
# ---------------------------------------------------------------------------
class _AssetLibrary:
    """Capacity-bounded triplet store with nearest-neighbour merge (Eq. 13)."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("IDEA K_MAX must be positive")
        self.prompts = []          # each [L, C]
        self.gammas = []           # each [2C]
        self.uncertainties = []    # each scalar float
        self.merge_count = 0
        self.add_count = 0

    def __len__(self):
        return len(self.prompts)

    def stacked(self, device, dtype):
        """Return ``(prompts[K,L,C], gammas[K,2C], u[K])`` on the given device."""
        prompts = torch.stack(self.prompts, dim=0).to(device=device, dtype=dtype)
        gammas = torch.stack(self.gammas, dim=0).to(device=device, dtype=dtype)
        uncertainties = torch.tensor(
            self.uncertainties, device=device, dtype=dtype
        )
        return prompts, gammas, uncertainties

    def add(self, prompt, gamma, uncertainty):
        """Insert a new asset, merging into the nearest neighbour when full."""
        prompt = prompt.detach().cpu().clone()
        gamma = gamma.detach().cpu().clone()
        uncertainty = float(uncertainty)
        if len(self.prompts) < self.capacity:
            self.prompts.append(prompt)
            self.gammas.append(gamma)
            self.uncertainties.append(uncertainty)
            self.add_count += 1
            return
        # Nearest neighbour by descriptor distance, then averaging merge.
        stored = torch.stack(self.gammas, dim=0)
        distances = (stored - gamma.unsqueeze(0)).norm(dim=1)
        index = int(distances.argmin().item())
        self.prompts[index] = 0.5 * (self.prompts[index] + prompt)
        self.gammas[index] = 0.5 * (self.gammas[index] + gamma)
        self.uncertainties[index] = 0.5 * (
            self.uncertainties[index] + uncertainty
        )
        self.merge_count += 1

    def clear(self):
        self.prompts = []
        self.gammas = []
        self.uncertainties = []
        self.merge_count = 0
        self.add_count = 0

    def payload(self, prompt_length, feature_dim, metadata=None):
        """Return a portable, versioned representation of the asset library."""
        return {
            "schema": "navtta.idea.asset_library",
            "version": 1,
            "capacity": self.capacity,
            "prompt_length": int(prompt_length),
            "feature_dim": int(feature_dim),
            "metadata": dict(metadata or {}),
            "assets": [
                {
                    "prompt": prompt.tolist(),
                    "gamma": gamma.tolist(),
                    "uncertainty": float(uncertainty),
                }
                for prompt, gamma, uncertainty in zip(
                    self.prompts, self.gammas, self.uncertainties
                )
            ],
        }

    @staticmethod
    def payload_sha256(payload):
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def digest(self, prompt_length, feature_dim):
        return self.payload_sha256(self.payload(prompt_length, feature_dim))

    def load_payload(self, payload, prompt_length, feature_dim):
        if payload.get("schema") != "navtta.idea.asset_library":
            raise ValueError("invalid IDEA asset-library schema")
        if int(payload.get("version", -1)) != 1:
            raise ValueError("unsupported IDEA asset-library version")
        if int(payload.get("capacity", -1)) != self.capacity:
            raise ValueError("IDEA asset-library capacity mismatch")
        if int(payload.get("prompt_length", -1)) != int(prompt_length):
            raise ValueError("IDEA asset prompt-length mismatch")
        if int(payload.get("feature_dim", -1)) != int(feature_dim):
            raise ValueError("IDEA asset feature-dimension mismatch")
        assets = payload.get("assets")
        if not isinstance(assets, list) or len(assets) > self.capacity:
            raise ValueError("invalid IDEA asset list")
        prompts, gammas, uncertainties = [], [], []
        for asset in assets:
            prompt = torch.as_tensor(asset.get("prompt"), dtype=torch.float32)
            gamma = torch.as_tensor(asset.get("gamma"), dtype=torch.float32)
            uncertainty = float(asset.get("uncertainty"))
            if tuple(prompt.shape) != (int(prompt_length), int(feature_dim)):
                raise ValueError("invalid IDEA asset prompt shape")
            if tuple(gamma.shape) != (2 * int(feature_dim),):
                raise ValueError("invalid IDEA asset descriptor shape")
            if not bool(torch.isfinite(prompt).all() and torch.isfinite(gamma).all()):
                raise ValueError("IDEA asset tensors must be finite")
            if not torch.isfinite(torch.tensor(uncertainty)):
                raise ValueError("IDEA asset uncertainty must be finite")
            prompts.append(prompt)
            gammas.append(gamma)
            uncertainties.append(uncertainty)
        self.prompts = prompts
        self.gammas = gammas
        self.uncertainties = uncertainties
        self.add_count = len(prompts)
        self.merge_count = 0


# ---------------------------------------------------------------------------
# Adapter.
# ---------------------------------------------------------------------------
class IDEAAdapter(_AdapterDiagnostics):
    """Training-free online adaptation via a historical asset library.

    The base policy weights are never written; IDEA only maintains external soft
    prompts and a descriptor library.  All decision logic lives in
    :meth:`prepare_action`, because the prompt has to be resolved *before* the
    action is selected; :meth:`adapt` is therefore a no-op recorder.
    """

    # IDEA runs its own grad-enabled forwards through the fusion protocol and
    # never differentiates the trainer's source logits.
    requires_source_grad = False

    def __init__(
        self,
        model,
        fusion_protocol,
        prompt_length=4,
        capacity=32,
        lam=0.4,
        tau=0.7,
        fisher_beta=0.1,
        opt_steps=50,
        lr=3e-3,
        optimizer_name="AdamW",
        momentum=0.9,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        use_fisher=True,
        ridge=1e-4,
        max_grad_norm=0.0,
        episodic=False,
        prompt_init_std=0.02,
        seed=0,
    ):
        if fusion_protocol is None:
            raise ValueError("IDEA requires an IDEAFusionProtocol instance")
        if int(prompt_length) < 1:
            raise ValueError("IDEA PROMPT_LENGTH must be positive")
        if not 0.0 < float(tau):
            raise ValueError("IDEA TAU must be positive")
        if float(lam) < 0.0:
            raise ValueError("IDEA LAMBDA must be nonnegative")
        if not 0.0 <= float(fisher_beta) <= 1.0:
            raise ValueError("IDEA FISHER_BETA must be in [0, 1]")
        if int(opt_steps) < 1:
            raise ValueError("IDEA OPT_STEPS must be positive")
        if bool(episodic):
            # The whole point of IDEA is a library that outlives each episode.
            raise ValueError(
                "IDEA accumulates cross-domain assets and requires "
                "TTA.EPISODIC=False"
            )

        try:
            source_metadata = getattr(
                fusion_protocol, "source_statistics_metadata", None
            )
        except NotImplementedError:
            source_metadata = None
        if not isinstance(source_metadata, dict) or (
            source_metadata.get("mode") != "offline_artifact"
        ):
            raise ValueError(
                "IDEA evaluation requires a validated offline source-statistics "
                "artifact; target-stream warmup is forbidden"
            )
        source_digest = source_metadata.get("sha256")
        if not isinstance(source_digest, str) or len(source_digest) != 64:
            raise ValueError("IDEA source-statistics artifact SHA256 is missing")

        self.model = model
        self.protocol = fusion_protocol
        self.source_statistics_metadata = dict(source_metadata)
        self.feature_dim = int(fusion_protocol.feature_dim)
        self.num_layers = int(fusion_protocol.num_layers)
        self.prompt_length = int(prompt_length)
        self.lam = float(lam)
        self.tau = float(tau)
        self.fisher_beta = float(fisher_beta)
        self.opt_steps = int(opt_steps)
        self.base_lr = float(lr)
        self.use_fisher = bool(use_fisher)
        self.ridge = float(ridge)
        self.max_grad_norm = float(max_grad_norm)
        self.episodic = False
        self.prompt_init_std = float(prompt_init_std)
        self._optimizer_args = dict(
            name=optimizer_name,
            lr=self.base_lr,
            momentum=momentum,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
        )

        # IDEA writes no policy parameters; declare an empty adapted set so the
        # shared diagnostics/audit interface treats it as training-free.
        self.params = []
        self.names = []
        self._source_flat = torch.zeros(0)

        # IDEA is prompt-only adaptation.  Freeze every base parameter eagerly
        # and clear stale gradients so both accidental optimizer inclusion and
        # silent gradient accumulation fail the invariant checks below.
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        # Keep a compact content-addressed baseline in addition to PyTorch's
        # cheap tensor version counters.  The content digest closes the gaps
        # left by ``.data`` writes and parameter replacement, while the exact
        # name-set digest catches additions/removals.  The snapshot stores no
        # tensor clone and hashes only one parameter at a time.
        self._base_parameter_before = _base_parameter_snapshot(self.model)

        self.library = _AssetLibrary(capacity)
        # Fisher-guided layer weights alpha (Eq. 8), initialised uniform.
        self.alpha = torch.full(
            (self.num_layers,), 1.0 / self.num_layers
        )
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(seed))

        self.episode_count = 0
        self.covered_steps = 0
        self.new_domain_steps = 0
        self.cold_start_steps = 0
        self.last_bridge_weight_max = 0.0
        self.last_coverage_ratio = 0.0
        self.last_uncertainty = 0.0
        self.alignment_loss_sum = 0.0
        self.alignment_step_count = 0
        self.prompt_optimizer_attempts = 0
        self.prompt_optimizer_updates = 0
        self.prompt_relative_drift_sum = 0.0
        self.last_prompt_relative_drift = 0.0
        self.max_prompt_relative_drift = 0.0
        self._init_diagnostics()

    def _assert_base_policy_frozen(self):
        trainable = [name for name, parameter in self.model.named_parameters()
                     if parameter.requires_grad]
        gradients = [name for name, parameter in self.model.named_parameters()
                     if parameter.grad is not None]
        if trainable or gradients:
            raise RuntimeError(
                "IDEA base policy must remain frozen with grad=None; trainable={}, "
                "gradients={}".format(trainable[:3], gradients[:3])
            )

    # -- prompt helpers -----------------------------------------------------
    def _new_prompt(self, device, dtype, warm_start=None):
        prompt = nn.Parameter(
            torch.empty(
                self.prompt_length, self.feature_dim,
                device=device, dtype=dtype,
            )
        )
        with torch.no_grad():
            if warm_start is None:
                # First-ever domain: random Gaussian initialisation.
                noise = torch.randn(
                    prompt.shape, generator=self._generator, dtype=dtype
                ).to(device=device)
                prompt.copy_(self.prompt_init_std * noise)
            else:
                prompt.copy_(warm_start.to(device=device, dtype=dtype))
        return prompt

    def _final_gamma(self, layer_stats):
        mu, sigma = layer_stats[-1]
        return _stats_vector(mu, sigma)

    # -- Fisher-guided weighting (Eq. 7-8) ----------------------------------
    @torch.enable_grad()
    def _update_fisher_weights(self, policy_inputs):
        if not self.use_fisher:
            return
        fisher_result = self.protocol.fisher_forward(policy_inputs)
        if not isinstance(fisher_result, (tuple, list)) or len(fisher_result) != 3:
            raise ValueError(
                "IDEA fisher_forward must return "
                "(layer_features, logits, valid_masks)"
            )
        features, logits, valid_masks = fisher_result
        if len(features) != self.num_layers:
            raise ValueError(
                "IDEA fisher_forward returned {} layers, expected {}".format(
                    len(features), self.num_layers
                )
            )
        if torch.is_tensor(valid_masks):
            valid_masks = [valid_masks] * self.num_layers
        if not isinstance(valid_masks, (tuple, list)) or len(valid_masks) != (
            self.num_layers
        ):
            raise ValueError(
                "IDEA Fisher valid-mask count must match aligned layers"
            )
        if not logits.requires_grad:
            raise RuntimeError("IDEA Fisher logits are not gradient-connected")
        disconnected = [
            index for index, feature in enumerate(features)
            if not torch.is_tensor(feature) or not feature.requires_grad
        ]
        if disconnected:
            raise RuntimeError(
                "IDEA Fisher activations lack gradients at layers {}".format(
                    disconnected
                )
            )
        log_probs = logits.log_softmax(dim=-1).reshape(-1)
        probs = log_probs.detach().exp()
        traces = torch.zeros(self.num_layers)
        for action_index in range(log_probs.shape[0]):
            grads = torch.autograd.grad(
                log_probs[action_index],
                features,
                retain_graph=True,
                allow_unused=True,
            )
            missing = [index for index, grad in enumerate(grads) if grad is None]
            if missing:
                raise RuntimeError(
                    "IDEA Fisher logits are disconnected from aligned layers {}"
                    .format(missing)
                )
            weight = float(probs[action_index].item())
            for layer_index, (grad, valid_mask) in enumerate(
                zip(grads, valid_masks)
            ):
                if grad is not None:
                    if not torch.is_tensor(valid_mask):
                        raise ValueError("IDEA Fisher valid masks must be tensors")
                    expected_shape = tuple(grad.shape[:-1])
                    if tuple(valid_mask.shape) != expected_shape:
                        raise ValueError(
                            "IDEA Fisher valid mask shape {} does not match "
                            "activation shape {} at layer {}".format(
                                tuple(valid_mask.shape), tuple(grad.shape),
                                layer_index,
                            )
                        )
                    valid_mask = valid_mask.to(
                        device=grad.device, dtype=torch.bool
                    )
                    if not bool(valid_mask.any()):
                        raise ValueError(
                            "IDEA Fisher layer {} has no valid navigable tokens"
                            .format(layer_index)
                        )
                    squared_norm = grad.detach().pow(2).sum(dim=-1)
                    traces[layer_index] += weight * float(
                        squared_norm[valid_mask].sum().item()
                    )
        total = float(traces.sum().item())
        if total <= 1e-12:
            raise RuntimeError("IDEA Fisher trace is zero for every aligned layer")
        normalized = traces / total
        # Eq. (8): exponential moving average of the normalised Fisher trace.
        self.alpha = (
            (1.0 - self.fisher_beta) * self.alpha
            + self.fisher_beta * normalized
        )
        self.alpha = self.alpha / self.alpha.sum().clamp_min(1e-12)

    # -- prompt optimisation (Eq. 5) ----------------------------------------
    @torch.enable_grad()
    def _optimize_prompt(self, policy_inputs, source_stats, warm_start):
        device = source_stats[0][0].device
        dtype = source_stats[0][0].dtype
        prompt = self._new_prompt(device, dtype, warm_start=warm_start)
        initial_prompt = prompt.detach().clone()
        optimizer = _make_optimizer([prompt], **self._optimizer_args)
        alpha = self.alpha.to(device=device, dtype=dtype)
        last_loss = 0.0
        for _ in range(self.opt_steps):
            optimizer.zero_grad(set_to_none=True)
            target_stats, _ = self.protocol.fused_forward(policy_inputs, prompt)
            loss = _alignment_discrepancy(source_stats, target_stats, alpha)
            if not loss.requires_grad:
                raise RuntimeError(
                    "IDEA alignment loss is disconnected from the soft prompt"
                )
            loss.backward()
            if prompt.grad is None:
                raise RuntimeError("IDEA soft prompt did not receive a gradient")
            if not bool(torch.isfinite(prompt.grad).all()):
                raise FloatingPointError("IDEA soft-prompt gradient is non-finite")
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([prompt], self.max_grad_norm)
            self.prompt_optimizer_attempts += 1
            optimizer.step()
            self.prompt_optimizer_updates += 1
            last_loss = float(loss.detach().item())
        prompt_delta = (prompt.detach() - initial_prompt).norm()
        prompt_reference = initial_prompt.norm().clamp_min(1e-12)
        relative_drift = float((prompt_delta / prompt_reference).item())
        self.last_prompt_relative_drift = relative_drift
        self.max_prompt_relative_drift = max(
            self.max_prompt_relative_drift, relative_drift
        )
        self.prompt_relative_drift_sum += relative_drift
        self.alignment_loss_sum += last_loss
        self.alignment_step_count += 1
        return prompt.detach()

    # -- per-step decision (Algorithm 1) ------------------------------------
    def prepare_action(self, source_logits, policy_inputs=None, **kwargs):
        if policy_inputs is None:
            raise ValueError("IDEA requires policy_inputs to drive its fusion")
        self._assert_base_policy_frozen()

        # Line 4: measure the target prompt-free.  This pass can never update the
        # offline source anchor; protocols are required to be read-only here.
        with torch.no_grad():
            base_stats, base_logits = self.protocol.fused_forward(
                policy_inputs, None
            )
        gamma_target = self._final_gamma(base_stats)

        source_stats = self.protocol.source_statistics(
            gamma_target.device, gamma_target.dtype
        )
        if len(source_stats) != self.num_layers:
            raise ValueError(
                "IDEA source artifact returned {} layers, expected {}".format(
                    len(source_stats), self.num_layers
                )
            )
        for mu, sigma in source_stats:
            if tuple(mu.shape) != (self.feature_dim,) or tuple(sigma.shape) != (
                self.feature_dim,
            ):
                raise ValueError("IDEA source-statistics feature shape mismatch")
            if not bool(torch.isfinite(mu).all() and torch.isfinite(sigma).all()):
                raise ValueError("IDEA source statistics must be finite")
        gamma_source = self._final_gamma(source_stats)

        chosen_prompt = None
        final_logits = base_logits
        bridge_prompt = None
        covered = False

        if len(self.library) > 0:
            # Lines 5-6: closed-form bridge over the current library.
            prompts, gammas, uncertainties = self.library.stacked(
                gamma_target.device, gamma_target.dtype
            )
            weights = solve_bridge_weights(
                gammas, gamma_target, uncertainties, self.lam, self.ridge
            )
            bridge_prompt = torch.einsum("k,klc->lc", weights, prompts)
            self.last_bridge_weight_max = float(weights.max().item())

            # Lines 7-8: coverage gate.  d_0 is the raw target-vs-source gap and
            # d_p is the gap after injecting the bridge prompt; the paper deems
            # the domain covered when the bridge shrinks the gap below tau*d_0.
            # (The Algorithm-1 figure overloads the Gamma_t/Gamma_b symbols; we
            # follow the text's "injected vs not injected" definition so the
            # ratio test is well-posed.)
            with torch.no_grad():
                bridge_stats, bridge_logits = self.protocol.fused_forward(
                    policy_inputs, bridge_prompt
                )
            gamma_bridge = self._final_gamma(bridge_stats)
            d_zero = _gaussian_w2(
                gamma_target[: self.feature_dim],
                gamma_target[self.feature_dim:],
                gamma_source[: self.feature_dim],
                gamma_source[self.feature_dim:],
            )
            d_prompt = _gaussian_w2(
                gamma_bridge[: self.feature_dim],
                gamma_bridge[self.feature_dim:],
                gamma_source[: self.feature_dim],
                gamma_source[self.feature_dim:],
            )
            self.last_coverage_ratio = float(
                (d_prompt / d_zero.clamp_min(1e-12)).item()
            )
            covered = bool(d_prompt < self.tau * d_zero)
            if covered:
                # Line 9: reuse the bridge directly (training-free shortcut).
                chosen_prompt = bridge_prompt.detach()
                final_logits = bridge_logits
                self.covered_steps += 1

        if not covered:
            # Lines 10-14: treat as a new domain and refine a fresh asset,
            # warm-starting from the bridge prompt when one exists (Eq. 9).
            warm_start = None
            if bridge_prompt is not None:
                warm_start = bridge_prompt.detach()
            else:
                self.cold_start_steps += 1
            self._update_fisher_weights(policy_inputs)
            chosen_prompt = self._optimize_prompt(
                policy_inputs, source_stats, warm_start
            )
            with torch.no_grad():
                _, final_logits = self.protocol.fused_forward(
                    policy_inputs, chosen_prompt
                )
            uncertainty = float(softmax_entropy(final_logits).mean().item())
            self.last_uncertainty = uncertainty
            # Lines 15-19: store or nearest-neighbour merge the new asset.
            self.library.add(chosen_prompt, gamma_target, uncertainty)
            self.new_domain_steps += 1
        else:
            self.last_uncertainty = float(
                softmax_entropy(final_logits).mean().item()
            )

        self._record_loss(softmax_entropy(final_logits).mean())
        self._assert_base_policy_frozen()
        return final_logits.detach()

    def adapt(self, logits, **kwargs):
        """No-op: IDEA resolves its prompt inside :meth:`prepare_action`."""
        return None

    def reset(self):
        self.library.clear()
        self.alpha = torch.full(
            (self.num_layers,), 1.0 / self.num_layers
        )
        self.episode_count = 0
        self.covered_steps = 0
        self.new_domain_steps = 0
        self.cold_start_steps = 0
        self.last_bridge_weight_max = 0.0
        self.last_coverage_ratio = 0.0
        self.last_uncertainty = 0.0
        self.alignment_loss_sum = 0.0
        self.alignment_step_count = 0
        self.prompt_optimizer_attempts = 0
        self.prompt_optimizer_updates = 0
        self.prompt_relative_drift_sum = 0.0
        self.last_prompt_relative_drift = 0.0
        self.max_prompt_relative_drift = 0.0
        self._init_diagnostics()

    def episode_start(self):
        # Assets persist across episodes; only a full reset clears the library.
        return None

    def episode_end(self, episode_stats=None):
        self.episode_count += 1

    def save_assets(self, path, metadata=None):
        """Serialize the asset library and return the artifact file SHA256."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.library.payload(
            self.prompt_length, self.feature_dim, metadata=metadata
        )
        if path.suffix.lower() == ".json":
            path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        elif path.suffix.lower() in (".pt", ".pth"):
            torch.save(payload, path)
        else:
            raise ValueError("IDEA asset artifact must end in .json or .pt")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest

    def load_assets(self, path, expected_sha256):
        """Load an asset artifact only after verifying its external digest."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("IDEA asset artifact is missing: {}".format(path))
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("IDEA asset expected SHA256 is required")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise ValueError("IDEA asset artifact SHA256 mismatch")
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() in (".pt", ".pth"):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:  # pragma: no cover - older supported torch
                payload = torch.load(path, map_location="cpu")
        else:
            raise ValueError("IDEA asset artifact must end in .json or .pt")
        self.library.load_payload(payload, self.prompt_length, self.feature_dim)
        return actual

    def diagnostics(self, verify_base_content=True):
        # Formal final diagnostics hash every base tensor.  Intermediate VLN
        # progress snapshots may opt out because copying a multi-billion-byte
        # policy to CPU once per episode would dominate evaluation time.  The
        # final snapshot remains mandatory and is enforced by the campaign
        # validator.
        verify_base_content = bool(verify_base_content)
        if verify_base_content:
            base_after = _base_parameter_snapshot(self.model)
        else:
            named = sorted(self.model.named_parameters(), key=lambda item: item[0])
            names = tuple(name for name, _ in named)
            base_after = {
                "names": names,
                "name_set_sha256": hashlib.sha256(
                    b"navtta.idea.base_parameter_name_set.v1\0"
                    + b"".join(name.encode("utf-8") + b"\0" for name in names)
                ).hexdigest(),
                "content_sha256": None,
                "parameter_sha256": {},
                "versions": {
                    name: int(parameter._version) for name, parameter in named
                },
            }
        base_before = self._base_parameter_before
        before_names = set(base_before["names"])
        after_names = set(base_after["names"])
        added_names = sorted(after_names - before_names)
        removed_names = sorted(before_names - after_names)
        common_names = before_names & after_names
        version_changed_names = sorted(
            name for name in common_names
            if base_after["versions"][name] != base_before["versions"][name]
        )
        content_changed_names = (
            sorted(
                name for name in common_names
                if base_after["parameter_sha256"][name]
                != base_before["parameter_sha256"][name]
            )
            if verify_base_content else []
        )
        changed_base_parameters = sorted(set(
            added_names + removed_names + version_changed_names
            + content_changed_names
        ))
        trainable_base_parameters = sorted(
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )
        gradient_base_parameters = sorted(
            name for name, parameter in self.model.named_parameters()
            if parameter.grad is not None
        )
        name_set_unchanged = (
            base_after["name_set_sha256"]
            == base_before["name_set_sha256"]
        )
        content_hash_match = (
            base_after["content_sha256"] == base_before["content_sha256"]
            if verify_base_content else None
        )
        base_parameter_unchanged = (
            name_set_unchanged
            and content_hash_match is True
            and not version_changed_names
            and not trainable_base_parameters
            and not gradient_base_parameters
        ) if verify_base_content else None
        output = {
            "method": "idea",
            "action_steps": int(self.action_steps),
            "episodes": int(self.episode_count),
            "mean_entropy": self.loss_sum / max(1, self.action_steps),
            "last_entropy": self.last_loss,
            "current_lr": self.current_lr,
            # IDEA adapts one external LxC prompt at a time.  These generic
            # fields intentionally describe that trainable state, not the
            # frozen base policy, so experiment validators can distinguish a
            # live IDEA run from a Source-like no-op.
            "updates": int(self.prompt_optimizer_updates),
            "slow_updates": 0,
            "adapted_parameter_names": ["external_soft_prompt"],
            "adapted_parameter_count": (
                self.prompt_length * self.feature_dim
            ),
            "relative_param_drift": (
                self.prompt_relative_drift_sum
                / max(1, self.alignment_step_count)
            ),
            "parameter_drift_semantics": (
                "mean_relative_soft_prompt_optimization_drift"
            ),
            "prompt_optimizer_attempts": self.prompt_optimizer_attempts,
            "prompt_optimizer_updates": self.prompt_optimizer_updates,
            "prompt_optimizations": self.alignment_step_count,
            "last_prompt_relative_drift": self.last_prompt_relative_drift,
            "max_prompt_relative_drift": self.max_prompt_relative_drift,
            "library_size": len(self.library),
            "library_capacity": self.library.capacity,
            "asset_adds": self.library.add_count,
            "asset_merges": self.library.merge_count,
            "covered_steps": self.covered_steps,
            "new_domain_steps": self.new_domain_steps,
            "cold_start_steps": self.cold_start_steps,
            "coverage_rate": self.covered_steps / max(1, self.action_steps),
            "last_bridge_weight_max": self.last_bridge_weight_max,
            "last_coverage_ratio": self.last_coverage_ratio,
            "last_uncertainty": self.last_uncertainty,
            "mean_alignment_loss": (
                self.alignment_loss_sum / max(1, self.alignment_step_count)
            ),
            "fisher_alpha": [float(value) for value in self.alpha.tolist()],
            "prompt_length": self.prompt_length,
            "lambda": self.lam,
            "tau": self.tau,
            "opt_steps": self.opt_steps,
            "use_fisher": self.use_fisher,
            "base_parameter_integrity_schema": (
                "navtta.idea.base_parameter_integrity.v1"
            ),
            "base_parameter_integrity_complete": verify_base_content,
            "base_parameter_unchanged": base_parameter_unchanged,
            "trains_base_policy": bool(trainable_base_parameters),
            "base_parameter_requires_grad_names": trainable_base_parameters,
            "base_parameter_grads_none": not gradient_base_parameters,
            "base_parameter_gradient_names": gradient_base_parameters,
            "base_parameter_name_count_before": len(base_before["names"]),
            "base_parameter_name_count_after": len(base_after["names"]),
            "base_parameter_name_set_before_sha256": (
                base_before["name_set_sha256"]
            ),
            "base_parameter_name_set_after_sha256": (
                base_after["name_set_sha256"]
            ),
            "base_parameter_name_set_unchanged": name_set_unchanged,
            "base_parameter_added_names": added_names,
            "base_parameter_removed_names": removed_names,
            "base_parameter_content_before_sha256": (
                base_before["content_sha256"]
            ),
            "base_parameter_content_after_sha256": (
                base_after["content_sha256"]
            ),
            "base_parameter_content_hash_match": content_hash_match,
            "base_parameter_content_modified_names": content_changed_names,
            "base_parameter_content_hash_algorithm": (
                "sha256_of_sorted_named_parameter_sha256_v1"
            ),
            "base_parameter_versions_unchanged": (
                name_set_unchanged and not version_changed_names
            ),
            "base_parameter_version_changed_names": version_changed_names,
            "base_parameter_modified_names": changed_base_parameters,
            "source_statistics_mode": "offline_artifact",
            "source_statistics_schema": self.source_statistics_metadata.get(
                "schema"
            ),
            "source_statistics_sha256": self.source_statistics_metadata.get(
                "sha256"
            ),
            "source_statistics_trajectory_count": (
                self.source_statistics_metadata.get("trajectory_count")
            ),
            "source_statistics_moment_estimator": (
                self.source_statistics_metadata.get("moment_estimator")
            ),
            "source_statistics_provenance": dict(
                self.source_statistics_metadata.get("provenance", {})
            ),
            "asset_library_digest": self.library.digest(
                self.prompt_length, self.feature_dim
            ),
            "coverage_gate": "d_prompt < tau * d_zero",
            "coverage_gate_interpretation": (
                "prompted-vs-source over prompt-free-vs-source; Algorithm 1 "
                "overloads Gamma_t/Gamma_b, so the paper prose is canonical"
            ),
            "bridge_simplex_correction": "paper_clip_then_renormalize",
        }
        if self.zero_update_audit:
            # IDEA never writes policy parameters, so a frozen base model is the
            # expected, verifiable outcome of this audit.
            if self.audit_model_states_after_sha256 is None:
                self.audit_model_states_after_sha256 = (
                    self._capture_audit_model_states()
                )
            before = self.audit_model_states_before_sha256
            after = self.audit_model_states_after_sha256
            output.update({
                "audit_mode": "idea_frozen_base_policy",
                "primary_model_state_hash_match": (
                    None if before is None or after is None else
                    before["primary_model"] == after["primary_model"]
                ),
            })
        return output


__all__ = [
    "IDEAAdapter",
    "IDEAFusionProtocol",
    "solve_bridge_weights",
]
