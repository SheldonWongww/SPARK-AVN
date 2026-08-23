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
import torch
import torch.nn as nn

from .tta_core import (
    _AdapterDiagnostics,
    _make_optimizer,
    softmax_entropy,
)


# ---------------------------------------------------------------------------
# Fusion protocol: the single model-specific seam.
# ---------------------------------------------------------------------------
class IDEAFusionProtocol:
    """Model-specific contract that lets IDEA drive one navigation step.

    A task implements this by *calling* its (frozen) fusion sub-modules; it must
    not require any change to the underlying model definition.  Every method
    operates on a single streaming step (batch size 1) and pools statistics over
    the ``N`` candidate/navigable-node dimension, matching the online setting.

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

        Returns ``(layer_features, logits)`` where ``layer_features`` is a list
        of ``M`` tensors of shape ``[N, C]`` and ``logits`` has shape ``[1, A]``.
        Each layer feature must be part of the autograd graph of ``logits`` so
        that IDEA can form ``grad(log pi(a), Z_l)`` for the Fisher trace
        (Eq. 7-8).  A binding typically enables grad on the fused token
        embedding and captures the per-layer activations with forward hooks --
        no model edit and no parameter unfreezing is needed.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Pure, task-agnostic math (unit-tested in isolation).
# ---------------------------------------------------------------------------
def _stats_vector(mu, sigma):
    """Vectorise a Gaussian descriptor ``Gamma = [mu; sigma]`` into ``[2C]``."""
    return torch.cat([mu.reshape(-1), sigma.reshape(-1)], dim=0)


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

        self.model = model
        self.protocol = fusion_protocol
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
        self._init_diagnostics()

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
        features, logits = self.protocol.fisher_forward(policy_inputs)
        if len(features) != self.num_layers:
            raise ValueError(
                "IDEA fisher_forward returned {} layers, expected {}".format(
                    len(features), self.num_layers
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
            weight = float(probs[action_index].item())
            for layer_index, grad in enumerate(grads):
                if grad is not None:
                    traces[layer_index] += weight * float(
                        grad.detach().pow(2).sum().item()
                    )
        total = float(traces.sum().item())
        if total <= 1e-12:
            return
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
        optimizer = _make_optimizer([prompt], **self._optimizer_args)
        alpha = self.alpha.to(device=device, dtype=dtype)
        last_loss = 0.0
        for _ in range(self.opt_steps):
            optimizer.zero_grad(set_to_none=True)
            target_stats, _ = self.protocol.fused_forward(policy_inputs, prompt)
            loss = _alignment_discrepancy(source_stats, target_stats, alpha)
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([prompt], self.max_grad_norm)
            optimizer.step()
            last_loss = float(loss.detach().item())
        self.alignment_loss_sum += last_loss
        self.alignment_step_count += 1
        return prompt.detach()

    # -- per-step decision (Algorithm 1) ------------------------------------
    def prepare_action(self, source_logits, policy_inputs=None, **kwargs):
        if policy_inputs is None:
            raise ValueError("IDEA requires policy_inputs to drive its fusion")
        device = source_logits.device if torch.is_tensor(source_logits) else None
        dtype = (
            source_logits.dtype if torch.is_tensor(source_logits) else None
        )

        # Line 4: observe the prompt-free target statistics Gamma_t first, so a
        # protocol that bootstraps its source anchor from streamed prompt-free
        # passes has seen at least this step before we query the anchor.
        with torch.no_grad():
            base_stats, base_logits = self.protocol.fused_forward(
                policy_inputs, None
            )
        gamma_target = self._final_gamma(base_stats)

        source_stats = self.protocol.source_statistics(device, dtype)
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
        self._init_diagnostics()

    def episode_start(self):
        # Assets persist across episodes; only a full reset clears the library.
        return None

    def episode_end(self, episode_stats=None):
        self.episode_count += 1

    def diagnostics(self):
        output = {
            "method": "idea",
            "action_steps": int(self.action_steps),
            "episodes": int(self.episode_count),
            "mean_entropy": self.loss_sum / max(1, self.action_steps),
            "last_entropy": self.last_loss,
            "current_lr": self.current_lr,
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
            "trains_base_policy": False,
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
