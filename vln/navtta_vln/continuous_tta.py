"""Shared high-level VLN-CE decision adapter for online adaptation.

Only the navigation distribution over graph waypoints is adapted here.  The
waypoint predictor and Habitat low-level controller stay outside the TTA action
unit.  Batch size one lets us remove masked/visited actions before entropy
losses, avoiding ``0 * log(0)`` NaNs while retaining an exact native-index map.
"""

import copy
import json
import hashlib
import os
from pathlib import Path

import torch

from navtta_core.tta import build_adapter, module_state_sha256


TTA_METHODS = ("source", "tent", "fstta", "eam", "feedtta", "atena", "idea")
FEEDBACK_METHODS = ("feedtta", "atena")
FEEDTTA_SCOPE_PROFILES = (
    "configured_prefixes", "paper_full", "last_crossmodal", "action_head",
)


def make_continuous_tta_config(CN, trainable_prefixes):
    """Return the common YACS config surface used by both CE baselines."""
    config = CN()
    config.METHOD = "none"
    config.AUDIT_ZERO_UPDATE = False
    config.AUDIT_CONTROL = False
    config.AUDIT_EXPECTED_EPISODES = -1
    config.DIAGNOSTICS_FILE = ""
    config.ACTION_SELECTION = "argmax"
    config.MATCHED_FEEDTTA_SOURCE = False
    config.ACTION_SEED = 0
    config.EPISODIC = False
    config.LR = 1e-6
    config.STEPS = 1
    config.NORM_SCOPE = "last_k_ln"
    config.LAST_K_LN = 4
    config.RESET_BN_STATS = True
    config.OPTIMIZER = "Adam"
    config.MOMENTUM = 0.9
    config.BETA1 = 0.9
    config.BETA2 = 0.999
    config.WEIGHT_DECAY = 0.0
    config.MAX_GRAD_NORM = 1.0
    config.UPDATE_INTERVAL = 1
    config.MAX_UPDATES_PER_EPISODE = -1
    config.LOG_INTERVAL_EPISODES = 50

    config.FSTTA = CN()
    config.FSTTA.M = 3
    config.FSTTA.N = 4
    config.FSTTA.Q = 0.1
    config.FSTTA.LR_SLOW = 1e-4
    config.FSTTA.RHO = 0.9
    config.FSTTA.TAU = 0.5
    config.FSTTA.A = 0.5
    config.FSTTA.B = 1.5
    config.FSTTA.USE_SLOW = True
    config.FSTTA.FAST_GRAD_MODE = "concordant"
    config.FSTTA.USE_FAST_LR_SCALER = True
    config.FSTTA.OPTIMIZER = "AdamW"
    config.FSTTA.BETA1 = 0.9
    config.FSTTA.BETA2 = 0.99
    config.FSTTA.WEIGHT_DECAY = 0.0
    config.FSTTA.SLOW_OPTIMIZER = ""
    config.FSTTA.SLOW_MOMENTUM = -1.0
    config.FSTTA.RESET_OPTIMIZER_EACH_EPISODE = True
    config.FSTTA.RESET_VAR_HIST_EACH_EPISODE = True
    config.FSTTA.RESET_SLOW_OPTIMIZER_EACH_WINDOW = False
    config.FSTTA.EIGEN_EPS = 1e-6

    config.EAM = CN()
    config.EAM.LR = 1e-5
    config.EAM.CONFIDENCE_SCALE = 0.4
    config.EAM.MEMORY_SIZE = 32
    config.EAM.BATCH_SIZE = 8
    config.EAM.UPDATE_INTERVAL = 1
    config.EAM.PARAM_SCOPE = "module_prefixes"
    config.EAM.TRAINABLE_PREFIXES = list(trainable_prefixes)
    config.EAM.OPTIMIZER = "Adam"
    config.EAM.MOMENTUM = 0.9
    config.EAM.BETA1 = 0.9
    config.EAM.BETA2 = 0.999
    config.EAM.WEIGHT_DECAY = 0.0
    config.EAM.MAX_GRAD_NORM = 0.0

    config.FEEDTTA = CN()
    config.FEEDTTA.LR = 5e-6
    config.FEEDTTA.P = 0.05
    config.FEEDTTA.ALPHA = -0.2
    config.FEEDTTA.SGR_SEED = 0
    config.FEEDTTA.GAMMA = 0.99
    config.FEEDTTA.NORMALIZE_GRADIENT = False
    config.FEEDTTA.PARAM_SCOPE = "module_prefixes"
    config.FEEDTTA.TRAINABLE_PREFIXES = list(trainable_prefixes)
    # Preserve the pre-profile scope unless a job explicitly selects one of
    # the paper-aligned model-aware profiles below.
    config.FEEDTTA.SCOPE_PROFILE = "configured_prefixes"
    config.FEEDTTA.OPTIMIZER = "Adam"
    config.FEEDTTA.BETA1 = 0.9
    config.FEEDTTA.BETA2 = 0.999
    config.FEEDTTA.WEIGHT_DECAY = 0.0
    config.FEEDTTA.EPS = 1e-5
    config.FEEDTTA.MAX_GRAD_NORM = 0.0

    config.ATENA = CN()
    config.ATENA.LR_QUERY = 1e-6
    config.ATENA.LR_SELF = 1e-7
    config.ATENA.MIX_LAMBDA = 0.5
    config.ATENA.QUERY_THRESHOLD = 0.1
    config.ATENA.SELF_LOSS_WEIGHT = 0.1
    config.ATENA.PARAM_SCOPE = "all"
    config.ATENA.OPTIMIZER = "AdamW"
    config.ATENA.BETA1 = 0.9
    config.ATENA.BETA2 = 0.999
    config.ATENA.WEIGHT_DECAY = 0.01
    config.ATENA.MAX_GRAD_NORM = 0.0

    config.IDEA = CN()
    config.IDEA.PROMPT_LENGTH = 4
    config.IDEA.K_MAX = 32
    config.IDEA.LAMBDA = 0.4
    config.IDEA.TAU = 0.7
    config.IDEA.FISHER_BETA = 0.1
    config.IDEA.OPT_STEPS = 50
    config.IDEA.LR = 3e-3
    config.IDEA.OPTIMIZER = "AdamW"
    config.IDEA.BETA1 = 0.9
    config.IDEA.BETA2 = 0.999
    config.IDEA.WEIGHT_DECAY = 0.0
    config.IDEA.USE_FISHER = True
    config.IDEA.RIDGE = 1e-4
    config.IDEA.MAX_GRAD_NORM = 0.0
    config.IDEA.PROMPT_INIT_STD = 0.02
    config.IDEA.SEED = 0
    # 0 aligns every fusion layer; ETPNav/BEVBert default to num_x_layers=4.
    config.IDEA.PROMPT_LAYERS = 0
    config.IDEA.SOURCE_WARMUP_STEPS = 64
    return config


def _tree_detach(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _tree_detach(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_detach(item) for item in value)
    if isinstance(value, list):
        return [_tree_detach(item) for item in value]
    return value


def _feedtta_trainable_prefixes(model, variant, profile, configured_prefixes):
    """Resolve the paper's cross-modal-and-downstream scope for VLN-CE."""
    profile = str(profile).lower()
    if profile not in FEEDTTA_SCOPE_PROFILES:
        raise ValueError("Unknown continuous FeedTTA scope profile: {}".format(
            profile
        ))
    if profile == "configured_prefixes":
        prefixes = tuple(configured_prefixes)
        if not prefixes:
            raise ValueError("configured FeedTTA prefixes cannot be empty")
        return prefixes

    variant = str(variant).lower()
    if variant == "etpnav":
        branch_names = ("global_encoder",)
        head_names = ("global_sap_head",)
    elif variant == "bevbert":
        branch_names = ("global_encoder", "local_encoder")
        head_names = (
            "global_sap_head", "local_sap_head", "sap_fuse_linear",
        )
    else:
        raise ValueError("Unknown continuous VLN variant: {}".format(variant))

    missing_heads = [name for name in head_names if not hasattr(model, name)]
    if missing_heads:
        raise ValueError(
            "Continuous FeedTTA model is missing action heads: {}".format(
                ", ".join(missing_heads)
            )
        )
    if profile == "action_head":
        return head_names

    stacks = []
    for branch_name in branch_names:
        branch = getattr(model, branch_name, None)
        encoder = getattr(branch, "encoder", None)
        layers = getattr(encoder, "x_layers", None)
        if layers is None or len(layers) < 1:
            raise ValueError(
                "Could not infer {} FeedTTA cross-modal stack".format(
                    branch_name
                )
            )
        stack_prefix = "{}.encoder.x_layers".format(branch_name)
        stacks.append((stack_prefix, len(layers)))

    if profile == "paper_full":
        return tuple(prefix for prefix, _ in stacks) + head_names
    return tuple(
        "{}.{}".format(prefix, length - 1)
        for prefix, length in stacks
    ) + head_names


def _adapter_config_with_feedtta_scope(tta_cfg, model, variant):
    """Clone the frozen runtime config and bind an explicit FeedTTA scope."""
    feed_cfg = getattr(tta_cfg, "FEEDTTA", None)
    profile = str(
        getattr(feed_cfg, "SCOPE_PROFILE", "configured_prefixes")
    ).lower()
    configured = tuple(getattr(feed_cfg, "TRAINABLE_PREFIXES", ()))
    prefixes = _feedtta_trainable_prefixes(
        model, variant, profile, configured
    )
    adapter_cfg = (
        tta_cfg.clone() if hasattr(tta_cfg, "clone") else copy.deepcopy(tta_cfg)
    )
    was_frozen = bool(
        hasattr(adapter_cfg, "is_frozen") and adapter_cfg.is_frozen()
    )
    if hasattr(adapter_cfg, "defrost"):
        adapter_cfg.defrost()
    adapter_cfg.FEEDTTA.TRAINABLE_PREFIXES = list(prefixes)
    adapter_cfg.FEEDTTA.SCOPE_PROFILE = profile
    if was_frozen and hasattr(adapter_cfg, "freeze"):
        adapter_cfg.freeze()
    return adapter_cfg, profile, prefixes


class ContinuousVLNTTA:
    """Own one core adapter and the ETPNav/BEVBert decision protocol."""

    _ETP_KEYS = (
        "txt_embeds",
        "txt_masks",
        "gmap_vp_ids",
        "gmap_step_ids",
        "gmap_img_fts",
        "gmap_pos_fts",
        "gmap_masks",
        "gmap_visited_masks",
        "gmap_pair_dists",
    )
    _BEV_KEYS = _ETP_KEYS + (
        "bev_fts",
        "bev_pos_fts",
        "bev_masks",
        "bev_nav_masks",
        "bev_cand_idxs",
        "bev_cand_vpids",
    )

    def __init__(
        self,
        decision_model,
        tta_cfg,
        variant,
        diagnostics_path,
        stream_name=None,
    ):
        self.model = decision_model
        self.model.eval()
        self.tta_cfg = tta_cfg
        self.method = str(getattr(tta_cfg, "METHOD", "none")).lower()
        self.variant = str(variant).lower()
        if self.variant not in ("etpnav", "bevbert"):
            raise ValueError("Unknown continuous VLN variant: {}".format(variant))
        if self.method not in TTA_METHODS:
            raise ValueError(
                "Unknown continuous VLN TTA method: {}".format(self.method)
            )

        self.logits_key = (
            "global_logits" if self.variant == "etpnav" else "fused_logits"
        )
        self.diagnostics_path = str(diagnostics_path)
        self.stream_name = stream_name
        self.episode_count = 0
        self.action_steps = 0
        self.stop_actions = 0
        self.move_actions = 0
        self.max_probability_sum = 0.0
        self._episode_open = False
        self._trajectory_hasher = hashlib.sha256()
        self.trajectory_steps = 0
        self.audit_control = bool(getattr(tta_cfg, "AUDIT_CONTROL", False))
        self.audit_expected_episodes = (
            int(getattr(tta_cfg, "AUDIT_EXPECTED_EPISODES", -1))
            if self.audit_control else None
        )
        self.model_state_before_sha256 = (
            module_state_sha256(self.model) if self.audit_control else None
        )
        self.model_state_after_sha256 = None

        feed_cfg = getattr(tta_cfg, "FEEDTTA", None)
        self.action_seed = int(
            getattr(
                tta_cfg,
                "ACTION_SEED",
                getattr(feed_cfg, "ACTION_SEED", 0)
                if feed_cfg is not None else 0,
            )
        )
        requested_selection = str(
            getattr(tta_cfg, "ACTION_SELECTION", "argmax")
        ).lower()
        if requested_selection not in ("argmax", "sample", "matched_sample"):
            raise ValueError(
                "TTA.ACTION_SELECTION must be argmax, sample, or matched_sample"
            )
        matched_source = bool(
            getattr(tta_cfg, "MATCHED_FEEDTTA_SOURCE", False)
        )
        self.sample_actions = requested_selection in ("sample", "matched_sample")
        if self.method == "source" and matched_source:
            self.sample_actions = True
        if (
            requested_selection != "argmax"
            and self.method not in ("source", "feedtta")
        ):
            raise ValueError(
                "Only FeedTTA and its explicit matched Source control may "
                "sample actions; {} requires argmax".format(self.method.upper())
            )
        self.action_selection = (
            "policy_sampling"
            if self.method == "feedtta" and self.sample_actions
            else (
                "matched_policy_sampling_source_control"
                if self.method == "source" and self.sample_actions
                else "target_native_argmax"
            )
        )
        self._action_generator = torch.Generator(device="cpu")
        self._action_generator.manual_seed(self.action_seed)

        adapter_cfg = tta_cfg
        self.feedtta_scope_profile = None
        self.trainable_prefixes = ()
        if self.method == "feedtta":
            (
                adapter_cfg,
                self.feedtta_scope_profile,
                self.trainable_prefixes,
            ) = _adapter_config_with_feedtta_scope(
                tta_cfg, decision_model, self.variant
            )
        fusion_protocol = None
        if self.method == "idea":
            # IDEA injects a soft prompt into the frozen cross-modal fusion
            # branch that produces this variant's waypoint logits.
            from .idea_fusion import ContinuousIDEAProtocol

            idea_cfg = getattr(tta_cfg, "IDEA", None)
            prompt_layers = int(getattr(idea_cfg, "PROMPT_LAYERS", 0))
            fusion_protocol = ContinuousIDEAProtocol(
                decision_model,
                self._forward_policy,
                self.variant,
                num_layers=(prompt_layers if prompt_layers > 0 else None),
                warmup_steps=int(getattr(idea_cfg, "SOURCE_WARMUP_STEPS", 64)),
            )
        self.adapter = build_adapter(
            decision_model,
            adapter_cfg,
            forward_policy=self._forward_policy,
            fusion_protocol=fusion_protocol,
        )
        if self.method == "feedtta":
            self.adapter.action_selection_protocol = (
                "sample_from_policy"
                if self.sample_actions else "target_native_argmax"
            )

    @property
    def feedback_supervised(self):
        return self.method in FEEDBACK_METHODS

    def _forward_native(self, model, nav_inputs):
        keys = self._ETP_KEYS if self.variant == "etpnav" else self._BEV_KEYS
        missing = [key for key in keys if key not in nav_inputs]
        if missing:
            raise KeyError(
                "Missing high-level navigation inputs: {}".format(
                    ", ".join(missing)
                )
            )
        return model.forward_navigation(*(nav_inputs[key] for key in keys))

    def _compact_decision(self, outputs):
        native_logits = outputs[self.logits_key]
        if native_logits.ndim != 2 or native_logits.shape[0] != 1:
            raise ValueError(
                "Canonical VLN-CE TTA expects [1, actions] logits; got {}"
                .format(tuple(native_logits.shape))
            )
        if bool(torch.isnan(native_logits).any()):
            raise FloatingPointError("High-level navigation logits contain NaN")
        if bool(torch.isposinf(native_logits).any()):
            raise FloatingPointError(
                "High-level navigation logits contain positive infinity"
            )
        finite = torch.isfinite(native_logits[0])
        if not bool(finite.any()):
            raise FloatingPointError(
                "High-level navigation decision has no finite action logits"
            )
        native_indices = finite.nonzero(as_tuple=False).view(-1)
        compact_logits = native_logits.index_select(1, native_indices)
        features = outputs["gmap_embeds"][:, 0]
        if features.ndim != 2 or features.shape[0] != 1:
            raise ValueError(
                "Expected one fixed-width graph decision feature; got {}"
                .format(tuple(features.shape))
            )
        return features, compact_logits, native_indices

    def _forward_policy(self, model, nav_inputs):
        outputs = self._forward_native(model, nav_inputs)
        features, logits, _ = self._compact_decision(outputs)
        return features, logits

    def begin_episode(self):
        if self._episode_open:
            raise RuntimeError("TTA episode_start called twice")
        if self.sample_actions:
            # Re-key each canonical episode so different trajectory lengths in
            # earlier episodes cannot desynchronize paired Source/FeedTTA draws.
            episode_seed = self.action_seed + self.episode_count * 1_000_003
            self._action_generator.manual_seed(episode_seed)
        if self.adapter is not None:
            self.adapter.episode_start()
        self._trajectory_hasher.update(
            "episode:{}\0".format(self.episode_count).encode("ascii")
        )
        self._episode_open = True

    def _matched_sample(self, logits):
        """Sample with a dedicated uniform stream shared by paired runs."""
        probabilities = logits.detach().float().softmax(dim=-1).cpu()
        uniforms = torch.rand(
            (probabilities.shape[0], 1), generator=self._action_generator
        )
        cumulative = probabilities.cumsum(dim=-1)
        sampled = (uniforms > cumulative).sum(dim=-1)
        sampled.clamp_(max=probabilities.shape[-1] - 1)
        return sampled.to(device=logits.device)

    def step(self, nav_inputs):
        """Run, select, and adapt one high-level waypoint decision."""
        if not self._episode_open:
            raise RuntimeError("TTA action step occurred outside an episode")
        nav_inputs = _tree_detach(nav_inputs)

        if self.adapter is not None:
            self.adapter.before_inference(policy_inputs=nav_inputs)
        requires_grad = bool(
            self.adapter is not None and self.adapter.requires_source_grad
        )
        with torch.set_grad_enabled(requires_grad):
            outputs = self._forward_native(self.model, nav_inputs)
            features, source_logits, native_indices = self._compact_decision(
                outputs
            )
            if requires_grad and not source_logits.requires_grad:
                raise RuntimeError(
                    "High-level logits are detached from the selected TTA "
                    "parameters; choose a navigation-head norm/module scope"
                )
            action_logits = (
                source_logits
                if self.adapter is None
                else self.adapter.prepare_action(
                    source_logits,
                    features=features,
                    policy_inputs=nav_inputs,
                )
            )

        if action_logits.shape != source_logits.shape:
            raise RuntimeError(
                "TTA adapter changed the compact high-level action shape"
            )
        with torch.no_grad():
            if self.sample_actions:
                compact_action = self._matched_sample(action_logits)
            else:
                compact_action = action_logits.argmax(dim=-1)
            native_action = native_indices.index_select(0, compact_action)
            max_probability = float(
                action_logits.softmax(dim=-1).max(dim=-1)[0].mean().item()
            )
            action_values = native_action.detach().cpu().view(-1).tolist()
            self._trajectory_hasher.update(
                ("actions:" + ",".join(map(str, action_values)) + "\0").encode(
                    "ascii"
                )
            )
            self.trajectory_steps += len(action_values)

        if self.adapter is not None:
            self.adapter.adapt(
                source_logits,
                action=compact_action,
                features=features,
                policy_inputs=nav_inputs,
            )

        full_action_logits = outputs[self.logits_key].detach().clone()
        full_action_logits[0, native_indices] = action_logits.detach()[0]
        detached_outputs = _tree_detach(outputs)
        detached_outputs[self.logits_key] = full_action_logits

        self.action_steps += 1
        self.max_probability_sum += max_probability
        is_stop = int(native_action.item()) == 0
        self.stop_actions += int(is_stop)
        self.move_actions += int(not is_stop)
        return detached_outputs, full_action_logits, native_action.detach()

    def end_episode(self, episode_stats=None):
        if not self._episode_open:
            raise RuntimeError("TTA episode_end called without episode_start")
        if self.adapter is not None:
            self.adapter.episode_end(episode_stats=episode_stats)
        self._trajectory_hasher.update(b"episode_end\0")
        self._episode_open = False
        self.episode_count += 1
        self.write_diagnostics()

    def diagnostics(self):
        source_control = None
        if self.audit_control:
            if (
                self.episode_count >= self.audit_expected_episodes
                and self.model_state_after_sha256 is None
            ):
                self.model_state_after_sha256 = module_state_sha256(self.model)
            source_control = {
                "action_steps": self.action_steps,
                "episodes": self.episode_count,
                "updates": 0,
                "relative_param_drift": 0.0,
                "control": "source_no_update",
                "audit_expected_episodes": self.audit_expected_episodes,
                "model_state_before_sha256": (
                    self.model_state_before_sha256
                ),
                "model_state_after_sha256": self.model_state_after_sha256,
                "model_state_hash_match": (
                    None if self.model_state_after_sha256 is None else
                    self.model_state_after_sha256
                    == self.model_state_before_sha256
                ),
            }
        return {
            "schema": "navtta.vln_ce_tta.v1",
            "baseline": self.variant,
            "method": self.method,
            "audit_zero_update": bool(
                getattr(self.tta_cfg, "AUDIT_ZERO_UPDATE", False)
            ),
            "audit_control": bool(
                getattr(self.tta_cfg, "AUDIT_CONTROL", False)
            ),
            "stream": self.stream_name,
            "episode_count": self.episode_count,
            "batch_size": 1,
            "tta_step_unit": "high_level_waypoint_navigation_decision",
            "low_level_controller_adapted": False,
            "action_selection": self.action_selection,
            "action_seed": (
                self.action_seed if self.sample_actions else None
            ),
            "action_sampling_rng": (
                "per_episode_dedicated_torch_generator"
                if self.sample_actions else None
            ),
            "action_episode_seed_rule": (
                "ACTION_SEED + episode_index * 1000003"
                if self.sample_actions else None
            ),
            "feedback_supervision": (
                "binary_episode_success" if self.feedback_supervised else "none"
            ),
            "feedtta_scope_profile": self.feedtta_scope_profile,
            "trainable_prefixes": list(self.trainable_prefixes),
            "masked_action_entropy": "finite_logits_only",
            "action_steps": self.action_steps,
            "trajectory_steps": self.trajectory_steps,
            "trajectory_sha256": self._trajectory_hasher.hexdigest(),
            "stop_actions": self.stop_actions,
            "move_actions": self.move_actions,
            "mean_max_action_probability": (
                self.max_probability_sum / max(1, self.action_steps)
            ),
            "adapter": (
                source_control if self.adapter is None else
                self.adapter.diagnostics()
            ),
        }

    def write_diagnostics(self):
        path = Path(self.diagnostics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                self.diagnostics(),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(str(temporary), str(path))


def validate_continuous_tta_run(config, mode, envs):
    """Reject protocol drift before constructing a stateful TTA adapter."""
    tta_cfg = getattr(config, "TTA", None)
    method = str(getattr(tta_cfg, "METHOD", "none")).lower()
    audit_zero_update = bool(getattr(tta_cfg, "AUDIT_ZERO_UPDATE", False))
    audit_control = bool(getattr(tta_cfg, "AUDIT_CONTROL", False))
    if audit_zero_update and method == "source":
        raise ValueError("zero-update adapter audit requires a TTA method")
    if audit_control and method != "source":
        raise ValueError("adapter audit controls require method=source")
    if audit_zero_update and audit_control:
        raise ValueError("audit job cannot be both adapter and Source control")
    if (audit_zero_update or audit_control) and int(
        getattr(tta_cfg, "AUDIT_EXPECTED_EPISODES", -1)
    ) <= 0:
        raise ValueError("adapter audit requires expected episodes > 0")
    if method in ("none", ""):
        return method
    if method not in TTA_METHODS:
        raise ValueError("Unknown continuous VLN TTA method: {}".format(method))
    if mode not in ("eval", "infer"):
        raise ValueError("Continuous VLN TTA is evaluation-only")
    if int(config.GPU_NUMBERS) != 1 or int(config.NUM_ENVIRONMENTS) != 1:
        raise ValueError(
            "Canonical VLN-CE TTA requires GPU_NUMBERS=1 and "
            "NUM_ENVIRONMENTS=1"
        )
    if int(envs.num_envs) != 1:
        raise ValueError("Canonical VLN-CE TTA requires one active environment")
    manifest = (
        config.EVAL.EPISODE_ORDER_MANIFEST
        if mode == "eval" else config.INFERENCE.EPISODE_ORDER_MANIFEST
    )
    if not manifest:
        raise ValueError(
            "Canonical VLN-CE TTA requires an episode-order manifest"
        )
    if mode == "infer" and method in FEEDBACK_METHODS:
        raise ValueError(
            "{} requires binary episode success and cannot run on test-set "
            "inference without evaluator feedback".format(method.upper())
        )
    return method
