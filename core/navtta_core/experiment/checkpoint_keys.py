"""Strict, task-agnostic checkpoint key normalization.

Upstream navigation baselines often merge a checkpoint into a freshly
initialized ``state_dict`` before calling ``load_state_dict``.  That makes a
strict PyTorch load appear successful even when learned parameters are absent.
For final evaluation checkpoints we instead require exact key identity, while
still accepting the one harmless representation difference introduced by
``torch.nn.DataParallel``: an all-keys ``module.`` prefix.
"""

from collections import OrderedDict
from typing import Any, Mapping, Sequence


MODULE_PREFIX = "module."


def _copy_state_dict(state_dict: Mapping[str, Any], transform: Any) -> OrderedDict:
    copied = OrderedDict((transform(str(key)), value) for key, value in state_dict.items())

    # PyTorch state dictionaries may carry module-version metadata outside the
    # mapping itself.  Preserve it when possible and apply the same whole-tree
    # prefix conversion.  Loading also works without this metadata, but keeping
    # it avoids changing module-specific migration behavior.
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        copied._metadata = OrderedDict(
            (transform(str(key)), value) for key, value in metadata.items()
        )
    return copied


def _strip_module_prefix(key: str) -> str:
    if key == "module":
        return ""
    if key.startswith(MODULE_PREFIX):
        return key[len(MODULE_PREFIX) :]
    return key


def _add_module_prefix(key: str) -> str:
    if not key:
        return "module"
    return MODULE_PREFIX + key


def normalize_strict_checkpoint_state_dict(
    model_state: Mapping[str, Any],
    checkpoint_state: Mapping[str, Any],
    component: str = "model",
    allowed_unexpected_prefixes: Sequence[str] = (),
) -> OrderedDict:
    """Return checkpoint state with keys exactly matching ``model_state``.

    The function accepts exact keys, all checkpoint keys prefixed by
    ``module.``, or all model keys prefixed by ``module.``.  Partial/mixed
    prefixes and every other missing or unexpected key are rejected.  Tensor
    shapes remain PyTorch's responsibility when the returned mapping is loaded
    with ``strict=True``.  A caller may name narrow prefixes for known
    checkpoint-only auxiliary modules that are absent from the inference
    graph; every model key is still required and every other extra is rejected.
    """
    if not isinstance(model_state, Mapping) or not isinstance(
        checkpoint_state, Mapping
    ):
        raise TypeError("model and checkpoint state must be mappings")
    allowed_unexpected_prefixes = tuple(allowed_unexpected_prefixes)
    if any(
        not isinstance(prefix, str) or not prefix
        for prefix in allowed_unexpected_prefixes
    ):
        raise ValueError("allowed unexpected prefixes must be non-empty strings")

    model_keys = set(map(str, model_state.keys()))
    checkpoint_keys = set(map(str, checkpoint_state.keys()))
    normalized = _copy_state_dict(checkpoint_state, lambda key: key)

    if checkpoint_keys == model_keys:
        return normalized

    checkpoint_all_prefixed = bool(checkpoint_keys) and all(
        key.startswith(MODULE_PREFIX) for key in checkpoint_keys
    )
    model_all_prefixed = bool(model_keys) and all(
        key.startswith(MODULE_PREFIX) for key in model_keys
    )

    if checkpoint_all_prefixed and not model_all_prefixed:
        candidate = _copy_state_dict(checkpoint_state, _strip_module_prefix)
        if set(candidate) == model_keys:
            return candidate
        normalized = candidate
    elif model_all_prefixed and not checkpoint_all_prefixed:
        # Adding a prefix is allowed only when *none* of the loaded keys is
        # already prefixed.  Mixed-prefix checkpoints are always malformed.
        if not any(key.startswith(MODULE_PREFIX) for key in checkpoint_keys):
            candidate = _copy_state_dict(checkpoint_state, _add_module_prefix)
            if set(candidate) == model_keys:
                return candidate
            normalized = candidate

    normalized_keys = set(normalized)
    missing = sorted(model_keys.difference(normalized_keys))
    unexpected = sorted(normalized_keys.difference(model_keys))
    disallowed_unexpected = [
        key
        for key in unexpected
        if not key.startswith(allowed_unexpected_prefixes)
    ]
    if missing or disallowed_unexpected:
        raise RuntimeError(
            "{} checkpoint/model key mismatch: missing={} unexpected={}".format(
                component, missing, disallowed_unexpected
            )
        )

    filtered = OrderedDict(
        (key, value) for key, value in normalized.items() if key in model_keys
    )
    metadata = getattr(normalized, "_metadata", None)
    if metadata is not None:
        filtered._metadata = metadata
    return filtered


def validate_strict_checkpoint_loading_info(
    loading_info: Mapping[str, Any], component: str = "model"
) -> None:
    """Reject non-empty Hugging Face checkpoint loading diagnostics."""
    if not isinstance(loading_info, Mapping):
        raise TypeError("checkpoint loading info must be a mapping")
    fields = (
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    )
    problems = {
        field: list(loading_info.get(field) or [])
        for field in fields
        if loading_info.get(field)
    }
    if problems:
        raise RuntimeError(
            "{} checkpoint loading diagnostics are non-empty: {}".format(
                component, problems
            )
        )
