"""Shared RedNet task adapters for parameter-efficient semantic TTA."""

import torch
import torch.nn as nn


class RedNetNormView(nn.Module):
    """Register selected RedNet modules as a shared-parameter adapter view."""

    def __init__(self, rednet, prefixes):
        super().__init__()
        self.prefixes = tuple(prefixes)
        if not self.prefixes:
            raise ValueError("RedNetNormView requires at least one prefix")
        rednet.eval()
        rednet.requires_grad_(False)
        for prefix in self.prefixes:
            try:
                module = rednet.get_submodule(prefix)
            except AttributeError:
                module = rednet
                for item in prefix.split("."):
                    module = getattr(module, item)
            self.add_module(prefix.replace(".", "__"), module)


def adaptation_model(rednet, norm_prefixes):
    prefixes = tuple(norm_prefixes)
    return RedNetNormView(rednet, prefixes) if prefixes else rednet


def select_entropy_logits(
    logits_map,
    raw_depth,
    mode="all",
    min_pixels=128,
    background_index=0,
):
    """Return categorical logits selected for the unsupervised entropy loss."""

    mode = str(mode).lower()
    flat = logits_map.permute(0, 2, 3, 1).reshape(-1, logits_map.shape[1])
    total = int(flat.shape[0])
    if mode == "all":
        return flat, total, total
    valid_modes = ("valid_depth", "foreground_valid_depth")
    if mode not in valid_modes:
        raise ValueError(
            "Unknown entropy mode {!r}; expected all, valid_depth, or "
            "foreground_valid_depth".format(mode)
        )
    depth = raw_depth[:, 0]
    valid = (depth >= 1.0) & (depth <= 5.0)
    if mode == "foreground_valid_depth":
        foreground = logits_map.detach().argmax(dim=1) != int(background_index)
        valid = valid & foreground
    selected = flat[valid.reshape(-1)]
    count = int(selected.shape[0])
    if count < int(min_pixels):
        return None, count, total
    return selected, count, total


class MaskDiagnostics:
    def __init__(self):
        self.frames = 0
        self.adapted_frames = 0
        self.skipped_frames = 0
        self.selected_pixels = 0
        self.total_pixels = 0

    def record(self, selected, total, adapted):
        self.frames += 1
        self.adapted_frames += int(adapted)
        self.skipped_frames += int(not adapted)
        self.selected_pixels += int(selected)
        self.total_pixels += int(total)

    def as_dict(self):
        return {
            "frames": self.frames,
            "adapted_frames": self.adapted_frames,
            "skipped_frames": self.skipped_frames,
            "selected_pixels": self.selected_pixels,
            "total_pixels": self.total_pixels,
            "mean_selected_fraction": (
                self.selected_pixels / max(1, self.total_pixels)
            ),
        }

