"""VLN-specific online adaptation integration utilities."""

from .discrete_tta import DiscreteTTAAgentMixin, add_discrete_tta_args

__all__ = ["DiscreteTTAAgentMixin", "add_discrete_tta_args"]
