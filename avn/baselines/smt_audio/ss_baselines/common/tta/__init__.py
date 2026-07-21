"""Compatibility re-export of the shared NavTTA adapters."""

from navtta_core.tta import (
    ATENAAdapter,
    EAMAdapter,
    FEEDTTAAdapter,
    FSTTAAdapter,
    TentAdapter,
    build_adapter,
    configure_tta_model,
)

__all__ = [
    "TentAdapter",
    "FSTTAAdapter",
    "EAMAdapter",
    "FEEDTTAAdapter",
    "ATENAAdapter",
    "configure_tta_model",
    "build_adapter",
]
