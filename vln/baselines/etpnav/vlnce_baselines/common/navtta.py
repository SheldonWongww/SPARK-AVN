"""ETPNav import bridge for the task-shared VLN-CE TTA controller."""

import sys
from pathlib import Path


for _parent in Path(__file__).resolve().parents:
    if (_parent / "navtta_vln").is_dir():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break
else:  # pragma: no cover - installation/layout error
    raise ImportError("Could not locate the task-shared navtta_vln package")

from navtta_vln.continuous_tta import (  # noqa: E402,F401
    ContinuousVLNTTA,
    FEEDBACK_METHODS,
    TTA_METHODS,
    make_continuous_tta_config,
    validate_continuous_tta_run,
)


__all__ = [
    "ContinuousVLNTTA",
    "FEEDBACK_METHODS",
    "TTA_METHODS",
    "make_continuous_tta_config",
    "validate_continuous_tta_run",
]
