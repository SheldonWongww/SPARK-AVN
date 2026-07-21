"""Compatibility import for the pre-refactor SoundSpaces module path.

New code should import adapters from :mod:`navtta_core.tta` directly. This
module remains so upstream-style scripts and old checkpoints/configurations do
not require path-specific patches.
"""

from navtta_core.tta.tta_core import *  # noqa: F401,F403
