#!/usr/bin/env python3
"""Live terminal monitor for PONI + FSTTA experiments."""

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
TENT_DIR = SCRIPT_DIR.parent / "Tent"
sys.path.insert(0, str(TENT_DIR))

from monitor_tent import main  # noqa: E402


if __name__ == "__main__":
    main(
        default_method="FSTTA",
        default_eval_script="eval_poni_fstta.py",
        default_save_root=(SCRIPT_DIR.parents[1] / "experiments/FSTTA"),
    )
