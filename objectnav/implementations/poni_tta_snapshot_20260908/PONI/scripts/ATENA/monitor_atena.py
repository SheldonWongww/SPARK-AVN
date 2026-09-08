#!/usr/bin/env python3
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "Tent"))
from monitor_tent import main  # noqa: E402

if __name__ == "__main__":
    main("ATENA", "eval_poni_atena.py", SCRIPT_DIR.parents[1] / "experiments/ATENA")

