"""
runner.py — the ONE entry point for the flood-runner batch.
===========================================================
Reads tools/runners/runner_config.py, looks at PRECIP_METHOD, and runs the
matching pipeline over every flood event:

    thiessen / idw / uniform   → gauge pipeline  (rainfall from the .gag files)
    imerg_thiessen / imerg_idw → IMERG pipeline  (rainfall from GEE IMERG)

You only edit two files:
    tools/runners/runner_config.py   ← all options (start HERE)
    tools/runner.py                  ← this — just run it

Usage
-----
  cd /path/to/OPM
  conda run -n opm python tools/runner.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TOOLS_DIR))

# Model config paths (DEM_PATH, OUTPUT_DIR, …) are relative to the repo root.
os.chdir(REPO_ROOT)

from runners import runner_config as rcfg
from runners import gauge, imerg

GAUGE_METHODS = ("thiessen", "idw", "uniform")
IMERG_METHODS = ("imerg_thiessen", "imerg_idw")


def main():
    method = str(rcfg.PRECIP_METHOD).lower().strip()

    print("=" * 68)
    print(f"  FLOOD RUNNER   |   PRECIP_METHOD = '{method}'")
    print("=" * 68)

    if method in IMERG_METHODS:
        print("  → IMERG pipeline (satellite rainfall from GEE)\n")
        imerg.run()
    elif method in GAUGE_METHODS:
        print("  → gauge pipeline (rainfall from the .gag files)\n")
        gauge.run()
    else:
        print(f"[ERROR] Unknown PRECIP_METHOD '{method}' in runner_config.py.")
        print(f"        Expected one of:")
        print(f"          gauge : {', '.join(GAUGE_METHODS)}")
        print(f"          IMERG : {', '.join(IMERG_METHODS)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
