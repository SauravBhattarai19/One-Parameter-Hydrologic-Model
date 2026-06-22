"""
runners — flood-runner package
==============================
Run a batch of flood events through the VSA-OPM model.

You only ever touch two things:
  • runner_config.py  → ALL knobs (which rainfall source, where files live, …)
  • ../runner.py      → the single entry point:  python tools/runner.py

Internals (you normally don't import these directly):
  • common.py  → shared helpers (gag parsing, metrics, plotting)
  • gauge.py   → pipeline for gauge rainfall   (thiessen / idw / uniform)
  • imerg.py   → pipeline for IMERG rainfall   (imerg_thiessen / imerg_idw)

runner.py looks at runner_config.PRECIP_METHOD and calls gauge.run() or
imerg.run() for you.
"""

import sys
from pathlib import Path

# Make the repo root (for model modules: config, kinematic_wave_router, …) and
# the tools/ folder importable no matter how this package is entered.
_PKG   = Path(__file__).resolve().parent      # tools/runners
_TOOLS = _PKG.parent                          # tools
_REPO  = _TOOLS.parent                        # repo root
for _p in (str(_REPO), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
