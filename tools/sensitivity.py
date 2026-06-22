#!/usr/bin/env python
"""
sensitivity.py
==============
One-Factor-At-A-Time (OFAT) sensitivity & runoff-partition sweep for the 100 m
model over the same 4 flood events used by the flood runner.

Each *config* is the locked baseline with exactly one knob changed (or, for the
mechanism block, one/two toggles).  Every config runs all 4 events into its own
leaf folder under ``outputs collection/sensitivity_100m/`` and writes the usual
per-event outputs PLUS the Dunne/Horton/Impervious partition (in mass_balance.csv
and partition_<tag>.csv).

Speed: all configs reuse ONE pre-seeded watershed (DEM rasters) and the cached
GEE products (deficit / Ksat / LCZ / texture) copied from the existing baseline
run, so there is no DEM re-processing and no raster re-download.  Runs at
``TIME_STEP_SECONDS=5`` (a sensitivity study compares deltas, not absolute
calibration).

Usage
-----
    python tools/sensitivity.py            # run the whole study
    python tools/sensitivity.py A B        # only blocks whose folder starts A_/B_
    python tools/sensitivity.py --list     # list configs, run nothing
"""

import os
import sys
import time
import shutil
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)                      # model uses paths relative to repo root
sys.path.insert(0, str(REPO_ROOT))

from tools.runners import gauge


# ── Where shared rasters come from (the existing baseline run has them all) ──────
BASELINE_SRC = REPO_ROOT / "outputs collection/100m_diff_inflt_imperv/stations_100m_lcz"

ROOT   = "outputs collection/sensitivity_100m"      # relative (model resolves vs CWD)
SHARED = REPO_ROOT / ROOT / "_shared"

# Rasters every config reuses.  The 4 routing rasters let process_dem be skipped;
# the GEE products (deficit / Ksat / LCZ / texture) are cache-hit so no download.
# Big DEM intermediates (filled/inflated/reprojected/flow_accumulation) are NOT
# needed by the router and are skipped to save space.
SEED_FILES = [
    "clipped_dem.tif", "flow_direction.tif", "clipped_flow_accumulation.tif",
    "watershed.tif", "watershed.geojson",
    "ksat_hihydro.tif", "lulc_mannings_lcz.tif", "texture_sandclay.tif",
]


# ── Locked baseline.  EVERY swept knob appears here so each config fully
#    specifies it (prevents leakage from one config's override into the next). ──
BASELINE = dict(
    RUNOFF_SOURCE      = 'vsa_opm',
    OPM_INFILTRATION   = 'green_ampt',
    IMPERVIOUS_SOURCE  = 'lcz',
    ROUTING_SCHEME     = 'diffusive',
    DIFFUSION_THETA    = 1.0,
    OPM_SD_SOURCE      = 'gee',
    OPM_SD_MAX_INITIAL = 0.1,
    OPM_GA_KSAT_SCALE  = 1.0,
    # Routing is CFL-limited to ~0.9 s on this 100 m grid (Courant=5.68 at dt=5s
    # blows the hydrograph up 2-12x even though mass balance still closes).
    # dt=1s is the only stable choice — matches the validated baseline run.
    TIME_STEP_SECONDS  = 1,
)


def cfg(**kw):
    """Baseline merged with the given overrides (full, self-contained config)."""
    d = dict(BASELINE)
    d.update(kw)
    return d


# ── The experiment matrix.  (leaf folder, overrides) ────────────────────────────
# theta=1.0 (block C) and ksat_scale=1.0 (block D) are identical to A0_full and
# are intentionally omitted — the report reuses A0_full as those reference points.
EXPERIMENTS = [
    # A — mechanism partition + toggles
    ("A_mechanism/A0_full",       cfg()),
    ("A_mechanism/A1_no_horton",  cfg(OPM_INFILTRATION='none')),
    ("A_mechanism/A2_no_imperv",  cfg(IMPERVIOUS_SOURCE='none')),
    ("A_mechanism/A3_dunne_only", cfg(OPM_INFILTRATION='none', IMPERVIOUS_SOURCE='none')),

    # B — SD / soil-moisture sensitivity (manual root-depth sweep)
    ("B_sd/sd_0.05", cfg(OPM_SD_SOURCE='manual', OPM_SD_MAX_INITIAL=0.05)),
    ("B_sd/sd_0.10", cfg(OPM_SD_SOURCE='manual', OPM_SD_MAX_INITIAL=0.10)),
    ("B_sd/sd_0.20", cfg(OPM_SD_SOURCE='manual', OPM_SD_MAX_INITIAL=0.20)),
    ("B_sd/sd_0.40", cfg(OPM_SD_SOURCE='manual', OPM_SD_MAX_INITIAL=0.40)),
    ("B_sd/sd_0.80", cfg(OPM_SD_SOURCE='manual', OPM_SD_MAX_INITIAL=0.80)),

    # C — routing scheme + diffusion weight
    ("C_routing/kinematic",       cfg(ROUTING_SCHEME='kinematic')),
    ("C_routing/diff_theta_0.25", cfg(DIFFUSION_THETA=0.25)),
    ("C_routing/diff_theta_0.50", cfg(DIFFUSION_THETA=0.50)),
    ("C_routing/diff_theta_0.75", cfg(DIFFUSION_THETA=0.75)),

    # D — Ksat scale (Horton magnitude)
    ("D_ksat/ksat_0.5", cfg(OPM_GA_KSAT_SCALE=0.5)),
    ("D_ksat/ksat_2.0", cfg(OPM_GA_KSAT_SCALE=2.0)),
    ("D_ksat/ksat_4.0", cfg(OPM_GA_KSAT_SCALE=4.0)),
]


def _shared_raster_list():
    """SEED_FILES that exist in BASELINE_SRC + every deficit_serves_*.tif."""
    files = [BASELINE_SRC / f for f in SEED_FILES if (BASELINE_SRC / f).exists()]
    files += sorted(BASELINE_SRC.glob("deficit_serves_*.tif"))
    return files


def seed_shared():
    """Copy the shared rasters from the baseline run into _shared/ (once)."""
    SHARED.mkdir(parents=True, exist_ok=True)
    src = _shared_raster_list()
    if not src:
        sys.exit(f"[ERROR] No shared rasters found in {BASELINE_SRC}. "
                 f"Run the baseline gauge pipeline there first.")
    for f in src:
        dst = SHARED / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
    print(f"  _shared seeded with {len(src)} rasters → {SHARED.relative_to(REPO_ROOT)}")


def seed_leaf(leaf: Path):
    """Copy every _shared raster into a config leaf so the run finds the cache."""
    leaf.mkdir(parents=True, exist_ok=True)
    for f in SHARED.iterdir():
        dst = leaf / f.name
        if not dst.exists():
            shutil.copy2(f, dst)


def run_study(block_filter=None):
    seed_shared()
    rows = []
    selected = [(folder, ov) for folder, ov in EXPERIMENTS
                if not block_filter or any(folder.startswith(b) for b in block_filter)]
    print(f"\n{'='*68}\n  SENSITIVITY STUDY — {len(selected)} configs × 4 floods\n{'='*68}")
    for i, (folder, overrides) in enumerate(selected, 1):
        leaf = REPO_ROOT / ROOT / folder
        print(f"\n\n########## [{i}/{len(selected)}]  {folder} ##########")
        print(f"  overrides: { {k: overrides[k] for k in overrides if BASELINE.get(k) != overrides[k]} }")
        seed_leaf(leaf)
        t0 = time.time()
        try:
            gauge.run(output_dir=f"{ROOT}/{folder}/",
                      overrides=overrides,
                      skip_process_dem=True)
            status = "ok"
        except Exception as exc:                       # keep the sweep going
            status = f"FAIL: {exc}"
            traceback.print_exc()
        rows.append((folder, status, round(time.time() - t0, 1)))

    print(f"\n\n{'='*68}\n  STUDY COMPLETE\n{'='*68}")
    for folder, status, secs in rows:
        print(f"  {status:>6}  {secs:8.1f}s  {folder}")
    n_fail = sum(1 for _, s, _ in rows if s != "ok")
    if n_fail:
        print(f"\n  [WARN] {n_fail} config(s) failed — see tracebacks above.")
    print(f"\n  Next: python tools/sensitivity_report.py")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    if "--list" in args:
        for folder, ov in EXPERIMENTS:
            diff = {k: ov[k] for k in ov if BASELINE.get(k) != ov[k]}
            print(f"  {folder:32s}  {diff}")
        sys.exit(0)
    # Optional positional block filters, e.g. "A" "B" → folders starting "A"/"B".
    blocks = [a if a.endswith("_") or "/" in a else f"{a}_" for a in args] or None
    run_study(block_filter=blocks)
