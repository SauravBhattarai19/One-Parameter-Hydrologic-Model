#!/usr/bin/env python
"""
run_combinations.py
===================
Full-factorial SCENARIO sweep for the 100 m OPM model.  Runs every combination of

    channel routing   ×   routing scheme   ×   runoff-mechanism subset

over ALL 4 flood events, into one organised, self-describing folder.  This is the
"try everything" companion to tools/sensitivity.py (which varies one knob at a
time); here we enumerate the full cross-product so you can read off, e.g.,
"river + diffusive + VSA-only" directly from the folder tree.

Axes (edit the lists below to add/trim)
---------------------------------------
    CHANNEL   = [False, True]                       (channel cross-section on/off)
    SCHEMES   = ['kinematic', 'diffusive']          (routing scheme)
    MECH_SUBSETS = the 7 non-empty subsets of {vsa, horton, impervious}

    → 2 × 2 × 7 = 28 configs × 4 floods = 112 router runs.

Everything else (OPM_SD_SOURCE='gee', DIFFUSION_THETA, CFL, Manning's-n source,
SD reducer, …) comes from config.py; the gauge pipeline sets EVENT_START_UTC per
event so the SERVES/GEE soil-moisture deficit is genuinely active.

Output tree (under outputs collection/combinations_100m/)
---------------------------------------------------------
    _shared/                                   cached rasters (seeded once)
    chan_on/diffusive/vsa/                      ← one config = 4 floods
        hydrograph_<event>.csv  (×4)
        comparison_<event>.png  (×4)
        summary_all_floods.csv      (NSE / PBIAS / peaks per flood)
        mass_balance.csv            (runoff ratio + Dunne/Horton/Imperv split)
        partition_<event>.csv
    chan_on/diffusive/vsa+horton/
    chan_off/kinematic/imperv/
    ...
    master_summary.csv             ← EVERY (config × flood) row, one table
    run_log.csv                    ← per-config status + wall time

Speed / resume
--------------
All configs reuse ONE pre-seeded watershed + the cached GEE products (deficit /
Ksat / LCZ / texture) copied from an existing baseline run, so there is no DEM
re-processing and no GEE re-download.  Completed configs are skipped on re-run
(resumable) unless --force is given — safe to launch under tmux on the HPC.

Usage
-----
    python tools/run_combinations.py                 # run everything (resumable)
    python tools/run_combinations.py --list          # list configs, run nothing
    python tools/run_combinations.py --force         # re-run completed configs too
    python tools/run_combinations.py --aggregate     # only rebuild master_summary.csv
    python tools/run_combinations.py chan_on diffusive   # only matching leaf paths
"""

import os
import sys
import time
import shutil
import itertools
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)                      # model uses paths relative to repo root
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from tools.runners import gauge


# ══════════════════════════════════════════════════════════════════════════════
# Axes — EDIT HERE to change what gets run
# ══════════════════════════════════════════════════════════════════════════════
CHANNEL = [False, True]                       # channel cross-section routing on/off
SCHEMES = ['kinematic', 'diffusive']          # routing scheme

# The 7 non-empty subsets of the three runoff mechanisms.
MECH_SUBSETS = [
    ['vsa'],
    ['horton'],
    ['impervious'],
    ['vsa', 'horton'],
    ['vsa', 'impervious'],
    ['horton', 'impervious'],
    ['vsa', 'horton', 'impervious'],
]

# Fixed knobs (held constant across every combo).  Promote either of these to a
# loop axis in all_configs() if you want to sweep them too.
SD_REDUCER      = 'max'        # 'max' | 'mean' | 'divide'
DIFFUSION_THETA = 1.0          # only used when scheme == 'diffusive'


# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
ROOT   = "outputs collection/combinations_100m"
SHARED = REPO_ROOT / ROOT / "_shared"

# An existing run that already holds the cached rasters for all 4 events.
BASELINE_SRC = REPO_ROOT / "outputs collection/100m_diff_inflt_imperv/stations_100m_lcz"

# Rasters every config reuses: 4 routing rasters (lets process_dem be skipped) +
# the static GEE products (Ksat / LCZ / texture).  The date-stamped deficit
# rasters are added separately (one per event).
SEED_FILES = [
    "clipped_dem.tif", "flow_direction.tif", "clipped_flow_accumulation.tif",
    "watershed.tif", "watershed.geojson",
    "ksat_hihydro.tif", "lulc_mannings_lcz.tif", "texture_sandclay.tif",
]


# ══════════════════════════════════════════════════════════════════════════════
# Config enumeration + naming
# ══════════════════════════════════════════════════════════════════════════════
_MECH_TOKEN = {'vsa': 'vsa', 'horton': 'horton', 'impervious': 'imperv'}
_CANON      = ['vsa', 'horton', 'impervious']   # canonical order for names


def _mech_name(mset):
    return '+'.join(_MECH_TOKEN[m] for m in _CANON if m in mset)


def _leaf(chan, scheme, mset):
    return f"chan_{'on' if chan else 'off'}/{scheme}/{_mech_name(mset)}"


def all_configs():
    """Return [(leaf_path, overrides_dict, meta_dict), ...] for the full product."""
    configs = []
    for chan, scheme, mset in itertools.product(CHANNEL, SCHEMES, MECH_SUBSETS):
        leaf = _leaf(chan, scheme, mset)
        overrides = {
            'CHANNEL_ROUTING':   chan,
            'ROUTING_SCHEME':    scheme,
            'DIFFUSION_THETA':   DIFFUSION_THETA,
            'RUNOFF_MECHANISMS': list(mset),
            'OPM_SD_REDUCER':    SD_REDUCER,
            # Keep param resolution + mass_balance.csv metadata consistent with the
            # active mechanism set (RUNOFF_MECHANISMS is authoritative, but this
            # makes OPM_INFILTRATION / IMPERVIOUS_SOURCE honest and avoids resolving
            # layers a disabled mechanism does not need).
            'OPM_INFILTRATION':  'green_ampt' if 'horton' in mset else 'none',
            'IMPERVIOUS_SOURCE': 'lcz' if 'impervious' in mset else 'none',
        }
        meta = {
            'channel':     'on' if chan else 'off',
            'scheme':      scheme,
            'mechanisms':  _mech_name(mset),
        }
        configs.append((leaf, overrides, meta))
    return configs


# ══════════════════════════════════════════════════════════════════════════════
# Raster seeding (no DEM re-processing, no GEE re-download)
# ══════════════════════════════════════════════════════════════════════════════
def _shared_raster_list():
    files = [BASELINE_SRC / f for f in SEED_FILES if (BASELINE_SRC / f).exists()]
    files += sorted(BASELINE_SRC.glob("deficit_serves_*.tif"))   # one per event date
    return files


def seed_shared():
    SHARED.mkdir(parents=True, exist_ok=True)
    src = _shared_raster_list()
    if not src:
        sys.exit(f"[ERROR] No cached rasters found in {BASELINE_SRC}.\n"
                 f"        Run a baseline gauge pipeline there first (it produces the "
                 f"watershed + GEE deficit/Ksat/LCZ/texture rasters).")
    for f in src:
        dst = SHARED / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
    print(f"  _shared seeded with {len(src)} rasters → {SHARED.relative_to(REPO_ROOT)}")


def seed_leaf(leaf_path: Path):
    leaf_path.mkdir(parents=True, exist_ok=True)
    for f in SHARED.iterdir():
        dst = leaf_path / f.name
        if not dst.exists():
            shutil.copy2(f, dst)


def is_done(leaf_path: Path) -> bool:
    """A config is 'done' once the gauge pipeline wrote its grand summary."""
    return (leaf_path / "summary_all_floods.csv").exists()


# ══════════════════════════════════════════════════════════════════════════════
# Master aggregation: every (config × flood) row in one table
# ══════════════════════════════════════════════════════════════════════════════
def aggregate():
    """Walk every leaf, join its per-flood metrics with the mass-balance partition,
    and write ROOT/master_summary.csv."""
    rows = []
    for leaf, _ov, meta in all_configs():
        leaf_path = REPO_ROOT / ROOT / leaf
        summ = leaf_path / "summary_all_floods.csv"
        if not summ.exists():
            continue
        df = pd.read_csv(summ)

        # mass_balance.csv appends one row per event; keep the LAST per run_tag.
        mb_path = leaf_path / "mass_balance.csv"
        mb = None
        if mb_path.exists():
            mb = pd.read_csv(mb_path).drop_duplicates('run_tag', keep='last') \
                   .set_index('run_tag')

        for _, r in df.iterrows():
            ev  = str(r['event'])
            row = {
                'channel':     meta['channel'],
                'scheme':      meta['scheme'],
                'mechanisms':  meta['mechanisms'],
                'event':       ev,
                'nse':         r.get('nse'),
                'pbias_pct':   r.get('pbias_pct'),
                'obs_peak_Q':  r.get('obs_peak_Q'),
                'mod_peak_Q':  r.get('mod_peak_Q'),
                'obs_peak_hr': r.get('obs_peak_hr'),
                'mod_peak_hr': r.get('mod_peak_hr'),
            }
            if mb is not None and ev in mb.index:
                m = mb.loc[ev]
                for col in ('runoff_ratio', 'dunne_frac', 'horton_frac',
                            'imperv_frac', 'rel_error'):
                    if col in mb.columns:
                        row[col] = m[col]
            rows.append(row)

    if not rows:
        print("  [aggregate] no completed configs yet — nothing to write.")
        return
    out = REPO_ROOT / ROOT / "master_summary.csv"
    cols = ['channel', 'scheme', 'mechanisms', 'event', 'nse', 'pbias_pct',
            'obs_peak_Q', 'mod_peak_Q', 'obs_peak_hr', 'mod_peak_hr',
            'runoff_ratio', 'dunne_frac', 'horton_frac', 'imperv_frac', 'rel_error']
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out, index=False)
    print(f"\n  master_summary.csv → {out.relative_to(REPO_ROOT)}  ({len(df)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════
def run_study(filters=None, force=False):
    seed_shared()
    configs = all_configs()
    selected = [(leaf, ov, meta) for leaf, ov, meta in configs
                if not filters or all(f in leaf for f in filters)]

    print(f"\n{'='*70}\n  COMBINATION SWEEP — {len(selected)} configs × 4 floods"
          f"\n  axes: channel{CHANNEL}  scheme{SCHEMES}  mechanisms×{len(MECH_SUBSETS)}"
          f"\n{'='*70}")

    log = []
    for i, (leaf, overrides, meta) in enumerate(selected, 1):
        leaf_path = REPO_ROOT / ROOT / leaf
        print(f"\n\n########## [{i}/{len(selected)}]  {leaf} ##########")
        if is_done(leaf_path) and not force:
            print("  already complete — skipping (use --force to re-run).")
            log.append((leaf, "skip", 0.0))
            continue
        key = {k: overrides[k] for k in
               ('CHANNEL_ROUTING', 'ROUTING_SCHEME', 'RUNOFF_MECHANISMS')}
        print(f"  overrides: {key}")
        seed_leaf(leaf_path)
        t0 = time.time()
        try:
            gauge.run(output_dir=f"{ROOT}/{leaf}/",
                      overrides=overrides,
                      skip_process_dem=True)
            status = "ok"
        except Exception as exc:                       # keep the sweep going
            status = f"FAIL: {exc}"
            traceback.print_exc()
        log.append((leaf, status, round(time.time() - t0, 1)))

    # ── per-config status log ────────────────────────────────────────────────
    (REPO_ROOT / ROOT).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log, columns=['config', 'status', 'seconds']).to_csv(
        REPO_ROOT / ROOT / "run_log.csv", index=False)

    print(f"\n\n{'='*70}\n  SWEEP COMPLETE\n{'='*70}")
    for leaf, status, secs in log:
        print(f"  {status:>6}  {secs:8.1f}s  {leaf}")
    n_fail = sum(1 for _, s, _ in log if s.startswith("FAIL"))
    if n_fail:
        print(f"\n  [WARN] {n_fail} config(s) failed — see tracebacks above.")

    aggregate()
    print(f"\n  Done.  Re-run any time to resume; --aggregate rebuilds the summary.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list" in args:
        for leaf, ov, _ in all_configs():
            tag = {k: ov[k] for k in ('CHANNEL_ROUTING', 'ROUTING_SCHEME',
                                      'RUNOFF_MECHANISMS')}
            print(f"  {leaf:42s}  {tag}")
        print(f"\n  {len(all_configs())} configs × 4 floods.")
        sys.exit(0)
    if "--aggregate" in args:
        aggregate()
        sys.exit(0)
    force   = "--force" in args
    filters = [a for a in args if not a.startswith("--")] or None
    run_study(filters=filters, force=force)
