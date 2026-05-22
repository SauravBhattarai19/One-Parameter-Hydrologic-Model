"""
run_all_floods.py
=================
End-to-end pipeline for every FLOOD event:

  1. Convert .gag  → OPM precipitation CSVs  (gauges.csv + timeseries.csv)
  2. Run kinematic-wave router with VSA-OPM runoff engine
  3. Load observed hydrograph from discharge_YYYYMM_YYYYMM.csv
  4. Compute NSE / PBIAS / peak stats
  5. Save comparison plot  →  output/comparison_<EVENT>.png

Usage:
    python tools/run_all_floods.py

All parameters come from config.py; only PRECIP_GAUGE_FILE,
PRECIP_TIMESERIES_FILE, and TOTAL_SIMULATION_TIME_HOURS are overridden
per-event at runtime.
"""

import re
import csv
import sys
import importlib
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
GSSHA_DIR   = REPO_ROOT / "test_data" / "gssha_format"
OPM_DIR     = REPO_ROOT / "test_data" / "opm_format"
OUTPUT_DIR  = REPO_ROOT / "output"

sys.path.insert(0, str(REPO_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — GAG → OPM conversion  (inline; no subprocess needed)
# ══════════════════════════════════════════════════════════════════════════════

def parse_gag(filepath: Path) -> dict:
    gauges, records = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("COORD"):
                m = re.match(r'COORD\s+([\d.]+)\s+([\d.]+)\s+"([^"]+)"', line)
                if not m:
                    raise ValueError(f"Bad COORD: {line!r}")
                e, n, name = float(m.group(1)), float(m.group(2)), m.group(3)
                gauges.append((name.split()[0], name, e, n))
            elif line.startswith("GAGES"):
                parts = line.split()
                yr, mo, dy, hr, mn = (int(p) for p in parts[1:6])
                records.append((datetime(yr, mo, dy, hr, mn),
                                [float(v) for v in parts[6:]]))
    return {"gauges": gauges, "records": records}


def write_opm_csvs(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gauges, records = data["gauges"], data["records"]

    # derive interval from first two timestamps
    interval_s = (records[1][0] - records[0][0]).total_seconds() if len(records) > 1 else 1800.0

    with open(out_dir / "gauges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gauge_id", "name", "easting_m", "northing_m"])
        for gid, name, e, n in gauges:
            w.writerow([gid, name, f"{e:.6f}", f"{n:.6f}"])

    gauge_ids = [g[0] for g in gauges]
    with open(out_dir / "timeseries.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s"] + gauge_ids)
        for i, (_, vals) in enumerate(records):
            w.writerow([int(round(i * interval_s))] + [f"{v:.4f}" for v in vals])


def gag_event_start(data: dict) -> datetime:
    return data["records"][0][0]


def gag_sim_hours(data: dict) -> float:
    recs = data["records"]
    interval_s = (recs[1][0] - recs[0][0]).total_seconds() if len(recs) > 1 else 1800.0
    return len(recs) * interval_s / 3600.0


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Run router for one event
# ══════════════════════════════════════════════════════════════════════════════

def run_router(gauge_csv: Path, ts_csv: Path, sim_hours: float) -> list:
    """
    Import config + router modules, patch the per-event paths, run, return
    the hydrograph list of (time_s, Q_m3s) tuples.
    """
    import config
    import kinematic_wave_router as kwr

    # Override per-event settings (relative paths from REPO_ROOT)
    config.PRECIP_GAUGE_FILE      = str(gauge_csv.relative_to(REPO_ROOT))
    config.PRECIP_TIMESERIES_FILE = str(ts_csv.relative_to(REPO_ROOT))
    config.TOTAL_SIMULATION_TIME_HOURS = sim_hours

    # Re-init grid each run (precipitation engine must be rebuilt for new gauges)
    grid_data  = kwr.initialise_grid(config)
    hydrograph = kwr.run_time_loop(grid_data, config)
    return hydrograph


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Load observed hydrograph
# ══════════════════════════════════════════════════════════════════════════════

def load_otl(otl_path: Path) -> pd.DataFrame:
    """Load whitespace-delimited .otl file (time_min, Q_m3s columns, no header)."""
    df = pd.read_csv(otl_path, sep=r'\s+', header=None,
                     names=["time_min", "Q_m3s"])
    return df


def load_discharge_csv(csv_path: Path, event_start: datetime) -> pd.DataFrame:
    """
    Load a GSSHA-format discharge CSV with columns:
        dateTime, stage_m, discharge_m3s

    Converts absolute datetimes to minutes elapsed since event_start so the
    returned DataFrame has the same interface as load_otl():
        time_min  [float]  minutes from event start
        Q_m3s     [float]  discharge in m³/s
    """
    df = pd.read_csv(csv_path, parse_dates=["dateTime"])
    df["time_min"] = (df["dateTime"] - event_start).dt.total_seconds() / 60.0
    df = df.rename(columns={"discharge_m3s": "Q_m3s"})[["time_min", "Q_m3s"]]
    # Drop rows before the event start (negative time)
    df = df[df["time_min"] >= 0].reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Performance metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(otl: pd.DataFrame, hydrograph: list) -> dict:
    obs_s   = otl["time_min"].values * 60.0
    obs_Q   = otl["Q_m3s"].values
    mod_s   = np.array([h[0] for h in hydrograph], dtype=float)
    mod_Q   = np.array([h[1] for h in hydrograph], dtype=float)

    mod_interp = np.interp(obs_s, mod_s, mod_Q)
    nse   = 1 - np.sum((obs_Q - mod_interp)**2) / np.sum((obs_Q - obs_Q.mean())**2)
    pbias = 100 * (mod_interp - obs_Q).sum() / obs_Q.sum()

    obs_peak_idx = obs_Q.argmax()
    mod_peak_idx = mod_Q.argmax()

    return {
        "nse"        : nse,
        "pbias"      : pbias,
        "obs_peak_Q" : obs_Q[obs_peak_idx],
        "obs_peak_s" : obs_s[obs_peak_idx],
        "mod_peak_Q" : mod_Q[mod_peak_idx],
        "mod_peak_s" : mod_s[mod_peak_idx],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Comparison plot
# ══════════════════════════════════════════════════════════════════════════════

def make_plot(event_name: str, event_start: datetime,
              otl: pd.DataFrame, hydrograph: list,
              ts_csv: Path, metrics: dict, out_png: Path) -> None:

    mod_s = np.array([h[0] for h in hydrograph], dtype=float)
    mod_Q = np.array([h[1] for h in hydrograph], dtype=float)

    obs_dt = [event_start + timedelta(minutes=float(m)) for m in otl["time_min"]]
    mod_dt = [event_start + timedelta(seconds=float(s)) for s in mod_s]

    obs_peak_dt = event_start + timedelta(seconds=metrics["obs_peak_s"])
    mod_peak_dt = event_start + timedelta(seconds=metrics["mod_peak_s"])

    fig, (ax, ax_rain) = plt.subplots(
        2, 1, figsize=(13, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # ── Hydrograph ─────────────────────────────────────────────────────────
    ax.plot(obs_dt, otl["Q_m3s"], color="black", lw=1.8,
            label="Observed (GSSHA)", zorder=3)
    ax.plot(mod_dt, mod_Q, color="#1f6aa5", lw=1.6, ls="--",
            label="VSA-OPM (simulated)", zorder=2)
    ax.fill_between(mod_dt, mod_Q, alpha=0.12, color="#1f6aa5")

    # Peak arrows — place text away from peak so arrows don't overlap
    q_max = max(metrics["obs_peak_Q"], metrics["mod_peak_Q"])
    ax.annotate(
        f"Obs peak\n{metrics['obs_peak_Q']:.0f} m³/s\n{obs_peak_dt.strftime('%b %d %H:%M')}",
        xy=(obs_peak_dt, metrics["obs_peak_Q"]),
        xytext=(obs_peak_dt - timedelta(hours=18), metrics["obs_peak_Q"] * 0.82),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
        fontsize=8.5, color="black"
    )
    ax.annotate(
        f"Mod peak\n{metrics['mod_peak_Q']:.0f} m³/s\n{mod_peak_dt.strftime('%b %d %H:%M')}",
        xy=(mod_peak_dt, metrics["mod_peak_Q"]),
        xytext=(mod_peak_dt + timedelta(hours=6), metrics["mod_peak_Q"] * 0.78),
        arrowprops=dict(arrowstyle="-|>", color="#1f6aa5", lw=1.2),
        fontsize=8.5, color="#1f6aa5"
    )

    stats_txt = (f"NSE   = {metrics['nse']:.3f}\n"
                 f"PBIAS = {metrics['pbias']:.1f}%\n"
                 f"Obs $Q_p$ = {metrics['obs_peak_Q']:.0f} m³/s\n"
                 f"Mod $Q_p$ = {metrics['mod_peak_Q']:.0f} m³/s")
    ax.text(0.985, 0.97, stats_txt, transform=ax.transAxes,
            va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.85))

    ax.set_ylabel("Discharge [m³/s]", fontsize=11)
    ax.set_title(f"{event_name}  |  VSA-OPM vs Observed  |  Bagmati Basin",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(bottom=0)
    ax.grid(True, ls="--", alpha=0.35)

    # ── Rainfall bar panel ──────────────────────────────────────────────────
    ts = pd.read_csv(ts_csv)
    gauge_cols = [c for c in ts.columns if c != "time_s"]
    ts["mean_mmhr"] = ts[gauge_cols].mean(axis=1) * 2.0   # mm/30min → mm/hr
    ts["datetime"]  = [event_start + timedelta(seconds=float(s))
                       for s in ts["time_s"]]

    ax_rain.bar(ts["datetime"], ts["mean_mmhr"],
                width=timedelta(minutes=28), color="#4da6ff",
                alpha=0.7, align="edge", label="Mean areal rainfall [mm/hr]")
    ax_rain.invert_yaxis()
    ax_rain.set_ylabel("Rain [mm/hr]", fontsize=9)
    ax_rain.legend(fontsize=8, loc="lower right")
    ax_rain.grid(True, ls="--", alpha=0.3)

    ax_rain.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_rain.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    ax_rain.set_xlabel(f"Date ({event_start.year})", fontsize=11)

    t_min = min(obs_dt[0], mod_dt[0])
    t_max = max(obs_dt[-1], mod_dt[-1])
    ax.set_xlim(t_min, t_max)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {out_png.relative_to(REPO_ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    gag_files = sorted(GSSHA_DIR.glob("*.gag"))
    if not gag_files:
        print(f"No .gag files found in {GSSHA_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    all_metrics = []

    for gag_path in gag_files:
        # Extract the date tag (e.g. "202407_202407") from "rainfall_input_202407_202407.gag"
        date_tag   = re.search(r'(\d{6}_\d{6})', gag_path.stem)
        date_tag   = date_tag.group(1) if date_tag else gag_path.stem
        event_name = date_tag

        # Matching discharge CSV: discharge_YYYYMM_YYYYMM.csv
        disc_path  = GSSHA_DIR / f"discharge_{date_tag}.csv"

        print("\n" + "=" * 65)
        print(f"  EVENT: {event_name}")
        print("=" * 65)

        # ── 1. Convert GAG → OPM ──────────────────────────────────────────
        print(f"\n[1/4] Converting {gag_path.name} → OPM format...")
        gag_data    = parse_gag(gag_path)
        opm_out_dir = OPM_DIR / event_name
        write_opm_csvs(gag_data, opm_out_dir)
        event_start = gag_event_start(gag_data)
        sim_hours   = gag_sim_hours(gag_data)
        print(f"      Start: {event_start}  |  Duration: {sim_hours:.0f} h")

        # ── 2. Run router ─────────────────────────────────────────────────
        print(f"\n[2/4] Running VSA-OPM router ({sim_hours:.0f} h)...")
        hydrograph = run_router(
            opm_out_dir / "gauges.csv",
            opm_out_dir / "timeseries.csv",
            sim_hours
        )

        # Save per-event hydrograph CSV
        hyd_csv = OUTPUT_DIR / f"hydrograph_{event_name}.csv"
        times_s = [h[0] for h in hydrograph]
        Q_vals  = [h[1] for h in hydrograph]
        pd.DataFrame({"time_s": times_s,
                      "time_hr": [t / 3600 for t in times_s],
                      "Q_m3s": Q_vals}).to_csv(hyd_csv, index=False)
        print(f"      Hydrograph saved → {hyd_csv.relative_to(REPO_ROOT)}")

        # ── 3. Load observed ──────────────────────────────────────────────
        if not disc_path.exists():
            print(f"\n[3/4] WARNING: {disc_path.name} not found — skipping plot.")
            continue
        print(f"\n[3/4] Loading observed hydrograph from {disc_path.name}...")
        otl = load_discharge_csv(disc_path, event_start)

        # ── 4. Metrics ────────────────────────────────────────────────────
        print("\n[4/4] Computing performance metrics and plotting...")
        metrics = compute_metrics(otl, hydrograph)
        print(f"      NSE   = {metrics['nse']:.3f}")
        print(f"      PBIAS = {metrics['pbias']:.1f}%")
        print(f"      Obs Qp = {metrics['obs_peak_Q']:.1f} m³/s  "
              f"at t={metrics['obs_peak_s']/3600:.2f} h")
        print(f"      Mod Qp = {metrics['mod_peak_Q']:.1f} m³/s  "
              f"at t={metrics['mod_peak_s']/3600:.2f} h")

        # ── 5. Plot ───────────────────────────────────────────────────────
        out_png = OUTPUT_DIR / f"comparison_{event_name}.png"
        make_plot(event_name, event_start, otl, hydrograph,
                  opm_out_dir / "timeseries.csv", metrics, out_png)

        all_metrics.append({
            "event"       : event_name,
            "start"       : event_start.strftime("%Y-%m-%d"),
            "nse"         : round(metrics["nse"], 3),
            "pbias_pct"   : round(metrics["pbias"], 1),
            "obs_peak_Q"  : round(metrics["obs_peak_Q"], 1),
            "obs_peak_hr" : round(metrics["obs_peak_s"] / 3600, 2),
            "mod_peak_Q"  : round(metrics["mod_peak_Q"], 1),
            "mod_peak_hr" : round(metrics["mod_peak_s"] / 3600, 2),
        })

    # ── Summary table ──────────────────────────────────────────────────────────
    if all_metrics:
        print("\n" + "=" * 65)
        print("  SUMMARY")
        print("=" * 65)
        summary = pd.DataFrame(all_metrics)
        print(summary.to_string(index=False))
        summary_csv = OUTPUT_DIR / "summary_all_floods.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"\n  Summary saved → {summary_csv.relative_to(REPO_ROOT)}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
