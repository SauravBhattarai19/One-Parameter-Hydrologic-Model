"""
run_all_floods.py
=================
End-to-end GPU routing pipeline for every FLOOD event.

Stages timed
------------
  [0] process_dem     – reproject / fill / flow-dir / accumulation / watershed
                        (CPU-only via pysheds; runs once)
  [1] GAG → OPM       – convert .gag rainfall files to OPM CSVs   (CPU-only)
  [2a] GPU init       – initialise_grid (BACKEND='gpu')
  [2b] GPU loop       – run_time_loop   (BACKEND='gpu')

Output
------
  output/hydrograph_<EVENT>.csv
  output/comparison_<EVENT>.png
  output/summary_all_floods.csv
  Console: timing table + peak-Q stats

Usage
-----
  cd /path/to/OPM
  conda run -n opm python tools/run_all_floods.py
"""

import re
import csv
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
GSSHA_DIR   = REPO_ROOT / "test_data" / "gssha_format"
OPM_DIR     = REPO_ROOT / "test_data" / "opm_format"
OUTPUT_DIR  = REPO_ROOT / "output"

sys.path.insert(0, str(REPO_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def hms(seconds: float) -> str:
    """Format seconds as H:MM:SS or MM:SS.s for readability."""
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}:{m:02d}:{s:04.1f}"
    elif seconds >= 60:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}:{s:04.1f}"
    else:
        return f"{seconds:.2f}s"


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0 — process_dem  (CPU only, run once)
# ══════════════════════════════════════════════════════════════════════════════

def run_process_dem() -> float:
    import process_dem
    t0 = time.perf_counter()
    process_dem.main()
    return time.perf_counter() - t0


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — GAG → OPM conversion  (CPU, per-event)
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
# Stage 2 — Router (GPU)
# ══════════════════════════════════════════════════════════════════════════════

def run_router(gauge_csv: Path, ts_csv: Path, sim_hours: float,
               event_start_date: str = None) -> tuple:
    """
    Run the kinematic-wave router for one event on GPU.

    Returns
    -------
    hydrograph  : list of (time_s, Q_m3s) tuples
    t_init      : wall-clock seconds for initialise_grid
    t_loop      : wall-clock seconds for run_time_loop
    """
    import config
    import kinematic_wave_router as kwr

    config.PRECIP_GAUGE_FILE           = str(gauge_csv.relative_to(REPO_ROOT))
    config.PRECIP_TIMESERIES_FILE      = str(ts_csv.relative_to(REPO_ROOT))
    config.TOTAL_SIMULATION_TIME_HOURS = sim_hours
    config.BACKEND                     = 'gpu'
    if hasattr(config, 'SERVES_TARGET_DATE'):
        config.SERVES_TARGET_DATE      = event_start_date

    t0 = time.perf_counter()
    grid_data = kwr.initialise_grid(config)
    t_init    = time.perf_counter() - t0

    t0 = time.perf_counter()
    hydrograph = kwr.run_time_loop(grid_data, config)
    t_loop     = time.perf_counter() - t0

    return hydrograph, t_init, t_loop


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Load observed hydrograph
# ══════════════════════════════════════════════════════════════════════════════

def load_discharge_csv(csv_path: Path, event_start: datetime) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["dateTime"])
    df["time_min"] = (df["dateTime"] - event_start).dt.total_seconds() / 60.0
    df = df.rename(columns={"discharge_m3s": "Q_m3s"})[["time_min", "Q_m3s"]]
    return df[df["time_min"] >= 0].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Performance metrics
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
        "nse":         nse,
        "pbias":       pbias,
        "obs_peak_Q":  obs_Q[obs_peak_idx],
        "obs_peak_s":  obs_s[obs_peak_idx],
        "mod_peak_Q":  mod_Q[mod_peak_idx],
        "mod_peak_s":  mod_s[mod_peak_idx],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Comparison plot
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

    ax.plot(obs_dt, otl["Q_m3s"], color="black", lw=1.8,
            label="Observed (GSSHA)", zorder=4)
    ax.plot(mod_dt, mod_Q, color="#1f6aa5", lw=1.8, ls="--",
            label="VSA-OPM (GPU)", zorder=3)
    ax.fill_between(mod_dt, mod_Q, alpha=0.10, color="#1f6aa5")

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

    ts = pd.read_csv(ts_csv)
    gauge_cols = [c for c in ts.columns if c != "time_s"]
    ts["mean_mmhr"] = ts[gauge_cols].mean(axis=1) * 2.0
    ts["datetime"]  = [event_start + timedelta(seconds=float(s)) for s in ts["time_s"]]
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
# Timing table printers
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(t_dem: float, rows: list) -> None:
    W = 70
    print("\n" + "═" * W)
    print("  BENCHMARK SUMMARY  (GPU)")
    print("═" * W)
    print(f"  {'Event':<18}  {'Sim':>5}  {'Init':>9}  {'Loop':>9}  {'Total':>9}")
    print("  " + "─" * (W - 2))

    total_init = total_loop = 0.0
    for r in rows:
        total_init += r["t_init"]
        total_loop += r["t_loop"]
        total = r["t_init"] + r["t_loop"]
        print(f"  {r['event']:<18}  {r['sim_h']:>4.0f}h  "
              f"  {hms(r['t_init']):>9}  {hms(r['t_loop']):>9}  {hms(total):>9}")

    total = total_init + total_loop
    print("  " + "─" * (W - 2))
    print(f"  {'ALL EVENTS':<18}  {'':>5}  "
          f"  {hms(total_init):>9}  {hms(total_loop):>9}  {hms(total):>9}")
    print()
    print(f"  process_dem (one-time)           :  {hms(t_dem)}")
    print(f"  Total wall time (dem + routing)  :  {hms(t_dem + total)}")
    print("═" * W)


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import gpu_utils

    gag_files = sorted(GSSHA_DIR.glob("*.gag"))
    if not gag_files:
        print(f"No .gag files found in {GSSHA_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    if not gpu_utils.cupy_available():
        print("[ERROR] CuPy not available — this script requires a GPU.")
        sys.exit(1)

    import cupy as cp
    mem = cp.cuda.Device(0).mem_info
    print(f"GPU detected: {mem[1]/1e9:.1f} GB VRAM  ({mem[0]/1e9:.1f} GB free)")

    # ── Stage 0: process_dem ─────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  STAGE 0 — process_dem  (CPU only, one-time)")
    print("=" * 68)
    t_dem = run_process_dem()
    print(f"\n  process_dem finished in  {hms(t_dem)}")

    all_metrics  = []
    summary_rows = []

    for gag_path in gag_files:
        date_tag   = re.search(r'(\d{6}_\d{6})', gag_path.stem)
        date_tag   = date_tag.group(1) if date_tag else gag_path.stem
        event_name = date_tag
        disc_path  = GSSHA_DIR / f"discharge_{date_tag}.csv"

        print("\n" + "=" * 68)
        print(f"  EVENT: {event_name}")
        print("=" * 68)

        # ── Stage 1: GAG → OPM ───────────────────────────────────────────────
        print(f"\n[1] GAG → OPM  ({gag_path.name})")
        t_gag_0   = time.perf_counter()
        gag_data  = parse_gag(gag_path)
        opm_dir   = OPM_DIR / event_name
        write_opm_csvs(gag_data, opm_dir)
        t_gag     = time.perf_counter() - t_gag_0
        event_start = gag_event_start(gag_data)
        sim_hours   = gag_sim_hours(gag_data)
        print(f"    Start: {event_start}  |  Duration: {sim_hours:.0f} h  "
              f"|  Converted in {hms(t_gag)}")

        gauge_csv = opm_dir / "gauges.csv"
        ts_csv    = opm_dir / "timeseries.csv"

        # ── Stage 2: GPU routing ──────────────────────────────────────────────
        print(f"\n[2] GPU routing  (BACKEND='gpu')")
        hydrograph, t_init, t_loop = run_router(
            gauge_csv, ts_csv, sim_hours,
            event_start_date=event_start.strftime('%Y-%m-%d'),
        )
        print(f"    init={hms(t_init)}  loop={hms(t_loop)}  "
              f"total={hms(t_init + t_loop)}")

        # Save hydrograph
        hyd_csv = OUTPUT_DIR / f"hydrograph_{event_name}.csv"
        pd.DataFrame({
            "time_s":  [h[0] for h in hydrograph],
            "time_hr": [h[0] / 3600 for h in hydrograph],
            "Q_m3s":   [h[1] for h in hydrograph],
        }).to_csv(hyd_csv, index=False)
        print(f"    Hydrograph saved → {hyd_csv.relative_to(REPO_ROOT)}")

        summary_rows.append({
            "event":  event_name,
            "sim_h":  sim_hours,
            "t_init": t_init,
            "t_loop": t_loop,
        })

        # ── Stage 3: Observed data + metrics ─────────────────────────────────
        if not disc_path.exists():
            print(f"\n  [WARN] {disc_path.name} not found — skipping plot.")
            continue

        print(f"\n[3] Metrics & plot")
        otl     = load_discharge_csv(disc_path, event_start)
        metrics = compute_metrics(otl, hydrograph)
        print(f"    NSE={metrics['nse']:.3f}  PBIAS={metrics['pbias']:.1f}%  "
              f"Obs Qp={metrics['obs_peak_Q']:.1f} m³/s  "
              f"Mod Qp={metrics['mod_peak_Q']:.1f} m³/s")

        out_png = OUTPUT_DIR / f"comparison_{event_name}.png"
        make_plot(event_name, event_start, otl, hydrograph, ts_csv, metrics, out_png)

        all_metrics.append({
            "event":       event_name,
            "start":       event_start.strftime("%Y-%m-%d"),
            "nse":         round(metrics["nse"], 3),
            "pbias_pct":   round(metrics["pbias"], 1),
            "obs_peak_Q":  round(metrics["obs_peak_Q"], 1),
            "obs_peak_hr": round(metrics["obs_peak_s"] / 3600, 2),
            "mod_peak_Q":  round(metrics["mod_peak_Q"], 1),
            "mod_peak_hr": round(metrics["mod_peak_s"] / 3600, 2),
        })

    # ── Grand summary ─────────────────────────────────────────────────────────
    print_summary_table(t_dem, summary_rows)

    # Save timing CSV
    timing_csv = OUTPUT_DIR / "timing_gpu.csv"
    pd.DataFrame([{
        "event":       r["event"],
        "sim_hours":   r["sim_h"],
        "init_s":      round(r["t_init"], 2),
        "loop_s":      round(r["t_loop"], 2),
        "total_s":     round(r["t_init"] + r["t_loop"], 2),
    } for r in summary_rows]).to_csv(timing_csv, index=False)
    print(f"\n  Timing CSV saved → {timing_csv.relative_to(REPO_ROOT)}")

    if all_metrics:
        metrics_csv = OUTPUT_DIR / "summary_all_floods.csv"
        pd.DataFrame(all_metrics).to_csv(metrics_csv, index=False)
        print(f"  Metrics CSV saved → {metrics_csv.relative_to(REPO_ROOT)}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
