#!/usr/bin/env python
"""
sensitivity_report.py
=====================
Aggregate every leaf of the sensitivity study into ONE table + a set of figures.

Reads, for each config leaf under ``outputs collection/sensitivity_100m/``:
  • mass_balance.csv      → runoff ratio + Dunne/Horton/Impervious partition + knobs
  • summary_all_floods.csv→ NSE, PBIAS, observed/modelled peaks
joins them per event, and writes:
  • master_sensitivity.csv          one row per (block, config, flood)
  • figures/fig1_partition.png      headline Dunne/Horton/Imperv split (baseline)
  • figures/fig2_toggle_matrix.png  mechanism on/off redistribution (block A)
  • figures/fig3_sd_sweep.png       partition + NSE vs SD_max (block B)
  • figures/fig4_routing.png        peak attenuation + timing + NSE vs θ (block C)
  • figures/fig5_ksat.png           Horton fraction + peak vs Ksat scale (block D)
  • figures/fig6_tornado.png        OFAT sensitivity of key metrics

Usage:  python tools/sensitivity_report.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT      = REPO_ROOT / "outputs collection/sensitivity_100m"
FIG_DIR   = ROOT / "figures"
GSSHA_DIR = REPO_ROOT / "test_data" / "gssha_format"

EVENTS = ["202308_202308", "202407_202407", "202407_202408", "202409_202409"]

# Reference config reused as θ=1.0 (block C) and ksat_scale=1.0 (block D).
A0 = "A_mechanism/A0_full"

C_BLUE, C_ORANGE, C_GREEN, C_RED = "#3b6ea5", "#e08214", "#4d9221", "#c2453c"
PART_COLORS = {"dunne": C_BLUE, "horton": C_ORANGE, "imperv": "#7a7a7a"}


# ────────────────────────────────────────────────────────────────────────────
# Load
# ────────────────────────────────────────────────────────────────────────────
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_master() -> pd.DataFrame:
    """Walk every leaf, join mass_balance + summary per event → tidy DataFrame."""
    rows = []
    for mb_path in sorted(ROOT.glob("*/*/mass_balance.csv")):
        leaf   = mb_path.parent
        config = str(leaf.relative_to(ROOT))          # e.g. "B_sd/sd_0.20"
        block  = config.split("/")[0]
        mb = pd.read_csv(mb_path)
        # Last write wins if a leaf was re-run (mass_balance appends).
        mb = mb.drop_duplicates(subset="run_tag", keep="last")

        summ = {}
        sp = leaf / "summary_all_floods.csv"
        if sp.exists():
            sdf = pd.read_csv(sp)
            summ = {str(r["event"]): r for _, r in sdf.iterrows()}

        for _, r in mb.iterrows():
            tag = str(r.get("run_tag", ""))
            s   = summ.get(tag, {})
            rows.append(dict(
                block=block, config=config, event=tag,
                scheme=r.get("scheme"), theta=_num(r.get("theta")),
                sd_source=r.get("sd_source"), sd_max=_num(r.get("sd_max")),
                ksat_scale=_num(r.get("ksat_scale")),
                infiltration=r.get("infiltration"), impervious=r.get("impervious"),
                runoff_ratio=_num(r.get("runoff_ratio")),
                dunne_frac=_num(r.get("dunne_frac")),
                horton_frac=_num(r.get("horton_frac")),
                imperv_frac=_num(r.get("imperv_frac")),
                dunne_m3=_num(r.get("dunne_m3")),
                horton_m3=_num(r.get("horton_m3")),
                imperv_m3=_num(r.get("imperv_m3")),
                input_m3=_num(r.get("input_m3")),
                rain_m3=_num(r.get("rain_m3")),
                nse=_num(s.get("nse")), pbias_pct=_num(s.get("pbias_pct")),
                obs_peak_Q=_num(s.get("obs_peak_Q")), mod_peak_Q=_num(s.get("mod_peak_Q")),
                obs_peak_hr=_num(s.get("obs_peak_hr")), mod_peak_hr=_num(s.get("mod_peak_hr")),
            ))
    if not rows:
        sys.exit(f"[ERROR] No mass_balance.csv found under {ROOT}. Run tools/sensitivity.py first.")
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# Figures
# ────────────────────────────────────────────────────────────────────────────
def fig1_partition(df):
    """Headline: Dunne/Horton/Impervious volume fraction per flood (baseline A0)."""
    d = df[df.config == A0].sort_values("event")
    if d.empty:
        return
    events = d.event.tolist()
    x = np.arange(len(events))
    fig, ax = plt.subplots(figsize=(8, 5))
    bot = np.zeros(len(events))
    for key, label in [("dunne_frac", "Dunne (saturation-excess)"),
                       ("horton_frac", "Horton (infiltration-excess)"),
                       ("imperv_frac", "Impervious (urban)")]:
        v = d[key].values * 100
        ax.bar(x, v, bottom=bot, label=label,
               color=PART_COLORS[key.split("_")[0]], edgecolor="white")
        for xi, vi, bi in zip(x, v, bot):
            if vi > 4:
                ax.text(xi, bi + vi / 2, f"{vi:.0f}%", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        bot += v
    ax.set_xticks(x); ax.set_xticklabels(events, rotation=20, ha="right")
    ax.set_ylabel("Share of routed runoff (%)"); ax.set_ylim(0, 100)
    ax.set_title("Runoff partition by mechanism — baseline config (per flood)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    _save(fig, "fig1_partition.png")


def fig2_toggle_matrix(df):
    """Block A: how the partition + runoff ratio shift as mechanisms toggle off."""
    order = ["A_mechanism/A0_full", "A_mechanism/A1_no_horton",
             "A_mechanism/A2_no_imperv", "A_mechanism/A3_dunne_only"]
    labels = ["full", "no Horton", "no imperv", "Dunne only"]
    g = df[df.config.isin(order)].groupby("config").mean(numeric_only=True)
    g = g.reindex(order).dropna(how="all")
    if g.empty:
        return
    x = np.arange(len(g)); lab = [labels[order.index(c)] for c in g.index]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bot = np.zeros(len(g))
    for key in ["dunne_frac", "horton_frac", "imperv_frac"]:
        v = g[key].values * 100
        ax1.bar(x, v, bottom=bot, label=key.split("_")[0].title(),
                color=PART_COLORS[key.split("_")[0]], edgecolor="white")
        bot += v
    ax1.set_xticks(x); ax1.set_xticklabels(lab); ax1.set_ylabel("Share of runoff (%)")
    ax1.set_title("Partition vs mechanism toggle"); ax1.legend(frameon=False)
    ax2.bar(x, g["runoff_ratio"].values, color=C_GREEN)
    for xi, vi in zip(x, g["runoff_ratio"].values):
        ax2.text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(lab)
    ax2.set_ylabel("Runoff ratio (runoff ÷ rain)")
    ax2.set_title("Total runoff response (mean of 4 floods)")
    _save(fig, "fig2_toggle_matrix.png")


def fig3_sd_sweep(df):
    """Block B: Dunne fraction, runoff ratio, NSE vs SD_max."""
    d = df[df.block == "B_sd"].groupby("sd_max").mean(numeric_only=True).sort_index()
    if d.empty:
        return
    x = d.index.values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, d.dunne_frac * 100, "-o", color=C_BLUE, label="Dunne fraction")
    ax.plot(x, d.horton_frac * 100, "-o", color=C_ORANGE, label="Horton fraction")
    ax.plot(x, d.runoff_ratio * 100, "-s", color=C_GREEN, label="Runoff ratio")
    ax.set_xlabel("SD$_{max}$ / root-zone depth (m)")
    ax.set_ylabel("% of runoff  /  % of rain"); ax.set_xscale("log")
    ax2 = ax.twinx()
    ax2.plot(x, d.nse, "--^", color=C_RED, label="NSE")
    ax2.set_ylabel("NSE", color=C_RED); ax2.tick_params(axis="y", colors=C_RED)
    ax.set_title("Soil-deficit sensitivity (mean of 4 floods)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", frameon=False)
    _save(fig, "fig3_sd_sweep.png")


def fig4_routing(df):
    """Block C: peak attenuation + time-to-peak + NSE vs diffusion θ (kinematic ref)."""
    c = df[df.block == "C_routing"].copy()
    a0 = df[df.config == A0].copy()           # θ=1.0 diffusive reference
    if c.empty and a0.empty:
        return
    # Diffusive points keyed by θ (+ A0 as θ=1.0); kinematic shown separately.
    diff = pd.concat([c[c.scheme == "diffusive"], a0])
    dg = diff.groupby("theta").mean(numeric_only=True).sort_index()
    kin = c[c.scheme == "kinematic"].mean(numeric_only=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if not dg.empty:
        ax1.plot(dg.index, dg.mod_peak_Q, "-o", color=C_BLUE, label="diffusive")
    if kin is not None and not np.isnan(kin.get("mod_peak_Q", np.nan)):
        ax1.axhline(kin["mod_peak_Q"], ls="--", color=C_RED, label="kinematic")
    if not dg.empty and "obs_peak_Q" in dg:
        ax1.axhline(dg.obs_peak_Q.mean(), ls=":", color="k", label="observed")
    ax1.set_xlabel("diffusion weight θ"); ax1.set_ylabel("modelled peak Q (m³/s)")
    ax1.set_title("Peak discharge vs routing"); ax1.legend(frameon=False)

    if not dg.empty:
        ax2.plot(dg.index, dg.nse, "-o", color=C_GREEN, label="diffusive NSE")
    if kin is not None and not np.isnan(kin.get("nse", np.nan)):
        ax2.axhline(kin["nse"], ls="--", color=C_RED, label="kinematic NSE")
    ax2.set_xlabel("diffusion weight θ"); ax2.set_ylabel("NSE")
    ax2.set_title("Skill vs routing (mean of 4 floods)"); ax2.legend(frameon=False)
    _save(fig, "fig4_routing.png")


def fig5_ksat(df):
    """Block D: Horton fraction + runoff ratio + peak vs Ksat scale (1.0 = A0)."""
    d = df[df.block == "D_ksat"].copy()
    a0 = df[df.config == A0].copy()
    a0 = a0.assign(ksat_scale=1.0)
    dd = pd.concat([d, a0]).groupby("ksat_scale").mean(numeric_only=True).sort_index()
    if dd.empty:
        return
    x = dd.index.values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, dd.horton_frac * 100, "-o", color=C_ORANGE, label="Horton fraction")
    ax.plot(x, dd.runoff_ratio * 100, "-s", color=C_GREEN, label="Runoff ratio")
    ax.set_xlabel("Ksat scale (× HiHydroSoil)"); ax.set_ylabel("% of runoff / rain")
    ax.set_xscale("log")
    ax2 = ax.twinx()
    ax2.plot(x, dd.mod_peak_Q, "--^", color=C_BLUE, label="peak Q")
    ax2.set_ylabel("modelled peak Q (m³/s)", color=C_BLUE)
    ax2.tick_params(axis="y", colors=C_BLUE)
    ax.set_title("Infiltration-capacity sensitivity (mean of 4 floods)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", frameon=False)
    _save(fig, "fig5_ksat.png")


def fig6_tornado(df):
    """OFAT range (max−min over each block's sweep) of key response metrics."""
    metrics = [("dunne_frac", "Dunne fraction", 100.0),
               ("runoff_ratio", "Runoff ratio", 100.0),
               ("mod_peak_Q", "Peak Q (m³/s)", 1.0),
               ("nse", "NSE", 1.0)]
    blocks = {"B_sd": "SD_max", "C_routing": "routing θ", "D_ksat": "Ksat scale",
              "A_mechanism": "mechanism toggles"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.5))
    for ax, (col, title, sc) in zip(axes, metrics):
        spans, labels = [], []
        for blk, name in blocks.items():
            g = df[df.block == blk].groupby("config")[col].mean()
            if g.notna().sum() >= 2:
                spans.append((g.max() - g.min()) * sc); labels.append(name)
        if spans:
            order = np.argsort(spans)
            yy = np.arange(len(spans))
            ax.barh(yy, np.array(spans)[order], color=C_BLUE)
            ax.set_yticks(yy); ax.set_yticklabels(np.array(labels)[order])
        ax.set_title(title); ax.set_xlabel("range (max − min)")
    fig.suptitle("OFAT sensitivity — response range of each metric to each factor",
                 fontsize=13)
    _save(fig, "fig6_tornado.png", tight_rect=(0, 0, 1, 0.94))


# ────────────────────────────────────────────────────────────────────────────
# Scenario-overlay hydrographs:  4 floods (2x2), each overlays several configs
# ────────────────────────────────────────────────────────────────────────────
def _event_starts():
    """tag → event-start datetime, read from any leaf's event_catalogue.csv."""
    for cat in ROOT.glob("*/*/event_catalogue.csv"):
        df = pd.read_csv(cat)
        return {str(r.event_tag): pd.to_datetime(r.start_local)
                for _, r in df.iterrows()}
    return {}


def _modelled(config, event):
    """(hours-since-start, Q) for one config + event, or None."""
    p = ROOT / config / f"hydrograph_{event}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df["time_hr"].values, df["Q_m3s"].values


def _observed(event, starts):
    """(hours-since-start, Q) observed, clipped to t>=0, or None."""
    disc = GSSHA_DIR / f"discharge_{event}.csv"
    if not disc.exists() or event not in starts:
        return None
    df = pd.read_csv(disc, parse_dates=["dateTime"])
    t_hr = (df["dateTime"] - starts[event]).dt.total_seconds() / 3600.0
    m = t_hr >= 0
    return t_hr[m].values, df["discharge_m3s"][m].values


def fig_overlay(scenarios, title, outname):
    """4 floods as 2x2 subplots; each overlays the scenario modelled hydrographs
    plus the observed.  `scenarios` = list of (config_path, label)."""
    starts = _event_starts()
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=False)
    for ax, event in zip(axes.ravel(), EVENTS):
        obs = _observed(event, starts)
        if obs is not None:
            ax.plot(obs[0], obs[1], color="black", lw=2.2, label="Observed", zorder=10)
        for i, (cfg_path, label) in enumerate(scenarios):
            md = _modelled(cfg_path, event)
            if md is not None:
                ax.plot(md[0], md[1], lw=1.5, color=colors[i % len(colors)],
                        label=label, alpha=0.9)
        ax.set_title(event, fontweight="bold")
        ax.set_xlabel("hours since event start")
        ax.set_ylabel("Discharge (m³/s)")
        ax.set_ylim(bottom=0)
        ax.grid(True, ls="--", alpha=0.3)
    axes.ravel()[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    _save(fig, outname, tight_rect=(0, 0, 1, 0.96))


def _save(fig, name, tight_rect=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=tight_rect) if tight_rect else fig.tight_layout()
    out = FIG_DIR / name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure → {out.relative_to(REPO_ROOT)}")


def main():
    df = load_master()
    out_csv = ROOT / "master_sensitivity.csv"
    df.sort_values(["block", "config", "event"]).to_csv(out_csv, index=False)
    print(f"  master table ({len(df)} rows) → {out_csv.relative_to(REPO_ROOT)}\n")

    fig1_partition(df)
    fig2_toggle_matrix(df)
    fig3_sd_sweep(df)
    fig4_routing(df)
    fig5_ksat(df)
    fig6_tornado(df)

    # Scenario-overlay hydrographs (4 floods × overlaid configs + observed).
    fig_overlay([
        ("A_mechanism/A0_full",       "Full (Horton+Imperv)"),
        ("A_mechanism/A1_no_horton",  "No Horton"),
        ("A_mechanism/A2_no_imperv",  "No impervious"),
        ("A_mechanism/A3_dunne_only", "Dunne only"),
    ], "Mechanism toggles — hydrograph per flood", "fig7_hydro_mechanism.png")

    fig_overlay([
        ("C_routing/kinematic",       "Kinematic"),
        ("C_routing/diff_theta_0.25", "Diffusive θ=0.25"),
        ("C_routing/diff_theta_0.50", "Diffusive θ=0.50"),
        ("C_routing/diff_theta_0.75", "Diffusive θ=0.75"),
        ("A_mechanism/A0_full",       "Diffusive θ=1.0"),
    ], "Routing scheme — hydrograph per flood", "fig8_hydro_routing.png")

    fig_overlay([
        ("D_ksat/ksat_0.5",     "Ksat ×0.5"),
        ("A_mechanism/A0_full", "Ksat ×1.0"),
        ("D_ksat/ksat_2.0",     "Ksat ×2.0"),
        ("D_ksat/ksat_4.0",     "Ksat ×4.0"),
    ], "Infiltration capacity (Ksat) — hydrograph per flood", "fig9_hydro_ksat.png")

    fig_overlay([
        ("B_sd/sd_0.05", "SD=0.05 m"),
        ("B_sd/sd_0.10", "SD=0.10 m"),
        ("B_sd/sd_0.20", "SD=0.20 m"),
        ("B_sd/sd_0.40", "SD=0.40 m"),
        ("B_sd/sd_0.80", "SD=0.80 m"),
    ], "Soil deficit (SD_max) — hydrograph per flood", "fig10_hydro_sd.png")

    # Console headline: baseline mean partition.
    base = df[df.config == A0]
    if not base.empty:
        print("\n  Baseline runoff partition (mean of 4 floods):")
        print(f"    Dunne  {base.dunne_frac.mean()*100:5.1f} %   "
              f"Horton {base.horton_frac.mean()*100:5.1f} %   "
              f"Imperv {base.imperv_frac.mean()*100:5.1f} %   "
              f"(runoff ratio {base.runoff_ratio.mean():.2f})")
    print("\n  Done.")


if __name__ == "__main__":
    main()
