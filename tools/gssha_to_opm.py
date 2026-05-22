"""
gssha_to_opm.py
===============
Convert GSSHA gauge rainfall files (.gag) to OPM precipitation format.

GSSHA format:
  EVENT "..."
  NRPDS <n_periods>
  NRGAG <n_gauges>
  COORD <easting> <northing> "<name>"
  GAGES <year> <month> <day> <hour> <min>  <val1> ... <valN>
  Values: mm depth accumulated during the 30-minute interval ending at timestamp.

OPM format (per-event folder):
  gauges.csv    → gauge_id, name, easting_m, northing_m
  timeseries.csv → time_s, <gauge_id1>, <gauge_id2>, ...
  Values: mm depth per interval (precip_input.py converts to m/s internally).

Usage:
  python tools/gssha_to_opm.py                    # convert all .gag in test_data/gssha_format/
  python tools/gssha_to_opm.py path/to/file.gag   # convert a single file
"""

import re
import sys
import csv
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parent.parent
GSSHA_DIR      = REPO_ROOT / "test_data" / "gssha_format"
OUTPUT_BASE    = REPO_ROOT / "test_data" / "opm_format"


def parse_gag(filepath: Path) -> dict:
    """Parse a GSSHA .gag file and return structured data."""
    gauges  = []   # list of (gauge_id, name, easting, northing)
    records = []   # list of (datetime, [float, ...])

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("COORD"):
                # COORD <easting> <northing> "<name>"
                m = re.match(
                    r'COORD\s+([\d.]+)\s+([\d.]+)\s+"([^"]+)"', line
                )
                if not m:
                    raise ValueError(f"Cannot parse COORD line: {line!r}")
                easting, northing, full_name = (
                    float(m.group(1)), float(m.group(2)), m.group(3)
                )
                # Extract gauge ID: first whitespace-delimited token (e.g. "TP001")
                gauge_id = full_name.split()[0]
                gauges.append((gauge_id, full_name, easting, northing))

            elif line.startswith("GAGES"):
                # GAGES <yr> <mo> <dy> <hr> <mn>  <v1> <v2> ...
                parts = line.split()
                yr, mo, dy, hr, mn = (int(p) for p in parts[1:6])
                values = [float(v) for v in parts[6:]]
                dt = datetime(yr, mo, dy, hr, mn)
                records.append((dt, values))

    if not gauges:
        raise ValueError(f"No COORD lines found in {filepath}")
    if not records:
        raise ValueError(f"No GAGES lines found in {filepath}")

    return {"gauges": gauges, "records": records, "source": filepath.stem}


def compute_interval_s(records: list) -> float:
    """Derive interval length (seconds) from the first two timestamps."""
    if len(records) < 2:
        return 1800.0  # default 30 min
    dt0, dt1 = records[0][0], records[1][0]
    delta = (dt1 - dt0).total_seconds()
    if delta <= 0:
        raise ValueError("Timestamps are not strictly increasing.")
    return delta


def write_opm(data: dict, out_dir: Path) -> None:
    """Write gauges.csv and timeseries.csv in OPM format."""
    out_dir.mkdir(parents=True, exist_ok=True)

    gauges  = data["gauges"]
    records = data["records"]

    interval_s = compute_interval_s(records)

    # ── gauges.csv ────────────────────────────────────────────────────────────
    gauges_path = out_dir / "gauges.csv"
    with open(gauges_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gauge_id", "name", "easting_m", "northing_m"])
        for gauge_id, name, easting, northing in gauges:
            writer.writerow([gauge_id, name, f"{easting:.6f}", f"{northing:.6f}"])

    # ── timeseries.csv ────────────────────────────────────────────────────────
    gauge_ids = [g[0] for g in gauges]
    ts_path   = out_dir / "timeseries.csv"
    with open(ts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s"] + gauge_ids)

        for i, (_, values) in enumerate(records):
            time_s = int(round(i * interval_s))
            row = [time_s] + [f"{v:.4f}" for v in values]
            writer.writerow(row)

    print(f"  [OK] {out_dir.relative_to(REPO_ROOT)}/")
    print(f"       {len(gauges)} gauges  |  {len(records)} time steps  |  "
          f"interval={int(interval_s)}s")


def convert_file(gag_path: Path) -> None:
    print(f"\nConverting: {gag_path.name}")
    data    = parse_gag(gag_path)
    out_dir = OUTPUT_BASE / gag_path.stem
    write_opm(data, out_dir)


def main():
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        targets = sorted(GSSHA_DIR.glob("*.gag"))
        if not targets:
            print(f"No .gag files found in {GSSHA_DIR}")
            sys.exit(1)

    for target in targets:
        convert_file(target)

    print(f"\nDone. Output written to: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
