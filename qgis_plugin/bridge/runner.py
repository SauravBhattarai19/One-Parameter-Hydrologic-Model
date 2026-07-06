# -*- coding: utf-8 -*-
"""
runner.py
=========
Background worker that runs the VSA-OPM pipeline in a QThread so the
QGIS GUI stays responsive.

Architecture
------------

    OpmWorker(QThread)
        Signals:
            progress(int)   – 0 … 100 progress percentage
            log(str)        – line of text from the model's stdout
            finished(dict)  – result dict with paths of generated files
            error(str)      – human-readable error message (run aborted)

        Slots (called from main thread):
            cancel()        – request graceful abort (sets _cancelled flag)

Stdout redirection
------------------
All existing model modules use print() for progress.  The worker redirects
sys.stdout to a _StdoutCapture object that emits the log() signal instead
of writing to the console.  The original stdout is restored on exit.

GPU memory management
---------------------
After each run, GPU memory pools are freed so QGIS's OpenGL renderer
gets the VRAM back.  This is idempotent when CuPy is not installed.

Path injection
--------------
The OPM repo root (parent of this file's parent) is inserted at the
front of sys.path so that  `import config`, `import routing_utils`, etc.
all resolve to the correct files even when QGIS's Python path doesn't
include the project directory.
"""

import io
import os
import sys
import traceback

from qgis.PyQt.QtCore import QThread, pyqtSignal

# Plugin root is 1 level up from bridge/: vsa_opm/bridge/ → vsa_opm/
_OPM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_opm_on_path():
    """Insert the OPM project root into sys.path if not already there."""
    if _OPM_ROOT not in sys.path:
        sys.path.insert(0, _OPM_ROOT)


# ── Stdout capture ─────────────────────────────────────────────────────────────

class _StdoutCapture(io.TextIOBase):
    """
    File-like object that forwards write() calls to a PyQt signal.

    Parameters
    ----------
    signal : pyqtSignal(str)
        Signal to emit for each non-empty line written.
    """

    def __init__(self, signal):
        super().__init__()
        self._signal = signal
        self._buffer = ""

    def write(self, text):
        self._buffer += text
        # Flush complete lines immediately so the log panel updates in real time
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._signal.emit(line)
        return len(text)

    def flush(self):
        if self._buffer:
            self._signal.emit(self._buffer)
            self._buffer = ""


# ── Worker thread ──────────────────────────────────────────────────────────────

class OpmWorker(QThread):
    """
    Runs the full VSA-OPM pipeline (or individual stages) in a background
    thread.

    Parameters
    ----------
    cfg : OpmConfig
        Configuration object built by the UI.
    stages : list of str
        Which pipeline stages to run.  Options (in order):
            'process_dem'  – DEM preprocessing (process_dem.main)
            'routing'      – kinematic-wave routing (initialise_grid + run_time_loop)
            'vsa_opm'      – standalone OPM run (vsa_opm.run_opm)
        Default: ['process_dem', 'routing']
    """

    # Qt signals (must be class-level)
    progress = pyqtSignal(int)          # 0–100
    log = pyqtSignal(str)               # one log line
    finished = pyqtSignal(dict)         # result dict
    error = pyqtSignal(str)             # error message

    def __init__(self, cfg, stages=None, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._stages = stages or ["process_dem", "routing"]
        self._cancelled = False
        self._result = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def cancel(self):
        """Request graceful cancellation (checked between stages)."""
        self._cancelled = True
        self.log.emit("[INFO] Cancellation requested — will stop after current stage.")

    # ── QThread entry point ───────────────────────────────────────────────────

    def run(self):
        """Called by QThread.start().  Runs in the worker thread."""
        _ensure_opm_on_path()

        # Redirect stdout so model print() calls go to the log signal
        original_stdout = sys.stdout
        sys.stdout = _StdoutCapture(self.log)

        try:
            self._run_pipeline()
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            self.log.emit(tb)
            self.error.emit(
                "An error occurred during the simulation.\n\n"
                f"{tb.splitlines()[-1]}"
            )
        finally:
            # Always restore stdout and free GPU memory
            sys.stdout = original_stdout
            self._release_gpu_memory()

    # ── Pipeline stages ───────────────────────────────────────────────────────

    def _run_pipeline(self):
        # Resolve OUTPUT_DIR to an ABSOLUTE path and pre-create it, so relative
        # defaults (e.g. an unedited config.py "outputs collection/…") are never
        # created under QGIS's read-only working directory.  Re-sync the derived
        # paths so ROUTING_*/hydrograph/etc. are absolute too.
        try:
            self._cfg.OUTPUT_DIR = os.path.abspath(self._cfg.OUTPUT_DIR or "output")
            if hasattr(self._cfg, "update_output_paths"):
                self._cfg.update_output_paths()
            os.makedirs(self._cfg.OUTPUT_DIR, exist_ok=True)
            self.log.emit(f"[INFO] Output directory: {self._cfg.OUTPUT_DIR}")
        except OSError as exc:
            raise RuntimeError(
                f"Cannot create the output directory '{self._cfg.OUTPUT_DIR}'. "
                "Pick a writable folder on the DEM & Watershed tab (e.g. under "
                "your user/Documents folder)."
            ) from exc

        stages = self._stages
        n = len(stages)

        for idx, stage in enumerate(stages):
            if self._cancelled:
                self.log.emit("[INFO] Run cancelled by user.")
                return

            self.log.emit(f"\n{'='*60}")
            self.log.emit(f"  STAGE {idx+1}/{n}: {stage.upper()}")
            self.log.emit(f"{'='*60}")

            if stage == "process_dem":
                self._stage_process_dem(base=0, span=30)

            elif stage == "routing":
                self._stage_routing(base=30, span=65)

            elif stage == "vsa_opm":
                self._stage_vsa_opm(base=30, span=65)

            else:
                raise ValueError(f"Unknown pipeline stage: '{stage}'")

        self.progress.emit(100)
        self.log.emit("\n[INFO] All stages complete.")
        self.finished.emit(self._result)

    def _stage_process_dem(self, base, span):
        """Run process_dem.main() with cfg applied."""
        import process_dem as pd_mod

        # Monkey-patch the module-level globals that process_dem.main() reads
        pd_mod.DEM_PATH = self._cfg.DEM_PATH
        pd_mod.TARGET_CRS_EPSG = self._cfg.TARGET_CRS_EPSG
        pd_mod.OUTPUT_POINT_LATLON = self._cfg.OUTPUT_POINT
        pd_mod.OUTPUT_DIR = self._cfg.OUTPUT_DIR

        self.progress.emit(base + span // 3)
        pd_mod.main()
        self.progress.emit(base + span)

        # Record outputs
        d = self._cfg.OUTPUT_DIR
        self._result.update({
            "watershed_tif": os.path.join(d, "watershed.tif"),
            "watershed_geojson": os.path.join(d, "watershed.geojson"),
            "clipped_dem": os.path.join(d, "clipped_dem.tif"),
            "flow_direction": os.path.join(d, "flow_direction.tif"),
            "flow_accumulation": os.path.join(d, "clipped_flow_accumulation.tif"),
        })

    def _stage_routing(self, base, span):
        """Run kinematic-wave routing: initialise_grid → run_time_loop → save_hydrograph."""
        import kinematic_wave_router as kwr

        self.log.emit("  Initialising grid …")
        self.progress.emit(base + span // 4)
        grid_data = kwr.initialise_grid(self._cfg)

        if self._cancelled:
            return

        self.log.emit("  Running time loop …")
        self.progress.emit(base + span // 2)
        hydrograph = kwr.run_time_loop(grid_data, self._cfg)

        self.log.emit("  Saving hydrograph …")
        self.progress.emit(base + span - 5)
        df = kwr.save_hydrograph(hydrograph, self._cfg)

        self.progress.emit(base + span)
        self._result["hydrograph_csv"] = self._cfg.HYDROGRAPH_CSV
        self._result["hydrograph_df"] = df

    def _stage_vsa_opm(self, base, span):
        """Run the standalone VSA-OPM model."""
        import vsa_opm

        self.progress.emit(base + span // 2)
        df = vsa_opm.run_opm(self._cfg)
        self.progress.emit(base + span)

        out_path = os.path.join(self._cfg.OUTPUT_DIR, "vsa_opm_results.csv")
        self._result["vsa_opm_csv"] = out_path
        self._result["vsa_opm_df"] = df

    # ── GPU cleanup ──────────────────────────────────────────────────────────

    @staticmethod
    def _release_gpu_memory():
        """Free CuPy memory pools.  No-op if CuPy is not installed."""
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:  # noqa: BLE001
            pass  # CuPy not available or no device — nothing to free
