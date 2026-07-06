# -*- coding: utf-8 -*-
"""
tab_runoff.py
=============
Tab 3 — Runoff Generation Engine.

Modes
-----
  none        – all rainfall is direct runoff
  coefficient – static spatial Cf raster
  raster      – pre-computed runoff raster time series
  scs_cn      – SCS Curve Number per-cell raster
  vsa_opm     – Variable Source Area (Pradhan & Ogden 2010)

The vsa_opm panel exposes the full OPM model: runoff mechanisms
(VSA / Horton-Green-Ampt / impervious), the SERVES satellite soil-moisture
source, Green-Ampt infiltration, urban impervious shedding, and baseflow.
"""

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QComboBox,
    QDoubleSpinBox, QSpinBox, QLabel, QStackedWidget, QCheckBox,
    QScrollArea, QHBoxLayout, QFrame,
)
from qgis.PyQt.QtCore import Qt
from qgis.gui import QgsFileWidget


class TabRunoff(QWidget):
    """Runoff generation engine configuration tab."""

    _MODES = ["none", "coefficient", "raster", "scs_cn", "vsa_opm"]
    _MODE_LABELS = [
        "None — all rainfall is runoff",
        "Runoff coefficient (static Cf raster)",
        "Pre-computed runoff raster time series",
        "SCS Curve Number (CN raster)",
        "VSA-OPM — Variable Source Area (Pradhan & Ogden 2010)",
    ]

    _SD_SOURCES = ["manual", "gee"]
    _SD_REDUCERS = ["mean", "max", "divide"]
    _SATELLITES = ["landsat", "sentinel2", "modis"]
    _SOILGRIDS_DEPTHS = ["b0", "b10", "b30", "b60", "b100", "b200"]
    _INFILTRATION = ["none", "green_ampt"]
    _SUCTION_SOURCES = ["scalar", "texture"]
    _KSAT_SOURCES = ["scalar", "gee", "raster"]
    _IMPERVIOUS_SOURCES = ["none", "lcz", "lulc", "raster"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        grp_mode = QGroupBox("Runoff Source")
        form_mode = QFormLayout(grp_mode)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self._MODE_LABELS)
        self.mode_combo.setCurrentIndex(4)   # default: vsa_opm
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form_mode.addRow("Mode:", self.mode_combo)

        root.addWidget(grp_mode)

        # Stacked panels — the vsa_opm panel is scrollable (it is tall).
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_none_panel())         # 0
        self._stack.addWidget(self._build_coefficient_panel())  # 1
        self._stack.addWidget(self._build_raster_panel())       # 2
        self._stack.addWidget(self._build_scs_cn_panel())       # 3
        self._stack.addWidget(self._build_vsa_opm_scroll())     # 4

        self._stack.setCurrentIndex(4)
        root.addWidget(self._stack)

    def _build_none_panel(self):
        w = QGroupBox("No Runoff Transformation")
        QFormLayout(w).addRow(QLabel("All rainfall reaches the channel as direct runoff.\nNo files required."))
        return w

    def _build_coefficient_panel(self):
        w = QGroupBox("Runoff Coefficient")
        form = QFormLayout(w)
        self._cf_file = QgsFileWidget()
        self._cf_file.setStorageMode(QgsFileWidget.GetFile)
        self._cf_file.setFilter("GeoTIFF (*.tif *.tiff);;All files (*)")
        form.addRow("Cf raster (0–1):", self._cf_file)
        return w

    def _build_raster_panel(self):
        w = QGroupBox("Pre-computed Runoff Raster Series")
        form = QFormLayout(w)
        self._raster_manifest = QgsFileWidget()
        self._raster_manifest.setStorageMode(QgsFileWidget.GetFile)
        self._raster_manifest.setFilter("CSV (*.csv);;All files (*)")
        form.addRow("Manifest CSV:", self._raster_manifest)
        return w

    def _build_scs_cn_panel(self):
        w = QGroupBox("SCS Curve Number")
        form = QFormLayout(w)
        self._cn_file = QgsFileWidget()
        self._cn_file.setStorageMode(QgsFileWidget.GetFile)
        self._cn_file.setFilter("GeoTIFF (*.tif *.tiff);;All files (*)")
        form.addRow("CN raster:", self._cn_file)
        self._ia_factor = QDoubleSpinBox()
        self._ia_factor.setRange(0.0, 0.5)
        self._ia_factor.setDecimals(3)
        self._ia_factor.setValue(0.2)
        self._ia_factor.setToolTip("SCS initial abstraction ratio (standard: 0.2)")
        form.addRow("Ia factor:", self._ia_factor)
        return w

    # ── VSA-OPM panel (scrollable) ─────────────────────────────────────────────

    def _build_vsa_opm_scroll(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self._build_vsa_opm_panel())
        return scroll

    def _build_vsa_opm_panel(self):
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(8)

        # ── Runoff mechanisms ──────────────────────────────────────────────
        grp_mech = QGroupBox("Runoff Mechanisms  (compose the OPM runoff model)")
        h_mech = QHBoxLayout(grp_mech)
        self._chk_vsa = QCheckBox("VSA (saturation-excess / Dunne)")
        self._chk_vsa.setChecked(True)
        self._chk_horton = QCheckBox("Horton (Green-Ampt infiltration-excess)")
        self._chk_horton.setChecked(True)
        self._chk_imperv = QCheckBox("Impervious (urban shedding)")
        self._chk_imperv.setChecked(True)
        for c in (self._chk_vsa, self._chk_horton, self._chk_imperv):
            h_mech.addWidget(c)
        h_mech.addStretch()
        v.addWidget(grp_mech)

        # ── Core OPM parameters ────────────────────────────────────────────
        grp_core = QGroupBox("Core OPM Parameters  (Pradhan & Ogden 2010)")
        form = QFormLayout(grp_core)

        self._sd_max = QDoubleSpinBox()
        self._sd_max.setRange(0.001, 5.0)
        self._sd_max.setDecimals(4)
        self._sd_max.setValue(0.10)
        self._sd_max.setSuffix(" m")
        self._sd_max.setToolTip(
            "Root-zone depth D at the catchment divide (physical height, not\n"
            "water volume).  Overridden per-zone when SD source = GEE/SERVES."
        )
        form.addRow("SD_max initial (root-zone depth):", self._sd_max)

        self._q_max = QDoubleSpinBox()
        self._q_max.setRange(0.002, 99999.0)
        self._q_max.setDecimals(4)
        self._q_max.setValue(100.0)
        self._q_max.setSuffix(" m³/s")
        self._q_max.setToolTip(
            "Observed baseflow / initial outlet discharge.\n"
            "Used in Eq 10 to calibrate the initial threshold area A_t."
        )
        form.addRow("Q_max (initial discharge):", self._q_max)

        self._phi = QDoubleSpinBox()
        self._phi.setRange(0.01, 0.99)
        self._phi.setDecimals(3)
        self._phi.setValue(0.35)
        self._phi.setToolTip("Drainable porosity (specific yield). Overridden when SD source = GEE.")
        form.addRow("phi (porosity):", self._phi)

        self._k_sat = QDoubleSpinBox()
        self._k_sat.setRange(0.001, 500.0)
        self._k_sat.setDecimals(3)
        self._k_sat.setValue(44.0)
        self._k_sat.setSuffix(" m/day")
        self._k_sat.setToolTip("LATERAL saturated hydraulic conductivity (drives sandbox Darcy drainage).")
        form.addRow("K_sat (lateral):", self._k_sat)

        self._per_polygon = QCheckBox("Per-polygon sandbox (each gauge/precip zone gets its own OPM)")
        self._per_polygon.setChecked(True)
        form.addRow(self._per_polygon)

        self._baseflow = QCheckBox("Seed baseflow from Q_max (hydrograph starts at pre-storm discharge)")
        self._baseflow.setChecked(False)
        form.addRow(self._baseflow)

        v.addWidget(grp_core)

        # ── SERVES / GEE soil-moisture deficit ─────────────────────────────
        grp_sd = QGroupBox("Soil-Moisture Deficit  (SD_max & phi source)")
        form_sd = QFormLayout(grp_sd)

        self._sd_source = QComboBox()
        self._sd_source.addItems([
            "Manual (use the values above)",
            "GEE / SERVES (satellite deficit + SoilGrids porosity + LULC/LCZ depth)",
        ])
        self._sd_source.setToolTip("GEE requires a project + event date on the DEM tab.")
        form_sd.addRow("SD source:", self._sd_source)

        self._sd_reducer = QComboBox()
        self._sd_reducer.addItems([
            "mean — zone-average deficit (robust)",
            "max — largest deficit / storage-capacity cell",
            "divide — deficit sampled at each zone's divide cell",
        ])
        form_sd.addRow("Zone reducer:", self._sd_reducer)

        self._satellite = QComboBox()
        self._satellite.addItems(self._SATELLITES)
        form_sd.addRow("SERVES satellite:", self._satellite)

        self._search_window = QSpinBox()
        self._search_window.setRange(1, 365)
        self._search_window.setValue(30)
        self._search_window.setSuffix(" days")
        self._search_window.setToolTip("Days backward from the event date to search for a clear scene.")
        form_sd.addRow("Search window:", self._search_window)

        self._soilgrids_depth = QComboBox()
        self._soilgrids_depth.addItems(self._SOILGRIDS_DEPTHS)
        self._soilgrids_depth.setCurrentText("b30")
        form_sd.addRow("SoilGrids depth band:", self._soilgrids_depth)

        v.addWidget(grp_sd)

        # ── Green-Ampt infiltration ────────────────────────────────────────
        grp_ga = QGroupBox("Infiltration — Green-Ampt (Horton mechanism)")
        form_ga = QFormLayout(grp_ga)

        self._infiltration = QComboBox()
        self._infiltration.addItems([
            "none — all rain infiltrates the sandbox (pure saturation-excess)",
            "green_ampt — per-cell infiltration capacity limits runoff",
        ])
        form_ga.addRow("Infiltration model:", self._infiltration)

        self._suction_source = QComboBox()
        self._suction_source.addItems([
            "scalar — uniform value below",
            "texture — per-cell from SoilGrids (needs GEE)",
        ])
        form_ga.addRow("Suction ψ source:", self._suction_source)

        self._suction_m = QDoubleSpinBox()
        self._suction_m.setRange(0.0, 2.0)
        self._suction_m.setDecimals(3)
        self._suction_m.setValue(0.15)
        self._suction_m.setSuffix(" m")
        self._suction_m.setToolTip("Wetting-front suction head ψ (loam ≈ 0.1–0.2 m). Uniform value / nodata fallback.")
        form_ga.addRow("Suction ψ (scalar/fallback):", self._suction_m)

        self._ksat_source = QComboBox()
        self._ksat_source.addItems([
            "scalar — uniform value below",
            "gee — HiHydroSoil v2.0 gridded Ksat (needs GEE)",
            "raster — pre-computed GeoTIFF",
        ])
        form_ga.addRow("Vertical Ksat source:", self._ksat_source)

        self._ga_ksat = QDoubleSpinBox()
        self._ga_ksat.setRange(0.01, 500.0)
        self._ga_ksat.setDecimals(2)
        self._ga_ksat.setValue(12.0)
        self._ga_ksat.setSuffix(" mm/hr")
        self._ga_ksat.setToolTip(
            "VERTICAL (surface) saturated conductivity — NOT the lateral K_sat.\n"
            "Sand ≈ 50, loam ≈ 10, clay ≈ 1 mm/hr.  Uniform value / nodata fallback."
        )
        form_ga.addRow("Vertical Ksat (scalar/fallback):", self._ga_ksat)

        self._ga_ksat_raster = QgsFileWidget()
        self._ga_ksat_raster.setStorageMode(QgsFileWidget.GetFile)
        self._ga_ksat_raster.setFilter("GeoTIFF (*.tif *.tiff);;All files (*)")
        form_ga.addRow("Ksat raster (source=raster):", self._ga_ksat_raster)

        self._ga_ksat_scale = QDoubleSpinBox()
        self._ga_ksat_scale.setRange(0.01, 100.0)
        self._ga_ksat_scale.setDecimals(2)
        self._ga_ksat_scale.setValue(1.0)
        self._ga_ksat_scale.setToolTip("Calibration multiplier on the gridded Ksat.")
        form_ga.addRow("Ksat calibration scale:", self._ga_ksat_scale)

        v.addWidget(grp_ga)

        # ── Impervious ─────────────────────────────────────────────────────
        grp_imp = QGroupBox("Impervious Fraction (urban shedding)")
        form_imp = QFormLayout(grp_imp)

        self._imperv_source = QComboBox()
        self._imperv_source.addItems([
            "none — no impervious contribution",
            "lcz — WUDAPT LCZ impervious column (needs GEE)",
            "lulc — ESA WorldCover impervious column (needs GEE)",
            "raster — pre-computed impervious-fraction GeoTIFF",
        ])
        form_imp.addRow("Impervious source:", self._imperv_source)

        self._imperv_raster = QgsFileWidget()
        self._imperv_raster.setStorageMode(QgsFileWidget.GetFile)
        self._imperv_raster.setFilter("GeoTIFF (*.tif *.tiff);;All files (*)")
        form_imp.addRow("Impervious raster (source=raster):", self._imperv_raster)

        v.addWidget(grp_imp)
        v.addStretch()
        return panel

    # ── Slot ─────────────────────────────────────────────────────────────────

    def _on_mode_changed(self, idx):
        self._stack.setCurrentIndex(idx)

    # ── Public getters ────────────────────────────────────────────────────────

    def get_mode(self) -> str:
        return self._MODES[self.mode_combo.currentIndex()]

    def _mechanisms(self):
        mechs = []
        if self._chk_vsa.isChecked():
            mechs.append("vsa")
        if self._chk_horton.isChecked():
            mechs.append("horton")
        if self._chk_imperv.isChecked():
            mechs.append("impervious")
        return mechs

    # ── Config I/O ────────────────────────────────────────────────────────────

    def apply_config(self, cfg):
        mode = getattr(cfg, "RUNOFF_SOURCE", "none")
        idx = self._MODES.index(mode) if mode in self._MODES else 0
        self.mode_combo.setCurrentIndex(idx)

        if cfg.RUNOFF_COEFFICIENT_PATH:
            self._cf_file.setFilePath(cfg.RUNOFF_COEFFICIENT_PATH)
        if cfg.RUNOFF_RASTER_MANIFEST:
            self._raster_manifest.setFilePath(cfg.RUNOFF_RASTER_MANIFEST)
        if cfg.RUNOFF_CN_PATH:
            self._cn_file.setFilePath(cfg.RUNOFF_CN_PATH)
        self._ia_factor.setValue(cfg.RUNOFF_SCS_Ia_FACTOR)

        # Mechanisms
        mechs = getattr(cfg, "RUNOFF_MECHANISMS", None) or ["vsa", "horton", "impervious"]
        self._chk_vsa.setChecked("vsa" in mechs)
        self._chk_horton.setChecked("horton" in mechs)
        self._chk_imperv.setChecked("impervious" in mechs)

        # Core
        self._sd_max.setValue(cfg.OPM_SD_MAX_INITIAL)
        self._q_max.setValue(cfg.OPM_Q_MAX)
        self._phi.setValue(cfg.OPM_PHI)
        self._k_sat.setValue(cfg.OPM_K_SAT)
        self._per_polygon.setChecked(bool(getattr(cfg, "OPM_PER_POLYGON", True)))
        self._baseflow.setChecked(bool(getattr(cfg, "OPM_BASEFLOW", False)))

        # SERVES / GEE
        self._sd_source.setCurrentIndex(self._idx(self._SD_SOURCES, getattr(cfg, "OPM_SD_SOURCE", "manual")))
        self._sd_reducer.setCurrentIndex(self._idx(self._SD_REDUCERS, getattr(cfg, "OPM_SD_REDUCER", "mean")))
        self._satellite.setCurrentIndex(self._idx(self._SATELLITES, getattr(cfg, "SERVES_SATELLITE", "landsat")))
        self._search_window.setValue(int(getattr(cfg, "SERVES_SEARCH_WINDOW", 30)))
        self._soilgrids_depth.setCurrentText(getattr(cfg, "OPM_SOILGRIDS_DEPTH", "b30"))

        # Green-Ampt
        self._infiltration.setCurrentIndex(self._idx(self._INFILTRATION, getattr(cfg, "OPM_INFILTRATION", "none")))
        self._suction_source.setCurrentIndex(self._idx(self._SUCTION_SOURCES, getattr(cfg, "OPM_GA_SUCTION_SOURCE", "scalar")))
        self._suction_m.setValue(float(getattr(cfg, "OPM_GA_SUCTION_M", 0.15)))
        self._ksat_source.setCurrentIndex(self._idx(self._KSAT_SOURCES, getattr(cfg, "OPM_GA_KSAT_SOURCE", "scalar")))
        self._ga_ksat.setValue(float(getattr(cfg, "OPM_GA_KSAT_MMHR", 12.0)))
        if getattr(cfg, "OPM_GA_KSAT_RASTER", None):
            self._ga_ksat_raster.setFilePath(cfg.OPM_GA_KSAT_RASTER)
        self._ga_ksat_scale.setValue(float(getattr(cfg, "OPM_GA_KSAT_SCALE", 1.0)))

        # Impervious
        self._imperv_source.setCurrentIndex(self._idx(self._IMPERVIOUS_SOURCES, getattr(cfg, "IMPERVIOUS_SOURCE", "none")))
        if getattr(cfg, "IMPERVIOUS_RASTER_PATH", None):
            self._imperv_raster.setFilePath(cfg.IMPERVIOUS_RASTER_PATH)

    def write_to_config(self, cfg):
        mode = self.get_mode()
        cfg.RUNOFF_SOURCE = mode
        cfg.RUNOFF_COEFFICIENT_PATH = self._cf_file.filePath()
        cfg.RUNOFF_RASTER_MANIFEST = self._raster_manifest.filePath()
        cfg.RUNOFF_CN_PATH = self._cn_file.filePath()
        cfg.RUNOFF_SCS_Ia_FACTOR = self._ia_factor.value()

        # Mechanisms
        cfg.RUNOFF_MECHANISMS = self._mechanisms()

        # Core
        cfg.OPM_SD_MAX_INITIAL = self._sd_max.value()
        cfg.OPM_Q_MAX = self._q_max.value()
        cfg.OPM_PHI = self._phi.value()
        cfg.OPM_K_SAT = self._k_sat.value()
        cfg.OPM_PER_POLYGON = self._per_polygon.isChecked()
        cfg.OPM_BASEFLOW = self._baseflow.isChecked()

        # SERVES / GEE
        cfg.OPM_SD_SOURCE = self._SD_SOURCES[self._sd_source.currentIndex()]
        cfg.OPM_SD_REDUCER = self._SD_REDUCERS[self._sd_reducer.currentIndex()]
        cfg.SERVES_SATELLITE = self._SATELLITES[self._satellite.currentIndex()]
        cfg.SERVES_SEARCH_WINDOW = self._search_window.value()
        cfg.OPM_SOILGRIDS_DEPTH = self._soilgrids_depth.currentText()

        # Green-Ampt
        cfg.OPM_INFILTRATION = self._INFILTRATION[self._infiltration.currentIndex()]
        cfg.OPM_GA_SUCTION_SOURCE = self._SUCTION_SOURCES[self._suction_source.currentIndex()]
        cfg.OPM_GA_SUCTION_M = self._suction_m.value()
        cfg.OPM_GA_KSAT_SOURCE = self._KSAT_SOURCES[self._ksat_source.currentIndex()]
        cfg.OPM_GA_KSAT_MMHR = self._ga_ksat.value()
        cfg.OPM_GA_KSAT_RASTER = self._ga_ksat_raster.filePath() or None
        cfg.OPM_GA_KSAT_SCALE = self._ga_ksat_scale.value()

        # Impervious
        cfg.IMPERVIOUS_SOURCE = self._IMPERVIOUS_SOURCES[self._imperv_source.currentIndex()]
        cfg.IMPERVIOUS_RASTER_PATH = self._imperv_raster.filePath() or None

    @staticmethod
    def _idx(options, value):
        return options.index(value) if value in options else 0
