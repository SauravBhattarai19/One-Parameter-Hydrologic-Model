# -*- coding: utf-8 -*-
"""
tab_precip.py
=============
Tab 2 — Precipitation Engine.

Methods
-------
  uniform         constant intensity + duration
  thiessen        Voronoi nearest-gauge weighting (CSV gauges)
  idw             Inverse Distance Weighting (CSV gauges)
  imerg_thiessen  NASA GPM IMERG V07 from GEE, Thiessen weighting
  imerg_idw       same IMERG source, IDW weighting

The IMERG event window is derived from the "Event start (UTC)" set on the
DEM & Watershed tab plus the simulation duration on the Routing tab, so the
IMERG panel here only carries the weighting/download knobs.
"""

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QComboBox,
    QDoubleSpinBox, QLabel, QStackedWidget, QCheckBox,
)
from qgis.gui import QgsFileWidget


class TabPrecip(QWidget):
    """Precipitation engine configuration tab."""

    _METHODS = ["uniform", "thiessen", "idw", "imerg_thiessen", "imerg_idw"]
    _METHOD_LABELS = [
        "Uniform (constant rate)",
        "Thiessen polygons (gauge CSV)",
        "Inverse Distance Weighting — IDW (gauge CSV)",
        "IMERG V07 satellite — Thiessen (Earth Engine)",
        "IMERG V07 satellite — IDW (Earth Engine)",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Method selector ───────────────────────────────────────────────────
        grp_method = QGroupBox("Precipitation Method")
        form_method = QFormLayout(grp_method)

        self.method_combo = QComboBox()
        self.method_combo.addItems(self._METHOD_LABELS)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form_method.addRow("Method:", self.method_combo)

        self.exclude_outside = QCheckBox(
            "Exclude gauge/pixel centroids that fall outside the watershed"
        )
        self.exclude_outside.setToolTip(
            "False (default): keep all stations; boundary cells use the nearest\n"
            "outside station.  True: drop outside stations entirely."
        )
        form_method.addRow(self.exclude_outside)

        root.addWidget(grp_method)

        # ── Stacked panel (one per method) ────────────────────────────────────
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_uniform_panel())          # 0 uniform
        self._stack.addWidget(self._build_gauge_panel("thiessen"))  # 1 thiessen
        self._stack.addWidget(self._build_gauge_panel("idw"))       # 2 idw
        self._stack.addWidget(self._build_imerg_panel("imerg_thiessen"))  # 3
        self._stack.addWidget(self._build_imerg_panel("imerg_idw"))       # 4

        root.addWidget(self._stack)
        root.addStretch()

    def _build_uniform_panel(self):
        w = QGroupBox("Uniform Rainfall Parameters")
        form = QFormLayout(w)

        self.intensity_spin = QDoubleSpinBox()
        self.intensity_spin.setRange(0.0, 9999.0)
        self.intensity_spin.setDecimals(2)
        self.intensity_spin.setValue(20.0)
        self.intensity_spin.setSuffix(" mm/hr")
        form.addRow("Rainfall intensity:", self.intensity_spin)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.0, 9999.0)
        self.duration_spin.setDecimals(2)
        self.duration_spin.setValue(3.0)
        self.duration_spin.setSuffix(" hours")
        form.addRow("Rainfall duration:", self.duration_spin)

        return w

    def _build_gauge_panel(self, method):
        w = QGroupBox("Gauge-Based Rainfall Parameters")
        form = QFormLayout(w)

        gauge_file = QgsFileWidget()
        gauge_file.setStorageMode(QgsFileWidget.GetFile)
        gauge_file.setFilter("CSV files (*.csv);;All files (*)")
        gauge_file.setDialogTitle("Select gauge metadata CSV (gauge_id, name, easting_m, northing_m)")
        form.addRow("Gauge metadata CSV:", gauge_file)

        ts_file = QgsFileWidget()
        ts_file.setStorageMode(QgsFileWidget.GetFile)
        ts_file.setFilter("CSV files (*.csv);;All files (*)")
        ts_file.setDialogTitle("Select timeseries CSV (time_s, G01, G02, …)")
        form.addRow("Timeseries CSV:", ts_file)

        power_spin = None
        if method == "idw":
            power_spin = QDoubleSpinBox()
            power_spin.setRange(0.1, 10.0)
            power_spin.setDecimals(1)
            power_spin.setValue(2.0)
            power_spin.setToolTip("IDW distance exponent p (standard: 2)")
            form.addRow("IDW power (p):", power_spin)

        if method == "thiessen":
            self._thiessen_gauge = gauge_file
            self._thiessen_ts = ts_file
        else:
            self._idw_gauge = gauge_file
            self._idw_ts = ts_file
            self._idw_power = power_spin

        return w

    def _build_imerg_panel(self, method):
        w = QGroupBox("IMERG Satellite Rainfall (NASA GPM V07 via Earth Engine)")
        form = QFormLayout(w)

        note = QLabel(
            "Downloads IMERG V07 pixels as pseudo-gauges over the watershed.\n"
            "Requires a GEE project + event start date (set on the DEM tab).\n"
            "The window is EVENT_START_UTC … +simulation-duration."
        )
        note.setWordWrap(True)
        form.addRow(note)

        force = QCheckBox("Force re-download (ignore cached IMERG CSVs)")
        form.addRow(force)

        power_spin = None
        if method == "imerg_idw":
            power_spin = QDoubleSpinBox()
            power_spin.setRange(0.1, 10.0)
            power_spin.setDecimals(1)
            power_spin.setValue(2.0)
            power_spin.setToolTip("IDW distance exponent p (standard: 2)")
            form.addRow("IDW power (p):", power_spin)

        if method == "imerg_thiessen":
            self._imerg_th_force = force
        else:
            self._imerg_idw_force = force
            self._imerg_idw_power = power_spin

        return w

    # ── Slot ─────────────────────────────────────────────────────────────────

    def _on_method_changed(self, idx):
        self._stack.setCurrentIndex(idx)

    # ── Public getters ────────────────────────────────────────────────────────

    def get_method(self) -> str:
        return self._METHODS[self.method_combo.currentIndex()]

    # ── Config I/O ────────────────────────────────────────────────────────────

    def apply_config(self, cfg):
        idx = self._METHODS.index(cfg.PRECIP_METHOD) if cfg.PRECIP_METHOD in self._METHODS else 0
        self.method_combo.setCurrentIndex(idx)
        self.exclude_outside.setChecked(bool(getattr(cfg, "PRECIP_EXCLUDE_OUTSIDE_STATIONS", False)))
        self.intensity_spin.setValue(cfg.RAIN_INTENSITY_MM_HR)
        self.duration_spin.setValue(cfg.RAIN_DURATION_HOURS)
        if cfg.PRECIP_GAUGE_FILE:
            self._thiessen_gauge.setFilePath(cfg.PRECIP_GAUGE_FILE)
            self._idw_gauge.setFilePath(cfg.PRECIP_GAUGE_FILE)
        if cfg.PRECIP_TIMESERIES_FILE:
            self._thiessen_ts.setFilePath(cfg.PRECIP_TIMESERIES_FILE)
            self._idw_ts.setFilePath(cfg.PRECIP_TIMESERIES_FILE)
        self._idw_power.setValue(cfg.PRECIP_IDW_POWER)
        self._imerg_idw_power.setValue(cfg.PRECIP_IDW_POWER)
        force = bool(getattr(cfg, "PRECIP_IMERG_FORCE_DOWNLOAD", False))
        self._imerg_th_force.setChecked(force)
        self._imerg_idw_force.setChecked(force)

    def write_to_config(self, cfg):
        method = self.get_method()
        cfg.PRECIP_METHOD = method
        cfg.PRECIP_EXCLUDE_OUTSIDE_STATIONS = self.exclude_outside.isChecked()
        cfg.RAIN_INTENSITY_MM_HR = self.intensity_spin.value()
        cfg.RAIN_DURATION_HOURS = self.duration_spin.value()

        if method == "thiessen":
            cfg.PRECIP_GAUGE_FILE = self._thiessen_gauge.filePath()
            cfg.PRECIP_TIMESERIES_FILE = self._thiessen_ts.filePath()
        elif method == "idw":
            cfg.PRECIP_GAUGE_FILE = self._idw_gauge.filePath()
            cfg.PRECIP_TIMESERIES_FILE = self._idw_ts.filePath()
            cfg.PRECIP_IDW_POWER = self._idw_power.value()
        elif method == "imerg_thiessen":
            cfg.PRECIP_IMERG_FORCE_DOWNLOAD = self._imerg_th_force.isChecked()
        elif method == "imerg_idw":
            cfg.PRECIP_IMERG_FORCE_DOWNLOAD = self._imerg_idw_force.isChecked()
            cfg.PRECIP_IDW_POWER = self._imerg_idw_power.value()
