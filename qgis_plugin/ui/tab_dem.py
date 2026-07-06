# -*- coding: utf-8 -*-
"""
tab_dem.py
==========
Tab 1 — DEM & Watershed Setup.

Widgets
-------
- DEM file picker  (QgsFileWidget)
- CRS selector     (QgsProjectionSelectionWidget)
- Outlet point: lat/lon spinboxes  +  "Pick from map" button
- Output directory (QgsFileWidget, folder mode)
- [Run DEM Processing] button (runs process_dem only)
"""

import os

from qgis.PyQt.QtWidgets import (
    QWidget, QFormLayout, QGroupBox, QVBoxLayout, QHBoxLayout,
    QPushButton, QDoubleSpinBox, QLabel, QSizePolicy,
    QLineEdit, QCheckBox, QDateTimeEdit,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal, QDateTime
from qgis.gui import QgsFileWidget, QgsProjectionSelectionWidget, QgsMapToolEmitPoint
from qgis.core import QgsCoordinateReferenceSystem, QgsPointXY


class TabDem(QWidget):
    """DEM & Watershed configuration tab."""

    # Emitted when the user picks a point on the map
    outlet_picked = pyqtSignal(float, float)   # lat, lon

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self._iface = iface
        self._map_tool = None   # QgsMapToolEmitPoint instance
        self._prev_tool = None  # map tool active before picking
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── DEM Input ────────────────────────────────────────────────────────
        grp_dem = QGroupBox("Digital Elevation Model")
        form_dem = QFormLayout(grp_dem)

        self.dem_widget = QgsFileWidget()
        self.dem_widget.setStorageMode(QgsFileWidget.GetFile)
        self.dem_widget.setFilter("GeoTIFF (*.tif *.tiff);;All files (*)")
        self.dem_widget.setDialogTitle("Select DEM raster")
        form_dem.addRow("DEM file:", self.dem_widget)

        self.crs_widget = QgsProjectionSelectionWidget()
        self.crs_widget.setCrs(QgsCoordinateReferenceSystem("EPSG:32645"))
        form_dem.addRow("Target CRS:", self.crs_widget)

        root.addWidget(grp_dem)

        # ── Outlet Point ─────────────────────────────────────────────────────
        grp_outlet = QGroupBox("Watershed Outlet Point")
        v_outlet = QVBoxLayout(grp_outlet)

        coord_row = QHBoxLayout()

        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setRange(-90.0, 90.0)
        self.lat_spin.setDecimals(6)
        self.lat_spin.setValue(27.632222)
        self.lat_spin.setSuffix("°  (lat)")

        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setRange(-180.0, 180.0)
        self.lon_spin.setDecimals(6)
        self.lon_spin.setValue(85.293333)
        self.lon_spin.setSuffix("°  (lon)")

        self.pick_btn = QPushButton("📍  Pick from map")
        self.pick_btn.setToolTip(
            "Click on the QGIS map canvas to set the outlet point.\n"
            "The map must be in a geographic CRS (EPSG:4326) or the\n"
            "coordinates will be reprojected automatically."
        )
        self.pick_btn.setCheckable(True)
        self.pick_btn.clicked.connect(self._toggle_map_pick)

        coord_row.addWidget(QLabel("Latitude:"))
        coord_row.addWidget(self.lat_spin)
        coord_row.addSpacing(10)
        coord_row.addWidget(QLabel("Longitude:"))
        coord_row.addWidget(self.lon_spin)
        coord_row.addSpacing(10)
        coord_row.addWidget(self.pick_btn)
        coord_row.addStretch()

        v_outlet.addLayout(coord_row)
        root.addWidget(grp_outlet)

        # ── Output Directory ─────────────────────────────────────────────────
        grp_out = QGroupBox("Output")
        form_out = QFormLayout(grp_out)

        self.output_dir_widget = QgsFileWidget()
        self.output_dir_widget.setStorageMode(QgsFileWidget.GetDirectory)
        self.output_dir_widget.setDialogTitle("Select output directory")
        # Default to OPM repo root / output
        default_out = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "output"
        )
        self.output_dir_widget.setFilePath(default_out)
        form_out.addRow("Output directory:", self.output_dir_widget)

        root.addWidget(grp_out)

        # ── Event & Earth Engine (needed for IMERG / SERVES) ─────────────────
        grp_evt = QGroupBox("Event & Earth Engine  (for IMERG rainfall / SERVES soil moisture)")
        form_evt = QFormLayout(grp_evt)

        self.use_event = QCheckBox("Set an event start date (UTC)")
        self.use_event.setToolTip(
            "Required for IMERG rainfall and SERVES soil-moisture deficit.\n"
            "The IMERG download window and the SERVES antecedent-moisture date\n"
            "are both derived from this single timestamp."
        )
        self.use_event.toggled.connect(self._on_use_event_toggled)
        form_evt.addRow(self.use_event)

        self.event_dt = QDateTimeEdit()
        self.event_dt.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.event_dt.setCalendarPopup(True)
        self.event_dt.setDateTime(QDateTime.currentDateTimeUtc())
        self.event_dt.setEnabled(False)
        form_evt.addRow("Event start (UTC):", self.event_dt)

        self.utc_offset = QDoubleSpinBox()
        self.utc_offset.setRange(-12.0, 14.0)
        self.utc_offset.setDecimals(2)
        self.utc_offset.setValue(5.75)   # Nepal Standard Time (UTC+5:45)
        self.utc_offset.setSuffix(" h")
        self.utc_offset.setToolTip(
            "UTC offset for local-time conversion of the IMERG window.\n"
            "Nepal Standard Time = UTC + 5:45 → 5.75.  Use 0 to work in UTC."
        )
        form_evt.addRow("UTC offset:", self.utc_offset)

        self.gee_project = QLineEdit()
        self.gee_project.setPlaceholderText("ee-yourusername  (or leave blank to use the GEE_PROJECT env var)")
        self.gee_project.setToolTip(
            "Google Earth Engine cloud project ID.\n"
            "Required for IMERG rainfall, SERVES deficit, gridded Ksat, and\n"
            "LULC/LCZ downloads.  Authenticate GEE once in the QGIS Python\n"
            "console (import ee; ee.Authenticate()) or place a key.json beside\n"
            "the plugin's serves_gee.py."
        )
        form_evt.addRow("GEE project:", self.gee_project)

        root.addWidget(grp_evt)
        root.addStretch()

    def _on_use_event_toggled(self, on):
        self.event_dt.setEnabled(on)

    # ── Map-pick tool ─────────────────────────────────────────────────────────

    def _toggle_map_pick(self, checked):
        canvas = self._iface.mapCanvas()
        if checked:
            self._prev_tool = canvas.mapTool()
            self._map_tool = QgsMapToolEmitPoint(canvas)
            self._map_tool.canvasClicked.connect(self._on_canvas_click)
            canvas.setMapTool(self._map_tool)
            self._iface.messageBar().pushMessage(
                "VSA-OPM", "Click on the map to set the outlet point.", duration=5
            )
        else:
            canvas.setMapTool(self._prev_tool)
            self._map_tool = None

    def _on_canvas_click(self, point: QgsPointXY, button):
        """Convert clicked map point to WGS-84 lat/lon."""
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsProject,
        )
        map_crs = self._iface.mapCanvas().mapSettings().destinationCrs()
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        if map_crs != wgs84:
            xform = QgsCoordinateTransform(map_crs, wgs84, QgsProject.instance())
            point = xform.transform(point)

        self.lat_spin.setValue(point.y())
        self.lon_spin.setValue(point.x())
        self.outlet_picked.emit(point.y(), point.x())

        # Un-check the button and restore previous tool
        self.pick_btn.setChecked(False)
        self._toggle_map_pick(False)

    # ── Public getters ────────────────────────────────────────────────────────

    def get_dem_path(self) -> str:
        return self.dem_widget.filePath()

    def get_target_crs(self) -> str:
        return self.crs_widget.crs().authid()

    def get_outlet_point(self) -> tuple:
        return (self.lat_spin.value(), self.lon_spin.value())

    def get_output_dir(self) -> str:
        return self.output_dir_widget.filePath()

    # ── Populate from config ──────────────────────────────────────────────────

    def apply_config(self, cfg):
        """Populate widgets from an OpmConfig object."""
        if cfg.DEM_PATH:
            self.dem_widget.setFilePath(cfg.DEM_PATH)
        self.crs_widget.setCrs(QgsCoordinateReferenceSystem(cfg.TARGET_CRS_EPSG))
        lat, lon = cfg.OUTPUT_POINT
        self.lat_spin.setValue(lat)
        self.lon_spin.setValue(lon)
        if cfg.OUTPUT_DIR:
            self.output_dir_widget.setFilePath(cfg.OUTPUT_DIR)

        # Event & GEE
        if getattr(cfg, "EVENT_START_UTC", None):
            self.use_event.setChecked(True)
            dt = QDateTime.fromString(str(cfg.EVENT_START_UTC).strip(), "yyyy-MM-dd HH:mm")
            if dt.isValid():
                self.event_dt.setDateTime(dt)
        else:
            self.use_event.setChecked(False)
        self.utc_offset.setValue(float(getattr(cfg, "IMERG_UTC_OFFSET_HOURS", 5.75)))
        if getattr(cfg, "GEE_PROJECT", None):
            self.gee_project.setText(str(cfg.GEE_PROJECT))

    def write_to_config(self, cfg):
        """Write widget values into an OpmConfig object."""
        cfg.DEM_PATH = self.get_dem_path()
        cfg.TARGET_CRS_EPSG = self.get_target_crs()
        cfg.OUTPUT_POINT = self.get_outlet_point()
        cfg.OUTPUT_DIR = self.get_output_dir()
        cfg.update_output_paths()

        # Event & GEE
        if self.use_event.isChecked():
            cfg.EVENT_START_UTC = self.event_dt.dateTime().toString("yyyy-MM-dd HH:mm")
        else:
            cfg.EVENT_START_UTC = None
        cfg.IMERG_UTC_OFFSET_HOURS = self.utc_offset.value()
        proj = self.gee_project.text().strip()
        cfg.GEE_PROJECT = proj or None
