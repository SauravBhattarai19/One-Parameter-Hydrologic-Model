# -*- coding: utf-8 -*-
"""
opm_plugin.py
=============
Main plugin class.  Registered with QGIS via classFactory in __init__.py.

Responsibilities
----------------
- Add toolbar button + menu item to open the main dialog.
- Register / deregister the Processing provider.
- Keep a reference to the main dialog so it can be shown/hidden.
"""

import os

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis.core import QgsApplication

from .processing.provider import OpmProcessingProvider


# Absolute path to THIS file's directory (= plugin root)
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))


class OpmPlugin:
    """QGIS Plugin class for the VSA-OPM Hydrological Model."""

    def __init__(self, iface):
        """
        Parameters
        ----------
        iface : QgisInterface
        """
        self.iface = iface
        self._action = None
        self._dialog = None
        self._provider = OpmProcessingProvider()

    # ── QGIS lifecycle ────────────────────────────────────────────────────────

    def initGui(self):  # noqa: N802
        """Called by QGIS when the plugin is loaded (after __init__)."""
        # ── Toolbar / menu action ────────────────────────────────────────────
        icon_path = os.path.join(PLUGIN_DIR, "resources", "icon.png")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()

        self._action = QAction(icon, "VSA-OPM Hydrological Model", self.iface.mainWindow())
        self._action.setToolTip(
            "Open the VSA-OPM kinematic-wave hydrological modelling dialog"
        )
        self._action.triggered.connect(self._open_dialog)

        # Add to Plugins menu and toolbar
        self.iface.addToolBarIcon(self._action)
        self.iface.addPluginToMenu("&VSA-OPM", self._action)

        # ── Register Processing provider ──────────────────────────────────────
        QgsApplication.processingRegistry().addProvider(self._provider)

    def unload(self):
        """Called by QGIS when the plugin is unloaded."""
        # Remove menu / toolbar
        self.iface.removePluginMenu("&VSA-OPM", self._action)
        self.iface.removeToolBarIcon(self._action)

        # Close dialog if open
        if self._dialog is not None:
            self._dialog.close()
            self._dialog = None

        # Deregister Processing provider
        QgsApplication.processingRegistry().removeProvider(self._provider)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _open_dialog(self):
        """Show (or bring to front) the main modelling dialog."""
        # Lazy import so QGIS loads faster
        from .ui.main_dialog import OpmMainDialog

        if self._dialog is None:
            self._dialog = OpmMainDialog(self.iface, parent=self.iface.mainWindow())
            # When the dialog is closed, clear the reference so next open
            # creates a fresh instance (state is reset).
            self._dialog.finished.connect(self._on_dialog_closed)

        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()

    def _on_dialog_closed(self):
        """Slot called when the dialog emits finished() (user closes it)."""
        self._dialog = None
