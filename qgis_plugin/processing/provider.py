# -*- coding: utf-8 -*-
"""
provider.py
===========
OpmProcessingProvider — registers all VSA-OPM algorithms with the
QGIS Processing Framework.

Registered algorithms appear under:
  Processing Toolbox → VSA-OPM Hydrological Model
    ├─ 1. DEM Pre-processing
    ├─ 2. Kinematic-Wave Routing
    └─ 3. VSA-OPM Standalone

Adding new algorithms later
---------------------------
1. Create a new alg_*.py in this package.
2. Import the class here and add an instance to _algorithms().
"""

import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .alg_process_dem import ProcessDemAlgorithm
from .alg_router import KinematicWaveAlgorithm

_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class OpmProcessingProvider(QgsProcessingProvider):
    """QGIS Processing provider for the VSA-OPM model."""

    def id(self):  # noqa: A003
        return "vsaopm"

    def name(self):
        return "VSA-OPM Hydrological Model"

    def longName(self):  # noqa: N802
        return "VSA-OPM  —  Variable Source Area One-Parameter Model"

    def icon(self):
        icon_path = os.path.join(_PLUGIN_DIR, "resources", "icon.png")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return super().icon()

    def loadAlgorithms(self):  # noqa: N802
        """Register all algorithms.  Called once by QGIS at provider load."""
        for alg in self._algorithms():
            self.addAlgorithm(alg)

    @staticmethod
    def _algorithms():
        return [
            ProcessDemAlgorithm(),
            KinematicWaveAlgorithm(),
        ]
