from .exact_gp import ExactGP
from .thermalboxes_gp import ThermalBoxesGP
from .spatial_exact_gp import SpatialExactGP
from .spatial_thermalboxes_gp import SpatialThermalBoxesGP
from .multi_exact_gp import MultiExactGP
from .multi_thermalboxes_gp import MultiThermalBoxesGP
from .multi_thermalboxes_gp_simple import SimpleMultiThermalBoxesGP

__all__ = ['ExactGP', 'ThermalBoxesGP', 'SpatialExactGP', 'SpatialThermalBoxesGP',
           'MultiExactGP', 'SimpleMultiThermalBoxesGP', 'MultiThermalBoxesGP']
