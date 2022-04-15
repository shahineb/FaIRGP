from .gas_parameters import get_gas_params
from .thermal_parameters import get_thermal_params
from .units import units_dict

thermal_params_df = get_thermal_params()
d = thermal_params_df.T.d.values
q = thermal_params_df.T.q.values
