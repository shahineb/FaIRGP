
import numpy as np
from .ancil import get_gas_params, get_thermal_params
from .forward import _run


def get_params():
    gas_params_df = get_gas_params()
    gas_kwargs = {k: np.asarray(list(v.values()))
                  for k, v in gas_params_df.T.to_dict().items()}
    thermal_params_df = get_thermal_params()
    d = thermal_params_df.T.d.values
    q = thermal_params_df.T.q.values
    base_kwargs = {'d': d,
                   'q': q,
                   **gas_kwargs}
    return base_kwargs


def run(time, emission, base_kwargs):
    timestep = np.append(np.diff(time), np.diff(time)[-1])
    ext_forcing = np.zeros_like(time)
    run_kwargs = {'inp_ar': emission,
                  'timestep': timestep,
                  'ext_forcing': ext_forcing,
                  **base_kwargs}
    res = _run(**run_kwargs)
    return res
