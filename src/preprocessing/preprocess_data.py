import os
import sys
import numpy as np
import torch
import xarray as xr
from .constants import GtC_to_GtCO2, Gt_to_Mt

base_dir = os.path.join(os.getcwd(), '..')
sys.path.append(base_dir)

from src.fair.ancil import get_gas_params, get_thermal_params
from src.structures import Scenario


def get_fair_params():
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


def load_emissions_dataset(filepath):
    inputs = xr.open_dataset(filepath).compute().isel(latitude=slice(0, 40), longitude=slice(0, 40))
    inputs.CO2.data = inputs.CO2.data / GtC_to_GtCO2
    inputs.CO2.attrs['units'] = 'GtC'
    inputs.CH4.data = inputs.CH4.data * Gt_to_Mt
    inputs.CH4.attrs['units'] = 'MtCH4'
    inputs.SO2.data = inputs.SO2.data * Gt_to_Mt
    inputs.SO2.attrs['units'] = 'MtSO2'
    inputs.BC.data = inputs.BC.data * Gt_to_Mt
    inputs.BC.attrs['units'] = 'MtBC'
    return inputs


def load_response_dataset(filepath):
    outputs = xr.open_dataset(filepath).compute().isel(lat=slice(0, 40), lon=slice(0, 40))
    return outputs


def extract_arrays(xr_input, xr_output):
    # Extract time steps array
    time = xr_input.time.values

    # Extract cumulative emissions
    cum_CO2_emissions = xr_input.CO2.values
    cum_emissions = cum_CO2_emissions

    # Compute emissions
    CO2_emissions = np.append(np.diff(cum_CO2_emissions)[0], np.diff(cum_CO2_emissions))
    CH4_emissions = xr_input.CH4.values
    weights = np.cos(np.deg2rad(xr_input.latitude))
    SO2_emissions = xr_input.SO2.weighted(weights).mean(['latitude', 'longitude']).data
    BC_emissions = xr_input.BC.weighted(weights).mean(['latitude', 'longitude']).data
    emissions = np.stack([CO2_emissions, CH4_emissions, SO2_emissions, BC_emissions])

    # Compute average temperature anomaly
    weights = np.cos(np.deg2rad(xr_output.lat))
    tas = xr_output.tas.weighted(weights).mean(['lat', 'lon', 'member']).data
    return time, cum_emissions, emissions, tas


def make_scenario(inputs, outputs, key, hist_scenario=None):
    time, _, emission, tas = extract_arrays(inputs[key], outputs[key])
    scenario = Scenario(name=key,
                        timesteps=torch.from_numpy(time).float(),
                        emissions=torch.from_numpy(emission).float().T,
                        tas=torch.from_numpy(tas).float(),
                        hist_scenario=hist_scenario)
    return scenario
