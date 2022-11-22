import os
import sys
import numpy as np
import torch
import xarray as xr
from .constants import GtC_to_GtCO2, Gt_to_Mt

base_dir = os.path.join(os.getcwd(), '../..')
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
    inputs = xr.open_dataset(filepath).compute()
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
    outputs = xr.open_dataset(filepath).compute()
    return outputs


def make_input_array(xr_input, xr_output):
    latitude = xr_input.latitude
    longitude = xr_input.longitude
    xr_input['CO2'] = xr_input.CO2.expand_dims(latitude=latitude, longitude=longitude).transpose('time', 'latitude', 'longitude')
    xr_input['CH4'] = xr_input.CH4.expand_dims(latitude=latitude, longitude=longitude).transpose('time', 'latitude', 'longitude')
    xr_input['tas'] = xr_output.tas.mean(['member'])
    return xr_input


def extract_arrays(xr_input):
    # Extract time steps, lat and lon arrays
    time = xr_input.time.values
    lat = xr_input.latitude.values
    lon = xr_input.longitude.values

    # Extract cumulative emissions
    cum_CO2_emissions = xr_input.CO2.values
    # cum_emissions = cum_CO2_emissions[:, :10, :10]
    cum_emissions = cum_CO2_emissions

    # Compute spatial emissions
    CO2_emissions = np.append(np.diff(cum_CO2_emissions, axis=0)[0][None, ...],
                              np.diff(cum_CO2_emissions, axis=0), axis=0)
    CH4_emissions = xr_input.CH4.values
    SO2_emissions = xr_input.SO2.values
    BC_emissions = xr_input.BC.values
    emissions = np.stack([CO2_emissions, CH4_emissions, SO2_emissions, BC_emissions])
    # emissions = np.stack([CO2_emissions, CH4_emissions, SO2_emissions, BC_emissions])[:, :, :10, :10]

    # Compute spatial temperature anomaly
    tas = xr_input.tas.data
    # tas = xr_input.tas.data[:, :10, :10]
    return time, lat, lon, cum_emissions, emissions, tas


def make_scenario(key, inputs, outputs, hist_scenario=None):
    xr_input = make_input_array(inputs[key], outputs[key])
    time, lat, lon, _, emission, tas = extract_arrays(xr_input)
    scenario = Scenario(name=key,
                        timesteps=torch.from_numpy(time).float(),
                        lat=torch.from_numpy(lat).float(),
                        lon=torch.from_numpy(lon).float(),
                        emissions=torch.from_numpy(emission).float().permute(1, 2, 3, 0),
                        tas=torch.from_numpy(tas).float(),
                        hist_scenario=hist_scenario)
    return scenario
