import os
import sys
from collections import namedtuple
import numpy as np
import torch
from gpytorch import lazy
from tqdm import tqdm
from joblib import Parallel, delayed

base_dir = os.path.join(os.getcwd(), '..')
sys.path.append(base_dir)

import src.fair as fair
from src.fair.tools import step_I, step_kernel
from src.fair.ancil import get_gas_params, get_thermal_params
from src.structures import Scenario, ScenarioDataset


def make_input_array(xr_input, xr_output):
    latitude = xr_input.latitude
    longitude = xr_input.longitude
    xr_input['CO2'] = xr_input.CO2.expand_dims(latitude=latitude, longitude=longitude).transpose('time', 'latitude', 'longitude')
    xr_input['CH4'] = xr_input.CH4.expand_dims(latitude=latitude, longitude=longitude).transpose('time', 'latitude', 'longitude')
    xr_input['tas'] = xr_output.tas.mean(['member'])
    return xr_input


def extract_arrays(xr_input):
    # Extract time steps array
    time = xr_input.time.values

    # Extract cumulative emissions
    cum_CO2_emissions = xr_input.CO2.values
    # cum_emissions = cum_CO2_emissions[:, :10, :10]
    cum_emissions = cum_CO2_emissions

    # Compute emissions
    CO2_emissions = np.append(np.diff(cum_CO2_emissions, axis=0)[0][None, ...],
                              np.diff(cum_CO2_emissions, axis=0), axis=0)
    CH4_emissions = xr_input.CH4.values
    SO2_emissions = xr_input.SO2.values
    BC_emissions = xr_input.BC.values
    emissions = np.stack([CO2_emissions, CH4_emissions, SO2_emissions, BC_emissions])
    # emissions = np.stack([CO2_emissions, CH4_emissions, SO2_emissions, BC_emissions])[:, :, :10, :10]

    # Compute temperature anomaly
    tas = xr_input.tas.data
    # tas = xr_input.tas.data[:, :10, :10]
    return time, cum_emissions, emissions, tas


def make_scenario(inputs, outputs, name, hist_scenario=None):
    xr_input = make_input_array(inputs[name], outputs[name])
    time, _, emission, tas = extract_arrays(xr_input)
    scenario = Scenario(name=name,
                        timesteps=torch.from_numpy(time).float(),
                        emissions=torch.from_numpy(emission).float().permute(1, 2, 3, 0),
                        tas=torch.from_numpy(tas).float(),
                        hist_scenario=hist_scenario)
    return scenario


def compute_means(scenario_dataset, d_map, q_map):
    base_kwargs = fair.get_params()
    means = dict()
    for name, scenario in scenario_dataset.scenarios.items():
        def run(d, q):
            base_kwargs.update({'d': d, 'q': q})
            res = fair.run(scenario.full_timesteps.numpy(),
                           scenario.full_emissions.T.numpy(),
                           base_kwargs)
            pixel = res['S']
            pixel = scenario.trim_hist(pixel)
            return pixel
        S = Parallel(n_jobs=4)(delayed(run)(d, q) for (d, q) in zip(d_map, q_map))
        # for i, (d, q) in enumerate(tqdm(zip(d_map, q_map), total=len(d_map))):
        #     base_kwargs.update({'d': d, 'q': q})
        #     res = fair.run(scenario.full_timesteps.numpy(),
        #                    scenario.full_emissions.T.numpy(),
        #                    base_kwargs)
        #     pixel = res['S']
        #     pixel = scenario.trim_hist(pixel)
        #     S.append(pixel)
        S = np.asarray(S)
        means.update({scenario: torch.from_numpy(S).float()})
    return means
