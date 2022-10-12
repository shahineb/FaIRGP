import os
import sys
from collections import namedtuple
import numpy as np
import torch
from gpytorch import lazy

base_dir = os.path.join(os.getcwd(), '..')
sys.path.append(base_dir)

from src.fair import forward
from src.fair.tools import step_I, step_kernel
from src.fair.ancil import get_gas_params, get_thermal_params


# Extrat emissions and temperature anomaly arrays
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


def get_fair_params():
    # Get FaIR parameters
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


def run_fair_forward(time, emission, base_kwargs):
    timestep = np.append(np.diff(time), np.diff(time)[-1])
    ext_forcing = np.zeros_like(time)
    run_kwargs = {'inp_ar': emission,
                  'timestep': timestep,
                  'ext_forcing': ext_forcing,
                  **base_kwargs}
    res = forward._run(**run_kwargs)
    forcing = res['RF']
    S = res['S']
    return forcing, S


def add_hist_to_ssp(emissions, times, tass, scenario):
    assert times[scenario][0] > times['historical'][-1]
    full_times = np.concatenate([times['historical'], times[scenario]])
    full_emissions = np.hstack([emissions['historical'], emissions[scenario]])
    full_tas = np.concatenate([tass['historical'], tass[scenario]])
    ssp_slice = slice(len(times['historical']), len(times['historical']) + len(times[scenario]))
    return full_times, full_emissions, full_tas, ssp_slice


TS = namedtuple(typename='TS',
                field_names=['times', 'emissions', 'tas', 'slices', 'scenarios'],
                defaults=(None,) * 4)


def make_emissions_timeseries(times, emissions, tass):
    times_ts = dict()
    emissions_ts = dict()
    tas_ts = dict()
    slices_ts = dict()
    for key in times.keys():
        if key == 'historical':
            time, emission, tas = times['historical'], emissions['historical'], tass['historical']
            output_slice = slice(0, emission.shape[-1])
        else:
            time, emission, tas, output_slice = add_hist_to_ssp(emissions, times, tass, key)
        times_ts.update({key: torch.from_numpy(time).float()})
        emissions_ts.update({key: torch.from_numpy(emission).T.float()})
        tas_ts.update({key: torch.from_numpy(tas).float()})
        slices_ts.update({key: output_slice})
    time_series = TS(times=times_ts, emissions=emissions_ts, tas=tas_ts, slices=slices_ts, scenarios=times.keys)
    return time_series


def make_stacked_emissions_timeseries(time_series):
    scenarios = time_series.scenarios()
    times = torch.cat([time_series.times[s] for s in scenarios])
    emissions = torch.cat([time_series.emissions[s] for s in scenarios])
    tas = torch.cat([time_series.tas[s] for s in scenarios])
    slices = dict()
    idx = 0
    for scenario in time_series.scenarios():
        if scenario == 'historical':
            scenario_slice = slice(idx, idx + len(time_series.times['historical']))
            slices.update({'historical': scenario_slice})
        else:
            start = time_series.slices[scenario].start
            stop = time_series.slices[scenario].stop
            hist_and_ssp_slice = slice(idx, idx + stop)
            ssp_only_slice = slice(idx + start, idx + stop)
            slices.update({'hist+' + scenario: hist_and_ssp_slice,
                            scenario: ssp_only_slice})
        idx += len(time_series.times[scenario])

    stacked_time_series = TS(times=times, emissions=emissions, tas=tas, slices=slices, scenarios=slices.keys)
    return stacked_time_series


def compute_mean(time_series):
    base_kwargs = get_fair_params()
    forcings = dict()
    means = dict()
    for scenario in time_series.scenarios():
        forcing, S = run_fair_forward(time_series.times[scenario].numpy(),
                                      time_series.emissions[scenario].T.numpy(),
                                      base_kwargs)
        forcing = forcing[:, time_series.slices[scenario]]
        S = S[time_series.slices[scenario]]
        forcings.update({scenario: torch.from_numpy(forcing).T.float()})
        means.update({scenario: torch.from_numpy(S).float()})
    return forcings, means


def compute_I_scenario(stacked_ts, ts, scenario, ks, d):
    scenario_emissions = ts.emissions[scenario]
    Ks = [k(stacked_ts.emissions, scenario_emissions).unsqueeze(0) for k in ks]
    K = lazy.CatLazyTensor(*Ks, dim=0)
    I = torch.zeros(K.shape)
    for t in range(1, len(scenario_emissions)):
        I_old = I[:, :, t - 1]
        K_new = K[:, :, t]
        I_new = step_I(I_old, K_new, d.unsqueeze(-1))
        I[:, :, t] = I_new.squeeze()
    return I


def compute_I(stacked_ts, ts, ks, d):
    I = [compute_I_scenario(stacked_ts, ts, s, ks, d) for s in ts.scenarios()]
    I = torch.cat(I, dim=-1)
    return I


def get_slices(stacked_ts, ts, scenario):
    if scenario == 'historical':
        stacked_scenario = scenario
    else:
        stacked_scenario = 'hist+' + scenario
    stacked_ts_slice = stacked_ts.slices[stacked_scenario]
    ts_slice = ts.slices[scenario]
    return stacked_ts_slice, ts_slice


def trim_Kj(Kj, stacked_ts, ts, scenario):
    # Remove hist columns used for iterative scheme only
    Kj = [Kj[..., stacked_ts.slices[s]] for s in ts.scenarios()]
    Kj = torch.cat(Kj, dim=-1)
    # Remove hist rows used for iterative scheme only
    Kj = Kj[:, ts.slices[scenario]]
    return Kj


def compute_covariance_scenario(I, stacked_ts, ts, scenario, q, d):
    stacked_ts_slice, ts_slice = get_slices(stacked_ts, ts, scenario)
    size = stacked_ts_slice.stop - stacked_ts_slice.start
    I_ts = I[:, stacked_ts_slice]
    Kj = torch.zeros_like(I_ts)
    for t in range(1, size):
        Kj_old = Kj[:, t - 1]
        I_new = I_ts[:, t]
        Kj_new = step_kernel(Kj_old, I_new, q.unsqueeze(-1), d.unsqueeze(-1))
        Kj[:, t] = Kj_new
    Kj = trim_Kj(Kj, stacked_ts, ts, scenario)
    return Kj.permute(0, 2, 1)


def compute_covariance(I, stacked_ts, ts, q, d):
    Kj = [compute_covariance_scenario(I, stacked_ts, ts, s, q, d) for s in ts.scenarios()]
    Kj = torch.cat(Kj, dim=-1)
    return Kj






























###########
