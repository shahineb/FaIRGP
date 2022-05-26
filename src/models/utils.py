import os
import sys
from collections import namedtuple
import torch

base_dir = os.path.join(os.getcwd(), '..')
sys.path.append(base_dir)

import src.fair as fair
from src.fair.tools import step_I, step_kernel


TS = namedtuple(typename='TS',
                field_names=['times', 'emissions', 'tas', 'slices', 'scenarios'],
                defaults=(None,) * 4)


def make_stacked_timeseries(time_series):
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
    base_kwargs = fair.get_params()
    forcings = dict()
    means = dict()
    for scenario in time_series.scenarios():
        res = fair.run(time_series.times[scenario].numpy(),
                       time_series.emissions[scenario].T.numpy(),
                       base_kwargs)
        forcing, S = res['RF'], res['S']
        forcing = forcing[:, time_series.slices[scenario]]
        S = S[time_series.slices[scenario]]
        forcings.update({scenario: torch.from_numpy(forcing).T.float()})
        means.update({scenario: torch.from_numpy(S).float()})
    return forcings, means


def compute_I_scenario(stacked_ts, ts, scenario, ks, d, mu, sigma):
    scenario_emissions_std = (ts.emissions[scenario] - mu) / sigma
    stacked_ts_emissions_std = (stacked_ts.emissions - mu) / sigma
    Ks = [k(stacked_ts_emissions_std, scenario_emissions_std).evaluate() for k in ks]
    K = torch.stack(Ks, dim=-1)
    I = torch.zeros(K.shape)
    for t in range(1, len(scenario_emissions_std)):
        I_old = I[:, t - 1]
        K_new = K[:, t]
        I_new = step_I(I_old, K_new, d.unsqueeze(0))
        I[:, t] = I_new.squeeze()
    return I


def compute_I(stacked_ts, ts, ks, d, mu, sigma):
    I = [compute_I_scenario(stacked_ts, ts, s, ks, d, mu, sigma) for s in ts.scenarios()]
    I = torch.cat(I, dim=-2)
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
    Kj = [Kj[:, stacked_ts.slices[s]] for s in ts.scenarios()]
    Kj = torch.cat(Kj, dim=-2)
    # Remove hist rows used for iterative scheme only
    Kj = Kj[ts.slices[scenario]]
    return Kj


def compute_covariance_scenario(I, stacked_ts, ts, scenario, q, d):
    stacked_ts_slice, ts_slice = get_slices(stacked_ts, ts, scenario)
    size = stacked_ts_slice.stop - stacked_ts_slice.start
    I_ts = I[stacked_ts_slice]
    Kj = torch.zeros_like(I_ts)
    for t in range(1, size):
        Kj_old = Kj[t - 1]
        I_new = I_ts[t]
        Kj_new = step_kernel(Kj_old, I_new, q.unsqueeze(0), d.unsqueeze(0))
        Kj[t] = Kj_new
    Kj = trim_Kj(Kj, stacked_ts, ts, scenario)
    return Kj.permute(1, 0, 2)


def compute_covariance(I, stacked_ts, ts, q, d):
    Kj = [compute_covariance_scenario(I, stacked_ts, ts, s, q, d) for s in ts.scenarios()]
    Kj = torch.cat(Kj, dim=-2)
    return Kj


def add_ts(ts1, ts2):
    times = {**ts1.times, **ts2.times}
    emissions = {**ts1.emissions, **ts2.emissions}
    slices = {**ts1.slices, **ts2.slices}
    tas = {**ts1.tas, **ts2.tas}
    scenarios = times.keys
    ts = TS(times=times,
            emissions=emissions,
            slices=slices,
            tas=tas,
            scenarios=scenarios)
    return ts
