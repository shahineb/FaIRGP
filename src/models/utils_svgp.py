import numpy as np
import torch


def compute_mean(scenario, FaIR_model, d_map, q_map):
    timestep = torch.cat([torch.ones(1), torch.diff(scenario.full_timesteps)])
    emissions = scenario.full_glob_emissions.T
    weights = torch.cos(torch.deg2rad(scenario.lat.double()))
    res = FaIR_model(emissions, timestep, d_map, q_map, weights)
    T = scenario.trim_hist(res['T'].float())
    return T


def compute_means(scenario_dataset, FaIR_model, d_map, q_map):
    means = dict()
    for name, scenario in scenario_dataset.scenarios.items():
        T = compute_mean(scenario, FaIR_model, d_map, q_map)
        means.update({name: T})
    return means


def compute_Kxx(scenario,
                time_idx,
                lat_idx,
                lon_idx,
                kernel,
                d_map,
                q_map,
                mu,
                sigma):
    I = compute_I_scenario(scenario1=scenario,
                           scenario2=scenario,
                           time_idx1=time_idx,
                           time_idx2=time_idx,
                           lat_idx1=lat_idx,
                           lat_idx2=lat_idx,
                           lon_idx1=lon_idx,
                           lon_idx2=lon_idx,
                           kernel=kernel,
                           d_map=d_map,
                           mu=mu,
                           sigma=sigma)
    covar = compute_covariance_scenario(scenario1=scenario,
                                        scenario2=scenario,
                                        time_idx1=time_idx,
                                        time_idx2=time_idx,
                                        lat_idx1=lat_idx,
                                        lat_idx2=lat_idx,
                                        lon_idx1=lon_idx,
                                        lon_idx2=lon_idx,
                                        I=I,
                                        d_map=d_map,
                                        q_map=q_map)
    n = len(time_idx) * len(lat_idx) * len(lon_idx)
    covar = covar.permute(0, 4, 5, 1, 2, 3).reshape(n, -1)
    return covar


def compute_Kwx(inducing_scenario,
                scenario,
                time_idx,
                lat_idx,
                lon_idx,
                kernel,
                d_map,
                q_map,
                mu,
                sigma):
    I = compute_I_scenario(scenario1=inducing_scenario,
                           scenario2=scenario,
                           time_idx1=inducing_scenario.idx_inducing_times,
                           time_idx2=time_idx,
                           lat_idx1=inducing_scenario.idx_inducing_lats,
                           lat_idx2=lat_idx,
                           lon_idx1=inducing_scenario.idx_inducing_lons,
                           lon_idx2=lon_idx,
                           kernel=kernel,
                           d_map=d_map,
                           mu=mu,
                           sigma=sigma)
    covar = compute_covariance_scenario(scenario1=inducing_scenario,
                                        scenario2=scenario,
                                        time_idx1=inducing_scenario.idx_inducing_times,
                                        time_idx2=time_idx,
                                        lat_idx1=inducing_scenario.idx_inducing_lats,
                                        lat_idx2=lat_idx,
                                        lon_idx1=inducing_scenario.idx_inducing_lons,
                                        lon_idx2=lon_idx,
                                        I=I,
                                        d_map=d_map,
                                        q_map=q_map)
    # print(covar.shape)
    covar = covar.permute(0, 4, 5, 1, 2, 3).reshape(inducing_scenario.n_inducing_points, -1)
    return covar


def compute_Kww(inducing_scenario, kernel, d_map, q_map, mu, sigma):
    Kww = compute_Kxx(inducing_scenario,
                      inducing_scenario.idx_inducing_times,
                      inducing_scenario.idx_inducing_lats,
                      inducing_scenario.idx_inducing_lons,
                      kernel,
                      d_map,
                      q_map,
                      mu,
                      sigma)
    return Kww


def compute_I_scenario(scenario1, scenario2,
                       time_idx1, time_idx2,
                       lat_idx1, lat_idx2,
                       lon_idx1, lon_idx2,
                       kernel,
                       d_map,
                       mu,
                       sigma):
    scenario1_emissions_std = (scenario1.full_glob_inputs - mu) / sigma
    scenario2_emissions_std = (scenario2.full_glob_inputs - mu) / sigma

    K = kernel(scenario1_emissions_std, scenario2_emissions_std).evaluate()[:, :, None, None, None]
    I = torch.zeros(K.size(0), len(time_idx2), d_map.size(0), len(lat_idx2), len(lon_idx2))
    d_map2 = d_map[:, lat_idx2][..., lon_idx2]

    I_old = I[:, 0]
    i = int(torch.any(time_idx2 == 0.).item())  # If includes initial time, we want to skip it

    for t in range(1, time_idx2.max() + 1):
        K_new = K[:, t]
        I_new = step_I(I_old, K_new, d_map2)
        if torch.any(time_idx2 == t).item():
            I[:, i] = I_new
            i += 1
        I_old = I_new
    return I


def compute_covariance_scenario(scenario1, scenario2,
                                time_idx1, time_idx2,
                                lat_idx1, lat_idx2,
                                lon_idx1, lon_idx2,
                                I, q_map, d_map):
    covar = torch.zeros(len(time_idx1), len(time_idx2),
                        len(lat_idx2), len(lon_idx2),
                        len(lat_idx1), len(lon_idx1))
    q_map1 = q_map[:, lat_idx1][..., lon_idx1]
    d_map1 = d_map[:, lat_idx1][..., lon_idx1]
    q_d_ratio_2 = q_map[:, lat_idx2][..., lon_idx2].div(d_map[:, lat_idx2][..., lon_idx2])

    Kj_old = covar[0].unsqueeze(1).repeat(1, d_map.size(0), 1, 1, 1, 1)
    i = int(torch.any(time_idx1 == 0.).item())

    for t in range(1, time_idx1.max() + 1):
        I_new = I[t]
        Kj_new = step_kernel(Kj_old, I_new, q_d_ratio_2, q_map1, d_map1)
        if torch.any(time_idx1 == t).item():
            covar[i] = Kj_new.sum(dim=1)
            i += 1
        Kj_old = Kj_new
    return covar


def step_I(I_old, K_new, d_map, dt=1):
    decay_factor = torch.exp(-dt / d_map)
    I_new = I_old * decay_factor + d_map * K_new * (1 - decay_factor)
    I_new = (I_new + I_old) / 2
    return I_new


def step_kernel(Kj_old, I_new, inducing_q_d_ratio, q_map, d_map, dt=1):
    decay_factor = torch.exp(-dt / d_map)
    update = [torch.kron(I_new[:, j] * inducing_q_d_ratio[j], q_map[j] * (1 - decay_factor[j])) for j in range(q_map.size(0))]
    update = torch.stack(update, dim=1)
    update = update.reshape(Kj_old.size(0), Kj_old.size(1), Kj_old.size(2), Kj_old.size(4), Kj_old.size(3), Kj_old.size(5))
    update = update.permute(0, 1, 2, 4, 3, 5)
    Kj_new = (Kj_old * (1 + decay_factor[:, None, None, :, :]) + update) / 2
    return Kj_new


def sample_scenario(scenario_dataset, seed=None):
    if seed:
        np.random.seed(seed)
    name = np.random.choice(list(scenario_dataset.scenarios.keys()))
    return name, scenario_dataset[name]


def sample_indices(scenario, n_time, n_lat, n_lon, seed=None):
    if seed:
        torch.random.manual_seed(seed)
    # time_idx = torch.randperm(len(scenario.timesteps))[:n_time]
    # lat_idx = torch.randperm(len(scenario.lat))[:n_lat]
    # lon_idx = torch.randperm(len(scenario.lon))[:n_lon]

    time_idx = torch.randperm(len(scenario.timesteps) - n_time + 1)[0]
    time_idx = torch.arange(time_idx, time_idx + n_time)
    lat_idx = torch.randperm(len(scenario.lat) - n_lat + 1)[0]
    lat_idx = torch.arange(lat_idx, lat_idx + n_lat)
    lon_idx = torch.randperm(len(scenario.lon) - n_lon + 1)[0]
    lon_idx = torch.arange(lon_idx, lon_idx + n_lon)
    # lat_idx = torch.zeros(1).long()
    # lon_idx = torch.zeros(1).long()
    return time_idx, lat_idx, lon_idx


# def compute_I(scenario_dataset, inducing_scenario, kernel):
#     I = [compute_I_scenario(scenario_dataset, inducing_scenario, scenario, kernel)
#          for scenario in scenario_dataset.scenarios.values()]
#     I = torch.cat(I, dim=0)
#     return I


# def compute_I_scenario(scenario1, scenario2, idx1, idx2, kernel, d_map, mu, sigma, I_hist=None):
#     scenario1_emissions_std = (scenario1.full_glob_inputs - mu) / sigma
#     scenario2_emissions_std = (scenario2.full_glob_inputs - mu) / sigma
#
#     K = kernel(scenario1_emissions_std, scenario2_emissions_std).evaluate()[:, :, None, None, None]
#     I = torch.zeros((K.size(0), len(idx2)) + d_map.shape)
#     if I_hist is None:
#         I_old = I[:, 0]
#         t0 = 1
#     else:
#         I_old = I_hist
#         t0 = len(scenario2.hist_scenario)
#
#     i = 0
#
#     for t in range(t0, idx2.max()):
#         K_new = K[:, t]
#         I_new = step_I(I_old, K_new, d_map)
#         if torch.any(idx2 == t).item():
#             i += 1
#             I[:, i] = I_new
#         I_old = I_new
#     return I


# def step_I(I_old, K_new, d_map, dt=1):
#     decay_factor = torch.exp(-dt / d_map)
#     I_new = I_old * decay_factor + d_map * K_new * (1 - decay_factor)
#     I_new = (I_new + I_old) / 2
#     return I_new


# def compute_covariance_scenario(scenario1, scenario2, idx1, idx2, kernel, I, q_map, d_map, Kj_hist=None):
#     q_d_ratio = q_map.div(d_map)
#
#     covar = torch.zeros(len(idx1), len(idx2), I.size(3), I.size(4), q_map.size(1), q_map.size(2))
#     Kj_old = covar[0].unsqueeze(1).repeat(1, I.size(2), 1, 1, 1, 1)
#     if Kj_hist is None:
#         Kj_old = covar[0].unsqueeze(1).repeat(1, I.size(2), 1, 1, 1, 1)
#         t0 = 1
#     else:
#         Kj_old = Kj_hist
#         t0 = len(scenario1.hist_scenario)
#
#     i = 0
#
#     for t in range(t0, idx1.max()):
#         I_new = I[t]
#         Kj_new = step_kernel(Kj_old, I_new, q_d_ratio, q_map, d_map)
#         if torch.any(idx2 == t).item():
#             i += 1
#             covar[t] = Kj_new.sum(dim=1)
#         Kj_old = Kj_new
#     return covar
#
#
# def compute_covariance_scenario(scenario_dataset, inducing_scenario, scenario, I, q_map, d_map):
#     inducing_q_d_ratio = inducing_scenario.q_map / inducing_scenario.d_map
#     Kj = torch.zeros(I.size(0), I.size(1), I.size(3), I.size(4), q_map.size(1), q_map.size(2))
#     Kj_old = Kj[0].unsqueeze(1).repeat(1, 3, 1, 1, 1, 1)
#     for t in range(1, I.size(0)):
#         I_new = I[t]
#         Kj_new = step_kernel(Kj_old, I_new, inducing_q_d_ratio, q_map, d_map)
#         Kj_old = Kj_new
#         Kj[t] = Kj_new.sum(dim=1)
#     Kj = scenario.trim_hist(Kj)
#     return Kj
#


# def compute_I(scenario_dataset, inducing_scenario, kernel):
#     I = [compute_I_scenario(scenario_dataset, inducing_scenario, scenario, kernel)
#          for scenario in scenario_dataset.scenarios.values()]
#     I = torch.cat(I, dim=0)
#     return I

#
# def compute_I_scenario(scenario_dataset, inducing_scenario, scenario, kernel):
#     mu, sigma = scenario_dataset.mu_glob_inputs, scenario_dataset.sigma_glob_inputs
#     scenario_emissions_std = (scenario.full_glob_inputs - mu) / sigma
#     inducing_emissions_std = (inducing_scenario.full_glob_inputs - mu) / sigma
#
#     inducing_d_map = inducing_scenario.d_map
#
#     K = kernel(inducing_emissions_std, scenario_emissions_std).evaluate()[:, :, None, None, None]
#     I = torch.zeros(K.size(0), K.size(1), inducing_d_map.size(0), inducing_d_map.size(1), inducing_d_map.size(2))
#     for t in range(1, len(scenario_emissions_std)):
#         I_old = I[:, t - 1]
#         K_new = K[:, t]
#         I_new = step_I(I_old, K_new, inducing_d_map)
#         I[:, t] = I_new
#     I = inducing_scenario.trim_noninducing_times(I.permute(1, 0, 2, 3, 4)).permute(1, 0, 2, 3, 4)
#     return I
#
#
# def step_I(I_old, K_new, inducing_d_map, dt=1):
#     decay_factor = torch.exp(-dt / inducing_d_map)
#     I_new = I_old * decay_factor + inducing_d_map * K_new * (1 - decay_factor)
#     I_new = (I_new + I_old) / 2
#     return I_new


# def compute_covariance(scenario_dataset, inducing_scenario, I, q_map, d_map):
#     Kj = [compute_covariance_scenario(scenario_dataset, inducing_scenario, scenario, I, q_map, d_map)
#           for scenario in scenario_dataset.scenarios.values()]
#     Kj = torch.cat(Kj, dim=0)
#     return Kj


# def compute_covariance_scenario(scenario_dataset, inducing_scenario, scenario, I, q_map, d_map):
#     inducing_q_d_ratio = inducing_scenario.q_map / inducing_scenario.d_map
#     Kj = torch.zeros(I.size(0), I.size(1), I.size(3), I.size(4), q_map.size(1), q_map.size(2))
#     Kj_old = Kj[0].unsqueeze(1).repeat(1, 3, 1, 1, 1, 1)
#     for t in range(1, I.size(0)):
#         I_new = I[t]
#         Kj_new = step_kernel(Kj_old, I_new, inducing_q_d_ratio, q_map, d_map)
#         Kj_old = Kj_new
#         Kj[t] = Kj_new.sum(dim=1)
#     Kj = scenario.trim_hist(Kj)
#     return Kj


# def compute_inducing_covariance(scenario_dataset, inducing_scenario, inducing_I):
#     inducing_q_d_ratio = inducing_scenario.q_map / inducing_scenario.d_map
#     Kj = torch.zeros(inducing_I.size(0), inducing_I.size(1), inducing_I.size(3), inducing_I.size(4), inducing_q_d_ratio.size(1), inducing_q_d_ratio.size(2))
#     Kj_old = Kj[0].unsqueeze(1).repeat(1, 3, 1, 1, 1, 1)
#     for t in range(1, inducing_I.size(0)):
#         I_new = inducing_I[t]
#         Kj_new = step_kernel(Kj_old, I_new, inducing_q_d_ratio, inducing_scenario.q_map, inducing_scenario.d_map)
#         Kj_old = Kj_new
#         Kj[t] = Kj_new.sum(dim=1)
#     Kj = inducing_scenario.trim_noninducing_times(Kj)
#     return Kj
