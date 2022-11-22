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
                sigma,
                diag=False):
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
    if diag:
        covar_diag = compute_covariance_scenario_diag(scenario=scenario,
                                                      time_idx=time_idx,
                                                      lat_idx=lat_idx,
                                                      lon_idx=lon_idx,
                                                      I=I,
                                                      q_map=q_map,
                                                      d_map=d_map)
        output = covar_diag.flatten()
    else:
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
        output = covar.permute(0, 4, 5, 1, 2, 3).reshape(n, -1)
    return output


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


def compute_covariance_scenario_diag(scenario,
                                     time_idx,
                                     lat_idx,
                                     lon_idx,
                                     I,
                                     q_map,
                                     d_map):
    covar_diag = torch.zeros(len(time_idx), len(lat_idx), len(lon_idx))
    rq_map = q_map[:, lat_idx][..., lon_idx]
    rd_map = d_map[:, lat_idx][..., lon_idx]
    q_d_ratio = rq_map.div(rd_map)

    Kj_old = covar_diag.unsqueeze(1).repeat(1, d_map.size(0), 1, 1)
    i = int(torch.any(time_idx == 0.).item())

    for t in range(1, time_idx.max() + 1):
        I_new = I[t]
        Kj_new = step_kernel_diag(Kj_old, I_new, q_d_ratio, rq_map, rd_map)
        if torch.any(time_idx == t).item():
            covar_diag[i] = Kj_new.sum(dim=1)[i]
            i += 1
        Kj_old = Kj_new
    return covar_diag


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


def step_kernel_diag(Kj_old, I_new, inducing_q_d_ratio, q_map, d_map, dt=1):
    decay_factor = torch.exp(-dt / d_map)
    update = [I_new[:, j] * inducing_q_d_ratio[j] * q_map[j] * (1 - decay_factor[j]) for j in range(q_map.size(0))]
    update = torch.stack(update, dim=1)
    Kj_new = (Kj_old * (1 + decay_factor) + update) / 2
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
    return time_idx, lat_idx, lon_idx


def compute_ell(qf, targets, batch_size, model):
    # Compute ELL scaling term
    eps = torch.finfo(torch.float32).eps
    α = batch_size * torch.log(2 * np.pi * model.likelihood.noise + eps)

    # Compute ELL covariance term
    β = qf.variance.div(model.likelihood.noise).sum()

    # Compute ELL mean term
    γ = (targets.view(-1) - qf.mean)**2
    γ = γ.div(model.likelihood.noise).sum()

    # Combine and return
    ell = -0.5 * torch.sum(α + β + γ)
    return ell, α, β, γ
