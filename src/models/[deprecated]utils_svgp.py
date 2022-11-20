import os
import sys
import torch

base_dir = os.path.join(os.getcwd(), '..')
sys.path.append(base_dir)


def compute_means(scenario_dataset, FaIR_model, d_map, q_map):
    means = dict()
    for name, scenario in scenario_dataset.scenarios.items():
        timestep = torch.cat([torch.ones(1), torch.diff(scenario.full_timesteps)])
        emissions = scenario.full_glob_emissions.T
        weights = torch.cos(torch.deg2rad(scenario.lat.double()))
        res = FaIR_model(emissions, timestep, d_map, q_map, weights)
        S = scenario.trim_hist(res['S'].float())
        means.update({scenario: S})
    return means


def compute_I_data_data(scenario_dataset, kernel, d_map):
    I = [compute_I_scenario_data_data(scenario_dataset, scenario, kernel, d_map)
         for scenario in scenario_dataset.scenarios.values()]
    I = torch.cat(I, dim=1)
    return I


def compute_I_scenario_data_data(scenario_dataset, scenario, kernel, d_map):
    mu, sigma = scenario_dataset.mu_glob_inputs, scenario_dataset.sigma_glob_inputs
    dataset_emissions_std = (scenario_dataset.full_glob_inputs - mu) / sigma
    scenario_emissions_std = (scenario.full_glob_inputs - mu) / sigma

    K = kernel(dataset_emissions_std, scenario_emissions_std).evaluate()[:, :, None, None, None]
    I = torch.zeros(K.size(0), K.size(1), d_map.size(0), d_map.size(1), d_map.size(2))
    for t in range(1, len(scenario_emissions_std)):
        I_old = I[:, t - 1]
        K_new = K[:, t]
        I_new = step_I(I_old, K_new, d_map)
        I[:, t] = I_new
    I = scenario.trim_hist(I.permute(1, 0, 2, 3, 4)).permute(1, 0, 2, 3, 4)
    return I


def compute_I(scenario_dataset, inducing_scenario, kernel):
    I = [compute_I_scenario(scenario_dataset, inducing_scenario, scenario, kernel)
         for scenario in scenario_dataset.scenarios.values()]
    I = torch.cat(I, dim=0)
    return I


def compute_I_scenario(scenario_dataset, inducing_scenario, scenario, kernel):
    mu, sigma = scenario_dataset.mu_glob_inputs, scenario_dataset.sigma_glob_inputs
    scenario_emissions_std = (scenario.full_glob_inputs - mu) / sigma
    inducing_emissions_std = (inducing_scenario.full_glob_inputs - mu) / sigma

    inducing_d_map = inducing_scenario.d_map

    K = kernel(inducing_emissions_std, scenario_emissions_std).evaluate()[:, :, None, None, None]
    I = torch.zeros(K.size(0), K.size(1), inducing_d_map.size(0), inducing_d_map.size(1), inducing_d_map.size(2))
    for t in range(1, len(scenario_emissions_std)):
        I_old = I[:, t - 1]
        K_new = K[:, t]
        I_new = step_I(I_old, K_new, inducing_d_map)
        I[:, t] = I_new
    I = inducing_scenario.trim_noninducing_times(I.permute(1, 0, 2, 3, 4)).permute(1, 0, 2, 3, 4)
    return I


def step_I(I_old, K_new, inducing_d_map, dt=1):
    decay_factor = torch.exp(-dt / inducing_d_map)
    I_new = I_old * decay_factor + inducing_d_map * K_new * (1 - decay_factor)
    I_new = (I_new + I_old) / 2
    return I_new


def compute_covariance(scenario_dataset, inducing_scenario, I, q_map, d_map):
    Kj = [compute_covariance_scenario(scenario_dataset, inducing_scenario, scenario, I, q_map, d_map)
          for scenario in scenario_dataset.scenarios.values()]
    Kj = torch.cat(Kj, dim=0)
    return Kj


def compute_covariance_scenario(scenario_dataset, inducing_scenario, scenario, I, q_map, d_map):
    inducing_q_d_ratio = inducing_scenario.q_map / inducing_scenario.d_map
    Kj = torch.zeros(I.size(0), I.size(1), I.size(3), I.size(4), q_map.size(1), q_map.size(2))
    Kj_old = Kj[0].unsqueeze(1).repeat(1, 3, 1, 1, 1, 1)
    for t in range(1, I.size(0)):
        I_new = I[t]
        Kj_new = step_kernel(Kj_old, I_new, inducing_q_d_ratio, q_map, d_map)
        Kj_old = Kj_new
        Kj[t] = Kj_new.sum(dim=1)
    Kj = scenario.trim_hist(Kj)
    return Kj


def step_kernel(Kj_old, I_new, inducing_q_d_ratio, q_map, d_map, dt=1):
    decay_factor = torch.exp(-dt / d_map)
    update = [torch.kron(I_new[:, j] * inducing_q_d_ratio[j], q_map[j] * (1 - decay_factor[j])) for j in range(q_map.size(0))]
    update = torch.stack(update, dim=1)
    update = update.reshape(Kj_old.size(0), Kj_old.size(1), Kj_old.size(2), Kj_old.size(4), Kj_old.size(3), Kj_old.size(5))
    update = update.permute(0, 1, 2, 4, 3, 5)
    Kj_new = (Kj_old * (1 + decay_factor[:, None, None, :, :]) + update) / 2
    return Kj_new


def compute_inducing_covariance(scenario_dataset, inducing_scenario, inducing_I):
    inducing_q_d_ratio = inducing_scenario.q_map / inducing_scenario.d_map
    Kj = torch.zeros(inducing_I.size(0), inducing_I.size(1), inducing_I.size(3), inducing_I.size(4), inducing_q_d_ratio.size(1), inducing_q_d_ratio.size(2))
    Kj_old = Kj[0].unsqueeze(1).repeat(1, 3, 1, 1, 1, 1)
    for t in range(1, inducing_I.size(0)):
        I_new = inducing_I[t]
        Kj_new = step_kernel(Kj_old, I_new, inducing_q_d_ratio, inducing_scenario.q_map, inducing_scenario.d_map)
        Kj_old = Kj_new
        Kj[t] = Kj_new.sum(dim=1)
    Kj = inducing_scenario.trim_noninducing_times(Kj)
    return Kj
