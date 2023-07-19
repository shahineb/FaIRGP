import os
import sys
import torch

base_dir = os.path.join(os.getcwd(), '../..')
sys.path.append(base_dir)

from src.fair.tools import step_I, step_kernel


def compute_mean(scenario, FaIR_model, S0, q_map, d_map):
    timestep = torch.cat([torch.ones(1).to(scenario.full_timesteps.device), torch.diff(scenario.full_timesteps)])
    emissions = scenario.full_glob_emissions.T
    weights = torch.cos(torch.deg2rad(scenario.lat.double()))
    res = FaIR_model(emissions, timestep, q_map, d_map, weights, S0)
    T = scenario.trim_hist(res['T'].float())
    return T


def compute_means(scenario_dataset, FaIR_model, S0, q_map, d_map):
    means = dict()
    for name, scenario in scenario_dataset.scenarios.items():
        T = compute_mean(scenario, FaIR_model, S0, q_map, d_map)
        means.update({name: T})
    return means


def compute_I(scenario_dataset, kernel, q, d):
    I = [compute_I_scenario(scenario_dataset, scenario, kernel, q, d)
         for scenario in scenario_dataset.scenarios.values()]
    I = torch.cat(I, dim=-2)
    return I


def compute_I_scenario(scenario_dataset, scenario, kernel, q, d):
    mu, sigma = scenario_dataset.mu_glob_inputs, scenario_dataset.sigma_glob_inputs
    scenario_emissions_std = (scenario.full_glob_inputs - mu) / sigma
    dataset_emissions_std = (scenario_dataset.full_glob_inputs - mu) / sigma

    K = kernel(dataset_emissions_std, scenario_emissions_std).evaluate().unsqueeze(-1)
    I = torch.zeros((K.size(0), K.size(1), len(d)))
    for t in range(1, len(scenario_emissions_std)):
        I_old = I[:, t - 1]
        K_new = K[:, t]
        I_new = step_I(I_old, K_new, q.unsqueeze(0), d.unsqueeze(0))
        I[:, t] = I_new.squeeze()
    return I


def compute_covariance(scenario_dataset, I, q, d):
    Kj = [compute_covariance_scenario(scenario_dataset, scenario, I, q, d)
          for scenario in scenario_dataset.scenarios.values()]
    Kj = torch.cat(Kj, dim=-1)
    Kj = scenario_dataset.trim_hist(Kj)
    Kj = 0.5 * (Kj + Kj.T)
    return Kj


def compute_covariance_scenario(scenario_dataset, scenario, I, q, d):
    I_scenario = I[scenario_dataset.full_slices[scenario.name]]
    Kj = torch.zeros_like(I_scenario)
    for t in range(1, I_scenario.size(0)):
        Kj_old = Kj[t - 1]
        I_new = I_scenario[t]
        Kj_new = step_kernel(Kj_old, I_new, q.unsqueeze(0), d.unsqueeze(0))
        Kj[t] = Kj_new
    Kj = scenario.trim_hist(Kj).sum(dim=-1)
    return Kj.T
