import os
import sys
import torch

base_dir = os.path.join(os.getcwd(), '..')
sys.path.append(base_dir)

import src.fair as fair
from src.fair.tools import step_I, step_kernel


def compute_means(scenario_dataset):
    base_kwargs = fair.get_params()
    means = dict()
    for name, scenario in scenario_dataset.scenarios.items():
        res = fair.run(scenario.full_timesteps.numpy(),
                       scenario.full_emissions.T.numpy(),
                       base_kwargs)
        S = res['S']
        S = scenario.trim_hist(S)
        means.update({scenario: torch.from_numpy(S).float()})
    return means


def compute_I(scenario_dataset, kernel, q, d):
    I = [compute_I_scenario(scenario_dataset, scenario, kernel, q, d)
         for scenario in scenario_dataset.scenarios.values()]
    I = torch.cat(I, dim=-2)
    return I


def compute_I_scenario(scenario_dataset, scenario, kernel, q, d):
    mu, sigma = scenario_dataset.mu_inputs, scenario_dataset.sigma_inputs
    scenario_emissions_std = (scenario.full_inputs - mu) / sigma
    dataset_emissions_std = (scenario_dataset.full_inputs - mu) / sigma

    K = kernel(dataset_emissions_std, scenario_emissions_std).evaluate().unsqueeze(-1)
    I = torch.zeros((K.size(0), K.size(1), len(d)))
    for t in range(1, len(scenario_emissions_std)):
        I_old = I[:, t - 1]
        K_new = K[:, t]
        I_new = step_I(I_old, K_new, q.unsqueeze(0), d.unsqueeze(0))
        I[:, t] = I_new.squeeze()
    return I


def compute_covariance(scenario_dataset, I, q, d):
    nboxes = len(q)
    I = I[..., None].repeat(1, 1, 1, nboxes).view(I.size(0), I.size(1), -1)
    d = d.repeat(nboxes)
    q = q.repeat(nboxes)
    Kj = [compute_covariance_scenario(scenario_dataset, scenario, I, q, d)
          for scenario in scenario_dataset.scenarios.values()]
    Kj = torch.cat(Kj, dim=-2)
    Kj = scenario_dataset.trim_hist(Kj).sum(dim=-1)
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
    Kj = scenario.trim_hist(Kj)
    return Kj.permute(1, 0, 2)


def compute_mF(scenario_dataset):
    base_kwargs = fair.get_params()
    means = dict()
    for name, scenario in scenario_dataset.scenarios.items():
        res = fair.run(scenario.full_timesteps.numpy(),
                       scenario.full_emissions.T.numpy(),
                       base_kwargs)
        mF = torch.from_numpy(res['RF'].sum(axis=0)).float()
        mF = scenario.trim_hist(mF)
        means.update({scenario: mF})
    return means


def compute_kFT(scenario_dataset_F, scenario_dataset_T, kernel, q, d):
    full_scenario_dataset = scenario_dataset_T + scenario_dataset_F
    kFT = compute_I(full_scenario_dataset, kernel, q, d).sum(dim=-1)
    kFT = full_scenario_dataset.trim_hist(kFT)
    kFT = full_scenario_dataset.trim_hist(kFT.T).T
    kFT = kFT[len(scenario_dataset_T.timesteps):, :len(scenario_dataset_T.timesteps)]
    return kFT
