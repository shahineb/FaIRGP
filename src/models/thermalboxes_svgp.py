import os
import sys
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch import variational
from .utils_svgp import compute_mean, compute_means, compute_Kxx, compute_Kwx, compute_Kww, sample_scenario, sample_indices

base_dir = os.path.join(os.getcwd(), '..')
sys.path.append(base_dir)

from src.variational import ScenarioVariationalStrategy


class ThermalBoxesSVGP(ApproximateGP):
    def __init__(self, scenario_dataset, inducing_scenario, kernel, likelihood, FaIR_model, q_map, d_map):
        variational_strategy = self._set_variational_strategy(inducing_scenario)
        super().__init__(variational_strategy=variational_strategy)
        self.kernel = kernel
        self.likelihood = likelihood
        self.train_scenarios = scenario_dataset
        self.FaIR_model = FaIR_model
        self.register_buffer('mu', scenario_dataset.mu_glob_inputs)
        self.register_buffer('sigma', scenario_dataset.sigma_glob_inputs)
        self.register_buffer('q_map', q_map)
        self.register_buffer('d_map', d_map)
        self.train_means = self._compute_means(scenario_dataset)
        self.train_targets = {name: scenario_dataset[name].tas - self.train_means[name]
                              for name in scenario_dataset.scenarios.keys()}
        self.register_buffer('mu_targets', torch.cat([v for v in self.train_targets.values()]).mean())
        self.register_buffer('sigma_targets', torch.cat([v for v in self.train_targets.values()]).std())
        self.train_targets = {name: (target - self.mu_targets) / self.sigma_targets
                              for (name, target) in self.train_targets.items()}

    def _set_variational_strategy(self, inducing_scenario):
        # Use gaussian variational family
        # variational_distribution = variational.CholeskyVariationalDistribution(num_inducing_points=inducing_scenario.n_inducing_points)
        variational_distribution = variational.CholeskyVariationalDistribution(num_inducing_points=inducing_scenario.n_inducing_points)
        # Set default variational approximation strategy
        variational_strategy = ScenarioVariationalStrategy(model=self,
                                                           inducing_scenario=inducing_scenario,
                                                           variational_distribution=variational_distribution)
        return variational_strategy

    def _compute_mean(self, scenario):
        mean = compute_mean(scenario, self.FaIR_model, self.d_map, self.q_map)
        return mean

    def _compute_means(self, scenario_dataset):
        means_dict = compute_means(scenario_dataset, self.FaIR_model, self.d_map, self.q_map)
        return means_dict

    def _compute_covariance(self, scenario, time_idx, lat_idx, lon_idx, diag=False):
        Kxx = compute_Kxx(scenario=scenario,
                          time_idx=time_idx,
                          lat_idx=lat_idx,
                          lon_idx=lon_idx,
                          kernel=self.kernel,
                          d_map=self.d_map,
                          q_map=self.q_map,
                          mu=self.mu,
                          sigma=self.sigma,
                          diag=diag)
        Kww = compute_Kww(inducing_scenario=self.inducing_scenario,
                          kernel=self.kernel,
                          d_map=self.d_map,
                          q_map=self.q_map,
                          mu=self.mu,
                          sigma=self.sigma)
        Kwx = compute_Kwx(inducing_scenario=self.inducing_scenario,
                          scenario=scenario,
                          time_idx=time_idx,
                          lat_idx=lat_idx,
                          lon_idx=lon_idx,
                          kernel=self.kernel,
                          d_map=self.d_map,
                          q_map=self.q_map,
                          mu=self.mu,
                          sigma=self.sigma)
        Kww = gpytorch.add_jitter(Kww)
        return Kxx, Kww, Kwx

    def sample_batch(self, n_time, n_lat, n_lon, seed=None):
        name, scenario = sample_scenario(self.train_scenarios, seed)
        time_idx, lat_idx, lon_idx = sample_indices(scenario, n_time, n_lat, n_lon, seed)
        targets = self.train_targets[name][time_idx][:, lat_idx][:, :, lon_idx]
        return scenario, time_idx, lat_idx, lon_idx, targets

    def __call__(self, scenario, time_idx, lat_idx, lon_idx, diag=False, **kwargs):
        Kxx, Kww, Kwx = self._compute_covariance(scenario, time_idx, lat_idx, lon_idx, diag)
        qf = self.variational_strategy.__call__(Kww=Kww, Kwx=Kwx, Kxx=Kxx, diag=diag, **kwargs)
        return qf

    @property
    def inducing_scenario(self):
        return self.variational_strategy.inducing_scenario

    def to(self, device):
        super().to(device)
        self.train_means = {k: v.to(device) for (k, v) in self.train_means.items()}
        self.train_targets = {k: v.to(device) for (k, v) in self.train_targets.items()}

    def cpu(self):
        self.to(device='cpu')
