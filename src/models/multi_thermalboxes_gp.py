import torch
import gpytorch
from gpytorch.models import GP
from .utils_multi import compute_means, compute_mean


class MultiThermalBoxesGP(GP):
    def __init__(self, scenario_dataset, kernel, FaIR_model, S0, q_map, d_map, likelihood):
        super().__init__()
        # Register input data
        self.train_scenarios = scenario_dataset

        # Setup mean, kernel and likelihood
        self.FaIR_model = FaIR_model
        self.register_buffer('train_mean', self._compute_mean(self.train_scenarios))
        self.kernel = kernel
        self.likelihood = likelihood

        self.register_buffer('mu', scenario_dataset.mu_glob_inputs)
        self.register_buffer('sigma', scenario_dataset.sigma_glob_inputs)
        self.register_buffer('q_map', q_map)
        self.register_buffer('d_map', d_map)
        self.register_buffer('S0', S0)

        self.train_means = self._compute_means(scenario_dataset)
        self.train_targets = {name: scenario_dataset[name].tas - self.train_means[name]
                              for name in scenario_dataset.scenarios.keys()}
        self.register_buffer('mu_targets', torch.cat([v for v in self.train_targets.values()]).mean())
        self.register_buffer('sigma_targets', torch.cat([v for v in self.train_targets.values()]).std())
        self.train_targets = {name: (target - self.mu_targets) / self.sigma_targets
                              for (name, target) in self.train_targets.items()}

    def _compute_mean(self, scenario):
        mean = compute_mean(scenario, self.FaIR_model, self.d_map, self.q_map)
        return mean

    def _compute_means(self, scenario_dataset):
        means_dict = compute_means(scenario_dataset, self.FaIR_model, self.d_map, self.q_map)
        return means_dict

    def _compute_covariance(self, scenario, time_idx, lat_idx, lon_idx, diag=False):
        

    def to(self, device):
        self = super().to(device)
        self.train_means = {k: v.to(device) for (k, v) in self.train_means.items()}
        self.train_targets = {k: v.to(device) for (k, v) in self.train_targets.items()}
        self.FaIR_model = self.FaIR_model.to(device)
        return self

    def cpu(self):
        self.to(device='cpu')
        return self
