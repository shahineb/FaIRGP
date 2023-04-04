import torch
import gpytorch
from gpytorch import distributions
from gpytorch.models import GP
from linear_operator.utils.cholesky import psd_safe_cholesky
from .utils_multi_simple import compute_means, compute_mean, compute_I, compute_covariance


class SimpleMultiThermalBoxesGP(GP):
    def __init__(self, scenario_dataset, kernel, FaIR_model, S0, q_map, d_map, likelihood):
        super().__init__()
        # Register input data
        self.train_scenarios = scenario_dataset

        # Setup mean, kernel and likelihood
        self.FaIR_model = FaIR_model
        self.kernel = kernel
        self.likelihood = likelihood

        self.register_buffer('mu', scenario_dataset.mu_glob_inputs)
        self.register_buffer('sigma', scenario_dataset.sigma_glob_inputs)
        self.register_buffer('q_map', q_map)
        self.register_buffer('d_map', d_map)
        self.register_buffer('q', q_map.mean(dim=(1, 2)))
        self.register_buffer('d', d_map.mean(dim=(1, 2)))
        self.register_buffer('S0', S0)

        self.train_means = self._compute_means(scenario_dataset)
        train_targets = {name: scenario_dataset[name].tas - self.train_means[name]
                         for name in scenario_dataset.scenarios.keys()}
        train_targets = torch.cat([v for v in train_targets.values()])
        self.register_buffer('mu_targets', train_targets.mean())
        self.register_buffer('sigma_targets', train_targets.std())
        self.register_buffer('train_targets', (train_targets - self.mu_targets) / self.sigma_targets)

    def _compute_mean(self, scenario):
        mean = compute_mean(scenario, self.FaIR_model, self.S0, self.q_map, self.d_map)
        return mean

    def _compute_means(self, scenario_dataset):
        means_dict = compute_means(scenario_dataset, self.FaIR_model, self.S0, self.q_map, self.d_map)
        return means_dict

    def _compute_covariance(self, scenario_dataset):
        I = compute_I(scenario_dataset, self.kernel, self.q, self.d)
        covar = compute_covariance(scenario_dataset, I, self.q, self.d)
        covar = gpytorch.add_jitter(covar)
        return covar

    def train_prior_dist(self):
        train_mean = torch.zeros_like(self.train_scenarios.tas)
        train_covar = self._compute_covariance(self.train_scenarios)
        train_prior_dist = distributions.MultivariateNormal(train_mean, train_covar)
        return train_prior_dist

    def forward(self, scenario_dataset):
        mean = torch.zeros(len(scenario_dataset.timesteps))
        covar = self._compute_covariance(scenario_dataset)
        prior_dist = distributions.MultivariateNormal(mean, covar)
        return prior_dist

    def posterior(self, test_scenarios, diag=True):
        ntrain = len(self.train_scenarios.timesteps)
        train_test_scenarios = self.train_scenarios + test_scenarios
        full_output = self.forward(train_test_scenarios)
        noisy_full_output = self.likelihood(full_output)

        full_covar = full_output.covariance_matrix
        train_test_covar = full_covar[:ntrain, ntrain:]
        test_test_covar = full_covar[ntrain:, ntrain:]
        train_train_covar = noisy_full_output.covariance_matrix[:ntrain, :ntrain]

        chol = psd_safe_cholesky(train_train_covar)
        interp = torch.cholesky_solve(train_test_covar, chol)

        posterior_mean = interp.T @ self.train_targets.view(ntrain, -1)
        if diag:
            posterior_var = test_test_covar.diag() - interp.mul(train_test_covar).sum(dim=0)
            output = torch.distributions.Normal(posterior_mean.T, posterior_var)
        else:
            posterior_covar = test_test_covar - train_test_covar.T @ interp
            output = gpytorch.distributions.MultivariateNormal(posterior_mean.T, posterior_covar)
        return output
