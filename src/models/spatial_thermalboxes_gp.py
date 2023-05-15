import torch
import gpytorch
from gpytorch import distributions
from gpytorch.models import GP
from linear_operator.utils.cholesky import psd_safe_cholesky
from sklearn.linear_model import LinearRegression
from .utils_spatial import compute_means, compute_I, compute_covariance, compute_mF, compute_kFT


class SpatialThermalBoxesGP(GP):
    def __init__(self, scenario_dataset, kernel, d, q, likelihood):
        super().__init__()
        # Register input data
        self.train_scenarios = scenario_dataset

        # Setup mean, kernel and likelihood
        self.pattern_scaling = self._fit_pattern_scaling()
        self.beta = torch.from_numpy(self.pattern_scaling.coef_).float().reshape(self.train_scenarios.tas.size(1), self.train_scenarios.tas.size(2))
        self.kernel = kernel
        self.likelihood = likelihood

        self.register_buffer('mu', scenario_dataset.mu_glob_inputs)
        self.register_buffer('sigma', scenario_dataset.sigma_glob_inputs)
        self.register_buffer('q', torch.from_numpy(q).float())
        self.register_buffer('d', torch.from_numpy(d).float())

        self.train_means = self._compute_means(scenario_dataset)
        train_targets = {name: scenario_dataset[name].tas - self.train_means[name]
                         for name in scenario_dataset.scenarios.keys()}
        train_targets = torch.cat([v for v in train_targets.values()]).div(self.beta.unsqueeze(0))
        self.register_buffer('mu_targets', train_targets.mean(dim=0))
        self.register_buffer('sigma_targets', train_targets.std(dim=0))
        self.register_buffer('train_targets', (train_targets - self.mu_targets) / self.sigma_targets)

    def _fit_pattern_scaling(self):
        all_tas = self.train_scenarios.tas
        wlat = torch.cos(torch.deg2rad(self.train_scenarios[0].lat)).clip(min=torch.finfo(torch.float64).eps)[:, None]
        glob_tas = torch.sum(all_tas * wlat, dim=(1, 2)) / (all_tas.size(2) * wlat.sum())
        pattern_scaling = LinearRegression()
        pattern_scaling.fit(glob_tas[:, None], all_tas.reshape(all_tas.size(0), -1))
        return pattern_scaling

    def _compute_means(self, scenario_dataset):
        means_dict = compute_means(scenario_dataset, self.pattern_scaling)
        return means_dict

    def _compute_covariance(self, scenario_dataset):
        I = compute_I(scenario_dataset, self.kernel, self.q, self.d)
        covar = compute_covariance(scenario_dataset, I, self.q, self.d)
        covar = gpytorch.add_jitter(covar)
        return covar

    def train_prior_dist(self):
        prior_dist = self.forward(self.train_scenarios)
        # output = self.likelihood(prior_dist)
        return prior_dist

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

    def _compute_forcing_mean(self, scenario_dataset):
        mFs = compute_mF(scenario_dataset)
        mF = torch.cat([v for v in mFs.values()])
        return mF

    def forcing_posterior(self, test_scenarios, diag=True):
        ntrain = len(self.train_scenarios.timesteps)
        mF = self._compute_forcing_mean(test_scenarios).view(-1, 1)

        # mu, sigma = self.train_scenarios.mu_glob_inputs, self.train_scenarios.sigma_glob_inputs
        # test_scenario_emissions_std = (test_scenarios.glob_inputs - mu) / sigma
        # kFF = self.kernel(test_scenario_emissions_std).evaluate()

        kFT = compute_kFT(test_scenarios, self.train_scenarios, self.kernel, self.q, self.d)

        covar = self._compute_covariance(self.train_scenarios)
        covar = covar + self.likelihood.compute_covariance(covar.size(0))
        chol = torch.linalg.cholesky(gpytorch.add_jitter(covar))
        kFT_covarinv = torch.cholesky_solve(kFT.T, chol).T

        posterior_mF = mF + kFT_covarinv @ self.train_targets.view(ntrain, -1)
        # if diag:
        #     posterior_var = kFF.diag() - kFT_covarinv.mul(kFT).sum(dim=1)
        #     posterior_F = torch.distributions.Normal(posterior_mF.T, posterior_var)
        # else:
        #     posterior_covar = kFF - kFT_covarinv @ kFT.T
        #     while True:
        #         try:
        #             posterior_F = distributions.MultivariateNormal(posterior_mF.T, posterior_covar)
        #             break
        #         except torch._C._LinAlgError:
        #             posterior_covar = gpytorch.add_jitter(posterior_covar)
        return posterior_mF
