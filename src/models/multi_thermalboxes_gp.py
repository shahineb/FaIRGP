import torch
import gpytorch
from gpytorch import distributions
from gpytorch.models import GP
from linear_operator.utils.cholesky import psd_safe_cholesky
from .utils_multi import compute_means, compute_mean, compute_I, compute_covariance


class MultiThermalBoxesGP(GP):
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

    def _compute_covariance(self, scenario_dataset, lat_idx, lon_idx):
        I = compute_I(scenario_dataset, self.kernel, self.q_map, self.d_map, lat_idx, lon_idx)
        covar = compute_covariance(scenario_dataset, I, self.q_map, self.d_map, lat_idx, lon_idx)
        covar = gpytorch.add_jitter(covar)
        return covar

    def _init_lat_lon_idx(self, lat_idx, lon_idx):
        if lat_idx is None:
            full_lats = self.train_scenarios[0].lat
            nlat = len(full_lats)
            lat_idx = torch.arange(nlat).to(full_lats.device)
        if lon_idx is None:
            full_lons = self.train_scenarios[0].lon
            nlon = len(full_lons)
            lon_idx = torch.arange(nlon).to(full_lons.device)
        return lat_idx, lon_idx

    def train_prior_dist(self, lat_idx=None, lon_idx=None):
        return self.forward(self.train_scenarios, lat_idx, lon_idx)

    def forward(self, scenario_dataset, lat_idx=None, lon_idx=None):
        lat_idx, lon_idx = self._init_lat_lon_idx(lat_idx, lon_idx)

        ntime = len(scenario_dataset.timesteps)
        nlat = len(lat_idx)
        nlon = len(lon_idx)

        mean = torch.zeros(nlat * nlon, ntime)
        covar = self._compute_covariance(scenario_dataset, lat_idx, lon_idx)

        prior_dist = distributions.MultivariateNormal(mean, covar)
        return prior_dist

    def posterior(self, test_scenarios, lat_idx=None, lon_idx=None, diag=True):
        lat_idx, lon_idx = self._init_lat_lon_idx(lat_idx, lon_idx)

        ntrain = len(self.train_scenarios.timesteps)
        train_test_scenarios = self.train_scenarios + test_scenarios
        full_output = self.forward(train_test_scenarios, lat_idx, lon_idx)
        noisy_full_output = self.likelihood(full_output)

        full_covar = full_output.covariance_matrix
        train_test_covar = full_covar[:, :ntrain, ntrain:]
        test_test_covar = full_covar[:, ntrain:, ntrain:]
        train_train_covar = noisy_full_output.covariance_matrix[:, :ntrain, :ntrain]

        chol = psd_safe_cholesky(train_train_covar)
        interp = torch.cholesky_solve(train_test_covar, chol)

        train_targets = self.train_targets[:, lat_idx][..., lon_idx].view(ntrain, -1).T.unsqueeze(-1)
        posterior_mean = (interp.permute(0, 2, 1) @ train_targets).squeeze()

        if diag:
            test_test_var = torch.diagonal(test_test_covar, dim1=1, dim2=2)
            posterior_var = test_test_var - interp.mul(train_test_covar).sum(dim=1)
            output = torch.distributions.Normal(posterior_mean, posterior_var)
        else:
            posterior_covar = test_test_covar - train_test_covar.permute(0, 2, 1) @ interp
            output = gpytorch.distributions.MultivariateNormal(posterior_mean, posterior_covar)
        return output

    def to(self, device):
        self = super().to(device)
        self.train_means = {k: v.to(device) for (k, v) in self.train_means.items()}
        self.FaIR_model = self.FaIR_model.to(device)
        return self

    def cpu(self):
        self.to(device='cpu')
        return self
