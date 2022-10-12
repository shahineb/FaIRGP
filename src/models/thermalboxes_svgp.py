import torch
import gpytorch
from gpytorch import settings, distributions
from .svgp import SVGP
from .utils import compute_means, compute_I, compute_covariance


class ThermalBoxesSVGP(SVGP):
    def __init__(self, scenario_dataset, inducing_points, kernel, likelihood, d_fn, q_fn):
        super().__init__(inducing_points=inducing_points,
                         kernel=kernel,
                         likelihood=likelihood)
        # Register input data
        self.train_scenarios = scenario_dataset

        # Register d and q functions
        self.d_fn = d_fn
        self.q_fn = q_fn

    def _compute_d(self, scenario_dataset):
        lat = scenario_dataset.lat
        ocean_mask = scenario_dataset.ocean_mask
        d = self.d_fun(lat=lat, ocean_mask=ocean_mask)
        return d

    def _compute_q(self, scenario_dataset):
        lat = scenario_dataset.lat
        ocean_mask = scenario_dataset.ocean_mask
        q = self.q_fun(lat=lat, ocean_mask=ocean_mask)
        return q

    def _compute_mean(self, scenario_dataset):
        d = self._compute_d(scenario_dataset)
        q = self._compute_q(scenario_dataset)
        mean = self.compute_mean(scenario_dataset, d, q)
        return mean

    def _compute_covariance(self, scenario_dataset):
        # somehow compute gram matrix
        return Kj

    def forward(self, scenario_dataset):
        mean = self._compute_mean(scenario_dataset)
        Kj = self._compute_covariance(scenario_dataset)
        covar = gpytorch.add_jitter(Kj.sum(dim=-1))
        prior_dist = distributions.MultivariateNormal(mean, covar)
        return prior_dist
