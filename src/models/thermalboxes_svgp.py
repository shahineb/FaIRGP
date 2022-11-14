import torch
import gpytorch
from gpytorch import settings, distributions
from .svgp import SVGP
from .utils_svgp import compute_means, compute_I, compute_covariance


class ThermalBoxesSVGP(SVGP):
    def __init__(self, scenario_dataset, inducing_points, kernel, likelihood, FaIR_model, d_fn, q_fn):
        super().__init__(inducing_points=inducing_points,
                         kernel=kernel,
                         likelihood=likelihood)
        # Register input data
        self.train_scenarios = scenario_dataset

        # Register FaIR module
        self.FaIR_model = FaIR_model

        # Register d and q functions
        self.d_fn = d_fn
        self.q_fn = q_fn

    def _compute_d(self, *args, **kwargs):
        d = self.d_fun(*args, **kwargs)
        return d

    def _compute_q(self, *args, **kwargs):
        q = self.q_fun(*args, **kwargs)
        return q

    def _compute_mean(self, scenario_dataset):
        d = self._compute_d()
        q = self._compute_q()
        mean = compute_means(scenario_dataset, self.FaIR_model, d, q)
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
