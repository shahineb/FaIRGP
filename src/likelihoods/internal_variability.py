import torch
from gpytorch import likelihoods, distributions
from scipy.spatial import distance_matrix


class InternalVariability(likelihoods.GaussianLikelihood):

    def __init__(self, d, q, add_observation_noise=False):
        super().__init__()
        self.d = torch.from_numpy(d).float()
        self.q = torch.from_numpy(q).float()
        self.add_observation_noise = add_observation_noise
        if self.add_observation_noise:
            self.observation_likelihood = likelihoods.GaussianLikelihood()

    @property
    def outputscale(self):
        return self.noise

    def _get_distance_matrix(self, size):
        v = torch.arange(0, size).reshape(-1, 1)
        mat = torch.from_numpy(distance_matrix(v, v)).float()
        return mat

    def compute_covariance(self, size, diag=False):
        nboxes = len(self.d)
        dist = self._get_distance_matrix(size)
        dist = dist.view(size, size, 1).repeat(1, 1, nboxes)
        covar = 0.5 * (self.q / self.d)**2 * torch.exp(-dist / self.d)
        covar = self.outputscale * covar.sum(dim=-1)
        if self.add_observation_noise:
            covar = covar + self.observation_likelihood.noise * torch.eye(size)
        return covar

    def forward(self, function_samples, *args, **kwargs):
        internal_covar = self.compute_covariance(len(function_samples))
        function_dist = distributions.MultivariateNormal(function_samples, internal_covar)
        return function_dist

    def marginal(self, function_dist, *args, **kwargs):
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        internal_covar = self.compute_covariance(mean.size(-1))
        full_covar = covar + internal_covar
        return function_dist.__class__(mean, full_covar)
