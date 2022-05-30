from gpytorch import models, distributions


class ExactGP(models.ExactGP):
    def __init__(self, X, y, mean, kernel, likelihood, mu, sigma, mu_targets, sigma_targets):
        super(ExactGP, self).__init__(X, y, likelihood)
        self.mean = mean
        self.kernel = kernel
        self.mu_targets = mu_targets
        self.sigma_targets = sigma_targets
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        mean_x = self.mean(x)
        covar_x = self.kernel(x)
        return distributions.MultivariateNormal(mean_x, covar_x)
