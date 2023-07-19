import torch
import gpytorch
from linear_operator.utils.cholesky import psd_safe_cholesky


class MultiExactGP(gpytorch.models.ExactGP):
    def __init__(self, X, y, mean, kernel, likelihood, mu, sigma, mu_targets, sigma_targets):
        super().__init__(X, y, likelihood)
        self.mean = mean
        self.kernel = kernel
        self.mu_targets = mu_targets
        self.sigma_targets = sigma_targets
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        mean_x = self.mean(x)
        covar_x = self.kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(self, x, diag=True):
        ntrain = len(self.train_inputs[0])
        full_data = torch.cat([self.train_inputs[0], x])
        full_output = self.forward(full_data)
        noisy_full_output = self.likelihood(full_output)

        full_covar = full_output.lazy_covariance_matrix
        train_test_covar = full_covar[:ntrain, ntrain:]
        test_test_covar = full_covar[ntrain:, ntrain:]
        train_train_covar = noisy_full_output.lazy_covariance_matrix[:ntrain, :ntrain]

        train_mean = full_output.mean[:ntrain].view(-1, 1)
        test_mean = full_output.mean[ntrain:].view(-1, 1)

        chol = psd_safe_cholesky(train_train_covar.to_dense())
        interp = torch.cholesky_solve(train_test_covar.to_dense(), chol)

        posterior_mean = test_mean + interp.T @ (self.train_targets - train_mean)
        if diag:
            posterior_var = test_test_covar.diag() - interp.mul(train_test_covar).sum(dim=0)
            output = torch.distributions.Normal(posterior_mean.T, posterior_var)
        else:
            posterior_covar = test_test_covar - train_test_covar.T @ interp
            output = gpytorch.distributions.MultivariateNormal(posterior_mean.T, posterior_covar)
        return output
