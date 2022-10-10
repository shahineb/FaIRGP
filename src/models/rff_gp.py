import numpy as np
import torch
import gpytorch
from gpytorch import settings, distributions
from gpytorch.models import GP
from .rff_prediction_strategy import prediction_strategy


class RFFGP(GP):
    def __init__(self, X, y, mean, kernel, likelihood, mu, sigma, mu_targets, sigma_targets, seed):
        super().__init__()
        self.register_buffer('train_inputs', X)
        self.register_buffer('train_targets', y)
        self.mean = mean
        self.kernel = kernel
        self.likelihood = likelihood
        self.mu_targets = mu_targets
        self.sigma_targets = sigma_targets
        self.mu = mu
        self.sigma = sigma

        # Initialize random fourier features weights
        self._setup_rffs(seed=seed)

        # Initialize prediction strategy
        self.prediction_strategy = None

    def _setup_rffs(self, seed):
        # Sample random fourier features weights
        torch.random.manual_seed(seed)
        self.kernel._init_weights(self.train_inputs.size(-1), self.kernel.num_samples)

    def _clear_cache(self):
        self.prediction_strategy = None

    def _setup_prediction_strategy(self):
        # Compute train random fourier features - shape = (n_train, self.kernel.num_samples)
        Z = self.kernel._featurize(self.train_inputs, normalize=True)

        # Create the prediction strategy
        self.prediction_strategy = prediction_strategy(train_rffs=Z,
                                                       train_targets=self.train_targets,
                                                       likelihood=self.likelihood)

    def forward(self, x):
        mean = self.mean(x)
        K = self.kernel(x, x)
        covar = gpytorch.add_jitter(K)
        prior_dist = distributions.MultivariateNormal(mean, covar)
        return prior_dist

    def mll(self):
        # Compute train random fourier features - shape = (n_train, self.kernel.num_samples)
        n_train = self.train_inputs.size(0)
        Z = self.kernel._featurize(self.train_inputs, normalize=True)

        # Compute RFF gram matrix
        ZTZ = Z.t() @ Z

        # Compute noisy RFF gram matrix
        σ2 = self.likelihood.noise
        ZTZ_σ2 = torch.clone(ZTZ)
        ZTZ_σ2.view(-1)[::ZTZ.size(0) + 1].add_(σ2)

        # Compute y terms
        ZTy = Z.t() @ self.train_targets
        norm_y = torch.square(self.train_targets).sum()

        # Compute inverse quadratic and logdet terms
        yTZ_ZTZ_σ2_inv_ZTy, logdet = gpytorch.inv_quad_logdet(input=ZTZ_σ2, rhs=ZTy, logdet=True)
        inverse_quad = -yTZ_ZTZ_σ2_inv_ZTy / σ2
        inverse_quad.view(-1)[::inverse_quad.size(0) + 1].add_(norm_y / σ2)

        # Warp terms together
        res = -0.5 * (n_train * torch.log(np.pi) + logdet + inverse_quad)
        return res

    def __call__(self, *args, **kwargs):
        # Training mode: optimizing
        if self.training:
            res = NotImplemented()  # way to evaluate marginal loglikelihood
            return res

        # Prior mode
        elif settings.prior_mode.on():
            res = super().__call__(*args, **kwargs)
            return res

        # Posterior mode
        else:
            if self.prediction_strategy is None:
                self._setup_prediction_strategy()

            # Extract test data
            x_test = args[0]
            if x_test.ndim < 3:
                x_test = x_test.unsqueeze(0)

            # Get the random fourier features on the test data - shape = (batch_size, n_samples, self.kernel.num_samples)
            Z_test = self.kernel._featurize(x_test, normalize=True)

            # Determine the shape of the posterior distribution
            batch_shape = torch.Size([x_test.size(0)])
            test_shape = torch.Size([x_test.size(1)])

            # Make the prediction
            with settings._use_eval_tolerance():
                predictive_mean, predictive_covar = self.prediction_strategy.rff_prediction(Z_test)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            predictive_dist = distributions.MultivariateNormal(predictive_mean, predictive_covar)
            return predictive_dist
