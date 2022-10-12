import math
import torch
import gpytorch
from gpytorch import settings, distributions
from gpytorch.models import GP
from .rff_prediction_strategy import prediction_strategy


class RFFGP(GP):
    def __init__(self, X, y, mean, kernel, likelihood, mu, sigma, mu_targets, sigma_targets, seed):
        super().__init__()
        self.register_buffer('train_inputs', X.view(X.size(0), -1))
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

    def mll(self, X=None, y=None):
        # Replace none args
        if X is None or y is None:
            X = self.train_inputs
            y = self.train_targets

        # Compute train random fourier features - shape = (n_train, self.kernel.num_samples)
        Z = self.kernel._featurize(X, normalize=True)
        n, p = Z.shape

        # Compute RFF gram matrix
        ZTZ = Z.t() @ Z

        # Compute noisy RFF gram matrix
        σ2 = self.likelihood.noise
        ZTZ_σ2 = torch.clone(ZTZ)
        ZTZ_σ2.view(-1)[::ZTZ.size(0) + 1].add_(σ2)

        # Compute y terms
        ZTy = Z.t() @ y
        norm_y = torch.square(y).sum()

        # Compute inverse quadratic term
        # yTZ_ZTZ_σ2_inv_ZTy, logdet = gpytorch.inv_quad_logdet(mat=ZTZ_σ2, inv_quad_rhs=ZTy.unsqueeze(-1), logdet=True)
        # inverse_quad = (norm_y - yTZ_ZTZ_σ2_inv_ZTy) / σ2
        L = torch.linalg.cholesky(ZTZ_σ2)
        ZTZ_σ2_inv_ZTy = torch.cholesky_solve(ZTy.unsqueeze(-1), L).squeeze()
        inv_quad = (norm_y - ZTy.t() @ ZTZ_σ2_inv_ZTy) / σ2

        # Compute logdet
        tσ = σ2 ** (n / p)
        sylvester = ZTZ.mul(tσ / σ2)
        sylvester.view(-1)[::sylvester.size(0) + 1].add_(tσ)
        logdet = torch.logdet(sylvester)

        # Warp terms together
        # return (np.log(np.pi), logdet / n_train, inverse_quad / n_train)
        res = -0.5 * (math.log(2 * math.pi) + logdet.div(n) + inv_quad.div(n))
        return res

    def __call__(self, *args, **kwargs):
        # Training mode: optimizing
        if self.training:
            res = self.mll()
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

            # Get the random fourier features on the test data - shape = (batch_size, n_samples, self.kernel.num_samples)
            Z_test = self.kernel._featurize(x_test.view(x_test.size(0), -1), normalize=True)

            # Determine the shape of the posterior distribution
            test_shape = torch.Size([x_test.size(0)])

            # Make the prediction
            with settings._use_eval_tolerance():
                predictive_mean, predictive_covar = self.prediction_strategy.rff_prediction(Z_test)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*test_shape).contiguous()
            predictive_dist = distributions.MultivariateNormal(predictive_mean, predictive_covar)
            return predictive_dist
