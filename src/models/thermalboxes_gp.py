import torch
import gpytorch
from gpytorch import settings, distributions
from gpytorch.models import GP
from .exact_prediction_strategy import prediction_strategy
from .utils import make_stacked_timeseries, compute_mean, compute_I, compute_covariance, add_ts


class ThermalBoxesGP(GP):
    def __init__(self, timeseries, kernels, q, d, likelihood):
        super().__init__()
        # Register input data
        stacked_ts = make_stacked_timeseries(timeseries)
        self.train_ts = timeseries
        self.train_stacked_ts = stacked_ts
        self.register_buffer('train_tas', torch.cat([tas[self.train_ts.slices[key]] for key, tas in self.train_ts.tas.items()]))

        # Compute train data mean and stddev
        data = torch.cat([e[self.train_ts.slices[key]] for (key, e) in self.train_ts.emissions.items()])
        self.mu = data.mean(dim=0)
        self.sigma = data.std(dim=0)

        # Setup mean, kernel and likelihood
        self.register_buffer('train_mean', self._compute_mean(self.train_ts))
        self.kernels = torch.nn.ModuleList(kernels)
        self.likelihood = likelihood

        # Create training targets
        train_targets = self.train_tas - self.train_mean
        self.mu_targets = train_targets.mean()
        self.sigma_targets = train_targets.std()
        self.register_buffer('train_targets', train_targets)

        # Register thermal boxes parameters
        self.d = torch.from_numpy(d).float()
        self.q = torch.from_numpy(q).float()
        self.nboxes = len(d)

        # Initialize prediction strategy
        self.prediction_strategy = None

    def _clear_cache(self):
        self.prediction_strategy = None

    def _setup_prediction_strategy(self):
        train_prior_dist = self.train_prior_dist()

        # Create the prediction strategy for
        self.prediction_strategy = prediction_strategy(train_prior_dist=train_prior_dist,
                                                       train_targets=self.train_targets,
                                                       likelihood=self.likelihood)

    def _compute_mean(self, ts):
        _, means = compute_mean(ts)
        means = torch.cat([v for v in means.values()]).sum(dim=-1)
        return means

    def _compute_covariance(self, stacked_ts, ts):
        I = compute_I(stacked_ts, ts, self.kernels, self.d, self.mu, self.sigma)
        Kj = compute_covariance(I, stacked_ts, ts, self.q, self.d)
        return Kj

    def train_prior_dist(self):
        Kj = self._compute_covariance(self.train_stacked_ts, self.train_ts)
        # train_mean = self.train_mean.sum(dim=-1)
        train_mean = torch.zeros(self.train_mean.size(0))
        train_covar = gpytorch.add_jitter(Kj.sum(dim=-1))
        prior = distributions.MultivariateNormal(train_mean, train_covar)
        train_prior_dist = self.likelihood(prior)
        return train_prior_dist

    def forward(self, ts):
        # mean = self._compute_mean(ts)
        # mean = mean.sum(dim=-1)
        mean = torch.zeros(sum([s.stop - s.start for s in ts.slices.values()]))
        stacked_ts = make_stacked_timeseries(ts)
        Kj = self._compute_covariance(stacked_ts, ts)
        covar = gpytorch.add_jitter(Kj.sum(dim=-1))
        mvn = distributions.MultivariateNormal(mean, covar)
        return mvn

    def __call__(self, *args, **kwargs):
        # Training mode: optimizing
        if self.training:
            res = self.train_prior_dist()
            return res

        # Prior mode
        elif settings.prior_mode.on():
            res = super().__call__(*args, **kwargs)
            return res

        # Posterior mode
        else:
            if self.prediction_strategy is None:
                self._setup_prediction_strategy()

            # Concatenate the input to the training input
            full_ts = add_ts(self.train_ts, args[0])

            # Get the joint distribution for training/test data
            full_prior_dist = super().__call__(full_ts, **kwargs)
            full_mean, full_covar = full_prior_dist.loc, full_prior_dist.lazy_covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_prior_dist.batch_shape
            joint_shape = full_prior_dist.event_shape
            test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0]])

            # Make the prediction
            with settings._use_eval_tolerance():
                predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(full_mean, full_covar)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_prior_dist.__class__(predictive_mean, predictive_covar)
