from gpytorch.models import ApproximateGP
from gpytorch import means, distributions
from gpytorch import variational


class SVGP(ApproximateGP):
    def __init__(self, inducing_points, kernel, likelihood):
        variational_strategy = self._set_variational_strategy(inducing_points)
        super().__init__(variational_strategy=variational_strategy)
        self.mean = means.ZeroMean()
        self.kernel = kernel
        self.likelihood = likelihood

    def _set_variational_strategy(self, inducing_points):
        # Use gaussian variational family
        variational_distribution = variational.CholeskyVariationalDistribution(num_inducing_points=inducing_points.size(0))
        # Set default variational approximation strategy
        variational_strategy = variational.VariationalStrategy(model=self,
                                                               inducing_points=inducing_points,
                                                               variational_distribution=variational_distribution,
                                                               learn_inducing_locations=True)
        return variational_strategy

    def forward(self, inputs):
        # Compute mean vector and covariance matrix on input samples
        mean = self.mean(inputs)
        covar = self.kernel(inputs)
        # Build multivariate normal distribution of model evaluated on input samples
        prior_distribution = distributions.MultivariateNormal(mean=mean,
                                                              covariance_matrix=covar)
        return prior_distribution
