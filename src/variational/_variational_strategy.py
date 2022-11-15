from abc import ABC, abstractproperty
import torch
import torch.nn as nn
from gpytorch import settings
from gpytorch.utils.memoize import cached, clear_cache_hook


class _ScenarioVariationalStrategy(nn.Module, ABC):
    """
    Reimplements https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/variational/_variational_strategy.py but
    adapting it to our needs
    """

    def __init__(self, model, inducing_scenario, variational_distribution, jitter_val=None):
        super().__init__()

        if jitter_val is None:
            self.jitter_val = settings.variational_cholesky_jitter.value(inducing_scenario.inputs.dtype)
        else:
            self.jitter_val = jitter_val

        # Model
        object.__setattr__(self, "model", model)

        # Inducing points
        self.inducing_scenario = inducing_scenario

        # Variational distribution
        self._variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(0))

    def _clear_cache(self):
        clear_cache_hook(self)

    @abstractproperty
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.
        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`p( \mathbf u)`
        """
        raise NotImplementedError

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self):
        return self._variational_distribution()

    def forward(self, x, inducing_scenario, inducing_values, variational_inducing_covar=None, **kwargs):
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
        inducing point function values. Specifically, forward defines how to transform a variational distribution
        over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
        specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`
        :param torch.Tensor x: Locations :math:`\mathbf X` to get the
            variational posterior of the function values at.
        :param torch.Tensor inducing_points: Locations :math:`\mathbf Z` of the inducing points
        :param torch.Tensor inducing_values: Samples of the inducing function values :math:`\mathbf u`
            (or the mean of the distribution :math:`q(\mathbf u)` if q is a Gaussian.
        :param ~linear_operator.operators.LinearOperator variational_inducing_covar: If
            the distribuiton :math:`q(\mathbf u)` is
            Gaussian, then this variable is the covariance matrix of that Gaussian.
            Otherwise, it will be None.
        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`q( \mathbf f(\mathbf X))`
        """
        raise NotImplementedError

    def kl_divergence(self):
        r"""
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.
        :rtype: torch.Tensor
        """
        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return kl_divergence

    def __call__(self, **kwargs):
        # Delete previously cached items from the training distribution
        if self.training:
            self._clear_cache()
        # (Maybe) initialize variational distribution
        if not self.variational_params_initialized.item():
            prior_dist = self.prior_distribution
            self._variational_distribution.initialize_variational_distribution(prior_dist)
            self.variational_params_initialized.fill_(1)

        # Get p(u)/q(u)
        variational_dist_u = self.variational_distribution

        # Get q(f)
        return super().__call__(
            inducing_values=variational_dist_u.mean,
            variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
            **kwargs,
        )
