"""
Stolen from https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/models/exact_prediction_strategies.py and
(slightly) adapted to our needs
"""
import functools
import torch
from gpytorch import settings
from gpytorch.lazy import (
    LazyEvaluatedKernelTensor,
    MatmulLazyTensor,
    RootLazyTensor,
    ZeroLazyTensor,
    delazify,
    lazify,
)
from gpytorch.utils.memoize import add_to_cache, cached, clear_cache_hook


def prediction_strategy(train_prior_dist, train_targets, likelihood):
    train_train_covar = train_prior_dist.lazy_covariance_matrix
    if isinstance(train_train_covar, LazyEvaluatedKernelTensor):
        cls = train_train_covar.kernel.prediction_strategy
    else:
        cls = DefaultPredictionStrategy
    return cls(train_prior_dist, train_targets, likelihood)


class DefaultPredictionStrategy(object):
    def __init__(self, train_prior_dist, train_targets, likelihood, root=None, inv_root=None):
        # Get training shape
        self._train_shape = train_prior_dist.event_shape
        self.train_prior_dist = train_prior_dist
        self.train_targets = train_targets
        self.likelihood = likelihood
        self._last_test_train_covar = None
        mvn = self.likelihood(train_prior_dist)
        self.lik_train_train_covar = mvn.lazy_covariance_matrix

        if root is not None:
            add_to_cache(self.lik_train_train_covar, "root_decomposition", RootLazyTensor(root))

        if inv_root is not None:
            add_to_cache(self.lik_train_train_covar, "root_inv_decomposition", RootLazyTensor(inv_root))

    def _exact_predictive_covar_inv_quad_form_cache(self, train_train_covar_inv_root, test_train_covar):
        """
        Computes a cache for K_X*X (K_XX + sigma^2 I)^-1 K_X*X if possible. By default, this does no work and returns
        the first argument.
        Args:
            train_train_covar_inv_root (:obj:`torch.tensor`): a root of (K_XX + sigma^2 I)^-1
            test_train_covar (:obj:`torch.tensor`): the observed noise (from the likelihood)
        Returns
            - A precomputed cache
        """
        res = train_train_covar_inv_root
        if settings.detach_test_caches.on():
            res = res.detach()

        if res.grad_fn is not None:
            wrapper = functools.partial(clear_cache_hook, self)
            functools.update_wrapper(wrapper, clear_cache_hook)
            res.grad_fn.register_hook(wrapper)

        return res

    def _exact_predictive_covar_inv_quad_form_root(self, precomputed_cache, test_train_covar):
        r"""
        Computes :math:`K_{X^{*}X} S` given a precomputed cache
        Where :math:`S` is a tensor such that :math:`SS^{\top} = (K_{XX} + \sigma^2 I)^{-1}`
        Args:
            precomputed_cache (:obj:`torch.tensor`): What was computed in _exact_predictive_covar_inv_quad_form_cache
            test_train_covar (:obj:`torch.tensor`): The observed noise (from the likelihood)
        Returns
            :obj:`~gpytorch.lazy.LazyTensor`: :math:`K_{X^{*}X} S`
        """
        # Here the precomputed cache represents S,
        # where S S^T = (K_XX + sigma^2 I)^-1
        return test_train_covar.matmul(precomputed_cache)

    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        train_train_covar = self.lik_train_train_covar
        train_train_covar_inv_root = delazify(train_train_covar.root_inv_decomposition().root)
        return self._exact_predictive_covar_inv_quad_form_cache(train_train_covar_inv_root, self._last_test_train_covar)

    @property
    @cached(name="mean_cache")
    def mean_cache(self):
        mvn = self.likelihood(self.train_prior_dist)
        train_mean, train_train_covar = mvn.loc, mvn.lazy_covariance_matrix

        train_targets_offset = (self.train_targets - train_mean).unsqueeze(-1)
        mean_cache = train_train_covar.evaluate_kernel().inv_matmul(train_targets_offset).squeeze(-1)

        if settings.detach_test_caches.on():
            mean_cache = mean_cache.detach()

        if mean_cache.grad_fn is not None:
            wrapper = functools.partial(clear_cache_hook, self)
            functools.update_wrapper(wrapper, clear_cache_hook)
            mean_cache.grad_fn.register_hook(wrapper)

        return mean_cache

    @property
    def num_train(self):
        return self._train_shape.numel()

    @property
    def train_shape(self):
        return self._train_shape

    def exact_prediction(self, joint_mean, joint_covar):
        # Find the components of the distribution that contain test data
        test_mean = joint_mean[..., self.num_train:]
        # For efficiency - we can make things more efficient
        if joint_covar.size(-1) <= settings.max_eager_kernel_size.value():
            test_covar = joint_covar[..., self.num_train:, :].evaluate()
            test_test_covar = test_covar[..., self.num_train:]
            test_train_covar = test_covar[..., : self.num_train]
        else:
            test_test_covar = joint_covar[..., self.num_train:, self.num_train:]
            test_train_covar = joint_covar[..., self.num_train:, :self.num_train]

        return (
            self.exact_predictive_mean(test_mean, test_train_covar),
            self.exact_predictive_covar(test_test_covar, test_train_covar),
        )

    def exact_predictive_mean(self, test_mean, test_train_covar):
        """
        Computes the posterior predictive covariance of a GP
        Args:
            test_mean (:obj:`torch.tensor`): The test prior mean
            test_train_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test and train inputs
        Returns:
            :obj:`torch.tensor`: The predictive posterior mean of the test points
        """
        # NOTE TO FUTURE SELF:
        # You **cannot* use addmv here, because test_train_covar may not actually be a non lazy tensor even for an exact
        # GP, and using addmv requires you to delazify test_train_covar, which is obviously a huge no-no!
        res = (test_train_covar @ self.mean_cache.unsqueeze(-1)).squeeze(-1)
        res = res + test_mean

        return res

    def exact_predictive_covar(self, test_test_covar, test_train_covar):
        """
        Computes the posterior predictive covariance of a GP
        Args:
            test_train_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test and train inputs
            test_test_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test inputs
        Returns:
            :obj:`gpytorch.lazy.LazyTensor`: A LazyTensor representing the predictive posterior covariance of the
                                               test points
        """
        if settings.fast_pred_var.on():
            self._last_test_train_covar = test_train_covar

        if settings.skip_posterior_variances.on():
            return ZeroLazyTensor(*test_test_covar.size())

        if settings.fast_pred_var.off():
            dist = self.train_prior_dist.__class__(
                torch.zeros_like(self.train_prior_dist.mean), self.train_prior_dist.lazy_covariance_matrix
            )
            if settings.detach_test_caches.on():
                train_train_covar = self.likelihood(dist).lazy_covariance_matrix.detach()
            else:
                train_train_covar = self.likelihood(dist).lazy_covariance_matrix

            test_train_covar = delazify(test_train_covar)
            train_test_covar = test_train_covar.transpose(-1, -2)
            covar_correction_rhs = train_train_covar.inv_matmul(train_test_covar)
            # For efficiency
            if torch.is_tensor(test_test_covar):
                # We can use addmm in the 2d case
                if test_test_covar.dim() == 2:
                    return lazify(
                        torch.addmm(test_test_covar, test_train_covar, covar_correction_rhs, beta=1, alpha=-1)
                    )
                else:
                    return lazify(test_test_covar + test_train_covar @ covar_correction_rhs.mul(-1))
            # In other cases - we'll use the standard infrastructure
            else:
                return test_test_covar + MatmulLazyTensor(test_train_covar, covar_correction_rhs.mul(-1))

        precomputed_cache = self.covar_cache
        covar_inv_quad_form_root = self._exact_predictive_covar_inv_quad_form_root(precomputed_cache, test_train_covar)
        if torch.is_tensor(test_test_covar):
            return lazify(
                torch.add(
                    test_test_covar, covar_inv_quad_form_root @ covar_inv_quad_form_root.transpose(-1, -2), alpha=-1
                )
            )
        else:
            return test_test_covar + MatmulLazyTensor(
                covar_inv_quad_form_root, covar_inv_quad_form_root.transpose(-1, -2).mul(-1)
            )
