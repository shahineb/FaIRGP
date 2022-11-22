import torch
from linear_operator import to_dense
from linear_operator.operators import (
    DiagLinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
    SumLinearOperator,
    TriangularLinearOperator,
)
from linear_operator import to_linear_operator
from linear_operator.utils.cholesky import psd_safe_cholesky
from gpytorch.distributions import MultivariateNormal
from gpytorch.settings import _linalg_dtype_cholesky, trace_mode
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from ._variational_strategy import _ScenarioVariationalStrategy


class ScenarioVariationalStrategy(_ScenarioVariationalStrategy):
    r"""
    Adapts code from https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/variational/variational_strategy.py
    to out needs
    """
    def __init__(self, model, inducing_scenario, variational_distribution, jitter_val=None):
        super().__init__(model, inducing_scenario, variational_distribution, jitter_val=jitter_val)
        self.register_buffer("updated_strategy", torch.tensor(True))

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar):
        L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()))
        return TriangularLinearOperator(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLinearOperator(ones))
        return res

    def forward(self, Kww, Kwx, Kxx, inducing_values, variational_inducing_covar=None, diag=False, **kwargs):
        if diag:
            return self.diag_forward(Kww, Kwx, Kxx, inducing_values, variational_inducing_covar, **kwargs)
        else:
            return self.full_forward(Kww, Kwx, Kxx, inducing_values, variational_inducing_covar, **kwargs)

    def diag_forward(self, Kww, Kwx, Kxx_diag, inducing_values, variational_inducing_covar, **kwargs):
        # Covariance terms
        test_mean = torch.zeros_like(Kxx_diag)
        induc_induc_covar = to_linear_operator(Kww).add_jitter(self.jitter_val)
        induc_data_covar = to_linear_operator(Kwx).to_dense()
        data_data_var = Kxx_diag

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.solve(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(test_mean.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean

        # Compute the variance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        middle_term = SumLinearOperator(variational_inducing_covar, middle_term)
        predictive_var = data_data_var + interp_term.mul(middle_term @ interp_term).sum(dim=0)
        diag_predictive_covar = DiagLinearOperator(predictive_var)
        # Return the distribution
        return MultivariateNormal(predictive_mean, diag_predictive_covar)

    def full_forward(self, Kww, Kwx, Kxx, inducing_values, variational_inducing_covar=None, **kwargs):
        # Covariance terms
        test_mean = torch.zeros_like(Kxx[0])
        induc_induc_covar = to_linear_operator(Kww).add_jitter(self.jitter_val)
        induc_data_covar = to_linear_operator(Kwx).to_dense()
        data_data_covar = to_linear_operator(Kxx)

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.solve(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(test_mean.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(self.jitter_val).to_dense()
                + interp_term.transpose(-1, -2) @ middle_term.to_dense() @ interp_term
            )
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                MatmulLinearOperator(interp_term.transpose(-1, -2), middle_term @ interp_term),
            )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, Kww, Kwx, Kxx, diag=False, **kwargs):
        if not self.updated_strategy.item():
            with torch.no_grad():
                # Get unwhitened p(u)
                prior_function_dist = MultivariateNormal(torch.zeros_like(Kww[0]), Kww)
                prior_mean = prior_function_dist.loc
                L = self._cholesky_factor(prior_function_dist.lazy_covariance_matrix.add_jitter(self.jitter_val))

                # Temporarily turn off noise that's added to the mean
                orig_mean_init_std = self._variational_distribution.mean_init_std
                self._variational_distribution.mean_init_std = 0.0

                # Change the variational parameters to be whitened
                variational_dist = self.variational_distribution
                mean_diff = (variational_dist.loc - prior_mean).unsqueeze(-1).type(_linalg_dtype_cholesky.value())
                whitened_mean = L.solve(mean_diff).squeeze(-1).to(variational_dist.loc.dtype)
                covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root.to_dense()
                covar_root = covar_root.type(_linalg_dtype_cholesky.value())
                whitened_covar = RootLinearOperator(L.solve(covar_root).to(variational_dist.loc.dtype))
                whitened_variational_distribution = variational_dist.__class__(whitened_mean, whitened_covar)
                self._variational_distribution.initialize_variational_distribution(whitened_variational_distribution)

                # Reset the random noise parameter of the model
                self._variational_distribution.mean_init_std = orig_mean_init_std

                # Reset the cache
                clear_cache_hook(self)

                # Mark that we have updated the variational strategy
                self.updated_strategy.fill_(True)

        return super().__call__(Kww=Kww, Kwx=Kwx, Kxx=Kxx, diag=diag, **kwargs)
