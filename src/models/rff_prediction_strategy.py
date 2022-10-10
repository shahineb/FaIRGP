"""
Stolen from https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/models/exact_prediction_strategies.py and
adapted to our needs
"""
import torch
from gpytorch.lazy import lazify, MatmulLazyTensor
from gpytorch.utils.memoize import cached


def prediction_strategy(train_rffs, train_targets, likelihood):
    return RFFPredictionStrategy(train_rffs, train_targets, likelihood)


class RFFPredictionStrategy(object):
    def __init__(self, train_rffs, train_targets, likelihood):
        # Get training shape
        self._train_shape = train_rffs.size(0)
        self.train_rffs = train_rffs
        self.train_targets = train_targets
        self.likelihood = likelihood

    @property
    @cached(name="covar_cache")
    def ZTZ(self):
        ZTZ = self.train_rffs.t() @ self.train_rffs
        return ZTZ

    @property
    @cached(name="covar_cache")
    def ZTZ_σ2_inv(self):
        ZTZ_σ2 = torch.clone(self.ZTZ)
        ZTZ_σ2.view(-1)[::self.ZTZ.size(0) + 1].add_(self.likelihood.noise)
        ZTZ_σ2_inv = torch.cholesky_inverse(ZTZ_σ2)
        return ZTZ_σ2_inv

    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        covar_cache = self.ZTZ_σ2_inv @ self.ZTZ
        return covar_cache

    @property
    @cached(name="mean_cache")
    def mean_cache(self):
        ZTy = self.train_rffs.t() @ self.train_targets
        mean_cache = self.ZTZ_σ2_inv @ ZTy
        return mean_cache

    @property
    def num_train(self):
        return self._train_shape.numel()

    @property
    def train_shape(self):
        return self._train_shape

    def rff_predictive_mean(self, test_mean, test_rffs):
        res = test_rffs @ self.mean_cache
        res = res + test_mean
        return res

    def rff_predictive_covar(self, test_test_covar, test_rffs):
        covar_correction_rhs = test_rffs @ self.covar_cache @ test_rffs.t()
        if torch.is_tensor(test_test_covar):
            return lazify(
                torch.add(
                    test_test_covar, covar_correction_rhs, alpha=-1
                )
            )
        else:
            return test_test_covar + MatmulLazyTensor(
                covar_correction_rhs.mul(-1)
            )
