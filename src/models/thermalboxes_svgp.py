import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch import means, distributions
from gpytorch import variational

from .utils_svgp import compute_means, compute_I, compute_I_scenario, compute_covariance, compute_inducing_covariance


class ThermalBoxesSVGP(ApproximateGP):
    def __init__(self, scenario_dataset, inducing_scenario, kernel, likelihood, FaIR_model):
        variational_strategy = self._set_variational_strategy(inducing_scenario)
        super().__init__(variational_strategy=variational_strategy)
        self.kernel = kernel
        self.train_scenarios = scenario_dataset
        self.FaIR_model = FaIR_model

    def _set_variational_strategy(self, inducing_scenario):
        # Use gaussian variational family
        variational_distribution = variational.CholeskyVariationalDistribution(num_inducing_points=inducing_scenario.num_inducing_points)
        # Set default variational approximation strategy
        variational_strategy = variational.VariationalStrategy(model=self,
                                                               inducing_scenario=inducing_scenario,
                                                               variational_distribution=variational_distribution)
        return variational_strategy

    def _compute_mean(self, scenario_dataset):
        d = scenario_dataset.d_map
        q = scenario_dataset.q_map
        mean = compute_means(scenario_dataset, self.FaIR_model, d, q)
        return mean

    def _compute_covariance(self, scenario_dataset):
        I = compute_I(scenario_dataset,
                      self.inducing_scenario,
                      self.kernel)
        Kwx = compute_covariance(scenario_dataset,
                                 self.inducing_scenario,
                                 I,
                                 scenario_dataset.q_map,
                                 scenario_dataset.d_map)
        Kwx.permute(1, 2, 3, 0, 4, 5).reshape(self.inducing_scenario.n_inducing_points, -1)

        inducing_I = compute_I_scenario(scenario_dataset,
                                        self.inducing_scenario,
                                        self.inducing_scenario,
                                        self.kernel)
        Kww = compute_inducing_covariance(scenario_dataset,
                                          self.inducing_scenario,
                                          inducing_I)
        Kww = Kww.permute(1, 2, 3, 0, 4, 5).reshape(self.inducing_scenario.n_inducing_points, -1)
        return Kwx, Kww

    def __call__(self, scenario_dataset, **kwargs):
        Kwx, Kww = self._compute_covariance(scenario_dataset)
        Kxx = torch.eye(Kwx.size(1))
        return self.variational_strategy.__call__(Kww=Kww, Kwx=Kwx, Kxx=Kxx)
