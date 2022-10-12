import os
import sys
import torch
import torch.nn as nn
import functools

base_dir = os.path.join(os.getcwd(), '..')
sys.path.append(base_dir)

import src.fair as fair


class ScenarioDataset(nn.Module):
    """Utility class that encapsulates multiple scenario timeseries.

    Args:
        scenarios (list[Scenario])
        hist_scenario (Scenario): historical scenario.
    """
    def __init__(self, scenarios, hist_scenario):
        super().__init__()
        self.scenarios = nn.ModuleDict({s.name: s for s in scenarios})
        self.hist_scenario = hist_scenario
        self._init_trimming_slices()

    def _init_trimming_slices(self):
        slices_tensor = []
        self.full_slices = dict()
        start_idx = 0
        stop_idx = 0
        for name, scenario in self.scenarios.items():
            stop_idx += len(scenario.hist_scenario)
            slices_tensor.append(torch.arange(stop_idx, stop_idx + len(scenario)))
            stop_idx += len(scenario)
            self.full_slices.update({name: slice(start_idx, stop_idx)})
            start_idx = stop_idx
        self.register_buffer('slices_tensor', torch.cat(slices_tensor))

    def trim_hist(self, timeserie):
        output = timeserie[self.slices_tensor]
        return output

    def __getitem__(self, idx):
        if isinstance(idx, str):
            output = self.scenarios[idx]
        elif isinstance(idx, int):
            output = self.scenarios[self.names[idx]]
        else:
            raise TypeError
        return output

    def __setitem__(self, key, value):
        if isinstance(key, str) and isinstance(value, Scenario):
            self.scenarios.update({key: value})
            self._clear_cache()
            self._init_trimming_slices()
        else:
            raise TypeError

    def update(self, kwargs):
        for key, value in kwargs.items():
            self.__setitem__(key, value)

    def __add__(self, scenario_dataset):
        left_scenarios = list(self.scenarios.values())
        right_scenarios = list(scenario_dataset.scenarios.values())
        return self.__class__(left_scenarios + right_scenarios, self.hist_scenario)

    def _clear_cache(self):
        try:
            del self.names
        except AttributeError:
            pass
        try:
            del self.timesteps
        except AttributeError:
            pass
        try:
            del self.emissions
        except AttributeError:
            pass
        try:
            del self.tas
        except AttributeError:
            pass
        try:
            del self.cum_emissions
        except AttributeError:
            pass
        try:
            del self.concentrations
        except AttributeError:
            pass
        try:
            del self.inputs
        except AttributeError:
            pass
        try:
            del self.full_timesteps
        except AttributeError:
            pass
        try:
            del self.full_emissions
        except AttributeError:
            pass
        try:
            del self.full_tas
        except AttributeError:
            pass
        try:
            del self.full_cum_emissions
        except AttributeError:
            pass
        try:
            del self.full_concentrations
        except AttributeError:
            pass
        try:
            del self.full_inputs
        except AttributeError:
            pass
        # try:
        #     del self.mu_emissions
        # except AttributeError:
        #     pass
        # try:
        #     del self.sigma_emissions
        # except AttributeError:
        #     pass
        # try:
        #     del self.mu_concentrations
        # except AttributeError:
        #     pass
        # try:
        #     del self.sigma_concentrations
        # except AttributeError:
        #     pass
        # try:
        #     del self.mu_inputs
        # except AttributeError:
        #     pass
        # try:
        #     del self.sigma_inputs
        # except AttributeError:
        #     pass
        # try:
        #     del self.mu_tas
        # except AttributeError:
        #     pass
        # try:
        #     del self.sigma_tas
        # except AttributeError:
        #     pass

    @functools.cached_property
    def mu_tas(self):
        return self.tas.mean()

    @functools.cached_property
    def sigma_tas(self):
        return self.tas.std().clip(min=torch.finfo(torch.float32).eps)

    @functools.cached_property
    def mu_emissions(self):
        return self.emissions.mean(dim=0)

    @functools.cached_property
    def sigma_emissions(self):
        return self.emissions.std(dim=0).clip(min=torch.finfo(torch.float32).eps)

    @functools.cached_property
    def mu_concentrations(self):
        return self.concentrations.mean(dim=0)

    @functools.cached_property
    def sigma_concentrations(self):
        return self.concentrations.std(dim=0).clip(min=torch.finfo(torch.float32).eps)

    @functools.cached_property
    def mu_inputs(self):
        return self.inputs.mean(dim=0)

    @functools.cached_property
    def sigma_inputs(self):
        return self.inputs.std(dim=0).clip(min=torch.finfo(torch.float32).eps)

    @functools.cached_property
    def names(self):
        return list(self.scenarios.keys())

    @functools.cached_property
    def timesteps(self):
        timesteps = torch.cat([s.timesteps for s in self.scenarios.values()])
        return timesteps

    @functools.cached_property
    def emissions(self):
        emissions = torch.cat([s.emissions for s in self.scenarios.values()])
        return emissions

    @functools.cached_property
    def tas(self):
        tas = torch.cat([s.tas for s in self.scenarios.values()])
        return tas

    @functools.cached_property
    def cum_emissions(self):
        cum_emissions = torch.cat([s.cum_emissions for s in self.scenarios.values()])
        return cum_emissions

    @functools.cached_property
    def concentrations(self):
        concentrations = torch.cat([s.concentrations for s in self.scenarios.values()])
        return concentrations

    @functools.cached_property
    def inputs(self):
        inputs = torch.cat([s.inputs for s in self.scenarios.values()])
        return inputs

    @functools.cached_property
    def full_timesteps(self):
        full_timesteps = torch.cat([s.full_timesteps for s in self.scenarios.values()])
        return full_timesteps

    @functools.cached_property
    def full_emissions(self):
        full_emissions = torch.cat([s.full_emissions for s in self.scenarios.values()])
        return full_emissions

    @functools.cached_property
    def full_tas(self):
        full_tas = torch.cat([s.full_tas for s in self.scenarios.values()])
        return full_tas

    @functools.cached_property
    def full_cum_emissions(self):
        full_cum_emissions = torch.cat([s.full_cum_emissions for s in self.scenarios.values()])
        return full_cum_emissions

    @functools.cached_property
    def full_concentrations(self):
        full_concentrations = torch.cat([s.full_concentrations for s in self.scenarios.values()])
        return full_concentrations

    @functools.cached_property
    def full_inputs(self):
        full_inputs = torch.cat([s.full_inputs for s in self.scenarios.values()])
        return full_inputs

    def __len__(self):
        return len(self.scenarios)


class Scenario(nn.Module):
    """Utility class to represent a gas emission scenario timeserie with associated
    temperature response

    Args:
        timesteps (torch.Tensor): (n_timesteps,) tensor of dates of each time step as floats
        emissions (torch.Tensor): (n_timesteps, n_agents) or (n_timesteps, n_lat, n_lon, n_agents) tensor of emissions
        tas (torch.Tensor): (n_timesteps,) or (n_timesteps, n_lat, nlon) tensor of surface temperature anomaly
        name (str): name of time serie
        hist_scenario (Scenario): historical scenario needed to complete SSPs timeseries is this is a SSP scenario]

    """
    def __init__(self, timesteps, emissions, tas, lon=None, lat=None, name=None, hist_scenario=None):
        super().__init__()
        self.name = name
        self.register_buffer('timesteps', timesteps)
        self.register_buffer('emissions', emissions)
        self.register_buffer('tas', tas)
        self._register_buffer('lat', lat)
        self._register_buffer('lon', lon)
        self.hist_scenario = hist_scenario if hist_scenario else []

    def _register_buffer(self, name, tensor):
        if tensor is not None:
            self.register_buffer(name, tensor)

    def trim_hist(self, full_timeserie):
        """Takes in time serie of size (n_hist_timesteps + n_timesteps, -1) and truncates
            to timeserie with pi_scenario (also works if this is a historial scenario).

        Args:
            full_timeserie (torch.Tensor): (n_hist_timesteps + n_timesteps, -1)

        Returns:
            type: torch.Tensor

        """
        return full_timeserie[-len(self):]

    def _compute_fair_concentrations(self):
        if self.full_emissions.ndim == 2:
            emissions = self.full_emissions
        else:
            emissions = self.full_glob_emissions
        res = fair.run(time=self.full_timesteps.numpy(),
                       emission=emissions.T.numpy(),
                       base_kwargs=fair.get_params())
        full_concentrations = torch.from_numpy(res['C'].T).float()
        return full_concentrations

    def forward(self):
        return self

    @property
    def hist_timesteps(self):
        return self.hist_scenario.timesteps

    @property
    def hist_emissions(self):
        return self.hist_scenario.emissions

    @property
    def hist_tas(self):
        return self.hist_scenario.tas

    @property
    def glob_hist_emissions(self):
        return self.hist_scenario.glob_emissions

    @property
    def glob_hist_tas(self):
        return self.hist_scenario.glob_tas

    @functools.cached_property
    def weights(self):
        return torch.cos(torch.deg2rad(self.lat))

    @functools.cached_property
    def glob_emissions(self):
        weighted_emissions = self.emissions.mul(self.weights.view(1, -1, 1, 1))
        glob_emissions = weighted_emissions.sum(dim=(1, 2)).div(self.weights.sum() * len(self.lon))
        return glob_emissions

    @functools.cached_property
    def glob_tas(self):
        weighted_tas = self.tas.mul(self.weights.view(1, -1, 1))
        glob_tas = weighted_tas.sum(dim=(1, 2)).div(self.weights.sum() * len(self.lon))
        return glob_tas

    @functools.cached_property
    def full_timesteps(self):
        if self.hist_scenario:
            full_timesteps = torch.cat([self.hist_timesteps, self.timesteps])
        else:
            full_timesteps = self.timesteps
        return full_timesteps

    @functools.cached_property
    def full_emissions(self):
        if self.hist_scenario:
            full_emissions = torch.cat([self.hist_emissions, self.emissions])
        else:
            full_emissions = self.emissions
        return full_emissions

    @functools.cached_property
    def full_tas(self):
        if self.hist_scenario:
            full_tas = torch.cat([self.hist_tas, self.tas])
        else:
            full_tas = self.tas
        return full_tas

    @functools.cached_property
    def full_glob_emissions(self):
        if self.hist_scenario:
            full_glob_emissions = torch.cat([self.glob_hist_emissions, self.glob_emissions])
        else:
            full_glob_emissions = self.glob_emissions
        return full_glob_emissions

    @functools.cached_property
    def full_glob_tas(self):
        if self.hist_scenario:
            full_glob_tas = torch.cat([self.glob_hist_tas, self.glob_tas])
        else:
            full_glob_tas = self.glob_tas
        return full_glob_tas

    @functools.cached_property
    def full_cum_emissions(self):
        return torch.cumsum(self.full_emissions, dim=0)

    @functools.cached_property
    def full_glob_cum_emissions(self):
        return torch.cumsum(self.full_glob_emissions, dim=0)

    @functools.cached_property
    def full_concentrations(self):
        return self._compute_fair_concentrations()

    @functools.cached_property
    def full_inputs(self):
        # logCO2 = torch.log(self.full_concentrations[..., 0, None].clip(min=torch.finfo(torch.float32).eps) / 278.)
        # sqrtCO2 = torch.sqrt(self.full_concentrations[..., 0, None].clip(min=torch.finfo(torch.float32).eps))
        # sqrtCH4 = torch.sqrt(self.full_concentrations[..., 1, None].clip(min=torch.finfo(torch.float32).eps))
        if self.emissions.ndim == 2:
            full_inputs = torch.cat([self.full_timesteps.unsqueeze(-1),
                                     self.full_cum_emissions[..., 0, None],
                                     self.full_emissions[..., 1:]], dim=-1)
        else:
            full_time_lat_lon = torch.stack(torch.meshgrid(self.full_timesteps, self.lat, self.lon), dim=-1)
            full_inputs = torch.cat([full_time_lat_lon,
                                     self.full_cum_emissions[..., 0, None],
                                     self.full_emissions[..., 1:]], dim=-1)
        return full_inputs

    @functools.cached_property
    def cum_emissions(self):
        return self.full_cum_emissions[-len(self):]

    @functools.cached_property
    def concentrations(self):
        return self.trim_hist(self.full_concentrations)

    @functools.cached_property
    def inputs(self):
        # logCO2 = torch.log(self.concentrations[..., 0, None].clip(min=torch.finfo(torch.float32).eps) / 278.)
        # sqrtCO2 = torch.sqrt(self.concentrations[..., 0, None].clip(min=torch.finfo(torch.float32).eps))
        # sqrtCH4 = torch.sqrt(self.concentrations[..., 1, None].clip(min=torch.finfo(torch.float32).eps))
        if self.emissions.ndim == 2:
            inputs = torch.cat([self.timesteps.unsqueeze(-1),
                                self.cum_emissions[..., 0, None],
                                self.emissions[..., 1:]], dim=-1)
        else:
            time_lat_lon = torch.stack(torch.meshgrid(self.timesteps, self.lat, self.lon), dim=-1)
            inputs = torch.cat([time_lat_lon,
                                self.cum_emissions[..., 0, None],
                                self.emissions[..., 1:]], dim=-1)
        return inputs

    def __len__(self):
        return len(self.timesteps)

    def __repr__(self):
        try:
            output = f"Scenario({self.name}, time={len(self.timesteps)}, lat={len(self.lat)}, lon={len(self.lon)})"
        except AttributeError:
            output = f"Scenario({self.name})"
        return output
