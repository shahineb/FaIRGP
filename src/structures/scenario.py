import torch
import torch.nn as nn
import functools


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
            del self.mu
        except AttributeError:
            pass
        try:
            del self.sigma
        except AttributeError:
            pass

    @functools.cached_property
    def mu(self):
        return self.emissions.mean(dim=0)

    @functools.cached_property
    def sigma(self):
        return self.emissions.std(dim=0)

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

    def __len__(self):
        return len(self.scenarios)


class Scenario(nn.Module):
    """Utility class to represent a gas emission scenario timeserie with associated
    temperature response

    Args:
        timesteps (torch.Tensor): (n_timesteps,) tensor of dates of each time step as floats
        emissions (torch.Tensor): (n_timesteps, n_agents) tensor of emissions
        tas (torch.Tensor): (n_timesteps,) tensor of surface temperature anomaly
        name (str): name of time serie
        hist_scenario (Scenario): since pre-industrial scenario, needed to complete SSPs timeseries
    """
    def __init__(self, timesteps, emissions, tas, name=None, hist_scenario=None):
        super().__init__()
        self.name = name
        self.register_buffer('timesteps', timesteps)
        self.register_buffer('emissions', emissions)
        self.register_buffer('tas', tas)
        self.hist_scenario = hist_scenario if hist_scenario else []

    def trim_hist(self, full_timeserie):
        """Takes in time serie of size (n_hist_timesteps + n_timesteps, -1) and truncates
            to timeserie with pi_scenario (also works if this is a historial scenario).

        Args:
            full_timeserie (torch.Tensor): (n_hist_timesteps + n_timesteps, -1)

        Returns:
            type: torch.Tensor

        """
        return full_timeserie[-len(self):]

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

    def __len__(self):
        return len(self.timesteps)

    def __repr__(self):
        return f"Scenario({self.name})"
