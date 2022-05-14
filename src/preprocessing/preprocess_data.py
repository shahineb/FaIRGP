import xarray as xr
from .constants import GtC_to_GtCO2, Gt_to_Mt


def load_emissions_dataset(filepath):
    inputs = xr.open_dataset(filepath).compute()
    inputs.CO2.data = inputs.CO2.data / GtC_to_GtCO2
    inputs.CO2.attrs['units'] = 'GtC'
    inputs.CH4.data = inputs.CH4.data * Gt_to_Mt
    inputs.CH4.attrs['units'] = 'MtCH4'
    inputs.SO2.data = inputs.SO2.data * Gt_to_Mt
    inputs.SO2.attrs['units'] = 'MtSO2'
    inputs.BC.data = inputs.BC.data * Gt_to_Mt
    inputs.BC.attrs['units'] = 'MtBC'
    return inputs


def load_response_dataset(filepath):
    outputs = xr.open_dataset(filepath).compute()
    return outputs
