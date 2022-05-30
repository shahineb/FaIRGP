"""
Nomenclature:
    - `scenarios` : scenario dataset used for training
    - `fair_kwargs` : default keyed parameters to use in FaIR runs
"""
import os
import sys
from collections import namedtuple
from .preprocess_data import load_emissions_dataset, load_response_dataset, make_scenario, get_fair_params

base_dir = os.path.join(os.getcwd(), '..')
sys.path.append(base_dir)

from src.structures import ScenarioDataset


field_names = ['scenarios',
               'fair_kwargs',
               'misc']
Data = namedtuple(typename='Data', field_names=field_names, defaults=(None,) * len(field_names))


def make_data(cfg):
    """Prepares and formats data to be used for training and testing.
    Returns all data objects needed to run experiment encapsulated in a namedtuple.
    Returned elements are not comprehensive and minimially needed to run experiments,
        they can be subject to change depending on needs.

    Args:
        cfg (dict): configuration file

    Returns:
        type: Data

    """
    # Load emissions and temperature response xarrays
    inputs_filepaths = {key: os.path.join(cfg['dataset']['dirpath'], f"inputs_{key}.nc") for key in cfg['dataset']['keys']}
    outputs_filepaths = {key: os.path.join(cfg['dataset']['dirpath'], f"outputs_{key}.nc") for key in cfg['dataset']['keys']}
    input_xarrays = {key: load_emissions_dataset(filepath) for (key, filepath) in inputs_filepaths.items()}
    output_xarrays = {key: load_response_dataset(filepath) for (key, filepath) in outputs_filepaths.items()}

    # Create scenario instances
    hist_scenario = make_scenario('historical')
    scenarios = {'historical': hist_scenario}
    for key in cfg['dataset']['keys']:
        scenarios[key] = make_scenario(name=key,
                                       inputs=input_xarrays,
                                       outputs=output_xarrays,
                                       hist_scenario=hist_scenario)

    # Encapsulate into scenario dataset
    scenarios = ScenarioDataset(scenarios=list(scenarios.values()), hist_scenario=hist_scenario)

    # Load FaIR NORESM2-LM tuned parameters
    fair_kwargs = get_fair_params()

    # Encapsulate into named tuple object
    kwargs = {'scenarios': scenarios,
              'fair_kwargs': fair_kwargs}
    data = Data(**kwargs)
    return data
