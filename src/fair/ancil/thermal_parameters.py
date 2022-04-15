import os
import pandas as pd


def get_thermal_params():
    """Gets default thermal parameters.
    Returns
    -------
    thermal_params_df : :obj:`pandas.DataFrame`
        Default thermal parameters
    """
    thermal_params_filename = os.path.join(
        os.path.dirname(__file__), "thermal_parameters.csv"
    )
    thermal_params_df = pd.read_csv(
        thermal_params_filename, skiprows=1, index_col="Thermal Box"
    ).T
    return thermal_params_df
