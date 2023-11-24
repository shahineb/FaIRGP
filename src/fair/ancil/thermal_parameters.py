import os
import pandas as pd


def get_thermal_params(file_name=None):
    """Gets default thermal parameters.
    """
    if file_name is None:
        file_path = os.path.join(
            os.path.dirname(__file__), "thermal_parameters_NORESM2-LM.csv"
        )
    else:
        file_path = os.path.join(
            os.path.dirname(__file__), f"thermal_parameters_NORESM2-LM/{file_name}.csv"
        )
    thermal_params_df = pd.read_csv(
        file_path, skiprows=1, index_col="Thermal Box"
    ).T
    return thermal_params_df
