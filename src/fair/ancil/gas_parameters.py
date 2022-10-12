import os
import pandas as pd

gas_params_filename = os.path.join(os.path.dirname(__file__), "gas_parameters_NORESM2-LM.csv")
gas_params_df = pd.read_csv(gas_params_filename, skiprows=1, index_col="Gas").T


def get_gas_params(gas_list=None):
    """Get gas parameters from the defauls.
    Parameters
    ----------
    gas_list : list[str] or None
        list of gases to calculate. If None, use all defaults
    Returns
    -------
    gas_params_df : :obj:`pandas.DataFrame`
        data frame of requested gases
    """
    if gas_list is not None:
        res = gas_params_df[gas_list]
    else:
        res = gas_params_df
    return res
