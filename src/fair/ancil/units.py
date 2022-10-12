import os
import pandas as pd

units_filename = os.path.join(os.path.dirname(__file__), "units.csv")
units_df = pd.read_csv(units_filename, skiprows=1, index_col="Variable")
units_dict = units_df.to_dict()["Unit"]


class Units(dict):
    def __init__(self):
        self.update(units_dict)
