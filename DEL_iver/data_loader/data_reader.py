import pandas as pd

class DEL_Data_Reader: #TODO: MAKE SURE EACH METHOD CORRECTLY HANDLES AND LABELS ITS INPUT INTO A CORRECT DF

    def __init__(self, name=None, age=None):
        self.df : pd.DataFrame = None  # store the loaded dataframe

    def from_csv(self, filepath: str, **kwargs):
        self.df = pd.read_csv(filepath, **kwargs)
        return self.df

    def from_parquet(self, filepath: str, **kwargs):
        self.df = pd.read_parquet(filepath, **kwargs)
        return self.df

    def from_dict(self, data_dict: dict, **kwargs):
        self.df = pd.DataFrame(data_dict, **kwargs)
        return self.df
