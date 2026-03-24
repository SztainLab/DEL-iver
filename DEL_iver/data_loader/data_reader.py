import pandas as pd

class Data_Reader: #TODO: MAKE SURE EACH METHOD CORRECTLY HANDLES AND LABELS ITS INPUT INTO A CORRECT DF


    @classmethod
    def from_csv(cls, filepath: str, **kwargs):
        cls.df = pd.read_csv(filepath, **kwargs)
        return cls.df

    @classmethod
    def from_parquet(cls, filepath: str, **kwargs):
        cls.df = pd.read_parquet(filepath, **kwargs)
        return cls.df

    @classmethod
    def from_dict(cls, data_dict: dict, **kwargs):
        cls.df = pd.DataFrame(data_dict, **kwargs)
        return cls.df
