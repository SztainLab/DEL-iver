import pandas as pd

class Data_Reader: 
    #TODO: MAKE SURE EACH METHOD CORRECTLY HANDLES AND LABELS ITS INPUT INTO A CORRECT DF meaning having options to label what column 
    #is what, right now there 
    #is an implicit format in the csv we are using for testing but the data reader should take in enough information to ensure it makes this format


    @classmethod
    def from_csv(cls, filepath: str, **kwargs):
        """
        Reads a CSV file. Supports optional chunked reading.

        Returns:
            pd.DataFrame or TextFileReader: DataFrame if chunk_size is None, else iterator of DataFrames.
        """

        cls.df = pd.read_csv(filepath, **kwargs)
        return cls.df

    @classmethod
    def iterator_from_csv(cls, filepath: str, chunk_size: int, **kwargs):
        """
        Reads a CSV file in chunks and returns an iterator of DataFrames.

        Returns:
            TextFileReader: An iterator that yields DataFrames of size chunk_size.
        """
        if chunk_size <= 0 and not type(chunk_size) == int:
            raise ValueError("chunk_size must be a positive integer.")
        cls.df_iterator = pd.read_csv(filepath, chunksize=chunk_size, **kwargs)
        return cls.df_iterator

    @classmethod
    def from_parquet(cls, filepath: str, **kwargs):
        """
        Reads a Parquet file. Chunking is not supported. #pyarrow.dataset does support it maybe we should make that be the enginge for parquet.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        cls.df = pd.read_parquet(filepath, **kwargs)
        return cls.df
    
