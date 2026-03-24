import pandas as pd

class Data_Reader: 

    @classmethod
    def _format_columns(cls, df: pd.DataFrame,
                        building_blocks: list,
                        molecule_smiles: str,
                        id_col: str = None,
                        misc_cols: list = None):
        """
        Standardizes columns into canonical names and order.

        Requires `molecule_smiles` and `building_blocks`. 
        If `id_col` is not provided, generates a sequential ID column.
        """

        # ---- Required argument checks ----
        if not building_blocks or not isinstance(building_blocks, list):
            raise ValueError(f"`building_blocks` must be a non-empty list of column names, got {type(building_blocks)}")
        if not molecule_smiles or not isinstance(molecule_smiles, str):
            raise ValueError(f"`molecule_smiles` must be a string, got {type(molecule_smiles_col)}")

        # ---- Type checks for optional arguments ----
        if id_col is not None and not isinstance(id_col, str):
            raise TypeError(f"id_col must be a string or None, got {type(id_col)}")

        rename_map = {}
        rename_map[molecule_smiles] = "molecule_smiles"

        if id_col:
            rename_map[id_col] = "id"

        for i, bb_col in enumerate(building_blocks, start=1):
            rename_map[bb_col] = f"building_block{i}_smiles"

        # Check that the columns they specified exist in the dataframe
        for col in rename_map.keys():
            if col not in df.columns:
                raise ValueError(f"Specified column '{col}' not found in DataFrame columns: {df.columns}")

        # Handle missing id column
        if not id_col:
            print("Note: No ID column specified, generating sequential IDs.")
            df["id"] = range(len(df))

        # Rename columns
        df = df.rename(columns=rename_map)

        # ---- Column ordering ----
        bb_cols = [f"building_block{i}_smiles" for i in range(1, len(building_blocks)+1)]
        required_cols = ["id"] + bb_cols + ["molecule_smiles", "label"]  # label stays at the end of canonical order
        if misc_cols:
            all_cols = required_cols + misc_cols
        else:
            all_cols = required_cols

        # Include only columns that exist in df to avoid KeyError
        all_cols = [c for c in all_cols if c in df.columns]

        df = df[all_cols]

        return df

    @classmethod
    def from_csv(cls, filepath: str,
                 building_blocks: list = None,
                 molecule_smiles: str = None,
                 id_col: str = None,
                 misc_cols: list = None,
                 **kwargs) -> pd.DataFrame:
        """
        Reads CSV and auto-formats columns.
        """
        df = pd.read_csv(filepath, **kwargs)
        df = cls._format_columns(df,
                                building_blocks=building_blocks,
                                molecule_smiles=molecule_smiles,
                                id_col=id_col,
                                misc_cols=misc_cols)
        cls.df = df
        return df

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
    def iterator_from_csv(cls, filepath: str,
                        chunk_size: int,
                        building_blocks: list = None,
                        molecule_smiles: str = None,
                        id_col: str = None,
                        misc_cols: list = None,
                        **kwargs):
        """
        Reads a CSV file in chunks, formats columns, and returns an iterator of DataFrames
        with canonical column names and order.
        """
        if chunk_size <= 0 or not isinstance(chunk_size, int):
            raise ValueError("chunk_size must be a positive integer.")

        # First, read just the headers (or a small sample) to validate and format column names


        # Now read the full CSV in chunks, keeping only canonical columns
        reader = pd.read_csv(filepath, chunksize=chunk_size, **kwargs)
        
        def formatted_chunk_iterator():
            for chunk in reader:
                # Apply the same renaming & column ordering to each chunk
                chunk = cls._format_columns(chunk,
                                            building_blocks=building_blocks,
                                            molecule_smiles=molecule_smiles,
                                            id_col=id_col,
                                            misc_cols=misc_cols)
                yield chunk
                


        cls.df_iterator = formatted_chunk_iterator()
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
    
