import pandas as pd
import warnings
from enum import Enum
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from DEL_iver.utils.cache import get_cache_path, is_cached, clear_cache

class Data_Reader:
    """Class to hold data and related attributes"""

    DEFAULT_CHUNK_SIZE=10000

    class Fields(Enum):
        DATA = "data"
        BUILDING_BLOCKS = "building_blocks"
        MOLECULE_SMILES = "molecule_smiles"

    def __init__(self):
        self.filepath=None
        self.building_blocks = None
        self.molecule_smiles = None

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_csv(cls, filepath: str,
                        building_blocks: list ,
                        chunk_size: int = DEFAULT_CHUNK_SIZE,
                        molecule_smiles: str = None ,
                        data_cols: list = None,
                        **kwargs):
        """

        """
        source = Path(filepath)
        if not source.exists():
            raise FileNotFoundError(f"No file found at {filepath}")


        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive non-zero integer.")

        self = cls()
        self.building_blocks = building_blocks
        self.chunk_size = chunk_size
        self.molecule_smiles=molecule_smiles
        self.data_cols = data_cols
        self.n_building_blocks = len(building_blocks)




        parquet_path = get_cache_path(source)

        if is_cached(source):
            cached_chunk_size = pq.read_metadata(parquet_path).row_group(0).num_rows
            if cached_chunk_size != chunk_size:
                warnings.warn(
                    f"Cached parquet was written with row_group_size={cached_chunk_size}. "
                    f"Ignoring requested chunk_size={chunk_size}. "
                    f"Call deliv.clear_cache('{source.name}') to reconvert with new chunk size.",
                    UserWarning
                )
            self.chunk_size = cached_chunk_size
        else:
            warnings.warn(
                f"Converting {source.name} to parquet for efficient access. "
                f"Cached at {parquet_path} — this will not repeat on future calls.",
                UserWarning
            )
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = parquet_path.with_suffix(".tmp.parquet")
            writer = None

            try:
                for chunk in pd.read_csv(source, chunksize=self.chunk_size):

                    table = pa.Table.from_pandas(chunk)

                    if writer is None:

                        writer = pq.ParquetWriter(
                            tmp_path,
                            table.schema,
                        )

                    writer.write_table(
                        table,
                        row_group_size=self.chunk_size
                        )

            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

            finally:
                if writer:
                    writer.close()
                    tmp_path.rename(parquet_path)
                    
                    

        self.source_file = parquet_path
        return self








    @classmethod
    def from_parquet(cls, filepath: str, **kwargs): #!TO BE DONE
        """
        Reads a Parquet file. Chunking is not supported. #pyarrow.dataset does support it maybe we should make that be the enginge for parquet.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Data access                                                         #
    # ------------------------------------------------------------------ #

    @property
    def data(self):
        """Always returns a fresh iterable of DataFrames."""
        return pd.read_csv(self.source_file, chunksize=self.chunk_size)

    @property
    def n_chunks(self) -> int:
        if self.n_rows is not None:
            self.__chunks = -(-self.n_rows // self.chunk_size)  # ceiling div
        else:
            warnings.warn(
                "n_rows not specified. Counting rows via full file scan — "
                "this may be slow for large files. Pass n_rows to from_csv() to avoid this.",
                UserWarning
            )
            self.n_rows = self._count_rows()
            self._n_chunks = -(-self.n_rows // self.chunk_size)
        return self.n_chunks

    def get_chunk(self, n: int) -> pd.DataFrame:
        """Returns chunk n directly without iterating through previous chunks."""
        if n < 0 or n >= self._n_chunks:
            raise IndexError(f"Chunk index {n} out of range [0, {self.n_chunks - 1}]")
        
        skip = n * self.chunk_size + 1  # +1 to preserve header
        return pd.read_csv(
            self.source_file,
            skiprows=range(1, skip),  # skip rows but keep header
            nrows=self.chunk_size
        )

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #


    def validate_required_data(self, required_fields):
        """
        Validates that this Data_Reader instance has all required attributes
        defined and not None (or empty).
        """
        missing = []

        for field in required_fields:
            # if Enum, get its value; otherwise assume string
            field_name = field.value if hasattr(field, "value") else field
            try:
                value = getattr(self, field_name)
            except AttributeError:
                missing.append(f"{field_name} (not defined)")
                continue

            if value is None:
                missing.append(field_name)
            elif hasattr(value, "empty") and value.empty:
                missing.append(f"{field_name} (empty)")
            elif isinstance(value, (list, dict, set)) and len(value) == 0:
                missing.append(f"{field_name} (empty)")

        if missing:
            raise ValueError(
                f"Missing required data for {type(self).__name__}: {', '.join(missing)}"
            )


    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #


    def _compute_n_chunks(self): #!might be slow
        if self.chunk_size is None:
            return 1
        else:
            # count rows without loading full file
            row_count = sum(1 for _ in open(self.source_file)) - 1  
            return -(-row_count // self.chunk_size)   
    

    def _get_actual_chunk_sizes(self):
        if self.chunk_size is None:
            return [sum(1 for _ in open(self.source_file)) - 1]
        
        row_count = sum(1 for _ in open(self.source_file)) - 1
        n_chunks = -(-row_count // self.chunk_size)
        
        sizes = [self.chunk_size] * (n_chunks - 1)
        last_size = row_count - self.chunk_size * (n_chunks - 1)
        sizes.append(last_size)
        return sizes






