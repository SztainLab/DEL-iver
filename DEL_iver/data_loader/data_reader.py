import pandas as pd
import warnings
from enum import Enum
from pathlib import Path
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
from tqdm import tqdm

from DEL_iver.utils.cache import get_cache_path, is_cached, clear_cache



class DataReader:
    """Class to hold data and related attributes"""

    DEFAULT_MEMORY_MB=256

    #!google if this is needed
    def __init__(self):
        #self.filepath=None
        self.building_blocks = None
        self.molecule_smiles = None

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #


    #!give option to choose for it to go to to cache or not
    @classmethod
    def from_csv(cls, filepath: str,
                        building_blocks: list ,
                        memory_per_chunk_mb: int = DEFAULT_MEMORY_MB ,
                        molecule_smiles: str = None ,
                        data_cols: list = None,
                        **kwargs):
        """

        """
        #!source might not exist but could still be cached under that file name need to check for that
        source = Path(filepath)
        if not source.exists():
            raise FileNotFoundError(f"No file found at {filepath}")

        if not isinstance(memory_per_chunk_mb, int) or memory_per_chunk_mb <= 0:
            raise ValueError("memory_per_chunk_mb must be a positive non-zero integer.")

        self = cls()
        self.building_blocks = building_blocks
        self.molecule_smiles=molecule_smiles
        self.data_cols = data_cols
        self.n_building_blocks = len(building_blocks)

        parquet_path = get_cache_path(source)

        if is_cached(source):
            warnings.warn(
                        f"Loading {parquet_path} from cache"
                        f"To remove run clear_cache({parquet_path})",
                        UserWarning)
            cached_chunk_size = pq.read_metadata(parquet_path).row_group(0).num_rows
            self.chunk_size = cached_chunk_size
        else:
            warnings.warn(
                        f"Caching CSV to Parquet using ~{memory_per_chunk_mb}MB chunks. "
                        f"This is a one-time process.",
                        UserWarning)
            self._convert_csv_to_parquet(source, parquet_path,memory_per_chunk_mb) 


        self.source_file = parquet_path

        return self


    @classmethod
    def from_parquet(cls, filepath: str, **kwargs): #!TO BE DONE
        """

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Data access                                                         #
    # ------------------------------------------------------------------ #

    @property
    def data(self):
        return pq.ParquetFile(self.source_file)

    @property
    def n_chunks(self) -> int:
        return pq.ParquetFile(self.source_file).num_row_groups


    def data_splits(self):
        raise NotImplementedError
        #!check for 

    def get_chunk(self, n: int):
        return pq.ParquetFile(self.source_file).read_row_group(n)

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



    #!neeed to handle script getting cancelled gracefully delete partial file
    def _convert_csv_to_parquet(self, csv_path: Path, parquet_path: Path,memory_per_chunk_mb) -> None:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = parquet_path.with_suffix(".tmp.parquet")
            
        # Calculate exact bytes from MB
            block_size_bytes = memory_per_chunk_mb * 1024 * 1024
            read_options = pv.ReadOptions(block_size=block_size_bytes)

            writer = None
            try:
                
                with pv.open_csv(csv_path, read_options=read_options) as reader:
                    for batch in tqdm(reader, desc=f"Converting {csv_path.name}",unit="Chunk"):
                        table = pa.Table.from_batches([batch])
                        if writer is None:
                            writer = pq.ParquetWriter(tmp_path, table.schema)
                        writer.write_table(table)
                tmp_path.rename(parquet_path)
            except Exception:
                if tmp_path.exists(): tmp_path.unlink()
                raise
            finally:
                if writer: writer.close()






    def _get_actual_chunk_sizes(self):
        raise NotImplementedError






'''
Here's your todo list:

**`from_csv`**
- add `n_rows` as optional parameter, store as `self.n_rows`
- initialise `self.__n_chunks = None` before the cache block

**`from_parquet`**
- implement it — same signature shape as `from_csv` minus `chunk_size` warning logic
- validate parquet file is readable via `pq.read_metadata()` on entry
- set `self.chunk_size` from actual row group size in the file

**`data` property**
- switch from `pd.read_csv` to `pq.ParquetFile(self.source_file).iter_batches(batch_size=self.chunk_size)`
- yields Arrow record batches — decide if you want `.to_pandas()` inside the property or leave that to the caller

**`n_chunks` property**
- fix recursion bug — `return self.__n_chunks` not `return self.n_chunks`
- fix backing variable name inconsistency — `self.__chunks` vs `self.__n_chunks`
- use `math.ceil` instead of ceiling division trick

**`get_chunk`**
- replace `pd.read_csv` + `skiprows` with `pq.ParquetFile(self.source_file).read_row_group(n).to_pandas()`
- bounds check should use `self.n_chunks` not `self._n_chunks`

**`__init__`**
- add `self.__n_chunks = None`
- add `self.n_rows = None`
- add `self.source_file = None`
- remove `self.filepath` — it's never used anywhere

**Dead code to delete**
- `_compute_n_chunks` — replaced by `n_chunks` property
- `_get_actual_chunk_sizes` — replaced by parquet row group metadata

**`_count_rows`**
- implement it using binary newline counting with 64MB buffer
- only needed as fallback when `n_rows` not provided by user

**`validate_required_data`**
- `Fields.DATA` enum member points to `"data"` but `data` is now a property not an attribute — decide if validation of `data` still makes sense or remove it from `Fields`

'''