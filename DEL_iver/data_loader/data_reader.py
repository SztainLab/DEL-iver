import pandas as pd
import warnings
from enum import Enum
from pathlib import Path
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
from tqdm import tqdm
from itertools import combinations
from DEL_iver.utils.cache import CacheManager, CacheNames

    #TODO: check if building_blocks list is in the column od source_file if not raise error, this should be handle by cache_manager when instantiating the data reader

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



    @classmethod
    def from_csv(cls, filepath: str,
                        building_blocks: list ,
                        memory_per_chunk_mb: int = DEFAULT_MEMORY_MB ,
                        molecule_smiles: str = None ,
                        data_cols: list = None,
                        output_dir: str = None,
                        label: str = None ,
                        **kwargs):
        """

        """
        #TODO:source might not exist but could still be cached under that file name need to check for that
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
        self.cache = CacheManager(source, output_dir=output_dir)
        self.label=label

        combos = list(combinations(range(self.n_building_blocks), 2))
        self.disynthons = [f"disynthon_{'_'.join(str(i + 1) for i in combo)}_id" for combo in combos]

        if self.cache.needs_conversion():
            self.cache.convert_csv_to_parquet(memory_per_chunk_mb)

        parquet_path = self.cache.get_parquet_path()
        self.source_file = parquet_path
        return self


    @classmethod
    def from_parquet(cls, filepath: str, **kwargs): #!TO BE DONE

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


    def get_chunk(self, n: int):
        return pq.ParquetFile(self.source_file).read_row_group(n)
