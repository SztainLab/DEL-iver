# DEL_iver/utils/cache.py
from pathlib import Path
from platformdirs import user_cache_dir
from enum import Enum
import shutil
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pv
from tqdm import tqdm

    #TODO: check if building_blocks list is in the column od source_file if not raise error, this should be handle by cache_manager when instantiating the data reader

class CacheNames(Enum):
    ROOT = Path(user_cache_dir("DEL_iver"))
    BB_DICTIONARIES = "bb_dictionaries"
    SPLITS = "splits"
    COMPUTE= "analysis"
    MODELS = "models"
    SMILESEMBEDDING = "smiles_embedding"
    


#!TODO: if loading from cache do a check for building block list provided to be usable
class CacheManager:
    def __init__(self, source_file: Path, output_dir: Path = None):
        self.source_file = Path(source_file)
        self.root = Path(output_dir) if output_dir else CacheNames.ROOT.value / self.source_file.stem
        self.dirs = {d: self.root / d.value for d in CacheNames if d != CacheNames.ROOT}

    def _ensure_dirs(self):
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    def get_parquet_path(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / (self.source_file.stem + ".parquet")

    def get_path(self, cache_name: CacheNames, filename: str = None, ext: str = ".parquet") -> Path:
        self._ensure_dirs()
        directory = self.dirs[cache_name]
        return directory / (filename or (self.source_file.stem + ext))

    def is_cached(self, path: Path) -> bool:
        if not path.exists():
            return False
        if path.suffix == ".parquet":
            try:
                pq.read_metadata(path)
                return True
            except Exception:
                path.unlink(missing_ok=True)
                return False
        return path.stat().st_size > 0

    def write_to_cache():
        return NotImplementedError

    def needs_conversion(self) -> bool:
        """Returns True if CSV->Parquet conversion is needed, False if already cached."""
        parquet_path = self.get_parquet_path()
        if self.is_cached(parquet_path):
            warnings.warn(
                f"Loading {parquet_path} from cache. "
                f"To remove run clear_cache({parquet_path})",
                UserWarning)
            return False
        warnings.warn(
            "Caching CSV to Parquet. This is a one-time process.",
            UserWarning)
        return True

#TODO: Before conversion run a quick schema check across all chunks
    def convert_csv_to_parquet(self, memory_per_chunk_mb: int) -> Path:
        parquet_path = self.get_parquet_path()
        tmp_path = parquet_path.with_suffix(".tmp.parquet")
        block_size_bytes = memory_per_chunk_mb * 1024 * 1024
        read_options = pv.ReadOptions(block_size=block_size_bytes)
        writer = None
        try:
            with pv.open_csv(self.source_file, read_options=read_options) as reader:
                for batch in tqdm(reader, desc=f"Converting {self.source_file.name}", unit="Chunk"):
                    table = pa.Table.from_batches([batch])
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, table.schema)
                    writer.write_table(table)
            tmp_path.rename(parquet_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        finally:
            if writer:
                writer.close()
        return parquet_path

    def clear(self, cache_dir: CacheNames = None):
        """Clear a specific cache subdirectory, or the entire source cache if None."""
        if cache_dir is None:
            shutil.rmtree(self.root, ignore_errors=True)
        else:
            shutil.rmtree(self.dirs[cache_dir], ignore_errors=True)

    @staticmethod
    def clear_all():
        """Clear the entire DEL_iver cache."""
        shutil.rmtree(CacheNames.ROOT.value, ignore_errors=True)