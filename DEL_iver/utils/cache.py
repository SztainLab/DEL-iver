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
    """
    Enum defining all cache subdirectories and their named artifact filename templates.

    Each member holds a (dir_name, artifacts) tuple. dir_name is the subdirectory name
    under the cache root. artifacts maps artifact keys to filename templates where {stem}
    is replaced with the source file stem and {prefix} with a caller-supplied prefix.

    Use CacheManager.get_output_path(CacheNames.MEMBER, "artifact_key") to resolve
    a full path without constructing filenames manually.
    """
    ROOT            = (Path(user_cache_dir("DEL_iver")), {})
    BB_DICTIONARIES = ("bb_dictionaries", {
        "main":         "{stem}",
        "id_to_smiles": "{stem}_id_to_smiles",
        "descriptors":  "{stem}_descriptors",
    })
    SPLITS          = ("splits", {})
    COMPUTE         = ("analysis", {
        "bb_enrichment":        "bb_enrichment.{stem}",
        "disynthon_enrichment": "disynthon_enrichment.{stem}",
    })
    MODELS          = ("models", {
        "trainset":        "{prefix}_trainset",
        "testset":         "{prefix}_testset",
        "predictions":     "{prefix}_testset_predictions",
        "model_default":   "{prefix}_trained_defaulttmodel",
        "model_invariant": "{prefix}_trained_invariantmodel",
        "model":           "{prefix}_trained_model",
        "auroc_plot":      "{prefix}_AUROC_plot",
        "pr_plot":         "{prefix}_PR_plot",
    })
    SMILESEMBEDDING = ("smiles_embedding", {
        "fingerprints_bb1": "{prefix}_bb1_fingerprints",
        "fingerprints_bb2": "{prefix}_bb2_fingerprints",
        "fingerprints_bb3": "{prefix}_bb3_fingerprints",
    })
    MLPERFORMANCE   = ("ml_performance", {})
    ANALOGS         = ("analog_analysis", {
        "fingerprints": "{prefix}_enaminefingerprints",
        "umap":         "{prefix}_UMAP",
        "similar":      "{prefix}_similar_analogs",
        "predictions":  "{prefix}_analog_predictions",
    })

    def __init__(self, dir_name, artifacts):
        self.dir_name = dir_name
        self.artifacts = artifacts
    


#!TODO: if loading from cache do a check for building block list provided to be usable
class CacheManager:
    """
    Manages the on-disk cache layout for a single DEL source file.

    All cached artifacts (parquet tables, model files, plots) are organized under a root
    directory keyed to the source file stem, with subdirectories defined by CacheNames.
    Pass output_dir to override the default system cache location.
    """

    def __init__(self, source_file: Path, output_dir: Path = None):
        """
        Parameters:
        -----------
        source_file : Path
            Path to the original CSV or Parquet source file. Used to derive the cache stem.
        output_dir : Path, optional
            Root directory for all outputs. Defaults to the system user cache under the source file stem.
        """
        self.source_file = Path(source_file)
        self.root = Path(output_dir) if output_dir else CacheNames.ROOT.dir_name / self.source_file.stem
        self.dirs = {d: self.root / d.dir_name for d in CacheNames if d != CacheNames.ROOT}

    def _ensure_dirs(self):
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    def get_parquet_path(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / (self.source_file.stem + ".parquet")

    def get_output_path(self, cache_name: CacheNames, artifact: str, ext: str = ".parquet", prefix: str = None) -> Path:
        self._ensure_dirs()
        template = cache_name.artifacts[artifact]
        name = template.format(stem=self.source_file.stem, prefix=prefix) + ext
        return self.dirs[cache_name] / name

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
        """Clear a specific cache subdirectory, or the entire source root if called with no argument."""
        target = self.dirs[cache_dir] if cache_dir is not None else self.root
        shutil.rmtree(target, ignore_errors=True)

    def list_cache(self):
        """Print all cached files grouped by subdirectory."""
        print(f"Cache root: {self.root}")
        for cache_name, path in self.dirs.items():
            if not path.exists():
                continue
            files = sorted(path.iterdir())
            if not files:
                continue
            print(f"\n  [{cache_name.name}]")
            for f in files:
                size_kb = f.stat().st_size / 1024
                print(f"    {f.name}  ({size_kb:.1f} KB)")
        


    def clear_all():
        """Clear the entire DEL_iver cache."""
        shutil.rmtree(CacheNames.ROOT.dir_name, ignore_errors=True)