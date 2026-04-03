# DEL_iver/utils/cache.py

from pathlib import Path
from platformdirs import user_cache_dir
import shutil
import pyarrow.parquet as pq  

CACHE_DIR = Path(user_cache_dir("DEL_iver"))

def get_cache_path(source_file: Path) -> Path:
    """Returns the canonical parquet cache path for a given source file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / source_file.stem / (source_file.stem + ".parquet")

def is_cached(source_file: Path) -> bool:
    parquet_path = get_cache_path(source_file)
    if not parquet_path.exists():
        return False
    try:
        pq.read_metadata(parquet_path)
        return True
    except Exception:
        parquet_path.unlink(missing_ok=True)  # delete corrupt file
        return False

def clear_cache(source_file: Path = None) -> None:
    """Clear cache for one file, or the entire DEL_iver cache if None."""
    if source_file is None:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
    else:
        get_cache_path(source_file).unlink(missing_ok=True)

