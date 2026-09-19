"""Process-local cache; baseline calculations remain in src.pipeline."""

from copy import deepcopy
from functools import lru_cache

from src.config import RAW_PRICES_PATH
from src.pipeline import run_mpt_analysis


@lru_cache(maxsize=1)
def _load_snapshot(path: str, modified_ns: int, size: int) -> dict:
    return run_mpt_analysis().to_dict()


def load_analysis() -> dict:
    """Return an isolated copy; reload when the frozen file's stat changes.

    Restart the process or clear_analysis_cache() after configuration changes.
    Missing datasets fail normally, never triggering an implicit download.
    """
    stat = RAW_PRICES_PATH.stat()
    return deepcopy(_load_snapshot(str(RAW_PRICES_PATH), stat.st_mtime_ns, stat.st_size))


def clear_analysis_cache() -> None:
    _load_snapshot.cache_clear()
