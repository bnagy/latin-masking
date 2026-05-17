"""Response caching for UDPipe API calls."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path


def get_cache_path(input_path: Path, cache_dir: Path, model: str) -> Path:
    """Derive cache filename from input hash + model.

    Args:
        input_path: Path to the input file.
        cache_dir: Directory for cache files.
        model: UDPipe model name.

    Returns:
        Path to the cache file.

    """
    # Create a hash of the input path and model for the cache filename
    hash_input = f"{input_path}_{model}"
    hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    return cache_dir / f"{input_path.stem}_{hash_value}.pkl"


def load_cached_response(path: Path) -> str | None:
    """Load pickled UDPipe response.

    Args:
        path: Path to the cache file.

    Returns:
        Cached response string, or None if cache doesn't exist or is corrupted.

    """
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data: str = pickle.load(f)
            return data
    except (pickle.UnpicklingError, EOFError, OSError):
        return None


def save_cached_response(path: Path, response: str) -> None:
    """Pickle and save UDPipe response.

    Args:
        path: Path to save the cache file.
        response: Raw CoNLL-U response string.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(response, f)


def is_cache_valid(cache_path: Path, input_path: Path) -> bool:
    """Check if cache is newer than input.

    Args:
        cache_path: Path to the cache file.
        input_path: Path to the input file.

    Returns:
        True if cache exists and is newer than input.

    """
    if not cache_path.exists():
        return False
    if not input_path.exists():
        return False
    return cache_path.stat().st_mtime > input_path.stat().st_mtime
