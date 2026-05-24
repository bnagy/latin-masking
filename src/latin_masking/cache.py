"""Response caching for UDPipe API calls."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path


def _content_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file's content.

    Args:
        path: Path to the file.

    Returns:
        Hex digest of the file content.

    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_cache_path(input_path: Path, cache_dir: Path, model: str) -> Path:
    """Derive cache filename from content hash + model.

    The hash is computed from the file content, so the cache is valid
    as long as the file content hasn't changed — regardless of mtime.

    Args:
        input_path: Path to the input file.
        cache_dir: Directory for cache files.
        model: UDPipe model name.

    Returns:
        Path to the cache file.

    """
    content = _content_hash(input_path)
    hash_value = hashlib.sha256(f"{content}_{model}".encode()).hexdigest()[:12]
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
    """Check if a valid cached response exists.

    Since the cache filename is derived from the content hash,
    the mere existence of the cache file is sufficient proof
    that the cached response matches the current file content.

    Args:
        cache_path: Path to the cache file.
        input_path: Path to the input file (unused, kept for
            backwards compatibility).

    Returns:
        True if the cache file exists.

    """
    return cache_path.exists()
