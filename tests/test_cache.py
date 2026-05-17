"""Tests for cache.py module."""

from __future__ import annotations

from pathlib import Path

from udpipe_masking.cache import (
    get_cache_path,
    is_cache_valid,
    load_cached_response,
    save_cached_response,
)


class TestGetCachePath:
    """Tests for get_cache_path function."""

    def test_cache_path_generation(self, tmp_path: Path) -> None:
        """Test cache path is generated correctly."""
        input_path = tmp_path / "test_input.txt"
        cache_dir = tmp_path / "cache"
        result = get_cache_path(input_path, cache_dir, "latin-model")
        assert result.parent == cache_dir
        assert "test_input" in result.name
        assert result.suffix == ".pkl"

    def test_different_inputs_different_paths(self, tmp_path: Path) -> None:
        """Test different inputs produce different cache paths."""
        input1 = tmp_path / "file1.txt"
        input2 = tmp_path / "file2.txt"
        cache_dir = tmp_path / "cache"
        path1 = get_cache_path(input1, cache_dir, "model")
        path2 = get_cache_path(input2, cache_dir, "model")
        assert path1 != path2


class TestSaveAndLoadCachedResponse:
    """Tests for save_cached_response and load_cached_response functions."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading a cached response."""
        cache_path = tmp_path / "test.pkl"
        response = "# text = test\n1\ttest\ttest\tNOUN\t_\t_\t0\troot\t_\t_\n"
        save_cached_response(cache_path, response)
        loaded = load_cached_response(cache_path)
        assert loaded == response

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        """Test loading nonexistent cache returns None."""
        result = load_cached_response(tmp_path / "nonexistent.pkl")
        assert result is None

    def test_load_corrupted(self, tmp_path: Path) -> None:
        """Test loading corrupted cache returns None."""
        cache_path = tmp_path / "corrupted.pkl"
        cache_path.write_bytes(b"not a pickle")
        result = load_cached_response(cache_path)
        assert result is None


class TestIsCacheValid:
    """Tests for is_cache_valid function."""

    def test_valid_cache(self, tmp_path: Path) -> None:
        """Test valid cache returns True."""
        input_path = tmp_path / "input.txt"
        cache_path = tmp_path / "cache.pkl"
        input_path.touch()
        cache_path.touch()
        # Cache is newer (touch again)
        cache_path.touch()
        assert is_cache_valid(cache_path, input_path)

    def test_stale_cache(self, tmp_path: Path) -> None:
        """Test stale cache returns False."""
        input_path = tmp_path / "input.txt"
        cache_path = tmp_path / "cache.pkl"
        input_path.touch()
        cache_path.touch()
        # Input is newer
        input_path.touch()
        assert not is_cache_valid(cache_path, input_path)

    def test_nonexistent_cache(self, tmp_path: Path) -> None:
        """Test nonexistent cache returns False."""
        input_path = tmp_path / "input.txt"
        input_path.touch()
        assert not is_cache_valid(tmp_path / "cache.pkl", input_path)

    def test_nonexistent_input(self, tmp_path: Path) -> None:
        """Test nonexistent input returns False."""
        cache_path = tmp_path / "cache.pkl"
        cache_path.touch()
        assert not is_cache_valid(cache_path, tmp_path / "input.txt")
