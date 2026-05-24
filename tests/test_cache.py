"""Tests for cache.py module."""

from __future__ import annotations

from pathlib import Path

from latin_masking.cache import (
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
        input_path.write_text("hello world")
        cache_dir = tmp_path / "cache"
        result = get_cache_path(input_path, cache_dir, "latin-model")
        assert result.parent == cache_dir
        assert "test_input" in result.name
        assert result.suffix == ".pkl"

    def test_different_inputs_different_paths(self, tmp_path: Path) -> None:
        """Test different inputs produce different cache paths."""
        input1 = tmp_path / "file1.txt"
        input1.write_text("content A")
        input2 = tmp_path / "file2.txt"
        input2.write_text("content B")
        cache_dir = tmp_path / "cache"
        path1 = get_cache_path(input1, cache_dir, "model")
        path2 = get_cache_path(input2, cache_dir, "model")
        assert path1 != path2

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        """Test that identical content produces the same content hash."""
        input1 = tmp_path / "file1.txt"
        input1.write_text("same content")
        input2 = tmp_path / "file2.txt"
        input2.write_text("same content")
        cache_dir = tmp_path / "cache"
        path1 = get_cache_path(input1, cache_dir, "model")
        path2 = get_cache_path(input2, cache_dir, "model")
        # Stems differ but the hash portion should be identical
        hash1 = path1.stem.split("_", 1)[1]
        hash2 = path2.stem.split("_", 1)[1]
        assert hash1 == hash2

    def test_content_change_changes_path(self, tmp_path: Path) -> None:
        """Test that changing file content changes the cache path."""
        input_path = tmp_path / "test.txt"
        input_path.write_text("version 1")
        cache_dir = tmp_path / "cache"
        path1 = get_cache_path(input_path, cache_dir, "model")
        input_path.write_text("version 2")
        path2 = get_cache_path(input_path, cache_dir, "model")
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

    def test_existing_cache_is_valid(self, tmp_path: Path) -> None:
        """Test that an existing cache file is valid."""
        cache_path = tmp_path / "cache.pkl"
        cache_path.touch()
        assert is_cache_valid(cache_path, tmp_path / "input.txt")

    def test_nonexistent_cache_is_invalid(self, tmp_path: Path) -> None:
        """Test that a nonexistent cache file is invalid."""
        assert not is_cache_valid(tmp_path / "cache.pkl", tmp_path / "input.txt")
