"""Tests for adverbs.py module."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from latin_masking.adverbs import (
    collect_adverbs,
    generate_adverb_list,
    load_adverb_list,
    normalize_adverb_counts,
    save_adverb_list,
)


class TestCollectAdverbs:
    """Tests for collect_adverbs function."""

    def test_basic_collection(self) -> None:
        """Test basic adverb collection."""
        df = pd.DataFrame(
            {
                "word": ["etiam", "semper", "bonus"],
                "POS": ["ADV", "ADV", "ADJ"],
            }
        )
        result = collect_adverbs([df])
        assert result["etiam"] == 1
        assert result["semper"] == 1

    def test_case_insensitive(self) -> None:
        """Test that adverb counting is case-insensitive."""
        df = pd.DataFrame(
            {
                "word": ["Etiam", "etiam", "ETIAM"],
                "POS": ["ADV", "ADV", "ADV"],
            }
        )
        result = collect_adverbs([df])
        assert result["etiam"] == 3

    def test_empty_input(self) -> None:
        """Test empty input returns empty counter."""
        result = collect_adverbs([])
        assert result == Counter()


class TestNormalizeAdverbCounts:
    """Tests for normalize_adverb_counts function."""

    def test_normalization(self) -> None:
        """Test adverb count normalization."""
        counter = Counter({"seruus": 5, "servus": 3})
        result = normalize_adverb_counts(counter)
        # Both normalize to "seruus" (v→u), so counts are merged
        assert result["seruus"] == 8  # 5 + 3

    def test_no_change_needed(self) -> None:
        """Test counter without variants."""
        counter = Counter({"etiam": 5, "semper": 3})
        result = normalize_adverb_counts(counter)
        assert result["etiam"] == 5
        assert result["semper"] == 3


class TestGenerateAdverbList:
    """Tests for generate_adverb_list function."""

    def test_top_n(self) -> None:
        """Test getting top N adverbs."""
        counter = Counter({"a": 10, "b": 8, "c": 6, "d": 4})
        result = generate_adverb_list(counter, max_adverbs=2)
        assert len(result) == 2
        assert result[0] == ("a", 10)
        assert result[1] == ("b", 8)

    def test_default_max(self) -> None:
        """Test default max of 200."""
        counter = Counter({f"adv{i}": i for i in range(300)})
        result = generate_adverb_list(counter)
        assert len(result) == 200

    def test_none_returns_all(self) -> None:
        """Test that None returns all adverbs."""
        counter = Counter({"a": 10, "b": 8, "c": 6, "d": 4})
        result = generate_adverb_list(counter, max_adverbs=None)
        assert len(result) == 4
        assert result == [("a", 10), ("b", 8), ("c", 6), ("d", 4)]


class TestSaveAndLoadAdverbList:
    """Tests for save_adverb_list and load_adverb_list functions."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading adverb list."""
        adverbs = [("etiam", 10), ("semper", 8)]
        output_path = tmp_path / "adverbs.txt"
        save_adverb_list(adverbs, output_path)
        result = load_adverb_list(output_path, 200)
        assert "etiam" in result
        assert "semper" in result

    def test_threshold(self, tmp_path: Path) -> None:
        """Test threshold limits number of adverbs loaded."""
        adverbs = [("a", 1), ("b", 2), ("c", 3)]
        output_path = tmp_path / "adverbs.txt"
        save_adverb_list(adverbs, output_path)
        result = load_adverb_list(output_path, 2)
        assert len(result) == 2
