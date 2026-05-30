"""Tests for clitics.py module."""

from __future__ import annotations

from pathlib import Path

from latin_masking.clitics import (
    load_que_blacklist,
    split_que_blacklist,
)


class TestLoadQueBlacklist:
    """Tests for load_que_blacklist function."""

    def test_load_blacklist_basic(self, tmp_path: Path) -> None:
        """Test loading blacklist from file."""
        test_file = tmp_path / "que_blacklist.txt"
        test_file.write_text("atque\netiamque\ncircumque\n")
        result = load_que_blacklist(test_file)
        assert "atque" in result
        assert "etiamque" in result
        assert "circumque" in result

    def test_load_blacklist_with_markers(self, tmp_path: Path) -> None:
        """Test loading blacklist with ?! and ?? markers."""
        test_file = tmp_path / "que_blacklist.txt"
        test_file.write_text("atque\n?!etiamque\n??circumque\n")
        result = load_que_blacklist(test_file)
        assert "atque" in result
        assert "etiamque" in result
        assert "circumque" in result

    def test_load_blacklist_skips_comments(self, tmp_path: Path) -> None:
        """Test that comment lines are skipped."""
        test_file = tmp_path / "que_blacklist.txt"
        test_file.write_text("# This is a comment\natque\n")
        result = load_que_blacklist(test_file)
        assert "atque" in result
        assert len(result) == 1

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test nonexistent file returns empty set."""
        result = load_que_blacklist(tmp_path / "nonexistent.txt")
        assert result == set()


class TestSplitQueBlacklist:
    """Tests for split_que_blacklist function."""

    def test_basic_split(self) -> None:
        """Test basic -que splitting with blacklist."""
        text = "Marcus etiamque in horto sedet."
        blacklist: set[str] = set()
        result, count = split_que_blacklist(text, blacklist)
        assert count == 1
        assert "etiam" in result
        assert "-que" in result

    def test_blacklist_preserves_word(self) -> None:
        """Test that blacklisted words are not split."""
        text = "Marcus atque in horto sedet."
        blacklist = {"atque"}
        result, count = split_que_blacklist(text, blacklist)
        assert count == 0
        assert "atque" in result
        assert "-que" not in result

    def test_case_insensitive_blacklist(self) -> None:
        """Test that blacklist matching is case-insensitive."""
        text = "Marcus ATQUE in horto sedet."
        blacklist = {"atque"}
        result, count = split_que_blacklist(text, blacklist)
        assert count == 0
        assert "ATQUE" in result

    def test_multiple_que_words(self) -> None:
        """Test multiple -que words, some blacklisted."""
        text = "atque etiamque circumque"
        blacklist = {"atque", "circumque"}
        result, count = split_que_blacklist(text, blacklist)
        assert count == 1  # Only etiamque should be split
        assert "atque" in result
        assert "circumque" in result
        assert "etiam -que" in result

    def test_punctuation_after_que(self) -> None:
        """Test -que word with punctuation after."""
        text = "Marcus etiamque, in horto sedet."
        blacklist: set[str] = set()
        result, count = split_que_blacklist(text, blacklist)
        assert count == 1
        assert "etiam" in result

    def test_common_adverbs_ending_in_que(self) -> None:
        """Test that common adverbs ending in -que are preserved."""
        text = "Marcus itaque in horto sedet."
        blacklist: set[str] = set()
        common_adverbs = {"itaque", "namque"}
        result, count = split_que_blacklist(text, blacklist, common_adverbs)
        assert count == 0
        assert "itaque" in result

    def test_empty_text(self) -> None:
        """Test empty text."""
        result, count = split_que_blacklist("", set())
        assert count == 0
        assert result == ""

    def test_no_que_words(self) -> None:
        """Test text with no -que words."""
        text = "Marcus in horto sedet."
        result, count = split_que_blacklist(text, set())
        assert count == 0
        assert result == text

    def test_default_blacklist(self) -> None:
        """Test that default blacklist is used when none provided."""
        text = "Marcus atque in horto sedet."
        result, count = split_que_blacklist(text)
        # atque is in the default blacklist
        assert count == 0
        assert "atque" in result
