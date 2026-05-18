"""Tests for clitics.py module."""

from __future__ import annotations

from pathlib import Path

from latin_masking.clitics import (
    load_que_blacklist,
    load_que_words,
    load_que_whitelist,
    split_que,
    split_que_blacklist,
)


class TestSplitQue:
    """Tests for split_que function."""

    def test_basic_split(self) -> None:
        """Test basic -que splitting."""
        text = "Marcus etiamque in horto sedet."
        que_words = ["etiamque"]
        result, count = split_que(text, que_words)
        assert count == 1
        assert "etiam" in result
        assert "-que" in result

    def test_punctuation_after_que(self) -> None:
        """Test -que word with punctuation after."""
        text = "Marcus etiamque, in horto sedet."
        que_words = ["etiamque"]
        result, count = split_que(text, que_words)
        assert count == 1
        assert "etiam" in result

    def test_multiple_occurrences(self) -> None:
        """Test multiple -que words in text."""
        text = "etiamque etiamque"
        que_words = ["etiamque"]
        result, count = split_que(text, que_words)
        assert count == 2

    def test_no_match(self) -> None:
        """Test text with no -que words."""
        text = "Marcus in horto sedet."
        que_words = ["etiamque"]
        result, count = split_que(text, que_words)
        assert count == 0
        assert result == text

    def test_empty_text(self) -> None:
        """Test empty text."""
        result, count = split_que("", ["etiamque"])
        assert count == 0
        assert result == ""

    def test_empty_que_words(self) -> None:
        """Test with empty que_words list."""
        text = "Marcus etiamque in horto."
        result, count = split_que(text, [])
        assert count == 0
        assert result == text


class TestLoadQueWords:
    """Tests for load_que_words function."""

    def test_load_from_fixture(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Test loading -que words from file."""
        # Create a test file
        test_file = tmp_path / "que_words.txt"
        test_file.write_text("etiamque\tcomment\natque\tcomment\n")
        result = load_que_words(test_file)
        assert "etiamque" in result
        assert "atque" in result

    def test_skips_comments(self, tmp_path: Path) -> None:
        """Test that comment lines are skipped."""
        test_file = tmp_path / "que_words.txt"
        test_file.write_text("# This is a comment\natque\tcomment\n")
        result = load_que_words(test_file)
        assert "atque" in result
        assert len(result) == 1


class TestLoadQueWhitelist:
    """Tests for load_que_whitelist function."""

    def test_load_whitelist(self, tmp_path: Path) -> None:
        """Test loading whitelist."""
        test_file = tmp_path / "whitelist.txt"
        test_file.write_text("word1\t10\tADV\tnotes\nword2\t5\tADV\tnotes\n")
        result = load_que_whitelist(test_file)
        assert "word1" in result
        assert "word2" in result

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test nonexistent file returns empty list."""
        result = load_que_whitelist(tmp_path / "nonexistent.txt")
        assert result == []


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
