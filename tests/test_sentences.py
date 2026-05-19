"""Tests for sentences.py module."""

from __future__ import annotations

from latin_masking.sentences import (
    has_sufficient_punctuation,
    normalize_whitespace,
    preprocess_text,
    split_paren_content,
    split_sentences,
)


class TestHasSufficientPunctuation:
    """Tests for has_sufficient_punctuation function."""

    def test_sufficient_periods(self) -> None:
        """Test text with enough periods returns True."""
        text = (
            "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        )
        assert has_sufficient_punctuation(text)

    def test_insufficient_punctuation(self) -> None:
        """Test text with little punctuation returns False."""
        text = "No punctuation here"
        assert not has_sufficient_punctuation(text)

    def test_mixed_punctuation(self) -> None:
        """Test text with mixed punctuation."""
        text = "One. Two? Three! Four; Five."
        assert has_sufficient_punctuation(text)


class TestNormalizeWhitespace:
    """Tests for normalize_whitespace function."""

    def test_multiple_newlines(self) -> None:
        """Test multiple newlines are collapsed to single space."""
        text = "Line one\n\n\nLine two"
        result = normalize_whitespace(text)
        assert result == "Line one Line two"

    def test_multiple_tabs(self) -> None:
        """Test multiple tabs are collapsed to single space."""
        text = "Word1\t\t\tWord2"
        result = normalize_whitespace(text)
        assert result == "Word1 Word2"

    def test_mixed_whitespace(self) -> None:
        """Test mixed whitespace (newlines, tabs, spaces) is normalized."""
        text = "Word1\n\t \n\tWord2"
        result = normalize_whitespace(text)
        assert result == "Word1 Word2"

    def test_preserves_single_spaces(self) -> None:
        """Test that single spaces are preserved."""
        text = "Word1 Word2 Word3"
        result = normalize_whitespace(text)
        assert result == "Word1 Word2 Word3"

    def test_leading_trailing_whitespace(self) -> None:
        """Test leading and trailing whitespace is removed."""
        text = "  Word1  \n  Word2  "
        result = normalize_whitespace(text)
        assert result == "Word1 Word2"


class TestPreprocessText:
    """Tests for preprocess_text function."""

    def test_quote_removal(self) -> None:
        """Test quotation marks are removed."""
        text = 'He said "hello" to me.'
        result, _ = preprocess_text(text)
        assert '"' not in result

    def test_unicode_quote_removal(self) -> None:
        """Test unicode quotation marks are removed."""
        text = "He said «hello» to me."
        result, _ = preprocess_text(text)
        assert "«" not in result
        assert "»" not in result

    def test_dash_after_punctuation(self) -> None:
        """Test dashes after punctuation are removed."""
        text = "Sentence. -Next sentence."
        result, _ = preprocess_text(text)
        assert "-Next" not in result

    def test_parenthesis_protection(self) -> None:
        """Test parenthetical content is protected."""
        text = "Text (parenthetical content) more text."
        result, paren_map = preprocess_text(text)
        assert "__PAREN_" in result
        assert "parenthetical content" in paren_map.values()


class TestSplitParenContent:
    """Tests for split_paren_content function."""

    def test_basic_split(self) -> None:
        """Test basic parenthetical content splitting."""
        content = "First. Second; Third!"
        result = split_paren_content(content)
        assert len(result) == 3

    def test_colon_not_boundary(self) -> None:
        """Test colons are not treated as sentence boundaries."""
        content = "First: second"
        result = split_paren_content(content)
        # Colon should not split
        assert len(result) == 1


class TestSplitSentences:
    """Tests for split_sentences function."""

    def test_basic_splitting(self) -> None:
        """Test basic sentence splitting."""
        text = "Marcus est bonus. Puella legit."
        result = split_sentences(text)
        assert len(result) >= 1

    def test_skip_brackets(self) -> None:
        """Test sentences with brackets are skipped."""
        text = "Marcus [editorial note] est bonus."
        result = split_sentences(text)
        # Should skip sentences with brackets
        for sent in result:
            assert "[" not in sent
            assert "]" not in sent

    def test_skip_no_letters(self) -> None:
        """Test sentences with no letters are skipped."""
        text = "Marcus. ... Puella."
        result = split_sentences(text)
        for sent in result:
            assert any(c.isalpha() for c in sent)

    def test_empty_input(self) -> None:
        """Test empty input returns empty list."""
        result = split_sentences("")
        assert result == []
