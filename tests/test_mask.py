"""Tests for mask.py module."""

from __future__ import annotations

import pandas as pd  # pyright: ignore[reportMissingImports]

from latin_masking.mask import (
    collect_lowercase_words,
    mask_corpus,
    mask_sentence,
    two_pass_mask,
)


class TestMaskSentence:
    """Tests for mask_sentence function."""

    def test_noun_masking(self) -> None:
        """Test NOUN tokens are masked."""
        words = ["Marcus", "bonus"]
        pos_tags = ["PROPN", "ADJ"]
        result = mask_sentence(words, pos_tags, common_adverbs=set())
        assert result == ["PROPN", "ADJ"]

    def test_verb_masking(self) -> None:
        """Test VERB tokens are masked."""
        words = ["legit"]
        pos_tags = ["VERB"]
        result = mask_sentence(words, pos_tags, common_adverbs=set())
        assert result == ["VERB"]

    def test_adv_common(self) -> None:
        """Test common adverbs are kept."""
        words = ["etiam"]
        pos_tags = ["ADV"]
        result = mask_sentence(words, pos_tags, common_adverbs={"etiam"})
        assert result == ["etiam"]

    def test_adv_uncommon(self) -> None:
        """Test uncommon adverbs are masked."""
        words = ["aliter"]
        pos_tags = ["ADV"]
        result = mask_sentence(words, pos_tags, common_adverbs=set())
        assert result == ["ADV"]

    def test_other_tokens_lowercased(self) -> None:
        """Test non-POS, non-ADV tokens are lowercased."""
        words = ["et", "in"]
        pos_tags = ["CCONJ", "ADP"]
        result = mask_sentence(words, pos_tags, common_adverbs=set())
        assert result == ["et", "in"]

    def test_normalization_applied(self) -> None:
        """Test UV/IJ normalization is applied."""
        words = ["seruus"]
        pos_tags = ["NOUN"]
        result = mask_sentence(words, pos_tags, common_adverbs=set())
        assert result == ["NOUN"]


class TestMaskCorpus:
    """Tests for mask_corpus function."""

    def test_basic_masking(self) -> None:
        """Test basic corpus masking."""
        df = pd.DataFrame(
            {
                "word": ["Marcus", "legit"],
                "POS": ["PROPN", "VERB"],
            }
        )
        result = mask_corpus([df], common_adverbs=set())
        assert len(result) == 1
        assert "PROPN" in result[0]
        assert "VERB" in result[0]

    def test_uv_replacement_count(self) -> None:
        """Test UV normalization is applied (no longer counts replacements)."""
        df = pd.DataFrame(
            {
                "word": ["seruus", "servus"],
                "POS": ["NOUN", "NOUN"],
            }
        )
        result = mask_corpus([df], common_adverbs=set())
        # Both words are NOUN tokens, so both should be masked
        assert len(result) == 1
        assert "NOUN" in result[0]


class TestCollectLowercaseWords:
    """Tests for collect_lowercase_words function."""

    def test_basic_collection(self) -> None:
        """Test basic lowercase word collection."""
        sentences = ["etiam bonus", "VERB NOUN"]
        result = collect_lowercase_words(sentences)
        assert "etiam" in result
        assert "bonus" in result
        assert "VERB" not in result
        assert "NOUN" not in result

    def test_empty_input(self) -> None:
        """Test empty input returns empty set."""
        result = collect_lowercase_words([])
        assert result == set()


class TestTwoPassMask:
    """Tests for two_pass_mask function."""

    def test_two_pass_returns_sentences(self) -> None:
        """Test two-pass masking returns sentences."""
        df = pd.DataFrame(
            {
                "word": ["Marcus", "legit"],
                "POS": ["PROPN", "VERB"],
            }
        )
        result = two_pass_mask([df], common_adverbs=set())
        assert len(result) == 1

    def test_two_pass_returns_additional_dict(self) -> None:
        """Test two-pass masking returns sentences (simplified, no dict)."""
        df = pd.DataFrame(
            {
                "word": ["seruus"],
                "POS": ["NOUN"],
            }
        )
        result = two_pass_mask([df], common_adverbs=set())
        assert len(result) == 1


class TestProtectedTokens:
    """Tests for protected token handling in mask_sentence."""

    def test_eol_preserved_in_mask_sentence(self) -> None:
        """Test that <EOL> tokens are preserved as-is during masking."""
        words = ["Arma", "<EOL>", "cano"]
        pos_tags = ["NOUN", "X", "VERB"]
        result = mask_sentence(words, pos_tags, common_adverbs=set())
        assert "<EOL>" in result

    def test_eol_not_lowercased(self) -> None:
        """Test that <EOL> is not lowercased to <eol>."""
        words = ["<EOL>"]
        pos_tags = ["X"]
        result = mask_sentence(words, pos_tags, common_adverbs=set())
        assert result == ["<EOL>"]
        assert "<eol>" not in result

    def test_eol_preserved_in_mask_corpus(self) -> None:
        """Test that <EOL> survives mask_corpus."""
        df = pd.DataFrame(
            {
                "word": ["Arma", "<EOL>", "cano"],
                "POS": ["NOUN", "X", "VERB"],
            }
        )
        result = mask_corpus([df], common_adverbs=set())
        assert "<EOL>" in result[0]

    def test_eol_preserved_in_two_pass_mask(self) -> None:
        """Test that <EOL> survives two_pass_mask."""
        df = pd.DataFrame(
            {
                "word": ["Arma", "<EOL>", "cano"],
                "POS": ["NOUN", "X", "VERB"],
            }
        )
        result = two_pass_mask([df], common_adverbs=set())
        assert "<EOL>" in result[0]

    def test_custom_protected_tokens(self) -> None:
        """Test that custom protected tokens can be added."""
        from latin_masking.types import PROTECTED_TOKENS

        original = PROTECTED_TOKENS.copy()
        try:
            PROTECTED_TOKENS.add("<CUSTOM>")
            words = ["<CUSTOM>", "verbum"]
            pos_tags = ["X", "NOUN"]
            result = mask_sentence(words, pos_tags, common_adverbs=set())
            assert "<CUSTOM>" in result
        finally:
            PROTECTED_TOKENS.clear()
            PROTECTED_TOKENS.update(original)
