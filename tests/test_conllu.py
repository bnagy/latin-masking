"""Tests for conllu.py module."""

from __future__ import annotations

from latin_masking.conllu import (
    iter_conllu_sentences,
    parse_conllu,
    parse_conllu_light,
)


class TestParseConllu:
    """Tests for parse_conllu function."""

    def test_basic_parsing(self, sample_conllu: str) -> None:
        """Test basic CoNLL-U parsing."""
        frames, texts = parse_conllu(sample_conllu)
        assert len(frames) >= 1
        assert len(texts) >= 1

    def test_filters_punctuation(self, sample_conllu: str) -> None:
        """Test that PUNCT tokens are filtered out."""
        frames, _ = parse_conllu(sample_conllu)
        for frame in frames:
            assert "PUNCT" not in frame["POS"].values

    def test_text_extraction(self, sample_conllu: str) -> None:
        """Test sentence text extraction."""
        frames, texts = parse_conllu(sample_conllu)
        assert len(frames) == len(texts)

    def test_empty_response(self) -> None:
        """Test empty response handling."""
        frames, texts = parse_conllu("")
        assert frames == []
        assert texts == []


class TestParseConlluLight:
    """Tests for parse_conllu_light function."""

    def test_basic_parsing(self, sample_conllu: str) -> None:
        """Test basic light parsing."""
        result = parse_conllu_light(sample_conllu)
        assert len(result) >= 1
        assert "words" in result[0]
        assert "pos" in result[0]
        assert "lemmas" in result[0]

    def test_word_extraction(self, sample_conllu: str) -> None:
        """Test word extraction."""
        result = parse_conllu_light(sample_conllu)
        for sentence in result:
            assert len(sentence["words"]) > 0

    def test_empty_response(self) -> None:
        """Test empty response handling."""
        result = parse_conllu_light("")
        assert result == []


class TestIterConlluSentences:
    """Tests for iter_conllu_sentences generator."""

    def test_basic_iteration(self, sample_conllu: str) -> None:
        """Test basic iteration over sentences."""
        sentences = list(iter_conllu_sentences(sample_conllu))
        assert len(sentences) >= 1

    def test_tuple_structure(self, sample_conllu: str) -> None:
        """Test that each item is a tuple of (words, pos_tags, text)."""
        for words, pos_tags, text in iter_conllu_sentences(sample_conllu):
            assert isinstance(words, list)
            assert isinstance(pos_tags, list)
            assert isinstance(text, str)
            assert len(words) == len(pos_tags)

    def test_empty_response(self) -> None:
        """Test empty response yields nothing."""
        sentences = list(iter_conllu_sentences(""))
        assert sentences == []
