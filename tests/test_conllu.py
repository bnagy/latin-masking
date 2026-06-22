"""Tests for conllu.py module."""

from __future__ import annotations

from latin_masking.conllu import (
    extract_features_by_type,
    extract_full_features,
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


class TestExtractFullFeatures:
    """Tests for extract_full_features function."""

    def test_returns_all_feature_pairs(self, sample_conllu_rich_feats: str) -> None:
        """Test that all feature key-value pairs are returned."""
        frames, _ = parse_conllu(sample_conllu_rich_feats)
        features = extract_full_features(frames)
        assert len(features) == 1
        # Marcus: Case=Nom|Gender=Masc|Number=Sing|InflClass=IndEurO (4)
        # est: Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act (6)
        # in: _ (0)
        # horto: Case=Abl|Gender=Masc|Number=Sing|InflClass=IndEurO (4)
        assert len(features[0]) == 4 + 6 + 0 + 4
        assert "Case=Nom" in features[0]
        assert "InflClass=IndEurO" in features[0]
        assert "Voice=Act" in features[0]

    def test_preserves_feature_order(self, sample_conllu_rich_feats: str) -> None:
        """Test that feature order matches CoNLL-U order."""
        frames, _ = parse_conllu(sample_conllu_rich_feats)
        features = extract_full_features(frames)
        # First token (Marcus) has Case=Nom|Gender=Masc|Number=Sing|InflClass=IndEurO
        assert features[0][:4] == [
            "Case=Nom",
            "Gender=Masc",
            "Number=Sing",
            "InflClass=IndEurO",
        ]

    def test_no_features_contributes_nothing(self) -> None:
        """Test that tokens with Feats='_' contribute nothing."""
        import pandas as pd  # pyright: ignore[reportMissingImports]

        frame = pd.DataFrame({
            "Feats": ["_", "_", "_"],
        })
        features = extract_full_features([frame])
        assert features == [[]]

    def test_empty_frames(self) -> None:
        """Test with empty frames list."""
        features = extract_full_features([])
        assert features == []

    def test_multiple_sentences(self, sample_conllu_rich_feats: str) -> None:
        """Test with multiple sentences (same fixture repeated)."""
        frames, _ = parse_conllu(sample_conllu_rich_feats)
        # Duplicate frames to simulate two sentences
        features = extract_full_features(frames + frames)
        assert len(features) == 2
        assert features[0] == features[1]


class TestExtractFeaturesByType:
    """Tests for extract_features_by_type function."""

    def test_groups_by_type(self, sample_conllu_rich_feats: str) -> None:
        """Test that features are grouped by type name."""
        frames, _ = parse_conllu(sample_conllu_rich_feats)
        result = extract_features_by_type(frames)
        assert len(result) == 1
        d = result[0]
        assert "Case" in d
        assert "Gender" in d
        assert "Number" in d
        assert "InflClass" in d
        assert "Mood" in d
        assert "Person" in d
        assert "Tense" in d
        assert "VerbForm" in d
        assert "Voice" in d

    def test_collects_all_values(self, sample_conllu_rich_feats: str) -> None:
        """Test that all values for a given type are collected."""
        frames, _ = parse_conllu(sample_conllu_rich_feats)
        result = extract_features_by_type(frames)
        d = result[0]
        # Case appears on Marcus (Nom) and horto (Abl)
        assert d["Case"] == ["Nom", "Abl"]
        # Gender appears on Marcus (Masc) and horto (Masc)
        assert d["Gender"] == ["Masc", "Masc"]

    def test_no_features_returns_empty_dict(self) -> None:
        """Test that tokens with Feats='_' produce empty dict."""
        import pandas as pd  # pyright: ignore[reportMissingImports]

        frame = pd.DataFrame({
            "Feats": ["_", "_"],
        })
        result = extract_features_by_type([frame])
        assert result == [{}]

    def test_empty_frames(self) -> None:
        """Test with empty frames list."""
        result = extract_features_by_type([])
        assert result == []

    def test_multiple_sentences(self, sample_conllu_rich_feats: str) -> None:
        """Test with multiple sentences."""
        frames, _ = parse_conllu(sample_conllu_rich_feats)
        result = extract_features_by_type(frames + frames)
        assert len(result) == 2
        assert result[0] == result[1]
