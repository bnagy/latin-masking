"""Tests for preprocessor.py module — especially idempotency."""

from __future__ import annotations

from latin_masking.preprocessor import preprocess
from latin_masking.normalize import normalize_text, normalize_uv_ij, normalize_ch_h


class TestPreprocessIdempotency:
    """Tests that preprocess() is idempotent: preprocess(preprocess(x)) == preprocess(x)."""

    def test_idempotent_on_plain_latin(self) -> None:
        """Test idempotency on plain Latin text."""
        text = "Servus vivit in horto."
        once = preprocess(text)
        twice = preprocess(once)
        assert twice == once

    def test_idempotent_on_uv_ij_text(self) -> None:
        """Test idempotency on text with v/j that need normalizing."""
        text = "Jam venit servus."
        once = preprocess(text)
        twice = preprocess(once)
        assert twice == once

    def test_idempotent_on_ch_h_text(self) -> None:
        """Test idempotency on text with michi/nichil."""
        text = "Michi nichil dicit."
        once = preprocess(text)
        twice = preprocess(once)
        assert twice == once

    def test_idempotent_on_macrons(self) -> None:
        """Test idempotency on text with macrons."""
        text = "Servus vīvit in hortō."
        once = preprocess(text)
        twice = preprocess(once)
        assert twice == once

    def test_idempotent_on_punctuation_chars(self) -> None:
        """Test idempotency on text with editorial punctuation chars."""
        text = "Marcus est †bonus† [note] {gloss}."
        once = preprocess(text)
        twice = preprocess(once)
        assert twice == once

    def test_idempotent_on_mixed(self) -> None:
        """Test idempotency on text with all transformations at once."""
        text = "Jam michi †vīvit† [note] in hortō."
        once = preprocess(text)
        twice = preprocess(once)
        assert twice == once

    def test_idempotent_on_empty(self) -> None:
        """Test idempotency on empty string."""
        assert preprocess(preprocess("")) == preprocess("")

    def test_idempotent_on_already_clean(self) -> None:
        """Test idempotency on text that needs no changes."""
        text = "Marcus est bonus."
        once = preprocess(text)
        twice = preprocess(once)
        assert twice == once

    def test_idempotent_with_protected_tokens(self) -> None:
        """Test idempotency when <EOL> protected tokens are present."""
        text = "Arma cano. <EOL> Troiae primus."
        once = preprocess(text)
        twice = preprocess(once)
        assert twice == once


class TestNormalizeIdempotency:
    """Tests that normalize functions are idempotent."""

    def test_normalize_uv_ij_idempotent(self) -> None:
        """Test normalize_uv_ij is idempotent."""
        words = ["servus", "vivere", "major", "jam", "bonus"]
        for word in words:
            once = normalize_uv_ij(word)
            twice = normalize_uv_ij(once)
            assert twice == once, f"Not idempotent for {word!r}: {once!r} → {twice!r}"

    def test_normalize_ch_h_idempotent(self) -> None:
        """Test normalize_ch_h is idempotent."""
        words = ["michi", "nichil", "mihi", "nihil", "bonus"]
        for word in words:
            once = normalize_ch_h(word)
            twice = normalize_ch_h(once)
            assert twice == once, f"Not idempotent for {word!r}: {once!r} → {twice!r}"

    def test_normalize_text_idempotent(self) -> None:
        """Test normalize_text is idempotent."""
        texts = [
            "jam michi servus",
            "vivere in horto",
            "nichil dicit",
            "",
        ]
        for text in texts:
            once = normalize_text(text)
            twice = normalize_text(once)
            assert twice == once, f"Not idempotent for {text!r}: {once!r} → {twice!r}"


class TestPreprocessTransformations:
    """Tests that preprocess applies the expected transformations."""

    def test_v_to_u(self) -> None:
        """Test v→u normalization."""
        assert "u" in preprocess("vivit")
        assert "v" not in preprocess("vivit")

    def test_j_to_i(self) -> None:
        """Test j→i normalization."""
        assert "i" in preprocess("jam")
        assert "j" not in preprocess("jam")

    def test_michi_to_mihi(self) -> None:
        """Test michi→mihi normalization."""
        result = preprocess("michi")
        assert "mihi" in result
        assert "michi" not in result

    def test_macron_removal(self) -> None:
        """Test macron removal."""
        result = preprocess("vīvit")
        assert "\u0304" not in result  # no combining macron
        assert "\u012b" not in result  # no ī (i with macron)

    def test_dagger_stripped(self) -> None:
        """Test † (dagger) is stripped."""
        result = preprocess("†bonus†")
        assert "†" not in result

    def test_brackets_stripped(self) -> None:
        """Test [] are stripped."""
        result = preprocess("[note]")
        assert "[" not in result
        assert "]" not in result
