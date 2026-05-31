"""Tests for normalize.py module."""

from __future__ import annotations

from latin_masking.normalize import normalize_ch_h, normalize_text, normalize_uv_ij


class TestNormalizeUvIj:
    """Tests for normalize_uv_ij function."""

    def test_v_to_u(self) -> None:
        """Test v→u normalization."""
        assert normalize_uv_ij("servus") == "seruus"
        assert normalize_uv_ij("vivere") == "uiuere"

    def test_j_to_i(self) -> None:
        """Test j→i normalization."""
        assert normalize_uv_ij("major") == "maior"
        assert normalize_uv_ij("jam") == "iam"

    def test_combined(self) -> None:
        """Test combined v→u and j→i normalization."""
        assert normalize_uv_ij("vivere") == "uiuere"
        assert normalize_uv_ij("major") == "maior"

    def test_no_change(self) -> None:
        """Test words without v/j are unchanged."""
        assert normalize_uv_ij("bonus") == "bonus"
        assert normalize_uv_ij("terra") == "terra"

    def test_uppercase_preserved(self) -> None:
        """Test uppercase letters are normalized correctly."""
        assert normalize_uv_ij("VIVERE") == "UIUERE"
        assert normalize_uv_ij("MAJOR") == "MAIOR"


class TestNormalizeChH:
    """Tests for normalize_ch_h function (h→ch normalization)."""

    def test_michi_to_mihi(self) -> None:
        """Test michi→mihi normalization."""
        assert normalize_ch_h("michi") == "mihi"
        assert normalize_ch_h("Michi") == "Mihi"

    def test_nichil_to_nihil(self) -> None:
        """Test nichil→nihil normalization."""
        assert normalize_ch_h("nichil") == "nihil"
        assert normalize_ch_h("Nichil") == "Nihil"

    def test_no_match_returns_unchanged(self) -> None:
        """Test non-matching words return unchanged."""
        assert normalize_ch_h("bonus") == "bonus"
        assert normalize_ch_h("terra") == "terra"


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_basic_normalization(self) -> None:
        """Test basic UV/IJ normalization on full text."""
        text = "jam in horto"
        result = normalize_text(text)
        assert result == "iam in horto"

    def test_v_to_u(self) -> None:
        """Test v→u normalization on full text."""
        text = "servus vivit"
        result = normalize_text(text)
        assert result == "seruus uiuit"

    def test_ch_h_normalization(self) -> None:
        """Test ch→h normalization on full text."""
        text = "michi nichil"
        result = normalize_text(text)
        assert result == "mihi nihil"

    def test_combined_normalization(self) -> None:
        """Test combined UV/IJ + ch/h normalization."""
        text = "jam michi servus"
        result = normalize_text(text)
        assert result == "iam mihi seruus"

    def test_preserves_whitespace_structure(self) -> None:
        """Test that output is single-space separated."""
        text = "jam   in    horto"
        result = normalize_text(text)
        assert result == "iam in horto"

    def test_empty_text(self) -> None:
        """Test empty text returns empty string."""
        assert normalize_text("") == ""

    def test_single_token(self) -> None:
        """Test single token normalization."""
        assert normalize_text("jam") == "iam"

    def test_no_change_needed(self) -> None:
        """Test text that needs no normalization."""
        text = "in horto sedet"
        assert normalize_text(text) == text

    def test_jamque_normalizes_to_iamque(self) -> None:
        """Test that jamque normalizes to iamque (not jamaque)."""
        result = normalize_text("jamque")
        assert result == "iamque"


class TestProtectedTokens:
    """Tests for protect_tokens and restore_tokens."""

    def test_protect_eol(self) -> None:
        """Test that <EOL> is replaced with a placeholder."""
        from latin_masking.types import protect_tokens

        text = "Arma <EOL> cano"
        result, mapping = protect_tokens(text)
        assert "<EOL>" not in result
        assert "<EOL>" in mapping.values()

    def test_restore_eol(self) -> None:
        """Test that <EOL> is restored from placeholder."""
        from latin_masking.types import protect_tokens, restore_tokens

        text = "Arma <EOL> cano"
        protected, mapping = protect_tokens(text)
        restored = restore_tokens(protected, mapping)
        assert restored == text

    def test_protect_multiple_eols(self) -> None:
        """Test that multiple <EOL> tokens are all protected."""
        from latin_masking.types import protect_tokens, restore_tokens

        text = "Arma <EOL> cano <EOL> Troiae"
        protected, mapping = protect_tokens(text)
        # Both <EOL> tokens are replaced with the same placeholder
        assert protected.count("__PROTECTED_0__") == 2
        assert "<EOL>" not in protected
        restored = restore_tokens(protected, mapping)
        assert restored == text

    def test_protect_custom_tokens(self) -> None:
        """Test protecting custom tokens."""
        from latin_masking.types import protect_tokens, restore_tokens

        text = "Arma <CUSTOM> cano"
        protected, mapping = protect_tokens(text, tokens={"<CUSTOM>"})
        assert "<CUSTOM>" not in protected
        restored = restore_tokens(protected, mapping)
        assert restored == text

    def test_protect_empty_text(self) -> None:
        """Test protecting empty text."""
        from latin_masking.types import protect_tokens, restore_tokens

        text = ""
        protected, mapping = protect_tokens(text)
        assert protected == ""
        assert mapping == {}

    def test_protect_no_protected_tokens(self) -> None:
        """Test text with no protected tokens."""
        from latin_masking.types import protect_tokens, restore_tokens

        text = "Arma virumque cano"
        protected, mapping = protect_tokens(text)
        assert protected == text
        assert mapping == {}
