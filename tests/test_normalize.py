"""Tests for normalize.py module."""

from __future__ import annotations

from latin_masking.normalize import normalize_ch_h, normalize_uv_ij


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
