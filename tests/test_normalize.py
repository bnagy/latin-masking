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
    """Tests for normalize_ch_h function."""

    def test_mihi_to_michi(self) -> None:
        """Test mihi→michi normalization."""
        assert normalize_ch_h("mihi") == "michi"
        assert normalize_ch_h("Mihi") == "Michi"

    def test_nihil_to_nichil(self) -> None:
        """Test nihil→nichil normalization."""
        assert normalize_ch_h("nihil") == "nichil"
        assert normalize_ch_h("Nihil") == "Nichil"

    def test_no_match_returns_unchanged(self) -> None:
        """Test non-matching words return unchanged."""
        assert normalize_ch_h("bonus") == "bonus"
        assert normalize_ch_h("terra") == "terra"
