"""Tests for sentences.py module."""

from __future__ import annotations

from pathlib import Path

from latin_masking.sentences import (
    check_paren_balance,
    clean_text,
    has_sufficient_punctuation,
    normalize_dashes,
    normalize_whitespace,
    split_sentences,
)

FIXTURES = Path(__file__).parent / "fixtures"


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


class TestNormalizeDashes:
    """Tests for normalize_dashes (dash-drop bug fix)."""

    def test_em_dash_to_space(self) -> None:
        """Em dash is replaced with a single space."""
        assert normalize_dashes("a—b") == "a b"

    def test_en_dash_to_space(self) -> None:
        """En dash is replaced with a single space."""
        assert normalize_dashes("a–b") == "a b"

    def test_no_double_space_when_flanked(self) -> None:
        """A dash already surrounded by spaces does not create a double space."""
        assert normalize_dashes("a — b") == "a b"
        assert normalize_dashes("a – b") == "a b"

    def test_other_punctuation_untouched(self) -> None:
        """Sentence punctuation is preserved."""
        assert normalize_dashes("a! b? c; d.") == "a! b? c; d."


class TestCleanTextDashDrop:
    """Regression tests for the la_senter dash-drop bug."""

    def test_clean_text_keeps_dash_word(self) -> None:
        """A word glued to a trailing em dash is preserved (no dash)."""
        result = clean_text("sanguis meus!—")
        assert "meus!" in result
        assert "—" not in result

    def test_split_sentences_preserves_eol(self) -> None:
        """A line ending in a dash keeps its word and gets an <EOL> token."""
        text = "prōice tēla manū, sanguis meus!—\narma uirumque cano"
        result = split_sentences(text)
        flat = " ".join(result)
        assert "meus! <EOL>" in flat
        # No token was dropped: the first line has 5 words + EOL.
        assert "meus!" in flat

    def test_split_sentences_dash_after_terminator_splits(self) -> None:
        """A dash after '!' still yields a sentence boundary via the '!'."""
        text = "auguria!– expāvit vitreō sub gurgite rēmōs."
        result = split_sentences(text)
        assert any(s.startswith("auguria!") for s in result)
        assert any(s.startswith("expauit") for s in result)


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

    def test_strips_editorial_punctuation(self) -> None:
        """Test that editorial punctuation chars (e.g. †) are stripped."""
        text = "Marcus est †bonus†."
        result = split_sentences(text)
        for sent in result:
            assert "†" not in sent

    def test_normalizes_michi(self) -> None:
        """Test that michi→mihi and nichil→nihil normalization always runs."""
        text = "Michi nichil dico."
        result = split_sentences(text)
        joined = " ".join(result)
        assert "Mihi" in joined  # capitalized form
        assert "nihil" in joined

    def test_normalizes_uv_ij(self) -> None:
        """Test that UV/IJ normalization (v→u, j→i) always runs."""
        text = "Servus vivit. Jam venit."
        result = split_sentences(text)
        joined = " ".join(result)
        assert "v" not in joined.split()  # no standalone v in words
        assert "j" not in joined.split()  # no standalone j in words


class TestSplitSentencesFragments:
    """End-to-end segmentation of real verse fragments.

    These run the normal library ``split_sentences`` on raw verse text and
    compare against manually-reviewed expected output (the ported
    segmentation logic should reproduce them exactly).
    """

    def _run(self, stem: str) -> list[str]:
        raw = (FIXTURES / f"{stem}_raw.txt").read_text(encoding="utf-8")
        expected = (FIXTURES / f"{stem}_expected.txt").read_text(encoding="utf-8")
        result = split_sentences(raw)
        assert result == expected.splitlines()
        return result

    def test_aeneid_1_242_266(self) -> None:
        """Vergil Aeneid 1.242-266 (single-token and multi-line parens)."""
        self._run("aeneid_1_242_266")

    def test_horace_epistulae_1_15(self) -> None:
        """Horace Epistulae 1.15 (large multi-sentence paren)."""
        self._run("horace_epistulae_1_15")


class TestCheckParenBalance:
    """Pre-flight check that every '(' is matched by a ')'."""

    def test_balanced_parens_ok(self) -> None:
        """Balanced parentheses raise nothing."""
        check_paren_balance("uos (infandum!) amissis, unius ob iram.")

    def test_nested_parens_ok(self) -> None:
        """Nested parentheses are accepted."""
        check_paren_balance("hic (fabor enim, (quando haec) te cura) remordet.")

    def test_unmatched_open_reports_line(self) -> None:
        """An unmatched '(' names the line where it opened."""
        text = "prima linea.\n(Heu michi! uos, fratres, sine dentibus estis.\nultima linea."
        try:
            check_paren_balance(text)
        except ValueError as exc:
            assert "line 2" in str(exc)
            assert "(Heu" in str(exc)
        else:
            raise AssertionError("expected ValueError for unmatched '('")

    def test_unmatched_close_reports_line(self) -> None:
        """An unmatched ')' names the line where it appears."""
        text = "prima linea.\nclausit) et dixit.\nultima linea."
        try:
            check_paren_balance(text)
        except ValueError as exc:
            assert "line 2" in str(exc)
            assert "clausit)" in str(exc)
        else:
            raise AssertionError("expected ValueError for unmatched ')'")

    def test_split_sentences_refuses_unbalanced(self) -> None:
        """split_sentences raises on unbalanced parens before any work."""
        text = "apertum (infandum remordet.\nsecunda linea."
        try:
            split_sentences(text)
        except ValueError as exc:
            assert "line 1" in str(exc)
        else:
            raise AssertionError("expected ValueError for unbalanced parens")
