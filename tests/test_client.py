"""Tests for client.py module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from latin_masking.client import list_models, process_text, remove_macrons
from latin_masking.types import UDPipeError, UDPipeInputError


class TestRemoveMacrons:
    """Tests for remove_macrons function."""

    def test_remove_macron(self) -> None:
        """Test macron removal."""
        text = "mārcus"
        result = remove_macrons(text)
        assert "ā" not in result

    def test_no_macrons(self) -> None:
        """Test text without macrons unchanged."""
        text = "marcus"
        result = remove_macrons(text)
        assert result == text


class TestProcessText:
    """Tests for process_text function."""

    def test_empty_input_raises(self) -> None:
        """Test empty input raises UDPipeInputError."""
        with pytest.raises(UDPipeInputError):
            process_text("")

    def test_whitespace_only_raises(self) -> None:
        """Test whitespace-only input raises UDPipeInputError."""
        with pytest.raises(UDPipeInputError):
            process_text("   ")

    @patch("latin_masking.client._perform_request")
    def test_successful_processing(self, mock_request: MagicMock) -> None:
        """Test successful text processing."""
        mock_request.return_value = {
            "model": "latin-ittb",
            "result": "# text = test\n1\ttest\ttest\tNOUN\t_\t_\t0\troot\t_\t_\n",
        }
        result = process_text("test", raw=True)
        assert "# text" in result

    @patch("latin_masking.client._perform_request")
    def test_presegmented_mode(self, mock_request: MagicMock) -> None:
        """Test presegmented mode."""
        mock_request.return_value = {
            "model": "latin-ittb",
            "result": "# text = test\n1\ttest\ttest\tNOUN\t_\t_\t0\troot\t_\t_\n",
        }
        result = process_text("test sentence", presegmented=True, raw=True)
        assert result is not None


class TestListModels:
    """Tests for list_models function."""

    @patch("latin_masking.client._perform_request")
    def test_list_models(self, mock_request: MagicMock) -> None:
        """Test listing available models."""
        mock_request.return_value = {"models": ["latin-ittb", "latin-ud"]}
        result = list_models()
        assert "latin-ittb" in result
        assert "latin-ud" in result


class TestProcessTextEdgeCases:
    """Tests for process_text edge cases."""

    @patch("latin_masking.client._perform_request")
    def test_strip_punctuation(self, mock_request: MagicMock) -> None:
        """Test punctuation stripping."""
        mock_request.return_value = {
            "model": "latin-ittb",
            "result": "# text = test\n1\ttest\ttest\tNOUN\t_\t_\t0\troot\t_\t_\n",
        }
        result = process_text("test [word] <here>", raw=True)
        assert result is not None

    @patch("latin_masking.client._perform_request")
    def test_remove_macrons_flag(self, mock_request: MagicMock) -> None:
        """Test macron removal flag."""
        mock_request.return_value = {
            "model": "latin-ittb",
            "result": "# text = test\n1\ttest\ttest\tNOUN\t_\t_\t0\troot\t_\t_\n",
        }
        result = process_text("mārcus", remove_macrons_flag=True, raw=True)
        assert result is not None

    @patch("latin_masking.client._perform_request")
    def test_raw_false_returns_parsed(self, mock_request: MagicMock) -> None:
        """Test raw=False returns parsed DataFrames."""
        mock_request.return_value = {
            "model": "latin-ittb",
            "result": "# text = test\n1\ttest\ttest\tNOUN\t_\t_\t0\troot\t_\t_\n",
        }
        result = process_text("test", raw=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch("latin_masking.client._perform_request")
    def test_missing_model_in_response_raises_udpipe_error(
        self, mock_request: MagicMock
    ) -> None:
        """Test missing model field raises UDPipeError."""
        mock_request.return_value = {"result": "test"}

        with pytest.raises(UDPipeError):
            process_text("test", raw=True)

    @patch("latin_masking.client._perform_request")
    def test_missing_result_in_response_raises_udpipe_error(
        self, mock_request: MagicMock
    ) -> None:
        """Test missing result field raises UDPipeError."""
        mock_request.return_value = {"model": "latin-ittb"}

        with pytest.raises(UDPipeError):
            process_text("test", raw=True)


class TestRemoveMacronsEdgeCases:
    """Tests for remove_macrons edge cases."""

    def test_mixed_macron_no_macron(self) -> None:
        """Test text with mixed macron and non-macron characters."""
        text = "mārcus vīta"
        result = remove_macrons(text)
        assert "ā" not in result
        assert "ī" not in result
        assert "v" in result

    def test_empty_string(self) -> None:
        """Test empty string."""
        text = ""
        result = remove_macrons(text)
        assert result == ""

    def test_only_macrons(self) -> None:
        """Test string with only macron characters."""
        text = "āē"
        result = remove_macrons(text)
        assert "ā" not in result
        assert "ē" not in result


class TestProcessTextRawFalse:
    """Tests for process_text with raw=False (parsed output)."""

    @patch("latin_masking.client._perform_request")
    def test_parsed_output_structure(self, mock_request: MagicMock) -> None:
        """Test that raw=False returns tuple of (frames, texts)."""
        mock_request.return_value = {
            "model": "latin-ittb",
            "result": "# text = test\n1\ttest\ttest\tNOUN\t_\t_\t0\troot\t_\t_\n",
        }
        result = process_text("test", raw=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        frames, texts = result
        assert isinstance(frames, list)
        assert isinstance(texts, list)

    @patch("latin_masking.client._perform_request")
    def test_parsed_output_with_multiple_sentences(
        self, mock_request: MagicMock
    ) -> None:
        """Test parsing multiple sentences."""
        mock_request.return_value = {
            "model": "latin-ittb",
            "result": "# text = first\n1\tfirst\tfirst\tNOUN\t_\t_\t0\troot\t_\t_\n\n# text = second\n1\tsecond\tsecond\tNOUN\t_\t_\t0\troot\t_\t_\n",
        }
        result = process_text("first second", raw=False)
        frames, texts = result
        assert len(frames) == 2
        assert len(texts) == 2
        assert "first" in texts[0]
        assert "second" in texts[1]


class TestProcessTextPresegmented:
    """Tests for process_text with presegmented mode."""

    @patch("latin_masking.client._perform_request")
    def test_presegmented_tokenizer_modified(self, mock_request: MagicMock) -> None:
        """Test that presegmented mode modifies tokenizer argument."""
        mock_request.return_value = {
            "model": "latin-ittb",
            "result": "# text = test\n1\ttest\ttest\tNOUN\t_\t_\t0\troot\t_\t_\n",
        }
        process_text("line1\nline2", presegmented=True, raw=True)

        # Check that the call was made with presegmented tokenizer
        call_args = mock_request.call_args
        assert call_args is not None
        # The call is _perform_request(method, params, ...)
        # call_args is ((method, params), {})
        method = call_args[0][0]
        params = call_args[0][1]
        assert method == "process"
        assert "tokenizer" in params
        assert "presegmented" in params["tokenizer"]


class TestProcessTextStripPunctuation:
    """Tests for process_text punctuation stripping."""

    @patch("latin_masking.client._perform_request")
    def test_brackets_stripped(self, mock_request: MagicMock) -> None:
        """Test that brackets are stripped from input."""
        mock_request.return_value = {
            "model": "latin-ittb",
            "result": "# text = test\n1\ttest\ttest\tNOUN\t_\t_\t0\troot\t_\t_\n",
        }
        process_text("test [bracket]", raw=True)

        # Check that the call was made with stripped text
        call_args = mock_request.call_args
        assert call_args is not None
        params = call_args[0][1]
        assert "[" not in params["data"]
        assert "]" not in params["data"]

    @patch("latin_masking.client._perform_request")
    def test_angle_brackets_stripped(self, mock_request: MagicMock) -> None:
        """Test that angle brackets are stripped from input."""
        mock_request.return_value = {
            "model": "latin-ittb",
            "result": "# text = test\n1\ttest\ttest\tNOUN\t_\t_\t0\troot\t_\t_\n",
        }
        process_text("test <angle>", raw=True)

        call_args = mock_request.call_args
        assert call_args is not None
        params = call_args[0][1]
        assert "<" not in params["data"]
        assert ">" not in params["data"]

    @patch("latin_masking.client._perform_request")
    def test_curly_braces_stripped(self, mock_request: MagicMock) -> None:
        """Test that curly braces are stripped from input."""
        mock_request.return_value = {
            "model": "latin-ittb",
            "result": "# text = test\n1\ttest\ttest\tNOUN\t_\t_\t0\troot\t_\t_\n",
        }
        process_text("test {curly}", raw=True)

        call_args = mock_request.call_args
        assert call_args is not None
        params = call_args[0][1]
        assert "{" not in params["data"]
        assert "}" not in params["data"]

    @patch("latin_masking.client._perform_request")
    def test_dagger_stripped(self, mock_request: MagicMock) -> None:
        """Test that dagger symbol is stripped from input."""
        mock_request.return_value = {
            "model": "latin-ittb",
            "result": "# text = test\n1\ttest\ttest\tNOUN\t_\t_\t0\troot\t_\t_\n",
        }
        process_text("test †dagger", raw=True)

        call_args = mock_request.call_args
        assert call_args is not None
        params = call_args[0][1]
        assert "†" not in params["data"]
