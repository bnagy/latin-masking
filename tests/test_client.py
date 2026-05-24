"""Tests for client.py module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, mock_open

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


class TestRetryLogic:
    """Tests for retry logic in _perform_request."""

    @patch("latin_masking.client.time.sleep")
    @patch("urllib.request.urlopen")
    def test_retry_on_URLError(
        self, mock_urlopen: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Test that URLError triggers retry with backoff."""
        import urllib.error

        # First two calls fail, third succeeds
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = b'{"model": "test", "result": "test"}'

        mock_urlopen.side_effect = [
            urllib.error.URLError("Connection refused"),
            urllib.error.URLError("Connection refused"),
            mock_response,
        ]

        from latin_masking.client import _perform_request

        result = _perform_request("models", None)
        assert result is not None
        # Should have slept twice (2 retries)
        assert mock_sleep.call_count == 2

    @patch("latin_masking.client.time.sleep")
    @patch("urllib.request.urlopen")
    def test_max_retries_exceeded(
        self, mock_urlopen: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Test that max retries is respected."""
        import urllib.error

        from latin_masking.client import _perform_request, MAX_RETRIES

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(Exception):  # UDPipeAPIError
            _perform_request("models", None)

        # Should have slept MAX_RETRIES - 1 times
        assert mock_sleep.call_count == MAX_RETRIES - 1


class TestProcessFileWithCache:
    """Tests for process_file_with_cache function."""

    @patch("latin_masking.client.process_text")
    @patch("latin_masking.cache.load_cached_response")
    def test_returns_cached_response(
        self,
        mock_load: MagicMock,
        mock_process: MagicMock,
    ) -> None:
        """Test that cached response is returned when cache file exists."""
        from pathlib import Path

        from latin_masking.client import process_file_with_cache

        mock_load.return_value = "cached result"

        mock_cache_path = MagicMock()
        mock_cache_path.exists.return_value = True

        with patch("latin_masking.cache.get_cache_path", return_value=mock_cache_path):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "is_dir", return_value=False):
                    with patch("builtins.open", mock_open(read_data="test text")):
                        result = process_file_with_cache(
                            Path("/source/test.txt"),
                            "test-model",
                        )

        assert result == "cached result"
        mock_process.assert_not_called()

    @patch("latin_masking.client.process_text")
    @patch("latin_masking.cache.save_cached_response")
    def test_processes_and_saves(
        self,
        mock_save: MagicMock,
        mock_process: MagicMock,
    ) -> None:
        """Test that text is processed and saved when cache missing."""
        from pathlib import Path

        from latin_masking.client import process_file_with_cache

        mock_process.return_value = "new result"

        mock_cache_path = MagicMock()
        mock_cache_path.exists.return_value = False

        with patch("latin_masking.cache.get_cache_path", return_value=mock_cache_path):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "is_dir", return_value=False):
                    with patch("builtins.open", mock_open(read_data="test text")):
                        result = process_file_with_cache(
                            Path("/source/test.txt"),
                            "test-model",
                        )

        assert result == "new result"
        mock_save.assert_called_once()

    @patch("latin_masking.client.process_text")
    @patch("latin_masking.cache.save_cached_response")
    def test_passes_kwargs_to_process_text(
        self,
        mock_save: MagicMock,
        mock_process: MagicMock,
    ) -> None:
        """Test that kwargs are passed to process_text."""
        from pathlib import Path

        from latin_masking.client import process_file_with_cache

        mock_process.return_value = "result"

        mock_cache_path = MagicMock()
        mock_cache_path.exists.return_value = False

        with patch("latin_masking.cache.get_cache_path", return_value=mock_cache_path):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "is_dir", return_value=False):
                    with patch("builtins.open", mock_open(read_data="test text")):
                        process_file_with_cache(
                            Path("/source/test.txt"),
                            "test-model",
                            presegmented=True,
                            raw=True,
                        )

        mock_process.assert_called_once()
        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["presegmented"] is True
        assert call_kwargs["raw"] is True

    @patch("latin_masking.client.process_text")
    @patch("latin_masking.cache.save_cached_response")
    def test_force_refresh_bypasses_cache(
        self,
        mock_save: MagicMock,
        mock_process: MagicMock,
    ) -> None:
        """Test that force_refresh=True bypasses cache."""
        from pathlib import Path

        from latin_masking.client import process_file_with_cache

        mock_process.return_value = "fresh result"
        mock_cache_path = MagicMock()
        mock_cache_path.exists.return_value = True

        with patch("latin_masking.cache.get_cache_path", return_value=mock_cache_path):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "is_dir", return_value=False):
                    with patch("builtins.open", mock_open(read_data="test text")):
                        result = process_file_with_cache(
                            Path("/source/test.txt"),
                            "test-model",
                            force_refresh=True,
                        )

        assert result == "fresh result"

    def test_missing_input_file_raises_error(self) -> None:
        """Test that missing input file raises FileNotFoundError."""
        from pathlib import Path

        from latin_masking.client import process_file_with_cache

        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                process_file_with_cache(
                    Path("/nonexistent/file.txt"),
                    "test-model",
                )

    def test_directory_input_raises_error(self, tmp_path: Path) -> None:
        """Test that directory input raises IsADirectoryError."""
        from pathlib import Path

        from latin_masking.client import process_file_with_cache

        with pytest.raises(IsADirectoryError):
            process_file_with_cache(
                tmp_path,
                "test-model",
            )

    @patch("latin_masking.client.process_text")
    @patch("latin_masking.cache.save_cached_response")
    def test_default_cache_dir_same_as_input(
        self,
        mock_save: MagicMock,
        mock_process: MagicMock,
    ) -> None:
        """Test that cache_dir defaults to input_path (same directory)."""
        from pathlib import Path

        from latin_masking.client import process_file_with_cache

        mock_process.return_value = "result"

        mock_cache_path = MagicMock()
        mock_cache_path.exists.return_value = False
        mock_cache_path.__str__ = MagicMock(return_value="/source/test.txt/cache.pkl")

        with patch("latin_masking.cache.get_cache_path", return_value=mock_cache_path):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "is_dir", return_value=False):
                    with patch("builtins.open", mock_open(read_data="test text")):
                        process_file_with_cache(
                            Path("/source/test.txt"),
                            "test-model",
                        )

        # Verify save was called with a path under input_path
        call_args = mock_save.call_args[0]
        assert str(call_args[0]).startswith("/source/test.txt")

    @patch("latin_masking.client.process_text")
    @patch("latin_masking.cache.save_cached_response")
    def test_custom_cache_dir(
        self,
        mock_save: MagicMock,
        mock_process: MagicMock,
    ) -> None:
        """Test that custom cache_dir is used."""
        from pathlib import Path

        from latin_masking.client import process_file_with_cache

        mock_process.return_value = "result"

        mock_cache_path = MagicMock()
        mock_cache_path.exists.return_value = False
        mock_cache_path.__str__ = MagicMock(return_value="/custom/cache/cache.pkl")

        with patch("latin_masking.cache.get_cache_path", return_value=mock_cache_path):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "is_dir", return_value=False):
                    with patch("builtins.open", mock_open(read_data="test text")):
                        process_file_with_cache(
                            Path("/source/test.txt"),
                            "test-model",
                            cache_dir=Path("/custom/cache"),
                        )

        # Verify save was called with a path under custom cache_dir
        call_args = mock_save.call_args[0]
        assert str(call_args[0]).startswith("/custom/cache")

    @patch("latin_masking.client.process_text")
    @patch("latin_masking.cache.save_cached_response")
    def test_encoding_error_propagates(
        self,
        mock_save: MagicMock,
        mock_process: MagicMock,
    ) -> None:
        """Test that UnicodeDecodeError propagates."""
        from pathlib import Path

        from latin_masking.client import process_file_with_cache

        mock_cache_path = MagicMock()
        mock_cache_path.exists.return_value = False

        with patch("latin_masking.cache.get_cache_path", return_value=mock_cache_path):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "is_dir", return_value=False):
                    mock_file = MagicMock()
                    mock_file.__enter__ = MagicMock(return_value=mock_file)
                    mock_file.__exit__ = MagicMock(return_value=False)
                    mock_file.read.side_effect = UnicodeDecodeError(
                        "utf-8", b"", 0, 1, "invalid"
                    )

                    with patch("builtins.open", return_value=mock_file):
                        with pytest.raises(UnicodeDecodeError):
                            process_file_with_cache(
                                Path("/source/test.txt"),
                                "test-model",
                            )

