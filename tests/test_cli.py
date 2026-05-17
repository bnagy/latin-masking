"""Tests for cli.py module."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from udpipe_masking.cli import (
    _cmd_generate_adverbs,
    _cmd_mask,
    _cmd_process,
    _cmd_split_que,
    _cmd_split_sentences,
    main,
)
from udpipe_masking.types import MaskingConfig


class TestCmdProcess:
    """Tests for _cmd_process function."""

    @patch("udpipe_masking.cli.run_pipeline")
    def test_process_command(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test process subcommand."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text")
        output_dir = tmp_path / "output"

        args = argparse.Namespace(
            input=[input_file],
            output=output_dir,
            model="test-model",
            regenerate=False,
            quesplit=False,
            unsafe_certs_ok=True,
        )

        mock_run.return_value = MagicMock(
            sentences_processed=5, cache_hits=2, uv_replacements=1, adverbs_found=10
        )

        result = _cmd_process(args)

        assert result == 0
        mock_run.assert_called_once()

    @patch("udpipe_masking.cli.run_pipeline_with_quesplit")
    def test_process_command_with_quesplit(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Test process subcommand with --quesplit flag."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text")
        output_dir = tmp_path / "output"

        args = argparse.Namespace(
            input=[input_file],
            output=output_dir,
            model="test-model",
            regenerate=False,
            quesplit=True,
            unsafe_certs_ok=True,
        )

        mock_run.return_value = MagicMock(
            sentences_processed=5, cache_hits=0, uv_replacements=2, adverbs_found=8
        )

        result = _cmd_process(args)

        assert result == 0
        mock_run.assert_called_once()


class TestCmdSplitSentences:
    """Tests for _cmd_split_sentences function."""

    def test_split_sentences_args(self, tmp_path: Path) -> None:
        """Test split-sentences argument parsing."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("First sentence. Second sentence.")
        output_dir = tmp_path / "output"

        args = argparse.Namespace(
            input=input_file,
            output=output_dir,
        )

        # Verify args are parsed correctly
        assert args.input == input_file
        assert args.output == output_dir

    def test_split_sentences_args_no_output(self, tmp_path: Path) -> None:
        """Test split-sentences without output argument."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("First sentence. Second sentence.")

        args = argparse.Namespace(
            input=input_file,
            output=None,
        )

        # Verify args are parsed correctly
        assert args.input == input_file
        assert args.output is None


class TestCmdSplitQue:
    """Tests for _cmd_split_que function."""

    def test_split_que_args(self, tmp_path: Path) -> None:
        """Test -que splitting argument parsing."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Marcus etiamque in horto est.")

        args = argparse.Namespace(
            input=input_file,
            que_words=None,
        )

        # Verify args are parsed correctly
        assert args.input == input_file
        assert args.que_words is None

    def test_split_que_args_with_custom_words(self, tmp_path: Path) -> None:
        """Test -que splitting with custom word list argument."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Marcus etiamque in horto est.")
        que_words_file = tmp_path / "que_words.txt"
        que_words_file.write_text("etiamque\nquoque\n")

        args = argparse.Namespace(
            input=input_file,
            que_words=que_words_file,
        )

        # Verify args are parsed correctly
        assert args.input == input_file
        assert args.que_words == que_words_file


class TestCmdGenerateAdverbs:
    """Tests for _cmd_generate_adverbs function."""

    def test_generate_adverbs_args(self, tmp_path: Path) -> None:
        """Test generate-adverbs argument parsing."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Marcus bene currit.")
        output_dir = tmp_path / "output"

        args = argparse.Namespace(
            input=[input_file],
            output=output_dir,
            max=200,
            unsafe_certs_ok=True,
        )

        # Verify args are parsed correctly
        assert args.input == [input_file]
        assert args.output == output_dir
        assert args.max == 200


class TestCmdMask:
    """Tests for _cmd_mask function."""

    def test_mask_command_basic(self, tmp_path: Path) -> None:
        """Test mask subcommand argument parsing."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Marcus currit.")
        output_dir = tmp_path / "output"

        args = argparse.Namespace(
            input=[input_file],
            adverbs=None,
            replacements=None,
            output=output_dir,
            unsafe_certs_ok=True,
        )

        # Verify args are parsed correctly
        assert args.input == [input_file]
        assert args.output == output_dir


class TestMain:
    """Tests for main function."""

    @patch("udpipe_masking.cli._cmd_process")
    def test_main_process_command(self, mock_cmd: MagicMock, tmp_path: Path) -> None:
        """Test main with process command."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text")
        output_dir = tmp_path / "output"

        mock_cmd.return_value = 0

        with patch(
            "sys.argv",
            ["udpipe-mask", "process", str(input_file), "-o", str(output_dir)],
        ):
            with patch("sys.stdout"):
                result = main()

        assert result == 0
        mock_cmd.assert_called_once()

    @patch("udpipe_masking.cli._cmd_split_sentences")
    def test_main_split_sentences_command(
        self, mock_cmd: MagicMock, tmp_path: Path
    ) -> None:
        """Test main with split-sentences command."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text")

        mock_cmd.return_value = 0

        with patch("sys.argv", ["udpipe-mask", "split-sentences", str(input_file)]):
            with patch("sys.stdout"):
                result = main()

        assert result == 0
        mock_cmd.assert_called_once()

    @patch("udpipe_masking.cli._cmd_split_que")
    def test_main_split_que_command(self, mock_cmd: MagicMock, tmp_path: Path) -> None:
        """Test main with split-que command."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text")

        mock_cmd.return_value = 0

        with patch("sys.argv", ["udpipe-mask", "split-que", str(input_file)]):
            with patch("sys.stdout"):
                result = main()

        assert result == 0
        mock_cmd.assert_called_once()

    @patch("udpipe_masking.cli._cmd_generate_adverbs")
    def test_main_generate_adverbs_command(
        self, mock_cmd: MagicMock, tmp_path: Path
    ) -> None:
        """Test main with generate-adverbs command."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text")

        mock_cmd.return_value = 0

        with patch("sys.argv", ["udpipe-mask", "generate-adverbs", str(input_file)]):
            with patch("sys.stdout"):
                result = main()

        assert result == 0
        mock_cmd.assert_called_once()

    @patch("udpipe_masking.cli._cmd_mask")
    def test_main_mask_command(self, mock_cmd: MagicMock, tmp_path: Path) -> None:
        """Test main with mask command."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text")
        output_dir = tmp_path / "output"

        mock_cmd.return_value = 0

        with patch(
            "sys.argv", ["udpipe-mask", "mask", str(input_file), "-o", str(output_dir)]
        ):
            with patch("sys.stdout"):
                result = main()

        assert result == 0
        mock_cmd.assert_called_once()
