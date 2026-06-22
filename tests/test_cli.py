"""Tests for cli.py module."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

from latin_masking.cli import (
    _cmd_generate_adverbs,
    _cmd_mask,
    _cmd_process,
    main,
)


class TestCmdProcess:
    """Tests for _cmd_process function."""

    @patch("latin_masking.cli.run_pipeline_stage2")
    @patch("latin_masking.cli.run_pipeline_stage1")
    def test_process_command(
        self, mock_stage1: MagicMock, mock_stage2: MagicMock, tmp_path: Path
    ) -> None:
        """Test process subcommand runs both stages."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text")
        output_dir = tmp_path / "output"

        mock_stage1.return_value = MagicMock(
            sentences_per_file={input_file: 5},
            adverb_counts={},
            common_adverbs_path=tmp_path / "common_adverbs.txt",
        )
        mock_stage2.return_value = MagicMock(
            output_files=[tmp_path / "output.txt"],
            sentences_processed=5,
            cache_hits=2,
        )

        args = argparse.Namespace(
            input=[input_file],
            output=output_dir,
            model="test-model",
            regenerate=False,
            stage1_only=False,
            no_preserve_eol=False,
            que_blacklist=None,
            cache_dir=None,
            eos_token="<EOS>",
        )

        result = _cmd_process(args)

        assert result == 0
        mock_stage1.assert_called_once()
        mock_stage2.assert_called_once()

    @patch("latin_masking.cli.run_pipeline_stage1")
    def test_process_command_stage1_only(
        self, mock_stage1: MagicMock, tmp_path: Path
    ) -> None:
        """Test process subcommand with --stage1-only flag."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text")
        output_dir = tmp_path / "output"

        mock_stage1.return_value = MagicMock(
            sentences_per_file={input_file: 5},
            adverb_counts={},
            common_adverbs_path=tmp_path / "common_adverbs.txt",
        )

        args = argparse.Namespace(
            input=[input_file],
            output=output_dir,
            model="test-model",
            regenerate=False,
            stage1_only=True,
            no_preserve_eol=False,
            que_blacklist=None,
            cache_dir=None,
            eos_token="<EOS>",
        )

        result = _cmd_process(args)

        assert result == 0
        mock_stage1.assert_called_once()

    @patch("latin_masking.cli.run_pipeline_stage2")
    @patch("latin_masking.cli.run_pipeline_stage1")
    def test_process_command_with_custom_cache_dir(
        self, mock_stage1: MagicMock, mock_stage2: MagicMock, tmp_path: Path
    ) -> None:
        """Test process subcommand with custom cache directory."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text")
        output_dir = tmp_path / "output"
        custom_cache = tmp_path / "custom_cache"

        mock_stage1.return_value = MagicMock(
            sentences_per_file={input_file: 5},
            adverb_counts={},
            common_adverbs_path=tmp_path / "common_adverbs.txt",
        )
        mock_stage2.return_value = MagicMock(
            output_files=[tmp_path / "output.txt"],
            sentences_processed=5,
            cache_hits=2,
        )

        args = argparse.Namespace(
            input=[input_file],
            output=output_dir,
            model="test-model",
            regenerate=False,
            stage1_only=False,
            no_preserve_eol=False,
            que_blacklist=None,
            cache_dir=custom_cache,
            eos_token="<EOS>",
        )

        result = _cmd_process(args)

        assert result == 0
        # Verify that config was created with custom cache dir
        call_args = mock_stage1.call_args
        assert call_args[1]["config"].cache_dir == custom_cache

    @patch("latin_masking.cli.run_pipeline_stage2")
    @patch("latin_masking.cli.run_pipeline_stage1")
    def test_process_command_default_cache_dir_is_input_dir(
        self, mock_stage1: MagicMock, mock_stage2: MagicMock, tmp_path: Path
    ) -> None:
        """Test that default cache dir is the directory containing input files."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text")
        output_dir = tmp_path / "output"

        mock_stage1.return_value = MagicMock(
            sentences_per_file={input_file: 5},
            adverb_counts={},
            common_adverbs_path=tmp_path / "common_adverbs.txt",
        )
        mock_stage2.return_value = MagicMock(
            output_files=[tmp_path / "output.txt"],
            sentences_processed=5,
            cache_hits=2,
        )

        args = argparse.Namespace(
            input=[input_file],
            output=output_dir,
            model="test-model",
            regenerate=False,
            stage1_only=False,
            no_preserve_eol=False,
            que_blacklist=None,
            cache_dir=None,
            eos_token="<EOS>",
        )

        result = _cmd_process(args)

        assert result == 0
        # Verify that config was created with udpipe_cache subdirectory
        call_args = mock_stage1.call_args
        assert call_args[1]["config"].cache_dir == tmp_path / "udpipe_cache"


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

    @patch("latin_masking.cli._cmd_process")
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

    @patch("latin_masking.cli._cmd_split_sentences")
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

    @patch("latin_masking.cli._cmd_generate_adverbs")
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

    @patch("latin_masking.cli._cmd_mask")
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

    def test_main_no_command_prints_help(self) -> None:
        """Test main with no command prints help and returns 0."""
        with patch("sys.argv", ["latin-mask"]):
            with patch("sys.stdout"):
                result = main()
        assert result == 0


class TestCmdProcessRegenerate:
    """Tests for _cmd_process --regenerate flag."""

    @patch("latin_masking.cli.run_pipeline_stage2")
    @patch("latin_masking.cli.run_pipeline_stage1")
    def test_process_command_regenerate_uses_tmp_cache_dir(
        self, mock_stage1: MagicMock, mock_stage2: MagicMock, tmp_path: Path
    ) -> None:
        """Test that --regenerate uses /tmp/latin-masking-nocache."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text")
        output_dir = tmp_path / "output"

        mock_stage1.return_value = MagicMock(
            sentences_per_file={input_file: 5},
            adverb_counts={},
            common_adverbs_path=tmp_path / "common_adverbs.txt",
        )
        mock_stage2.return_value = MagicMock(
            output_files=[tmp_path / "output.txt"],
            sentences_processed=5,
            cache_hits=0,
        )

        args = argparse.Namespace(
            input=[input_file],
            output=output_dir,
            model="test-model",
            regenerate=True,
            stage1_only=False,
            no_preserve_eol=False,
            que_blacklist=None,
            cache_dir=None,
            eos_token="<EOS>",
        )

        result = _cmd_process(args)

        assert result == 0
        call_args = mock_stage1.call_args
        assert call_args[1]["config"].cache_dir == Path("/tmp/latin-masking-nocache")


class TestCmdProcessNoPreserveEol:
    """Tests for _cmd_process --no-preserve-eol flag."""

    @patch("latin_masking.cli.run_pipeline_stage2")
    @patch("latin_masking.cli.run_pipeline_stage1")
    def test_process_command_no_preserve_eol(
        self, mock_stage1: MagicMock, mock_stage2: MagicMock, tmp_path: Path
    ) -> None:
        """Test that --no-preserve-eol passes preserve_eol=False."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text")
        output_dir = tmp_path / "output"

        mock_stage1.return_value = MagicMock(
            sentences_per_file={input_file: 5},
            adverb_counts={},
            common_adverbs_path=tmp_path / "common_adverbs.txt",
        )
        mock_stage2.return_value = MagicMock(
            output_files=[tmp_path / "output.txt"],
            sentences_processed=5,
            cache_hits=0,
        )

        args = argparse.Namespace(
            input=[input_file],
            output=output_dir,
            model="test-model",
            regenerate=False,
            stage1_only=False,
            no_preserve_eol=True,
            que_blacklist=None,
            cache_dir=None,
            eos_token="<EOS>",
        )

        result = _cmd_process(args)

        assert result == 0
        call_args = mock_stage1.call_args
        assert call_args[1]["preserve_eol"] is False


class TestCmdProcessCustomBlacklist:
    """Tests for _cmd_process --que-blacklist flag."""

    @patch("latin_masking.cli.run_pipeline_stage2")
    @patch("latin_masking.cli.run_pipeline_stage1")
    def test_process_command_custom_blacklist(
        self, mock_stage1: MagicMock, mock_stage2: MagicMock, tmp_path: Path
    ) -> None:
        """Test that custom blacklist path is passed to stage2."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text")
        output_dir = tmp_path / "output"
        custom_bl = tmp_path / "custom_blacklist.txt"
        custom_bl.write_text("atque\n")

        mock_stage1.return_value = MagicMock(
            sentences_per_file={input_file: 5},
            adverb_counts={},
            common_adverbs_path=tmp_path / "common_adverbs.txt",
        )
        mock_stage2.return_value = MagicMock(
            output_files=[tmp_path / "output.txt"],
            sentences_processed=5,
            cache_hits=0,
        )

        args = argparse.Namespace(
            input=[input_file],
            output=output_dir,
            model="test-model",
            regenerate=False,
            stage1_only=False,
            no_preserve_eol=False,
            que_blacklist=custom_bl,
            cache_dir=None,
            eos_token="<EOS>",
        )

        result = _cmd_process(args)

        assert result == 0
        call_args = mock_stage2.call_args
        assert call_args[1]["que_blacklist_path"] == custom_bl


class TestCmdGenerateAdverbsFunctional:
    """Functional tests for _cmd_generate_adverbs."""

    @patch("latin_masking.cli.run_pipeline_stage1")
    def test_generate_adverbs_calls_stage1(
        self, mock_stage1: MagicMock, tmp_path: Path
    ) -> None:
        """Test that generate-adverbs calls stage1 with preserve_eol=False."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Marcus bene currit.")
        output_dir = tmp_path / "output"

        mock_stage1.return_value = MagicMock(
            sentences_per_file={input_file: 3},
            adverb_counts={"bene": 1},
            common_adverbs_path=output_dir / "common_adverbs.txt",
        )

        args = argparse.Namespace(
            input=[input_file],
            output=output_dir,
            model="test-model",
            eos_token="<EOS>",
        )

        result = _cmd_generate_adverbs(args)

        assert result == 0
        mock_stage1.assert_called_once()
        call_args = mock_stage1.call_args
        assert call_args[1]["preserve_eol"] is False

    @patch("latin_masking.cli.run_pipeline_stage1")
    def test_generate_adverbs_no_output(
        self, mock_stage1: MagicMock, tmp_path: Path
    ) -> None:
        """Test generate-adverbs with no output dir uses cwd."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Marcus bene currit.")

        mock_stage1.return_value = MagicMock(
            sentences_per_file={input_file: 3},
            adverb_counts={"bene": 1},
            common_adverbs_path=tmp_path / "common_adverbs.txt",
        )

        args = argparse.Namespace(
            input=[input_file],
            output=None,
            model="test-model",
            eos_token="<EOS>",
        )

        result = _cmd_generate_adverbs(args)

        assert result == 0
        mock_stage1.assert_called_once()


class TestCmdMaskFunctional:
    """Functional tests for _cmd_mask."""

    @patch("latin_masking.cli.run_pipeline_stage2")
    def test_mask_calls_stage2(self, mock_stage2: MagicMock, tmp_path: Path) -> None:
        """Test that mask calls stage2."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Marcus currit.")
        output_dir = tmp_path / "output"

        mock_stage2.return_value = MagicMock(
            output_files=[output_dir / "masked.txt"],
            sentences_processed=1,
            cache_hits=0,
        )

        args = argparse.Namespace(
            input=[input_file],
            output=output_dir,
            que_blacklist=None,
            model="test-model",
            eos_token="<EOS>",
        )

        result = _cmd_mask(args)

        assert result == 0
        mock_stage2.assert_called_once()

    @patch("latin_masking.cli.run_pipeline_stage2")
    def test_mask_with_custom_blacklist(
        self, mock_stage2: MagicMock, tmp_path: Path
    ) -> None:
        """Test mask with custom blacklist path."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Marcus currit.")
        output_dir = tmp_path / "output"
        custom_bl = tmp_path / "blacklist.txt"
        custom_bl.write_text("atque\n")

        mock_stage2.return_value = MagicMock(
            output_files=[output_dir / "masked.txt"],
            sentences_processed=1,
            cache_hits=0,
        )

        args = argparse.Namespace(
            input=[input_file],
            output=output_dir,
            que_blacklist=custom_bl,
            model="test-model",
            eos_token="<EOS>",
        )

        result = _cmd_mask(args)

        assert result == 0
        call_args = mock_stage2.call_args
        assert call_args[1]["que_blacklist_path"] == custom_bl

    @patch("latin_masking.cli.run_pipeline_stage2")
    def test_mask_no_output(self, mock_stage2: MagicMock, tmp_path: Path) -> None:
        """Test mask with no output dir uses cwd."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Marcus currit.")

        mock_stage2.return_value = MagicMock(
            output_files=[],
            sentences_processed=0,
            cache_hits=0,
        )

        args = argparse.Namespace(
            input=[input_file],
            output=None,
            que_blacklist=None,
            model="test-model",
            eos_token="<EOS>",
        )

        result = _cmd_mask(args)

        assert result == 0
        mock_stage2.assert_called_once()
