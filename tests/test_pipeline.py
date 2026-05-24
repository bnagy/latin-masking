"""Tests for pipeline.py module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from latin_masking.pipeline import (
    process_file,
    run_pipeline,
    run_pipeline_with_quesplit,
)
from latin_masking.types import MaskingConfig


class TestProcessFile:
    """Tests for process_file function."""

    @patch("latin_masking.pipeline.process_text")
    @patch("latin_masking.conllu.parse_conllu")
    def test_process_file_basic(
        self, mock_parse: MagicMock, mock_process: MagicMock, tmp_path: Path
    ) -> None:
        """Test basic file processing."""
        # Create input file
        input_file = tmp_path / "test.txt"
        input_file.write_text("Marcus est bonus.")

        # Mock UDPipe response
        mock_process.return_value = (
            "# text = test\n1\ttest\ttest\tNOUN\t_\t_\t0\troot\t_\t_\n"
        )
        mock_parse.return_value = ([], [])

        config = MaskingConfig()
        result = process_file(input_file, tmp_path, config=config)

        assert "sentences" in result


class TestRunPipeline:
    """Tests for run_pipeline function."""

    @patch("latin_masking.pipeline.process_file")
    def test_run_pipeline(self, mock_process: MagicMock, tmp_path: Path) -> None:
        """Test running the full pipeline."""
        input_file = tmp_path / "test.txt"
        input_file.write_text("Marcus est bonus.")

        mock_process.return_value = {
            "sentences": 1,
            "cache_hit": False,
            "output_file": str(tmp_path / "test_sentences.masked.txt"),
        }

        config = MaskingConfig()
        result = run_pipeline([input_file], tmp_path, config=config)

        assert result.sentences_processed == 1


class TestRunPipelineWithQuesplit:
    """Tests for run_pipeline_with_quesplit function."""

    @patch("latin_masking.pipeline.process_file")
    def test_run_pipeline_with_quesplit(
        self, mock_process: MagicMock, tmp_path: Path
    ) -> None:
        """Test running pipeline with -que splitting."""
        input_file = tmp_path / "test.txt"
        input_file.write_text("Marcus etiamque in horto.")

        mock_process.return_value = {
            "sentences": 1,
            "cache_hit": False,
            "output_file": str(tmp_path / "test_sentences.masked.txt"),
        }

        config = MaskingConfig()
        result = run_pipeline_with_quesplit(
            [input_file], tmp_path, config=config, que_blacklist_path=None
        )

        assert result.sentences_processed == 1


class TestEndToEndYsengrimus:
    """End-to-end test for Ysengrimus processing."""

    @pytest.fixture
    def ysengrimus_input(self) -> Path:
        """Return path to Ysengrimus raw text fixture."""
        return Path(__file__).parent / "fixtures" / "ysengrimus_raw.txt"

    @pytest.fixture
    def ysengrimus_expected(self) -> Path:
        """Return path to expected Ysengrimus masked output."""
        return Path(__file__).parent / "fixtures" / "ysengrimus_expected.masked.txt"

    @pytest.fixture
    def ysengrimus_sentences_expected(self) -> Path:
        """Return path to expected Ysengrimus sentences (sentence-split)."""
        return Path(__file__).parent / "fixtures" / "ysengrimus_sentences.txt"

    @pytest.fixture
    def ysengrimus_cache_dir(self, tmp_path: Path) -> Path:
        """Return cache directory for Ysengrimus test."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @pytest.fixture
    def ysengrimus_udpipe_response(self) -> Path:
        """Return path to cached UDPipe response for Ysengrimus (pre-que-split)."""
        return Path(__file__).parent / "fixtures" / "ysengrimus_udpipe_response.pkl"

    @pytest.fixture
    def ysengrimus_udpipe_response_quesplit(self) -> Path:
        """Return path to cached UDPipe response for Ysengrimus (post-que-split)."""
        return (
            Path(__file__).parent
            / "fixtures"
            / "ysengrimus_udpipe_response_quesplit.pkl"
        )

    def test_end_to_end_ysengrimus_with_cached_response(
        self,
        ysengrimus_input: Path,
        ysengrimus_expected: Path,
        ysengrimus_sentences_expected: Path,
        ysengrimus_udpipe_response_quesplit: Path,
        ysengrimus_cache_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test full pipeline on Ysengrimus text with cached UDPipe response.

        This test:
        1. Splits sentences from raw text and verifies against expected
        2. Uses cached UDPipe response (from -que-split text) instead of calling the API
        3. Processes through the full pipeline (masking, -que splitting)
        4. Verifies output matches expected masked output
        """
        if not ysengrimus_input.exists():
            pytest.skip("Ysengrimus fixture not found")

        if not ysengrimus_expected.exists():
            pytest.skip("Expected output file not found")

        if not ysengrimus_sentences_expected.exists():
            pytest.skip("Expected sentences file not found")

        if not ysengrimus_udpipe_response_quesplit.exists():
            pytest.skip("Cached UDPipe response (quesplit) not found")

        import pickle

        from latin_masking.sentences import split_sentences

        # Step 1: Verify sentence splitting
        raw_text = ysengrimus_input.read_text()
        actual_sentences = split_sentences(raw_text)
        expected_sentences = (
            ysengrimus_sentences_expected.read_text().strip().split("\n")
        )

        assert len(actual_sentences) == len(expected_sentences), (
            f"Sentence count mismatch: got {len(actual_sentences)}, "
            f"expected {len(expected_sentences)}"
        )

        # Step 2: Load cached UDPipe response (from -que-split text)
        with open(ysengrimus_udpipe_response_quesplit, "rb") as f:
            udpipe_response = pickle.load(f)

        # Step 3: Configure pipeline
        # Use the pre-defined common adverbs file from test fixtures
        # This matches how the expected output was generated
        common_adverbs_path = (
            Path(__file__).parent / "fixtures" / "common_adverbs_quesplit.txt"
        )
        config = MaskingConfig(
            cache_dir=ysengrimus_cache_dir,
            presegmented=True,  # Text is already segmented
            strip_punct=True,
            remove_macrons=True,
            common_adverbs_path=common_adverbs_path,
        )

        # Step 4: Write pre-segmented text to intermediate file
        intermediate_path = tmp_path / f"{ysengrimus_input.stem}_sentences.txt"
        with open(intermediate_path, "w", encoding="utf-8") as f:
            f.write("\n".join(actual_sentences))

        # Step 5: Mock the cache loading to return our cached response
        with (
            patch("latin_masking.pipeline.load_cached_response") as mock_load,
            patch("latin_masking.pipeline.get_cache_path") as mock_cache_path,
        ):
            mock_load.return_value = udpipe_response
            # Make cache_path.exists() return True
            mock_cache_path.return_value.exists.return_value = True

            # Step 6: Run the full pipeline with -que splitting
            # Use the default que blacklist file
            que_blacklist_path = (
                Path(__file__).parent.parent
                / "src"
                / "latin_masking"
                / "data"
                / "que_blacklist.txt"
            )
            _result = run_pipeline_with_quesplit(
                [intermediate_path],
                tmp_path,
                config=config,
                que_blacklist_path=que_blacklist_path,
            )

        # Step 7: Verify output
        # intermediate_path.stem = "ysengrimus_raw_sentences"
        # run_pipeline_with_quesplit creates {stem}.quesplit.txt
        # process_file creates {stem}.masked.txt
        output_file = tmp_path / f"{intermediate_path.stem}.quesplit.masked.txt"
        assert output_file.exists(), f"Output file not found: {output_file}"

        actual_lines = output_file.read_text().strip().split("\n")
        expected_lines = ysengrimus_expected.read_text().strip().split("\n")

        # Compare line counts
        assert len(actual_lines) == len(expected_lines), (
            f"Output line count mismatch: got {len(actual_lines)}, "
            f"expected {len(expected_lines)}"
        )

        # Compare each line (normalized for whitespace)
        for i, (actual, expected) in enumerate(
            zip(actual_lines, expected_lines, strict=True)
        ):
            actual_normalized = " ".join(actual.split())
            expected_normalized = " ".join(expected.split())
            assert actual_normalized == expected_normalized, (
                f"Line {i + 1} mismatch:\n"
                f"  Got: {actual_normalized[:80]}...\n"
                f"  Expected: {expected_normalized[:80]}..."
            )

    def test_sentence_splitting_matches_expected(
        self,
        ysengrimus_input: Path,
        ysengrimus_sentences_expected: Path,
    ) -> None:
        """Test that sentence splitting matches the expected output.

        This verifies that the sentence splitting in this package produces
        the same output as the reference ysengrimus_sentences.txt from
        Liber-Regum.
        """
        if not ysengrimus_input.exists():
            pytest.skip("Ysengrimus fixture not found")

        if not ysengrimus_sentences_expected.exists():
            pytest.skip("Expected sentences file not found")

        from latin_masking.sentences import split_sentences

        # Read raw text
        raw_text = ysengrimus_input.read_text()

        # Split into sentences
        actual_sentences = split_sentences(raw_text)

        # Load expected sentences
        expected_text = ysengrimus_sentences_expected.read_text()
        expected_sentences = expected_text.strip().split("\n")

        # Compare line counts
        assert len(actual_sentences) == len(expected_sentences), (
            f"Sentence count mismatch: got {len(actual_sentences)}, "
            f"expected {len(expected_sentences)}"
        )

        # Compare each sentence (normalized for whitespace)
        for i, (actual, expected) in enumerate(
            zip(actual_sentences, expected_sentences, strict=True)
        ):
            actual_normalized = " ".join(actual.split())
            expected_normalized = " ".join(expected.split())
            assert actual_normalized == expected_normalized, (
                f"Sentence {i + 1} mismatch:\n"
                f"  Got: {actual_normalized[:80]}...\n"
                f"  Expected: {expected_normalized[:80]}..."
            )

    def test_cache_file_is_used_when_content_matches(
        self,
        ysengrimus_sentences_expected: Path,
        ysengrimus_udpipe_response_quesplit: Path,
        ysengrimus_cache_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test that process_file uses a pre-existing cache file (no mocking).

        This is an integration test: it writes a real cache file to disk
        with the correct content-hash-derived filename, then calls
        process_file and verifies the cache was hit (process_text never
        called). This catches regressions where the cache key computation
        or cache lookup logic changes.
        """
        import pickle
        import shutil

        from latin_masking.cache import get_cache_path
        from latin_masking.clitics import load_que_blacklist, split_que_blacklist

        if not ysengrimus_sentences_expected.exists():
            pytest.skip("Expected sentences fixture not found")
        if not ysengrimus_udpipe_response_quesplit.exists():
            pytest.skip("Cached UDPipe response (quesplit) not found")

        # Step 1: Start from the sentence-split fixture and apply -que
        # splitting to reproduce the exact input that the cached response
        # was generated from.
        sentences_text = ysengrimus_sentences_expected.read_text()
        que_blacklist_path = (
            Path(__file__).parent.parent
            / "src"
            / "latin_masking"
            / "data"
            / "que_blacklist.txt"
        )
        que_blacklist = (
            load_que_blacklist(que_blacklist_path)
            if que_blacklist_path.exists()
            else set()
        )
        quesplit_text, _ = split_que_blacklist(sentences_text, que_blacklist)

        # Step 2: Write the quesplit text to a temp file (this is what
        # run_pipeline_with_quesplit would produce as intermediate output)
        quesplit_input = tmp_path / "ysengrimus_sentences.quesplit.txt"
        with open(quesplit_input, "w", encoding="utf-8") as f:
            f.write(quesplit_text)

        # Step 3: Compute the correct cache path from the content hash
        # and copy the saved response there.
        model = "latin-evalatin24-240520"
        cache_path = get_cache_path(quesplit_input, ysengrimus_cache_dir, model)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ysengrimus_udpipe_response_quesplit, cache_path)
        assert cache_path.exists(), "Cache file was not created"

        # Step 4: Load the cached response to verify it is valid
        with open(cache_path, "rb") as f:
            cached_response = pickle.load(f)
        assert isinstance(cached_response, str) and len(cached_response) > 0

        # Step 5: Configure the pipeline to use our pre-populated cache dir
        common_adverbs_path = (
            Path(__file__).parent / "fixtures" / "common_adverbs_quesplit.txt"
        )
        config = MaskingConfig(
            cache_dir=ysengrimus_cache_dir,
            presegmented=True,
            strip_punct=True,
            remove_macrons=True,
            common_adverbs_path=common_adverbs_path,
            model=model,
        )

        # Step 6: Run process_file with process_text mocked -- if the cache
        # is used, process_text should never be called.
        with patch("latin_masking.pipeline.process_text") as mock_process:
            result = process_file(quesplit_input, tmp_path, config=config)

        mock_process.assert_not_called()
        assert result.get("cache_hit") is True
        assert result.get("sentences", 0) > 0

        # Step 7: Verify the output file was actually written
        output_file = tmp_path / f"{quesplit_input.stem}.masked.txt"
        assert output_file.exists(), f"Output file not found: {output_file}"
        output_lines = output_file.read_text().strip().split("\n")
        assert len(output_lines) > 0
