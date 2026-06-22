"""Tests for pipeline.py module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest  # pyright: ignore[reportMissingImports]

from latin_masking.pipeline import (
    _fix_eol_placement,
    run_pipeline_stage1,
    run_pipeline_stage2,
)
from latin_masking.types import MaskingConfig


class TestRunPipelineStage1:
    """Tests for run_pipeline_stage1 function."""

    @patch("latin_masking.pipeline.process_file_with_cache")
    @patch("latin_masking.pipeline.split_sentences")
    def test_stage1_basic(
        self,
        mock_split,
        mock_udpipe,
        tmp_path: Path,
    ) -> None:
        """Test basic stage 1 processing."""
        input_file = tmp_path / "test.txt"
        input_file.write_text("Marcus est bonus.\nPuella legit.")

        mock_split.return_value = ["Marcus est bonus.", "Puella legit."]
        mock_udpipe.return_value = (
            (
                "# text = Marcus est bonus.\n"
                "1\tMarcus\tMarcus\tPROPN\t_\t_\t0\troot\t_\t_\n"
                "2\test\tsum\tVERB\t_\t_\t0\troot\t_\t_\n"
                "3\tbonus\tbonus\tADJ\t_\t_\t0\troot\t_\t_\n"
            ),
            False,
        )

        config = MaskingConfig(
            output_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            common_adverbs_path=tmp_path / "common_adverbs.txt",
        )
        result = run_pipeline_stage1([input_file], config=config)

        assert input_file in result.sentences_per_file
        assert result.sentences_per_file[input_file] == 2
        assert result.common_adverbs_path.exists()

    @patch("latin_masking.pipeline.process_file_with_cache")
    def test_stage1_preserves_eol(
        self,
        mock_udpipe,
        tmp_path: Path,
    ) -> None:
        """Test that preserve_eol inserts <EOL> tokens at verse line breaks."""
        input_file = tmp_path / "poem.txt"
        input_file.write_text("Arma virumque cano.\nTroiae qui primus ab oris.")

        mock_udpipe.return_value = ("", False)

        config = MaskingConfig(output_dir=tmp_path, cache_dir=tmp_path / "cache")
        run_pipeline_stage1([input_file], config=config, preserve_eol=True)

        sent_path = tmp_path / "poem_sentences.txt"
        assert sent_path.exists()
        lines = sent_path.read_text(encoding="utf-8").strip().split("\n")
        # Both sentences should have <EOL> (each line ends a verse line)
        for line in lines:
            assert "<EOL>" in line, f"Missing <EOL> in: {line!r}"
            assert not line.startswith("<EOL>"), f"Sentence starts with <EOL>: {line!r}"

    def test_fix_eol_placement_parenthetical(self, tmp_path: Path) -> None:
        """Test that <EOL> is not moved to parenthetical extractions."""
        # When a parenthetical extraction sits between two sentences,
        # the <EOL> from the next sentence should go to the containing
        # sentence (the one before the paren), not the parenthetical.
        result = _fix_eol_placement([
            "Carmina quod pleno saltari nostra theatro, <EOL> nil equidem feci theatris, <EOL> Musa nec in plausus ambitiosa mea est.",
            "tu scis hoc ipse",
            "<EOL> Non tamen ingratum est, quodcumque obliuia nostri impedit.",
        ])
        # The <EOL> from "Non tamen..." should go to the containing sentence,
        # not the parenthetical "tu scis hoc ipse".
        assert result[0].endswith("<EOL>")
        assert "<EOL>" not in result[1]
        assert result[2].endswith("<EOL>")

    def test_fix_eol_placement_basic(self, tmp_path: Path) -> None:
        """Unit test for _fix_eol_placement helper."""
        # Leading <EOL> on second sentence → moved to end of first.
        result = _fix_eol_placement([
            "Arma virumque cano.",
            "<EOL> Troiae qui primus ab oris.",
        ])
        assert result == [
            "Arma virumque cano. <EOL>",
            "Troiae qui primus ab oris. <EOL>",
        ]

        # Multiple sentences with leading <EOL>.
        result = _fix_eol_placement([
            "First sentence.",
            "<EOL> Second sentence.",
            "<EOL> Third sentence.",
        ])
        assert result == [
            "First sentence. <EOL>",
            "Second sentence. <EOL>",
            "Third sentence. <EOL>",
        ]

        # No <EOL> tags — last sentence gets <EOL> appended.
        result = _fix_eol_placement(["One.", "Two."])
        assert result == ["One.", "Two. <EOL>"]

        # Single sentence — gets <EOL> appended.
        result = _fix_eol_placement(["Only sentence."])
        assert result == ["Only sentence. <EOL>"]

        # Empty list.
        assert _fix_eol_placement([]) == []

        # Sentence that is only <EOL> — removed.
        result = _fix_eol_placement(["First.", "<EOL>", "Third."])
        assert result == ["First. <EOL>", "Third. <EOL>"]

        # Last sentence should get <EOL> appended (end of poem marker).
        result = _fix_eol_placement([
            "Arma virumque cano. <EOL>",
            "Troiae qui primus ab oris.",
        ])
        assert result == [
            "Arma virumque cano. <EOL>",
            "Troiae qui primus ab oris. <EOL>",
        ]


class TestRunPipelineStage2:
    """Tests for run_pipeline_stage2 function."""

    @patch("latin_masking.pipeline.process_file_with_cache")
    def test_stage2_basic(
        self,
        mock_udpipe,
        tmp_path: Path,
    ) -> None:
        """Test basic stage 2 processing."""
        sent_path = tmp_path / "test_sentences.txt"
        sent_path.write_text("Marcus etiamque in horto.\n")

        adv_path = tmp_path / "common_adverbs.txt"
        adv_path.write_text("saepe\t10\nbene\t5\n")

        mock_udpipe.return_value = (
            (
                "# text = Marcus etiam -que in horto.\n"
                "1\tMarcus\tMarcus\tPROPN\t_\t_\t0\troot\t_\t_\n"
                "2\tetiam\tetiam\tADV\t_\t_\t0\troot\t_\t_\n"
                "3\t-que\t-que\tADV\t_\t_\t0\troot\t_\t_\n"
                "4\tin\tin\tADP\t_\t_\t0\troot\t_\t_\n"
                "5\thorto\thortus\tNOUN\t_\t_\t0\troot\t_\t_\n"
            ),
            False,
        )

        config = MaskingConfig(
            output_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            common_adverbs_path=adv_path,
        )
        result = run_pipeline_stage2(
            [tmp_path / "test.txt"],
            config=config,
            que_blacklist_path=None,
        )

        assert len(result.output_files) == 1
        assert result.sentences_processed == 1

        qs_path = tmp_path / "test_sentences.quesplit.txt"
        assert qs_path.exists()
        qs_content = qs_path.read_text()
        assert "etiam -que" in qs_content

        masked_path = tmp_path / "test_sentences.quesplit.masked.txt"
        assert masked_path.exists()

    @patch("latin_masking.pipeline.process_file_with_cache")
    def test_stage2_blacklist_preserves_words(
        self,
        mock_udpipe,
        tmp_path: Path,
    ) -> None:
        """Test that stage 2 respects the blacklist."""
        sent_path = tmp_path / "test_sentences.txt"
        sent_path.write_text("Marcus atque in horto.\n")

        adv_path = tmp_path / "common_adverbs.txt"
        adv_path.write_text("saepe\t10\n")

        mock_udpipe.return_value = ("", False)

        config = MaskingConfig(
            output_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            common_adverbs_path=adv_path,
        )

        bl_path = tmp_path / "blacklist.txt"
        bl_path.write_text("atque\n")

        run_pipeline_stage2(
            [tmp_path / "test.txt"],
            config=config,
            que_blacklist_path=bl_path,
        )

        qs_path = tmp_path / "test_sentences.quesplit.txt"
        qs_content = qs_path.read_text()
        assert "atque" in qs_content
        assert "-que" not in qs_content

    @patch("latin_masking.pipeline.process_file_with_cache")
    def test_stage2_common_adverbs_que(
        self,
        mock_udpipe,
        tmp_path: Path,
    ) -> None:
        """Test that common adverbs ending in -que are not split."""
        sent_path = tmp_path / "test_sentences.txt"
        sent_path.write_text("Marcus itaque in horto.\n")

        adv_path = tmp_path / "common_adverbs.txt"
        adv_path.write_text("itaque\t10\n")

        mock_udpipe.return_value = ("", False)

        config = MaskingConfig(
            output_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            common_adverbs_path=adv_path,
        )

        run_pipeline_stage2(
            [tmp_path / "test.txt"],
            config=config,
            que_blacklist_path=None,
        )

        qs_path = tmp_path / "test_sentences.quesplit.txt"
        qs_content = qs_path.read_text()
        assert "itaque" in qs_content
        assert "-que" not in qs_content


class TestEosToken:
    """Tests for eos_token functionality."""

    @patch("latin_masking.pipeline.process_file_with_cache")
    def test_stage2_default_eos_token(
        self,
        mock_udpipe,
        tmp_path: Path,
    ) -> None:
        """Test that default eos_token='<EOS>' appends <EOS> to masked output."""
        sent_path = tmp_path / "test_sentences.txt"
        sent_path.write_text("Marcus etiamque in horto.\n")

        adv_path = tmp_path / "common_adverbs.txt"
        adv_path.write_text("saepe\t10\nbene\t5\n")

        mock_udpipe.return_value = (
            (
                "# text = Marcus etiam -que in horto.\n"
                "1\tMarcus\tMarcus\tPROPN\t_\t_\t0\troot\t_\t_\n"
                "2\tetiam\tetiam\tADV\t_\t_\t0\troot\t_\t_\n"
                "3\t-que\t-que\tADV\t_\t_\t0\troot\t_\t_\n"
                "4\tin\tin\tADP\t_\t_\t0\troot\t_\t_\n"
                "5\thorto\thortus\tNOUN\t_\t_\t0\troot\t_\t_\n"
            ),
            False,
        )

        config = MaskingConfig(
            output_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            common_adverbs_path=adv_path,
        )
        run_pipeline_stage2(
            [tmp_path / "test.txt"],
            config=config,
            que_blacklist_path=None,
        )

        masked_path = tmp_path / "test_sentences.quesplit.masked.txt"
        assert masked_path.exists()
        lines = masked_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            assert line.endswith("<EOS>"), f"Line missing <EOS>: {line!r}"

    @patch("latin_masking.pipeline.process_file_with_cache")
    def test_stage2_eos_token_none(
        self,
        mock_udpipe,
        tmp_path: Path,
    ) -> None:
        """Test that eos_token=None does not append anything."""
        sent_path = tmp_path / "test_sentences.txt"
        sent_path.write_text("Marcus etiamque in horto.\n")

        adv_path = tmp_path / "common_adverbs.txt"
        adv_path.write_text("saepe\t10\nbene\t5\n")

        mock_udpipe.return_value = (
            (
                "# text = Marcus etiam -que in horto.\n"
                "1\tMarcus\tMarcus\tPROPN\t_\t_\t0\troot\t_\t_\n"
                "2\tetiam\tetiam\tADV\t_\t_\t0\troot\t_\t_\n"
                "3\t-que\t-que\tADV\t_\t_\t0\troot\t_\t_\n"
                "4\tin\tin\tADP\t_\t_\t0\troot\t_\t_\n"
                "5\thorto\thortus\tNOUN\t_\t_\t0\troot\t_\t_\n"
            ),
            False,
        )

        config = MaskingConfig(
            output_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            common_adverbs_path=adv_path,
            eos_token=None,
        )
        run_pipeline_stage2(
            [tmp_path / "test.txt"],
            config=config,
            que_blacklist_path=None,
            eos_token=None,
        )

        masked_path = tmp_path / "test_sentences.quesplit.masked.txt"
        assert masked_path.exists()
        lines = masked_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            assert not line.endswith("<EOS>"), f"Line has <EOS> when disabled: {line!r}"

    @patch("latin_masking.pipeline.process_file_with_cache")
    def test_stage2_custom_eos_token(
        self,
        mock_udpipe,
        tmp_path: Path,
    ) -> None:
        """Test that a custom eos_token string is appended."""
        sent_path = tmp_path / "test_sentences.txt"
        sent_path.write_text("Marcus etiamque in horto.\n")

        adv_path = tmp_path / "common_adverbs.txt"
        adv_path.write_text("saepe\t10\nbene\t5\n")

        mock_udpipe.return_value = (
            (
                "# text = Marcus etiam -que in horto.\n"
                "1\tMarcus\tMarcus\tPROPN\t_\t_\t0\troot\t_\t_\n"
                "2\tetiam\tetiam\tADV\t_\t_\t0\troot\t_\t_\n"
                "3\t-que\t-que\tADV\t_\t_\t0\troot\t_\t_\n"
                "4\tin\tin\tADP\t_\t_\t0\troot\t_\t_\n"
                "5\thorto\thortus\tNOUN\t_\t_\t0\troot\t_\t_\n"
            ),
            False,
        )

        config = MaskingConfig(
            output_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            common_adverbs_path=adv_path,
            eos_token="<SENT>",
        )
        run_pipeline_stage2(
            [tmp_path / "test.txt"],
            config=config,
            que_blacklist_path=None,
            eos_token="<SENT>",
        )

        masked_path = tmp_path / "test_sentences.quesplit.masked.txt"
        assert masked_path.exists()
        lines = masked_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            assert line.endswith("<SENT>"), f"Line missing <SENT>: {line!r}"

    def test_masking_config_default_eos_token(self) -> None:
        """Test that MaskingConfig default eos_token is '<EOS>'."""
        config = MaskingConfig()
        assert config.eos_token == "<EOS>"

    def test_masking_config_eos_token_none(self) -> None:
        """Test that MaskingConfig accepts eos_token=None."""
        config = MaskingConfig(eos_token=None)
        assert config.eos_token is None

    def test_masking_config_eos_token_custom(self) -> None:
        """Test that MaskingConfig accepts a custom eos_token."""
        config = MaskingConfig(eos_token="<SENT>")
        assert config.eos_token == "<SENT>"


class TestStage1Stage2Integration:
    """Integration tests for stage 1 + stage 2."""

    @patch("latin_masking.pipeline.process_file_with_cache")
    @patch("latin_masking.pipeline.split_sentences")
    def test_full_pipeline(
        self,
        mock_split,
        mock_udpipe,
        tmp_path: Path,
    ) -> None:
        """Test running stage 1 then stage 2."""
        input_file = tmp_path / "test.txt"
        input_file.write_text("Arma virumque cano.\nTroiae qui primus ab oris.")

        mock_split.return_value = [
            "Arma uirumque cano. <EOL> Troiae qui primus ab oris."
        ]

        udpipe_response_unsplit = (
            (
                "# text = Arma uirumque cano. <EOL> Troiae qui primus ab oris.\n"
                "1\tArma\tarma\tNOUN\t_\t_\t0\troot\t_\t_\n"
                "2\tuirumque\tuirumque\tNOUN\t_\t_\t0\troot\t_\t_\n"
                "3\tcano\tcano\tVERB\t_\t_\t0\troot\t_\t_\n"
            ),
            False,
        )

        udpipe_response_quesplit = (
            (
                "# text = Arma uirum -que cano. <EOL> Troiae qui primus ab oris.\n"
                "1\tArma\tarma\tNOUN\t_\t_\t0\troot\t_\t_\n"
                "2\tuirum\tuirum\tNOUN\t_\t_\t0\troot\t_\t_\n"
                "3\t-que\t-que\tADV\t_\t_\t0\troot\t_\t_\n"
                "4\tcano\tcano\tVERB\t_\t_\t0\troot\t_\t_\n"
            ),
            False,
        )

        mock_udpipe.side_effect = [udpipe_response_unsplit, udpipe_response_quesplit]

        config = MaskingConfig(output_dir=tmp_path, cache_dir=tmp_path / "cache")

        result1 = run_pipeline_stage1([input_file], config=config, preserve_eol=True)
        assert result1.sentences_per_file[input_file] == 1

        result2 = run_pipeline_stage2(
            [input_file],
            config=config,
            que_blacklist_path=None,
        )
        assert len(result2.output_files) == 1

        qs_path = tmp_path / "test_sentences.quesplit.txt"
        assert qs_path.exists()
        qs_content = qs_path.read_text()
        assert "uirum -que" in qs_content

        masked_path = tmp_path / "test_sentences.quesplit.masked.txt"
        assert masked_path.exists()


class TestEndToEndAeneid:
    """End-to-end test for Aeneid 1 using cached UDPipe responses.

    Cache files live in the fixtures directory alongside the raw text.
    No UDPipe API calls are made; all responses come from cache.
    """

    @pytest.fixture
    def aeneid_fixtures_dir(self) -> Path:
        """Return path to test fixtures directory."""
        return Path(__file__).parent / "fixtures"

    @pytest.fixture
    def aeneid_raw(self, aeneid_fixtures_dir: Path) -> Path:
        return aeneid_fixtures_dir / "aeneid_1_raw.txt"

    @pytest.fixture
    def aeneid_sentences_expected(self, aeneid_fixtures_dir: Path) -> Path:
        return aeneid_fixtures_dir / "aeneid_1_sentences.txt"

    @pytest.fixture
    def aeneid_quesplit_expected(self, aeneid_fixtures_dir: Path) -> Path:
        return aeneid_fixtures_dir / "aeneid_1_quesplit.txt"

    @pytest.fixture
    def aeneid_masked_expected(self, aeneid_fixtures_dir: Path) -> Path:
        return aeneid_fixtures_dir / "aeneid_1_expected.masked.txt"

    def test_end_to_end_aeneid_with_cached_responses(
        self,
        aeneid_raw: Path,
        aeneid_sentences_expected: Path,
        aeneid_quesplit_expected: Path,
        aeneid_masked_expected: Path,
        aeneid_fixtures_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test full two-stage pipeline on Aeneid 1 with cached UDPipe responses."""
        if not aeneid_raw.exists():
            pytest.skip("Aeneid 1 fixture not found")

        # Copy raw text to tmp; cache lives in fixtures dir
        input_file = tmp_path / "aeneid_1_raw.txt"
        input_file.write_text(aeneid_raw.read_text())

        # Pre-write sentences and quesplit files so stage2 can find them
        sentences_file = tmp_path / "aeneid_1_raw_sentences.txt"
        sentences_file.write_text(aeneid_sentences_expected.read_text())
        quesplit_file = tmp_path / "aeneid_1_raw_sentences.quesplit.txt"
        quesplit_file.write_text(aeneid_quesplit_expected.read_text())

        config = MaskingConfig(
            output_dir=tmp_path,
            cache_dir=aeneid_fixtures_dir,
            common_adverbs_path=aeneid_fixtures_dir / "common_adverbs_quesplit.txt",
            eos_token=None,
        )

        # Stage 1: normalize, sentence-split, UDPipe (from cache), collect adverbs
        result1 = run_pipeline_stage1(
            [input_file],
            config=config,
            preserve_eol=True,
        )

        expected_sentences = [
            line.strip()
            for line in aeneid_sentences_expected.read_text().splitlines()
            if line.strip()
        ]
        assert result1.sentences_per_file[input_file] == len(expected_sentences), (
            f"Sentence count mismatch: got {result1.sentences_per_file[input_file]}, "
            f"expected {len(expected_sentences)}"
        )
        assert len(result1.adverb_counts) > 0, "No adverbs collected"

        # Stage 2: -que split, UDPipe (from cache), mask
        result2 = run_pipeline_stage2(
            [input_file],
            config=config,
            que_blacklist_path=None,
            eos_token=None,
        )

        assert len(result2.output_files) == 1

        actual_masked = result2.output_files[0].read_text().strip().splitlines()
        expected_masked = [
            line.strip()
            for line in aeneid_masked_expected.read_text().splitlines()
            if line.strip()
        ]

        assert len(actual_masked) == len(expected_masked), (
            f"Masked output line count mismatch: got {len(actual_masked)}, "
            f"expected {len(expected_masked)}"
        )

        for i, (actual, expected) in enumerate(
            zip(actual_masked, expected_masked, strict=True)
        ):
            actual_normalized = " ".join(actual.split())
            expected_normalized = " ".join(expected.split())
            assert actual_normalized == expected_normalized, (
                f"Line {i + 1} mismatch:\n"
                f"  Got:      {actual_normalized[:100]}...\n"
                f"  Expected: {expected_normalized[:100]}..."
            )

    def test_aeneid_sentence_splitting_matches_expected(
        self,
        aeneid_raw: Path,
        aeneid_sentences_expected: Path,
        aeneid_fixtures_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test that sentence splitting produces the expected output for Aeneid 1."""
        if not aeneid_raw.exists():
            pytest.skip("Aeneid 1 fixture not found")

        input_file = tmp_path / "aeneid_1_raw.txt"
        input_file.write_text(aeneid_raw.read_text())

        config = MaskingConfig(
            output_dir=tmp_path,
            cache_dir=aeneid_fixtures_dir,
        )

        run_pipeline_stage1(
            [input_file],
            config=config,
            preserve_eol=True,
        )

        sent_path = tmp_path / "aeneid_1_raw_sentences.txt"
        actual_sentences = [
            line.strip() for line in sent_path.read_text().splitlines() if line.strip()
        ]
        expected_sentences = [
            line.strip()
            for line in aeneid_sentences_expected.read_text().splitlines()
            if line.strip()
        ]

        assert len(actual_sentences) == len(expected_sentences)

        for i, (actual, expected) in enumerate(
            zip(actual_sentences, expected_sentences, strict=True)
        ):
            assert actual == expected, (
                f"Sentence {i + 1} mismatch:\n"
                f"  Got:      {actual[:100]}...\n"
                f"  Expected: {expected[:100]}..."
            )

    def test_aeneid_quesplit_matches_expected(
        self,
        aeneid_raw: Path,
        aeneid_sentences_expected: Path,
        aeneid_quesplit_expected: Path,
        aeneid_fixtures_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test that -que splitting produces the expected output for Aeneid 1."""
        if not aeneid_raw.exists():
            pytest.skip("Aeneid 1 fixture not found")

        input_file = tmp_path / "aeneid_1_raw.txt"
        input_file.write_text(aeneid_raw.read_text())

        sentences_file = tmp_path / "aeneid_1_raw_sentences.txt"
        sentences_file.write_text(aeneid_sentences_expected.read_text())

        config = MaskingConfig(
            output_dir=tmp_path,
            cache_dir=aeneid_fixtures_dir,
            common_adverbs_path=aeneid_fixtures_dir / "common_adverbs_quesplit.txt",
        )

        run_pipeline_stage1(
            [input_file],
            config=config,
            preserve_eol=True,
        )

        run_pipeline_stage2(
            [input_file],
            config=config,
            que_blacklist_path=None,
        )

        qs_path = tmp_path / "aeneid_1_raw_sentences.quesplit.txt"
        actual_qs = [line.strip() for line in qs_path.read_text().splitlines() if line.strip()]
        expected_qs = [
            line.strip()
            for line in aeneid_quesplit_expected.read_text().splitlines()
            if line.strip()
        ]

        assert len(actual_qs) == len(expected_qs)

        for i, (actual, expected) in enumerate(
            zip(actual_qs, expected_qs, strict=True)
        ):
            assert actual == expected, (
                f"Quesplit line {i + 1} mismatch:\n"
                f"  Got:      {actual[:100]}...\n"
                f"  Expected: {expected[:100]}..."
            )

    def test_aeneid_sentence_count_matches_masked_count(
        self,
        aeneid_raw: Path,
        aeneid_sentences_expected: Path,
        aeneid_quesplit_expected: Path,
        aeneid_masked_expected: Path,
        aeneid_fixtures_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Verify that masked output has same sentence count as input (presegmented=True)."""
        if not aeneid_raw.exists():
            pytest.skip("Aeneid 1 fixture not found")

        input_file = tmp_path / "aeneid_1_raw.txt"
        input_file.write_text(aeneid_raw.read_text())

        sentences_file = tmp_path / "aeneid_1_raw_sentences.txt"
        sentences_file.write_text(aeneid_sentences_expected.read_text())
        quesplit_file = tmp_path / "aeneid_1_raw_sentences.quesplit.txt"
        quesplit_file.write_text(aeneid_quesplit_expected.read_text())

        config = MaskingConfig(
            output_dir=tmp_path,
            cache_dir=aeneid_fixtures_dir,
            common_adverbs_path=aeneid_fixtures_dir / "common_adverbs_quesplit.txt",
        )

        result1 = run_pipeline_stage1(
            [input_file],
            config=config,
            preserve_eol=True,
        )
        n_input_sentences = result1.sentences_per_file[input_file]

        result2 = run_pipeline_stage2(
            [input_file],
            config=config,
            que_blacklist_path=None,
        )
        n_masked_sentences = result2.sentences_processed

        assert n_input_sentences == n_masked_sentences, (
            f"Sentence count mismatch: {n_input_sentences} input sentences "
            f"produced {n_masked_sentences} masked sentences. "
            f"With presegmented=True, these should be equal."
        )

    def test_ysengrimus_sentence_count_matches_masked_count(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify sentence count consistency for Ysengrimus (non-verse, no EOL).

        Uses cached UDPipe responses from the fixtures directory to avoid
        live API calls.
        """
        fixtures_dir = Path(__file__).parent / "fixtures"
        raw = fixtures_dir / "ysengrimus_raw.txt"
        if not raw.exists():
            pytest.skip("Ysengrimus fixture not found")

        input_file = tmp_path / "ysengrimus_raw.txt"
        input_file.write_text(raw.read_text())

        config = MaskingConfig(
            output_dir=tmp_path,
            cache_dir=fixtures_dir,
        )

        result1 = run_pipeline_stage1(
            [input_file],
            config=config,
            preserve_eol=False,
        )
        n_input_sentences = result1.sentences_per_file[input_file]

        result2 = run_pipeline_stage2(
            [input_file],
            config=config,
            que_blacklist_path=None,
        )
        n_masked_sentences = result2.sentences_processed

        assert n_input_sentences == n_masked_sentences, (
            f"Sentence count mismatch: {n_input_sentences} input sentences "
            f"produced {n_masked_sentences} masked sentences. "
            f"With presegmented=True, these should be equal."
        )
