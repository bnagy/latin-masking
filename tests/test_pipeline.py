"""Tests for pipeline.py module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest  # pyright: ignore[reportMissingImports]

from latin_masking.pipeline import (
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


class TestEndToEndRaw:
    """End-to-end pipeline tests on raw text, mirroring the --stage1/--stage2 CLI.

    Both Ysengrimus (prose, preserve_eol=False) and Aeneid 1 (verse,
    preserve_eol=True) are run through the real two-stage pipeline.  The
    intermediate files (``*_sentences.txt``, ``*_sentences.quesplit.txt``,
    ``*_sentences.quesplit.masked.txt``) and the regenerated UDPipe cache
    responses are written into the fixtures directory so they can be reviewed
    manually; they are NOT asserted against (exact output is locked in once
    the intermediates have been reviewed).

    The first run makes live UDPipe API calls (network on) and caches the
    responses in the fixtures directory; subsequent runs serve from cache.
    """

    @pytest.fixture
    def fixtures_dir(self) -> Path:
        return Path(__file__).parent / "fixtures"

    def _run(self, stem: str, *, preserve_eol: bool, fixtures_dir: Path) -> None:
        raw = fixtures_dir / f"{stem}.txt"
        if not raw.exists():
            pytest.skip(f"{stem} fixture not found")

        # Copy raw text into the fixtures dir under its real stem so the
        # generated intermediates/cache land alongside the other fixtures.
        input_file = fixtures_dir / f"{stem}.txt"
        assert input_file.exists()

        # Both stages use the curated common_adverbs_quesplit.txt gold list
        # (the old adverb list).  Stage 1 writes its regenerated list to a
        # temp path so the curated fixture is not clobbered; stage 2 reads
        # the curated list for -que splitting.  The regenerated list is not
        # asserted against the gold — the curated list is treated as fixed.
        adverb_tmp = fixtures_dir / f"{stem}_generated_adverbs.txt"

        config = MaskingConfig(
            output_dir=fixtures_dir,
            cache_dir=fixtures_dir,
            common_adverbs_path=adverb_tmp,
            eos_token=None,
        )

        result1 = run_pipeline_stage1(
            [input_file],
            config=config,
            preserve_eol=preserve_eol,
        )
        n_sentences = result1.sentences_per_file[input_file]
        assert n_sentences > 0, "Stage 1 produced no sentences"

        # No sentence should start with <EOL> (placement is always at the end).
        sent_path = fixtures_dir / f"{stem}_sentences.txt"
        sent_lines = [
            ln.strip() for ln in sent_path.read_text().splitlines() if ln.strip()
        ]
        for ln in sent_lines:
            assert not ln.startswith("<EOL>"), f"Sentence starts with <EOL>: {ln!r}"

        if preserve_eol:
            # Every verse line should end a sentence with <EOL>.  Count the
            # non-empty raw lines and the <EOL> tokens in the output.
            n_raw_lines = sum(1 for ln in raw.read_text().splitlines() if ln.strip())
            n_eol = sum(ln.count("<EOL>") for ln in sent_lines)
            assert n_eol == n_raw_lines, (
                f"<EOL> count ({n_eol}) != non-empty raw line count "
                f"({n_raw_lines}) for {stem}"
            )
        else:
            # Prose: no <EOL> tokens should appear at all.
            assert all("<EOL>" not in ln for ln in sent_lines), (
                f"Unexpected <EOL> in prose output for {stem}"
            )

        # Stage 2: -que split, UDPipe, mask.  Reads the curated adverb list.
        config.common_adverbs_path = fixtures_dir / "common_adverbs_quesplit.txt"
        result2 = run_pipeline_stage2(
            [input_file],
            config=config,
            que_blacklist_path=None,
            eos_token=None,
        )
        assert len(result2.output_files) == 1

        masked_path = result2.output_files[0]
        masked_lines = [
            ln.strip() for ln in masked_path.read_text().splitlines() if ln.strip()
        ]
        # With presegmented=True, masked line count == stage1 sentence count.
        assert len(masked_lines) == n_sentences, (
            f"Masked line count ({len(masked_lines)}) != stage1 sentence "
            f"count ({n_sentences}) for {stem}"
        )

    def test_ysengrimus_end_to_end(self, fixtures_dir: Path) -> None:
        """Ysengrimus: prose, line breaks meaningless, no <EOL> tokens."""
        self._run("ysengrimus_raw", preserve_eol=False, fixtures_dir=fixtures_dir)

    def test_aeneid_1_end_to_end(self, fixtures_dir: Path) -> None:
        """Aeneid 1: verse, <EOL> tokens at each verse-line end."""
        self._run("aeneid_1_raw", preserve_eol=True, fixtures_dir=fixtures_dir)
