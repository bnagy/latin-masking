"""Full pipeline orchestration for Latin text processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from udpipe_masking.adverbs import (
    collect_adverbs,
    generate_adverb_list,
    load_adverb_list,
    normalize_adverb_counts,
)
from udpipe_masking.cache import (
    get_cache_path,
    is_cache_valid,
    load_cached_response,
    save_cached_response,
)
from udpipe_masking.clitics import load_que_words, split_que
from udpipe_masking.client import process_text
from udpipe_masking.mask import two_pass_mask
from udpipe_masking.types import MaskingConfig, PipelineResult

logger = logging.getLogger(__name__)


def process_file(
    input_path: Path,
    output_dir: Path,
    *,
    config: MaskingConfig,
) -> dict[str, Any]:
    """Process a single file through all pipeline stages.

    Args:
        input_path: Path to input file.
        output_dir: Directory for output files.
        config: Pipeline configuration.

    Returns:
        Dictionary with processing statistics.

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Check cache
    cache_path = get_cache_path(input_path, config.cache_dir, config.model)
    cache_hit = False
    response: str | None = None

    if is_cache_valid(cache_path, input_path):
        response = load_cached_response(cache_path)
        cache_hit = True
    else:
        # Process through UDPipe
        result = process_text(
            text,
            model=config.model,
            presegmented=config.presegmented,
            strip_punct=config.strip_punct,
            remove_macrons_flag=config.remove_macrons,
            raw=True,
            unsafe_certs_ok=config.unsafe_certs_ok,
        )
        if result:
            response = result if isinstance(result, str) else ""
            save_cached_response(cache_path, response)

    if not response:
        return {"sentences": 0, "cache_hit": cache_hit}

    # Parse response
    from udpipe_masking.conllu import parse_conllu

    frames, texts = parse_conllu(response)

    # Generate adverbs if needed
    if config.common_adverbs_path and config.common_adverbs_path.exists():
        common_adverbs = load_adverb_list(
            config.common_adverbs_path, config.adverb_threshold
        )
    else:
        # Collect adverbs from this file
        adv_counter = collect_adverbs(frames)
        normalized_counter = normalize_adverb_counts(adv_counter)
        common_adverbs = {
            adv
            for adv, _ in generate_adverb_list(
                normalized_counter, config.adverb_threshold
            )
        }

    # Masking with universal UV/IJ normalization
    masked_sentences = two_pass_mask(
        frames,
        common_adverbs=common_adverbs,
    )

    # Write output
    output_path = output_dir / f"{input_path.stem}_sentences.masked.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(masked_sentences))

    return {
        "sentences": len(masked_sentences),
        "cache_hit": cache_hit,
        "output_file": str(output_path),
    }


def run_pipeline(
    input_paths: list[Path],
    output_dir: Path,
    *,
    config: MaskingConfig,
) -> PipelineResult:
    """End-to-end pipeline: sentence split → UDPipe → adverb generation → two-pass mask.

    Args:
        input_paths: List of input file paths.
        output_dir: Directory for output files.
        config: Pipeline configuration.

    Returns:
        PipelineResult with statistics.

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total_sentences = 0
    total_cache_hits = 0
    output_files: list[Path] = []

    for input_path in input_paths:
        result = process_file(input_path, output_dir, config=config)
        total_sentences += result.get("sentences", 0)
        if result.get("cache_hit"):
            total_cache_hits += 1
        if "output_file" in result:
            output_files.append(Path(result["output_file"]))

    return PipelineResult(
        output_files=output_files,
        sentences_processed=total_sentences,
        cache_hits=total_cache_hits,
    )


def run_pipeline_with_quesplit(
    input_paths: list[Path],
    output_dir: Path,
    *,
    config: MaskingConfig,
    que_words_path: Path | None = None,
) -> PipelineResult:
    """Same as run_pipeline but with -que splitting after sentence splitting.

    Args:
        input_paths: List of input file paths.
        output_dir: Directory for output files.
        config: Pipeline configuration.
        que_words_path: Path to -que words file.

    Returns:
        PipelineResult with statistics.

    """
    # Load -que words if provided
    que_words = []
    if que_words_path and que_words_path.exists():
        que_words = load_que_words(que_words_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_sentences = 0
    total_cache_hits = 0
    output_files: list[Path] = []

    for input_path in input_paths:
        # Read input
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Apply -que splitting
        if que_words:
            text, _ = split_que(text, que_words)

        # Write intermediate file
        intermediate_path = output_dir / f"{input_path.stem}_sentences.quesplit.txt"
        with open(intermediate_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Process through rest of pipeline
        result = process_file(intermediate_path, output_dir, config=config)
        total_sentences += result.get("sentences", 0)
        if result.get("cache_hit"):
            total_cache_hits += 1
        if "output_file" in result:
            output_files.append(Path(result["output_file"]))

    return PipelineResult(
        output_files=output_files,
        sentences_processed=total_sentences,
        cache_hits=total_cache_hits,
    )
