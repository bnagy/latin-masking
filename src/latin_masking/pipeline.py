"""Two-stage pipeline orchestration for Latin text processing.

Stage 1: Normalize → sentence-split → UDPipe → collect adverbs.
Stage 2: -que split → UDPipe → mask.

The caller reviews common_adverbs.txt between stages.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from latin_masking.adverbs import (
    collect_adverbs,
    generate_adverb_list,
    load_adverb_list,
    normalize_adverb_counts,
    save_adverb_list,
)
from latin_masking.cache import (
    get_cache_path,
    load_cached_response,
    save_cached_response,
)
from latin_masking.clitics import load_que_blacklist, split_que_blacklist
from latin_masking.client import process_file_with_cache
from latin_masking.conllu import parse_conllu
from latin_masking.mask import two_pass_mask
from latin_masking.preprocessor import preprocess
from latin_masking.sentences import split_sentences
from latin_masking.types import MaskingConfig, Stage1Result, Stage2Result

logger = logging.getLogger(__name__)


def run_pipeline_stage1(
    input_paths: list[Path],
    *,
    config: MaskingConfig,
    preserve_eol: bool = True,
) -> Stage1Result:
    """Stage 1: Normalize, sentence-split, UDPipe, collect adverbs.

    Per file:
    1. Read raw text
    2. If preserve_eol: join lines with <EOL> tokens
    3. Sentence-split (raw, no mangling yet)
    4. Preprocess each sentence (normalize, macrons, punct) — protected tokens preserved
    5. Write {stem}_sentences.txt to config.output_dir
    6. UDPipe (raw=True, presegmented=True) → populates cache
    7. Parse CoNLL-U, collect adverbs

    After all files:
    8. Aggregate → normalize counts → generate list → save common_adverbs.txt

    Args:
        input_paths: List of input file paths.
        config: Pipeline configuration (includes output_dir, cache_dir, model).
        preserve_eol: If True, join verse lines with <EOL> tokens.

    Returns:
        Stage1Result with adverb counts and sentence counts.
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_adverbs: Counter[str] = Counter()
    sentences_per_file: dict[Path, int] = {}

    for input_path in input_paths:
        logger.info("Stage 1: processing %s", input_path.name)

        # Read raw text
        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Optionally preserve verse linebreaks as <EOL> tokens
        if preserve_eol:
            raw_lines = [line.strip() for line in raw_text.split("\n")]
            raw_lines = [line for line in raw_lines if line]
            text = " <EOL> ".join(raw_lines)
        else:
            text = raw_text

        # Sentence-split (raw, no mangling yet)
        sentences = split_sentences(text)

        # Apply all text mangling (normalize, macrons, punct) in one place.
        # Protected tokens (<EOL>) are preserved throughout.
        mangled_sentences = [preprocess(sent) for sent in sentences]

        sent_path = output_dir / f"{input_path.stem}_sentences.txt"
        with open(sent_path, "w", encoding="utf-8") as f:
            for sent in mangled_sentences:
                f.write(sent + "\n")
        sentences_per_file[input_path] = len(mangled_sentences)

        # UDPipe (with caching) — text is already sentence-split, always presegmented
        response, cache_hit = process_file_with_cache(
            sent_path,
            model=config.model,
            cache_dir=config.cache_dir,
            presegmented=True,
            raw=True,
        )

        cache_label = "cached" if cache_hit else "fetched"
        print(f"  {input_path.name}: {len(mangled_sentences)} sentences ({cache_label})")

        # Parse and collect adverbs
        if response and isinstance(response, str):
            frames, _ = parse_conllu(response)
            advs = collect_adverbs(frames)
            all_adverbs.update(advs)
            print(f"    {len(advs)} adverbs collected")

    # Save adverbs
    normalized = normalize_adverb_counts(all_adverbs)
    adv_list = generate_adverb_list(normalized, max_adverbs=config.adverb_threshold)
    save_adverb_list(adv_list, config.common_adverbs_path)

    return Stage1Result(
        adverb_counts=all_adverbs,
        sentences_per_file=sentences_per_file,
        common_adverbs_path=config.common_adverbs_path,
    )


def run_pipeline_stage2(
    input_paths: list[Path],
    *,
    config: MaskingConfig,
    que_blacklist_path: Path | None = None,
) -> Stage2Result:
    """Stage 2: -que split, UDPipe, mask.

    Per file:
    1. Read {stem}_sentences.txt (written by stage 1)
    2. Load common adverbs; add -que adverbs to effective blacklist
    3. Apply split_que_blacklist() → write {stem}_sentences.quesplit.txt
    4. UDPipe (raw=True, presegmented=True) → populates cache
    5. Parse CoNLL-U → two_pass_mask() → write {stem}_sentences.quesplit.masked.txt

    Args:
        input_paths: List of input file paths (same as stage 1).
        config: Pipeline configuration (includes output_dir, cache_dir, model).
        que_blacklist_path: Path to -que blacklist file.

    Returns:
        Stage2Result with output file paths and statistics.
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load blacklist: use custom file if provided, otherwise None (triggers default)
    que_blacklist: set[str] | None = None
    if que_blacklist_path and que_blacklist_path.exists():
        que_blacklist = load_que_blacklist(que_blacklist_path)

    # Load common adverbs; -que adverbs are passed to split_que_blacklist
    # which adds them to the effective blacklist automatically
    common_adverbs = load_adverb_list(
        config.common_adverbs_path, config.adverb_threshold
    )
    common_adverbs_que = {adv for adv in common_adverbs if adv.endswith("que")}

    output_files: list[Path] = []
    total_sentences = 0
    total_cache_hits = 0

    for input_path in input_paths:
        logger.info("Stage 2: processing %s", input_path.name)

        # Read sentence-split file from stage 1
        sent_path = output_dir / f"{input_path.stem}_sentences.txt"
        with open(sent_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Apply -que splitting
        qs_text, qs_count = split_que_blacklist(
            text, que_blacklist, common_adverbs=common_adverbs_que
        )
        qs_path = output_dir / f"{input_path.stem}_sentences.quesplit.txt"
        with open(qs_path, "w", encoding="utf-8") as f:
            f.write(qs_text)
        # UDPipe (with caching) — text is already sentence-split, always presegmented
        response, cache_hit = process_file_with_cache(
            qs_path,
            model=config.model,
            cache_dir=config.cache_dir,
            presegmented=True,
            raw=True,
        )

        cache_label = "cached" if cache_hit else "fetched"
        print(f"  {input_path.name}: {qs_count} -que splits ({cache_label})")

        # Parse and mask
        if response and isinstance(response, str):
            frames, _ = parse_conllu(response)
            masked = two_pass_mask(frames, common_adverbs=common_adverbs)

            masked_path = (
                output_dir / f"{input_path.stem}_sentences.quesplit.masked.txt"
            )
            with open(masked_path, "w", encoding="utf-8") as f:
                f.write("\n".join(masked))
            output_files.append(masked_path)
            total_sentences += len(masked)
            print(f"    {len(masked)} masked sentences -> {masked_path.name}")

            if cache_hit:
                total_cache_hits += 1

    return Stage2Result(
        output_files=output_files,
        sentences_processed=total_sentences,
        cache_hits=total_cache_hits,
    )
