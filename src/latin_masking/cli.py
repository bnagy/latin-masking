"""Command-line interface for latin-masking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from latin_masking.types import MaskingConfig
from latin_masking.pipeline import run_pipeline_stage1, run_pipeline_stage2


def _cmd_process(args: argparse.Namespace) -> int:
    """Handle the process subcommand (two-stage pipeline)."""
    # Default cache dir
    if args.cache_dir:
        cache_dir = args.cache_dir
    elif args.regenerate:
        cache_dir = Path("/tmp/latin-masking-nocache")
    else:
        cache_dir = (
            args.input[0].parent / "udpipe_cache"
            if args.input
            else Path.cwd() / "udpipe_cache"
        )

    config = MaskingConfig(
        model=args.model,
        cache_dir=cache_dir,
    )

    # Stage 1: normalize, sentence-split, UDPipe, collect adverbs
    print("=== Stage 1: UDPipe + adverb collection ===")
    result1 = run_pipeline_stage1(
        args.input,
        config=config,
        preserve_eol=not args.no_preserve_eol,
    )
    print(f"Processed {len(result1.sentences_per_file)} files")
    print(f"Sentences: {sum(result1.sentences_per_file.values())}")
    print(f"Adverbs collected: {len(result1.adverb_counts)} unique")
    print(f"Saved to: {result1.common_adverbs_path}")

    if args.stage1_only:
        print("\nStage 1 complete. Review common_adverbs.txt before running stage 2.")
        return 0

    # Stage 2: quesplit, UDPipe, mask
    print("\n=== Stage 2: -que splitting + masking ===")

    que_blacklist_path = args.que_blacklist
    if que_blacklist_path is None:
        que_blacklist_path = Path(__file__).parent / "data" / "que_blacklist.txt"

    result2 = run_pipeline_stage2(
        args.input,
        config=config,
        que_blacklist_path=que_blacklist_path,
    )
    print(f"Processed {result2.sentences_processed} sentences")
    print(f"Cache hits: {result2.cache_hits}")
    print(f"Output files: {len(result2.output_files)}")
    return 0


def _cmd_split_sentences(args: argparse.Namespace) -> int:
    """Handle the split-sentences subcommand."""
    from latin_masking.sentences import split_sentences

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = split_sentences(text)

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        output_path = args.output / f"{args.input.stem}_sentences.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            for sent in sentences:
                f.write(sent + "\n")
        print(f"Wrote {len(sentences)} sentences to {output_path}")
    else:
        for sent in sentences:
            print(sent)
    return 0


def _cmd_generate_adverbs(args: argparse.Namespace) -> int:
    """Handle the generate-adverbs subcommand (stage 1 only)."""
    cache_dir = Path.home() / ".cache" / "latin-masking"
    config = MaskingConfig(
        model=args.model,
        cache_dir=cache_dir,
    )

    result = run_pipeline_stage1(
        args.input,
        config=config,
        preserve_eol=False,
    )

    print(f"Collected {len(result.adverb_counts)} unique adverbs")
    print(f"Saved to: {result.common_adverbs_path}")
    return 0


def _cmd_mask(args: argparse.Namespace) -> int:
    """Handle the mask subcommand (stage 2 only)."""
    cache_dir = Path.home() / ".cache" / "latin-masking"
    config = MaskingConfig(
        model=args.model,
        cache_dir=cache_dir,
    )

    que_blacklist_path = args.que_blacklist
    if que_blacklist_path is None:
        que_blacklist_path = Path(__file__).parent / "data" / "que_blacklist.txt"

    result = run_pipeline_stage2(
        args.input,
        config=config,
        que_blacklist_path=que_blacklist_path,
    )

    print(f"Processed {result.sentences_processed} sentences")
    print(f"Output files: {len(result.output_files)}")
    return 0


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="latin-mask",
        description="Latin text processing pipeline with UDPipe POS tagging and masking",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command (two-stage pipeline)
    process_parser = subparsers.add_parser(
        "process", help="Process input files through the two-stage pipeline"
    )
    process_parser.add_argument("input", nargs="+", type=Path, help="Input files")
    process_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output directory"
    )
    process_parser.add_argument(
        "--model", "-m", default="latin-evalatin24-240520", help="UDPipe model"
    )
    process_parser.add_argument(
        "--cache-dir", type=Path, help="Directory for caching UDPipe responses"
    )
    process_parser.add_argument(
        "--que-blacklist", type=Path, help="Path to -que blacklist file"
    )
    process_parser.add_argument(
        "--regenerate", action="store_true", help="Force regeneration (ignore cache)"
    )
    process_parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Run only stage 1 (adverb collection)",
    )
    process_parser.add_argument(
        "--no-preserve-eol",
        action="store_true",
        help="Do not insert <EOL> tokens between verse lines",
    )

    # Split sentences command
    split_parser = subparsers.add_parser(
        "split-sentences", help="Split text into sentences"
    )
    split_parser.add_argument("input", type=Path, help="Input file")
    split_parser.add_argument("--output", "-o", type=Path, help="Output directory")

    # Generate adverbs command (stage 1)
    adv_parser = subparsers.add_parser(
        "generate-adverbs", help="Generate adverb list from input (stage 1)"
    )
    adv_parser.add_argument("input", nargs="+", type=Path, help="Input files")
    adv_parser.add_argument("--output", "-o", type=Path, help="Output directory")
    adv_parser.add_argument(
        "--model", "-m", default="latin-evalatin24-240520", help="UDPipe model"
    )

    # Mask command (stage 2)
    mask_parser = subparsers.add_parser(
        "mask", help="Apply -que splitting and masking (stage 2)"
    )
    mask_parser.add_argument("input", nargs="+", type=Path, help="Input files")
    mask_parser.add_argument("--output", "-o", type=Path, help="Output directory")
    mask_parser.add_argument(
        "--que-blacklist", type=Path, help="Path to -que blacklist file"
    )
    mask_parser.add_argument(
        "--model", "-m", default="latin-evalatin24-240520", help="UDPipe model"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "process": _cmd_process,
        "split-sentences": _cmd_split_sentences,
        "generate-adverbs": _cmd_generate_adverbs,
        "mask": _cmd_mask,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
