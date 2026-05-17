"""Command-line interface for udpipe-masking."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from udpipe_masking.types import MaskingConfig
from udpipe_masking.pipeline import run_pipeline, run_pipeline_with_quesplit


def _cmd_process(args: argparse.Namespace) -> int:
    """Handle the process subcommand."""
    config = MaskingConfig(
        model=args.model,
        cache_dir=Path.home() / ".cache" / "udpipe-masking",
    )
    if args.regenerate:
        config.cache_dir = Path("/tmp/udpipe-masking-nocache")

    if args.quesplit:
        result = run_pipeline_with_quesplit(args.input, args.output, config=config)
    else:
        result = run_pipeline(args.input, args.output, config=config)

    print(f"Processed {result.sentences_processed} sentences")
    print(f"Cache hits: {result.cache_hits}")
    return 0


def _cmd_split_sentences(args: argparse.Namespace) -> int:
    """Handle the split-sentences subcommand."""
    from udpipe_masking.sentences import split_sentences

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


def _cmd_split_que(args: argparse.Namespace) -> int:
    """Handle the split-que subcommand."""
    from udpipe_masking.clitics import load_que_words, split_que

    que_words = []
    if args.que_words:
        que_words = load_que_words(args.que_words)

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    new_text, count = split_que(text, que_words)
    print(f"Made {count} replacements")
    print(new_text)
    return 0


def _cmd_generate_adverbs(args: argparse.Namespace) -> int:
    """Handle the generate-adverbs subcommand."""
    from udpipe_masking.client import process_text
    from udpipe_masking.adverbs import (
        collect_adverbs,
        normalize_adverb_counts,
        generate_adverb_list,
        save_adverb_list,
    )
    from udpipe_masking.conllu import parse_conllu

    all_adverbs: Counter[str] = Counter()
    for input_path in args.input:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        response = process_text(
            text, presegmented=True, raw=True, unsafe_certs_ok=args.unsafe_certs_ok
        )
        if response:
            frames, _ = parse_conllu(str(response))
            advs = collect_adverbs(frames)
            all_adverbs.update(advs)

    normalized = normalize_adverb_counts(all_adverbs)
    top_advs = generate_adverb_list(normalized, args.max)

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        output_path = args.output / "common_adverbs.txt"
        save_adverb_list(top_advs, output_path)
        print(f"Saved {len(top_advs)} adverbs to {output_path}")
    else:
        for adv, count in top_advs:
            print(f"{adv}\t{count}")
    return 0


def _cmd_mask(args: argparse.Namespace) -> int:
    """Handle the mask subcommand."""
    from udpipe_masking.client import process_text
    from udpipe_masking.conllu import parse_conllu
    from udpipe_masking.mask import two_pass_mask
    from udpipe_masking.adverbs import load_adverb_list

    common_adverbs = set()
    if args.adverbs:
        common_adverbs = load_adverb_list(args.adverbs, 200)

    for input_path in args.input:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        response = process_text(
            text, presegmented=True, raw=True, unsafe_certs_ok=args.unsafe_certs_ok
        )
        if response:
            frames, _ = parse_conllu(str(response))

            masked = two_pass_mask(
                frames,
                common_adverbs=common_adverbs,
            )

            if args.output:
                args.output.mkdir(parents=True, exist_ok=True)
                output_path = args.output / f"{input_path.stem}_masked.txt"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(masked))
                print(f"Wrote {len(masked)} sentences to {output_path}")
            else:
                for sent in masked:
                    print(sent)
    return 0


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="udpipe-mask",
        description="Latin text processing pipeline with UDPipe POS tagging and masking",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser(
        "process", help="Process input files through the full pipeline"
    )
    process_parser.add_argument("input", nargs="+", type=Path, help="Input files")
    process_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output directory"
    )
    process_parser.add_argument(
        "--model", "-m", default="latin-ittb-ud-2.5-191005", help="UDPipe model"
    )
    process_parser.add_argument(
        "--quesplit", action="store_true", help="Enable -que splitting"
    )
    process_parser.add_argument(
        "--regenerate", action="store_true", help="Force regeneration (ignore cache)"
    )
    process_parser.add_argument(
        "--unsafe-certs-ok",
        action="store_true",
        default=True,
        help="Accept self-signed SSL certificates (default: True for UDPipe)",
    )

    # Split sentences command
    split_parser = subparsers.add_parser(
        "split-sentences", help="Split text into sentences"
    )
    split_parser.add_argument("input", type=Path, help="Input file")
    split_parser.add_argument("--output", "-o", type=Path, help="Output directory")

    # Split que command
    que_parser = subparsers.add_parser("split-que", help="Split -que enclitics")
    que_parser.add_argument("input", type=Path, help="Input file")
    que_parser.add_argument("--que-words", type=Path, help="Path to -que words file")

    # Generate adverbs command
    adv_parser = subparsers.add_parser(
        "generate-adverbs", help="Generate adverb list from input"
    )
    adv_parser.add_argument("input", nargs="+", type=Path, help="Input files")
    adv_parser.add_argument("--output", "-o", type=Path, help="Output directory")
    adv_parser.add_argument(
        "--max", type=int, default=200, help="Maximum adverbs to save"
    )

    # Mask command
    mask_parser = subparsers.add_parser("mask", help="Apply masking to input")
    mask_parser.add_argument("input", nargs="+", type=Path, help="Input files")
    mask_parser.add_argument("--adverbs", type=Path, help="Path to adverbs file")
    mask_parser.add_argument(
        "--replacements", type=Path, help="Path to replacement dict"
    )
    mask_parser.add_argument("--output", "-o", type=Path, help="Output directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "process": _cmd_process,
        "split-sentences": _cmd_split_sentences,
        "split-que": _cmd_split_que,
        "generate-adverbs": _cmd_generate_adverbs,
        "mask": _cmd_mask,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
