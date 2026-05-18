#!/usr/bin/env python3
"""Extract -que words from quesplit files that were NOT split."""

import re
from pathlib import Path


def extract_que_words_from_file(filepath: Path) -> set[str]:
    """Extract -que words that remain unsplit in a quesplit file."""
    que_words = set()
    content = filepath.read_text(encoding="utf-8")

    # Find all words ending in -que that are NOT preceded by " -que" (which would indicate they were split)
    # We want words like "efficiuntque", "Inque", "sodomamque", etc.
    # But NOT "manu -que" (which was split)

    # Pattern: word characters followed by "que" at word boundary
    # But exclude cases where there's " -que" before (already split)
    pattern = r"\b(\w+que)\b"

    for match in re.finditer(pattern, content):
        word = match.group(1)
        # Check if this is part of a split pattern " -que"
        # by looking at the context
        start = match.start()
        if start >= 3 and content[start - 3 : start] == " -q":
            # This was a split word, skip it
            continue
        que_words.add(word)

    return que_words


def main():
    all_que_words = set()

    # Find all quesplit.masked.txt files in liber-regum
    # These contain POS-tagged tokens where -que words that were NOT split remain intact
    # (e.g., "cumque" stays as one token, while split words become "word -que")
    liber_regum_dir = Path("/Users/ben/code/Liber-Regum/Metrical analysis/corpus")

    quesplit_files = list(liber_regum_dir.glob("*_sentences.quesplit.masked.txt"))

    for filepath in quesplit_files:
        words = extract_que_words_from_file(filepath)
        all_que_words.update(words)
        print(f"From {filepath.name}: {len(words)} words")

    print(f"\nTotal unique -que words: {len(all_que_words)}")

    # Write to blacklist file
    output_file = Path(
        "/Users/ben/code/latin-masking/src/latin_masking/data/que_blacklist.txt"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        for word in sorted(all_que_words):
            f.write(f"{word}\n")
    print(f"\nWrote blacklist to {output_file}")


if __name__ == "__main__":
    main()
