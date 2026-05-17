"""-que enclitic splitting for Latin text."""

from __future__ import annotations

import re
from pathlib import Path


def load_que_words(path: Path) -> list[str]:
    """Load the list of -que words from the curated file.

    Args:
        path: Path to the que_conj_words.txt file.

    Returns:
        List of -que words.

    """
    words: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                word = line.split("\t")[0]
                words.append(word)
    return words


def load_que_whitelist(path: Path) -> list[str]:
    """Load additional -que words from whitelist file.

    Format: word<TAB>count<TAB>POS<TAB>notes (comments start with #)
    Only includes words marked for splitting (not commented out).

    Args:
        path: Path to the whitelist candidates file.

    Returns:
        List of -que words from whitelist.

    """
    words: list[str] = []
    if not path.exists():
        return words
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("\t")
                if parts:
                    word = parts[0]
                    if not word.startswith("#"):
                        words.append(word)
    return words


def split_que(text: str, que_words: list[str]) -> tuple[str, int]:
    """Split -que enclitics in text.

    Returns the modified text and the count of replacements made.

    Args:
        text: Text to process.
        que_words: List of -que words to split.

    Returns:
        Tuple of (modified text, replacement count).

    """
    total_replacements = 0

    for que_word in que_words:
        # Build regex pattern with word boundaries
        # Match the word followed by optional punctuation
        pattern = r"\b(" + re.escape(que_word) + r")([.,;:]?)\b"

        def replace_func(match: re.Match[str]) -> str:
            nonlocal total_replacements
            word = match.group(1)
            punct = match.group(2)
            total_replacements += 1
            # Split: word[:-3] + " -que" + punctuation
            base = word[:-3]  # Remove "que"
            return f"{base} -que{punct}"

        text = re.sub(pattern, replace_func, text)

    return text, total_replacements
