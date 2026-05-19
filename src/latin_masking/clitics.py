"""-que enclitic splitting for Latin text."""

from __future__ import annotations

import importlib.resources
import re
from pathlib import Path


def get_default_blacklist() -> set[str]:
    """Get the default -que blacklist from the package data.

    Returns the set of -que words that should NOT be split, loaded from
    the default que_blacklist.txt file bundled with the package.

    Returns:
        Set of -que words that should NOT be split.

    """
    words: set[str] = set()
    try:
        # Use importlib.resources for Python 3.9+ compatibility
        content = (
            importlib.resources.files("latin_masking")
            .joinpath("data", "que_blacklist.txt")
            .read_text(encoding="utf-8")
        )
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                word = line.lstrip("?!")
                if word:
                    words.add(word.lower())
    except (FileNotFoundError, TypeError):
        pass
    return words


def load_que_blacklist(path: Path) -> set[str]:
    """Load the blacklist of -que words that should NOT be split.

    The blacklist file contains words with optional markers:
    - No marker: word found in Wiktionary (verified)
    - ?! prefix: base word found in Wiktionary
    - ?? prefix: neither word nor base found in Wiktionary

    All words in the blacklist should be preserved as-is (not split).

    Args:
        path: Path to the que_blacklist.txt file.

    Returns:
        Set of -que words that should NOT be split.

    """
    words: set[str] = set()
    if not path.exists():
        return words
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Remove marker prefixes if present
                word = line.lstrip("?!")
                if word:
                    words.add(word.lower())
    return words


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


def split_que_blacklist(
    text: str,
    blacklist: set[str] | None = None,
    common_adverbs: set[str] | None = None,
) -> tuple[str, int]:
    """Split -que enclitics in text using blacklist approach.

    Split all -que words EXCEPT those in the blacklist. This is the inverse
    of the whitelist approach - we split by default and preserve exceptions.

    If blacklist is None, uses the default blacklist from package data.
    Common adverbs are automatically added to the effective blacklist to
    prevent splitting of common adverbs (which should be preserved as-is).

    Args:
        text: Text to process.
        blacklist: Set of -que words that should NOT be split. If None,
            uses the default blacklist.
        common_adverbs: Set of common adverbs to protect from splitting.
            These are added to the effective blacklist.

    Returns:
        Tuple of (modified text, replacement count).

    """
    # Use default blacklist if none provided
    if blacklist is None:
        blacklist = get_default_blacklist()

    # Combine blacklist with common adverbs that end in 'que'
    effective_blacklist = set(blacklist)
    if common_adverbs:
        for adv in common_adverbs:
            if adv.endswith("que"):
                effective_blacklist.add(adv.lower())

    total_replacements = 0

    # Pattern to find all -que words (word boundary + word ending in que + optional punctuation)
    # We need to capture the word and any following punctuation
    pattern = r"\b(\w+que)([.,;:]?)\b"

    def replace_func(match: re.Match[str]) -> str:
        nonlocal total_replacements
        word = match.group(1)
        punct = match.group(2)

        # Check if this word is in the effective blacklist (case-insensitive)
        if word.lower() in effective_blacklist:
            return match.group(0)  # Return unchanged

        total_replacements += 1
        # Split: word[:-3] + " -que" + punctuation
        base = word[:-3]  # Remove "que"
        return f"{base} -que{punct}"

    text = re.sub(pattern, replace_func, text)

    return text, total_replacements
