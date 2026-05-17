"""Adverb processing for Latin text."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from latin_masking.normalize import normalize_uv_ij


def collect_adverbs(parsed_sentences: list[pd.DataFrame]) -> Counter[str]:
    """Collect adverbs from parsed sentences.

    Args:
        parsed_sentences: List of DataFrames from CoNLL-U parsing.

    Returns:
        Counter of adverb tokens (case-insensitive).

    """
    counter: Counter[str] = Counter()
    for sentence in parsed_sentences:
        advs = sentence[sentence["POS"] == "ADV"]["word"]
        for adv in advs:
            counter[adv.lower()] += 1
    return counter


def normalize_adverb_counts(counter: Counter[str]) -> Counter[str]:
    """Merge adverb counts by normalized (u→v, i→j) form.

    Args:
        counter: Counter of adverb tokens.

    Returns:
        Counter with normalized forms.

    """
    normalized: Counter[str] = Counter()
    for adv, count in counter.items():
        normalized[normalize_uv_ij(adv)] += count
    return normalized


def generate_adverb_list(
    counter: Counter[str], max_adverbs: int = 200
) -> list[tuple[str, int]]:
    """Get top-N most frequent adverbs.

    Args:
        counter: Counter of adverb tokens.
        max_adverbs: Maximum number of adverbs to return.

    Returns:
        List of (adverb, count) tuples.

    """
    return counter.most_common(max_adverbs)


def save_adverb_list(adverbs: list[tuple[str, int]], path: Path) -> None:
    """Write adverb list to file.

    Args:
        adverbs: List of (adverb, count) tuples.
        path: Path to write the file.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for adv, count in adverbs:
            f.write(f"{adv}\t{count}\n")


def load_adverb_list(path: Path, threshold: int) -> set[str]:
    """Load adverb list and return top-threshold adverbs as a set.

    Args:
        path: Path to the adverb list file.
        threshold: Number of top adverbs to include.

    Returns:
        Set of adverb strings.

    """
    adverbs: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= threshold:
                break
            line = line.strip()
            if line:
                word = line.split("\t")[0]
                adverbs.add(word)
    return adverbs
