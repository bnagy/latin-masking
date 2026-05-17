"""Sentence splitting for Latin text using spaCy la_senter with NLTK fallback."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from spacy.language import Language

# Latin abbreviations for NLTK PunktSentenceTokenizer
LT_ABBREV = {
    "ti",
    "gn",
    "cn",
    "sp",
    "kal",
    "agr",
    "ap",
    "mam",
    "oct",
    "opet",
    "post",
    "pro",
    "ser",
    "st",
    "m",
}


@lru_cache(maxsize=1)
def get_senter() -> Any:
    """Get configured spaCy Latin senter with colon handling.

    Returns:
        Configured spaCy Language pipeline with prevent_colon_split component.

    """
    import la_senter

    senter = la_senter.load()
    senter.add_pipe("prevent_colon_split", after="senter")
    return senter


@Language.component("prevent_colon_split")
def prevent_colon_split(doc: Any) -> Any:
    """Prevent colons from being treated as sentence boundaries.

    Args:
        doc: spaCy document to process.

    Returns:
        Modified document with colon handling applied.

    """
    for i, token in enumerate(doc[:-1]):
        if token.text == ":":
            # Explicitly prevent the next token from being a sentence start
            doc[i + 1].is_sent_start = False
    return doc


def has_sufficient_punctuation(text: str) -> bool:
    """Check if text has enough punctuation to indicate sentence boundaries.

    Files that were preprocessed to remove punctuation will have very few
    sentence-ending characters. We require at least some periods or other
    sentence terminators.

    Args:
        text: Text to check for punctuation.

    Returns:
        True if text has sufficient punctuation (at least 5 sentence terminators).

    """
    periods = text.count(".")
    question_marks = text.count("?")
    exclamation = text.count("!")
    semicolons = text.count(";")

    total = periods + question_marks + exclamation + semicolons
    return total >= 5


def preprocess_text(text: str) -> tuple[str, dict[str, str]]:
    """Preprocess text before sentence segmentation.

    - Remove quotation marks
    - Remove dashes that follow sentence-terminating punctuation
    - Protect parenthetical content with placeholders

    Args:
        text: Raw text to preprocess.

    Returns:
        Tuple of (processed_text, paren_map) for later extraction.

    """
    # Remove quotation marks (various types)
    text = text.replace('"', "").replace("'", "").replace('"', "").replace("'", "")

    # Protect parenthetical content that starts with ( at word boundary
    paren_map: dict[str, str] = {}
    paren_counter = [0]

    def protect_parens(match: re.Match[str]) -> str:
        content = match.group(1)
        placeholder = f"__PAREN_{paren_counter[0]}__"
        paren_map[placeholder] = content
        paren_counter[0] += 1
        return placeholder

    # Match (content) where content doesn't contain unbalanced parens
    # Use non-greedy match with limit on content length (regex {0,500} caps at 500)
    text = re.sub(r"\(([^()]{0,500}?)\)", protect_parens, text)

    # Remove dashes that immediately follow sentence-terminating punctuation
    # Pattern: word. - or word! - or word? - or word; - or word: -
    text = re.sub(r"[.!?;:]+\s*-", "", text)

    return text, paren_map


def split_paren_content(content: str) -> list[str]:
    """Split parenthetical content into sentences.

    Splits on sentence-ending punctuation: . ; ! ?
    Colons are NOT sentence boundaries (consistent with prevent_colon_split).

    Args:
        content: Parenthetical content to split.

    Returns:
        List of sentence strings extracted from the content.

    """
    # Split on sentence-ending punctuation followed by space or end
    # Note: colons are excluded to be consistent with prevent_colon_split
    parts = re.split(r"[.!?;]+(\s+|$)", content)
    sentences = []
    for part in parts:
        part = part.strip()
        if part and re.search(r"[a-zA-ZÀ-ÿ]", part):
            sentences.append(part)
    return sentences


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using la_senter model.

    Filters out sentences containing square brackets (editorial markers)
    and sentences with no letters (e.g., ellipses).

    Handles parenthetical content by extracting and splitting it into sentences.

    Args:
        text: Text to split into sentences.

    Returns:
        List of sentence strings.

    """
    # Preprocess text before sentence segmentation
    text, paren_map = preprocess_text(text)

    senter = get_senter()
    doc = senter(text)
    sentences = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        # Skip sentences containing square brackets
        if not sent_text or "[" in sent_text or "]" in sent_text:
            continue
        # Skip sentences with no letters at all (e.g., ellipses)
        if not re.search(r"[a-zA-ZÀ-ÿ]", sent_text):
            continue

        # Check for parenthetical placeholders and restore them
        restored = False
        for placeholder, content in paren_map.items():
            if placeholder in sent_text:
                # Split the parenthetical content into sentences
                paren_sentences = split_paren_content(content)
                sentences.extend(paren_sentences)
                restored = True
                break

        if restored:
            continue

        sentences.append(sent_text)

    return sentences
