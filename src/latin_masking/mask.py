"""POS masking for Latin text.

Text should already be preprocessed (normalized, macrons removed,
punctuation stripped) before masking. Use
:func:`latin_masking.preprocessor.preprocess` for that.
"""

from __future__ import annotations

import pandas as pd

from latin_masking.types import PROTECTED_TOKENS


def mask_sentence(
    words: list[str],
    pos_tags: list[str],
    *,
    common_adverbs: set[str],
) -> list[str]:
    """Apply POS masking rules to one sentence.

    - NOUN/VERB/ADJ/PROPN/NUM/AUX → POS tag
    - ADV → lowercased word (if in common_adverbs) else "ADV"
    - Protected tokens (e.g. <EOL>) → preserved as-is
    - Everything else → lowercased word

    No normalization or macron removal is performed — text should
    already be clean.

    Args:
        words: List of words in the sentence.
        pos_tags: List of POS tags corresponding to words.
        common_adverbs: Set of common adverbs to keep unmasked.

    Returns:
        List of masked tokens.

    """
    pos_mask_tags = {"NOUN", "VERB", "ADJ", "PROPN", "NUM", "AUX"}
    final = []

    for i, w in enumerate(words):
        # Strip leading ( and trailing ) from tokens
        w_clean = w.lstrip("(").rstrip(")")

        # Preserve protected tokens (e.g. <EOL>) as-is
        if w_clean in PROTECTED_TOKENS:
            final.append(w_clean)
        elif pos_tags[i] in pos_mask_tags:
            final.append(pos_tags[i])
        elif pos_tags[i] == "ADV":
            w_lower = w_clean.lower()
            if w_lower in common_adverbs:
                final.append(w_lower)
            else:
                final.append("ADV")
        else:
            final.append(w_clean.lower())

    return final


def mask_corpus(
    parsed_sentences: list[pd.DataFrame],
    *,
    common_adverbs: set[str],
) -> list[str]:
    """Process all sentences and return masked lines.

    Args:
        parsed_sentences: List of DataFrames from CoNLL-U parsing.
        common_adverbs: Set of common adverbs to keep unmasked.

    Returns:
        List of masked sentences.

    """
    processed_sentences = []

    for sentence in parsed_sentences:
        words = list(sentence["word"])
        pos_tags = list(sentence["POS"])

        masked = mask_sentence(
            words,
            pos_tags,
            common_adverbs=common_adverbs,
        )
        processed_sentences.append(" ".join(masked))

    return processed_sentences


def collect_lowercase_words(masked_sentences: list[str]) -> set[str]:
    """Extract non-POS, non-ADV tokens from masked output.

    Args:
        masked_sentences: List of masked sentence strings.

    Returns:
        Set of lowercase words.

    """
    lowercase_words: set[str] = set()
    pos_tags = {"NOUN", "VERB", "ADJ", "PROPN", "NUM", "AUX", "ADV"}

    for sent in masked_sentences:
        for token in sent.split():
            if token not in pos_tags:
                lowercase_words.add(token)

    return lowercase_words


def two_pass_mask(
    parsed_sentences: list[pd.DataFrame],
    *,
    common_adverbs: set[str],
) -> list[str]:
    """Single-pass masking with universal UV/IJ normalization.

    Args:
        parsed_sentences: List of DataFrames from CoNLL-U parsing.
        common_adverbs: Set of common adverbs to keep unmasked.

    Returns:
        List of masked sentences.

    """
    return mask_corpus(parsed_sentences, common_adverbs=common_adverbs)
