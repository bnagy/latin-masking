"""Text normalization for UV/IJ variants."""

from __future__ import annotations

# Special case normalizations for ch→h variants
CH_TO_H_NORMALIZATIONS = {
    "michi": "mihi",
    "nichil": "nihil",
}


def normalize_uv_ij(word: str) -> str:
    """Generate normalized form by replacing v→u and j→i.

    This is applied universally after lowercasing in the pipeline.

    Args:
        word: Word to normalize.

    Returns:
        Normalized word with u/v and i/j standardized.

    """
    return word.replace("v", "u").replace("V", "U").replace("j", "i").replace("J", "I")


def normalize_ch_h(token: str) -> str:
    """Special case: normalize michi→mihi, nichil→nihil.

    Args:
        token: Token to normalize.

    Returns:
        Normalized token if it matches special cases, otherwise unchanged.

    """
    token_lower = token.lower()
    if token_lower in CH_TO_H_NORMALIZATIONS:
        normalized = CH_TO_H_NORMALIZATIONS[token_lower]
        if token and token[0].isupper():
            return normalized.capitalize()
        return normalized
    return token


def normalize_text(text: str) -> str:
    """Apply UV/IJ and ch/h normalization to entire text.

    Splits on whitespace, applies normalize_uv_ij and normalize_ch_h
    to each token, rejoins with single spaces. This is the first
    processing step applied to raw text before any other pipeline stage.

    Args:
        text: Raw text to normalize.

    Returns:
        Normalized text with UV/IJ and ch/h variants standardized.
    """
    tokens = text.split()
    normalized = [normalize_ch_h(normalize_uv_ij(t)) for t in tokens]
    return " ".join(normalized)
