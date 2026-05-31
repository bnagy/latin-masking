"""Text preprocessing: all mangling in one place.

Applies normalization, macron removal, and punctuation stripping
in a single pass, with protected-token handling.

After preprocessing, text is "clean" — downstream stages (UDPipe,
masking) receive text that needs no further mangling.
"""

from __future__ import annotations

import re

from latin_masking.normalize import normalize_ch_h, normalize_uv_ij
from latin_masking.types import PROTECTED_TOKENS, protect_tokens, restore_tokens

# Characters stripped by strip_punct (same set as the old client.py)
PUNCT_CHARS = r"[]<>{}†'\""


def preprocess(
    text: str,
    *,
    normalize: bool = True,
    remove_macrons: bool = True,
    strip_punct: bool = True,
    protected_tokens: set[str] | None = None,
) -> str:
    """Apply all text mangling in one place.

    Order of operations:
    1. Protect special tokens (e.g. <EOL>) with placeholders
    2. Normalize UV/IJ (v→u, j→i) and ch/h (michi→mihi)
    3. Remove macrons
    4. Strip punctuation characters
    5. Restore protected tokens

    Args:
        text: Raw text to preprocess.
        normalize: Apply UV/IJ + ch/h normalization.
        remove_macrons: Remove macron diacritics.
        strip_punct: Strip punctuation characters ([]<>{}†'").
        protected_tokens: Tokens to preserve as-is. Defaults to
            PROTECTED_TOKENS (which includes "<EOL>").

    Returns:
        Preprocessed text with protected tokens restored.
    """
    if protected_tokens is None:
        protected_tokens = PROTECTED_TOKENS

    # Step 1: Protect special tokens
    text, mapping = protect_tokens(text, protected_tokens)

    # Step 2: Normalize (UV/IJ + ch/h) — per-token
    if normalize:
        tokens = text.split()
        tokens = [normalize_ch_h(normalize_uv_ij(t)) for t in tokens]
        text = " ".join(tokens)

    # Step 3: Remove macrons
    if remove_macrons:
        text = _remove_macrons(text)

    # Step 4: Strip punctuation
    if strip_punct:
        text = text.translate(str.maketrans("", "", PUNCT_CHARS))

    # Step 5: Restore protected tokens
    text = restore_tokens(text, mapping)

    return text


def _remove_macrons(text: str) -> str:
    """Remove macrons from Latin text."""
    import unicodedata

    return unicodedata.normalize("NFD", text).replace("\u0304", "")
