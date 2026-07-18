"""Rule-based sentence boundary fixers for la_senter."""

from typing import Callable

from spacy.language import Language
from spacy.tokens import Doc, Token

CLOSING_QUOTES = {
    '"',          # straight double (U+0022) — ambiguous
    "”",     # right double quotation mark " — definitively closing
    "'",          # straight single (U+0027) — ambiguous
    "’",     # right single quotation mark ' — definitively closing
    "»",     # right-pointing double angle quotation mark » — definitively closing
    "›",     # single right-pointing angle quotation mark › — definitively closing
}

# These characters unambiguously encode direction — no whitespace check needed
DEFINITIVELY_CLOSING = {"”", "’", "»", "›"}

# Hard terminals after which a quote is a closer, not an opener.
# Colon is excluded: in Latin, colon introduces direct speech ("loquitur: 'Rex...'")
# and a quote after colon is always an opener.
CLOSING_TERMINALS = {".", "!", "?", ";"}


def _is_likely_closing(token: Token) -> bool:
    """Return True if a sentence-initial quote is closing an open speech.

    For unambiguous closing characters (curly quotes, guillemets): always True.
    For ambiguous straight/single quotes: True only when the immediately preceding
    token is a hard sentence terminal (. ! ? ;). Colon is excluded because in
    Latin it introduces direct speech and a quote after colon is an opener.
    """
    if token.text in DEFINITIVELY_CLOSING:
        return True
    return token.i > 0 and token.nbor(-1).text in CLOSING_TERMINALS


OPEN_BRACKETS = {"(", "["}
CLOSE_BRACKETS = {")", "]"}

# Unicode dash/hyphen characters used as parenthetical delimiters.
# Not added to OPEN_BRACKETS because bare - is too common in Latin
# (enclitic separators, verse markup) to treat as a bracket opener.
# Instead, _dash_paren_fix uses a stricter paired-with-terminal check.
DASH_CHARS = {
    "-",           # U+002D HYPHEN-MINUS
    "—",           # U+2014 EM DASH
    "–",           # U+2013 EN DASH
    "‐",           # U+2010 HYPHEN
    "‑",           # U+2011 NON-BREAKING HYPHEN
}
DASH_TERMINALS = {"!", "?"}


def paren_fix(doc: Doc) -> Doc:
    """Suppress sentence boundaries that fall inside unclosed parentheses/brackets.

    The senter treats '!' and '?' as terminals even inside parentheticals like
    '(mirabile dictu!)'. This rule scans backward from each predicted boundary
    to the previous sentence start, counting open vs. close brackets — if any
    bracket is still open, the boundary is spurious and is suppressed.
    """
    for token in doc[1:]:
        if not token.is_sent_start:
            continue
        depth = 0
        for t in reversed(list(doc[:token.i])):
            if t.text in CLOSE_BRACKETS:
                depth += 1
            elif t.text in OPEN_BRACKETS:
                if depth > 0:
                    depth -= 1
                else:
                    token.is_sent_start = False
                    break
            if t.is_sent_start and t.i > 0:
                break
    return doc


def _dash_paren_fix(doc: Doc) -> Doc:
    """Suppress boundaries where a dash-parenthetical contains a terminal.

    Handles '- content! -' and '— content? —' asides. Only fires when:
    1. The sent_start token is itself a dash character (the closing dash), AND
    2. Scanning backward finds an opening dash of any DASH_CHARS type, AND
    3. A '!' or '?' appears between the two dashes.

    This avoids the false-positive problem of bare '-' in Latin enclitic
    separators (e.g. '-que') which must never suppress a sentence boundary.
    """
    for token in doc[1:]:
        if not token.is_sent_start or token.text not in DASH_CHARS:
            continue
        found_terminal = False
        for t in reversed(list(doc[:token.i])):
            if t.is_sent_start and t.i > 0:
                break
            if t.text in DASH_TERMINALS:
                found_terminal = True
            elif t.text in DASH_CHARS and found_terminal:
                token.is_sent_start = False
                break
    return doc


def quote_fix(doc: Doc) -> Doc:
    for token in doc[1:]:
        if (
            token.is_sent_start
            and token.text in CLOSING_QUOTES
            and _is_likely_closing(token)
        ):
            token.is_sent_start = False
            if token.i + 1 < len(doc):
                doc[token.i + 1].is_sent_start = True
    return doc


def token_fix(doc: Doc) -> Doc:
    """Unified rules-based sentence boundary corrector for Latin text.

    Applies all punctuation-aware corrections in a single pipeline pass:
    - paren_fix: suppress boundaries inside unclosed parentheses/brackets
    - quote_fix: move boundaries past orphaned closing quotes

    Additional rules (em-dash parentheticals, etc.) should be added here.
    """
    doc = paren_fix(doc)
    doc = _dash_paren_fix(doc)
    doc = quote_fix(doc)
    return doc


@Language.factory("token_fix")
def create_token_fix(nlp: Language, name: str) -> Callable[[Doc], Doc]:
    """Factory for the token_fix pipeline component.

    Registered as a factory (not Language.component) so spaCy records this
    module as the factory's source. `spacy package` reads that module to derive
    model requirements; the component form records 'spacy.language' instead,
    which drops latincy-senter from the published model's dependencies.
    token_fix itself stays a plain Doc -> Doc callable.
    """
    return token_fix
