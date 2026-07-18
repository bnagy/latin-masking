"""Sentence splitting for Latin text using spaCy la_senter with NLTK fallback."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from spacy.language import Language  # pyright: ignore[reportMissingImports]

from latin_masking.normalize import normalize_text

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


@lru_cache(maxsize=4)
def get_senter(version: str | None = "3.9.2") -> Any:
    """Get configured spaCy Latin senter with colon handling.

    By default loads the vendored ``la_senter`` 3.9.2 model, which is bundled
    as a sub-package under ``latin_masking/_vendor/la_senter`` (no separate
    install or wheel extraction required). Pass ``version`` (e.g.
    ``"3.8.0"``) to load a specific model version instead, so multiple model
    versions can be compared.  Pass ``None`` to load whatever ``la_senter``
    package is importable in the environment.

    Args:
        version: Optional ``la_senter`` version string. If given (including the
            default ``"3.9.2"``), the model is loaded from the bundled
            sub-package ``latin_masking/_vendor/la_senter/la_senter-{version}``.

    Returns:
        Configured spaCy Language pipeline.

    """
    from spacy.util import load_model_from_path  # pyright: ignore[reportMissingImports]

    if version is None:
        import la_senter  # pyright: ignore[reportMissingImports]

        senter = la_senter.load()
    else:
        # Load a specific model version directly from the bundled sub-package,
        # without installing it into the environment. The vendored model's
        # fixers module registers the `token_fix` factory used by newer
        # la_senter versions (3.9.x); import it so the model config can resolve
        # the component.
        vendor_dir = Path(__file__).resolve().parent / "_vendor" / "la_senter"
        model_dir = vendor_dir / f"la_senter-{version}"
        import importlib.util

        fixers_path = vendor_dir / "fixers.py"
        fixers_spec = importlib.util.spec_from_file_location(
            "la_senter_fixers_tmp", fixers_path
        )
        if fixers_spec is None or fixers_spec.loader is None:
            raise RuntimeError(f"Could not load fixers module from {fixers_path}")
        fixers_mod = importlib.util.module_from_spec(fixers_spec)
        fixers_spec.loader.exec_module(fixers_mod)
        senter = load_model_from_path(model_dir)

    # Double the default max_length to handle longer texts (default: 1,000,000)
    senter.max_length = 2000000
    # The vendored model's `token_fix` component (paren_fix + dash_fix +
    # quote_fix) is disabled: paren_fix has a bug where a sentence boundary
    # that falls ON a closing ')' is wrongly merged (it scans doc[:token.i]
    # and never sees the closing paren), which cascades into a giant run.
    # Our own clean_text (quote/dash stripping) and split_parens (paren
    # extraction) already cover what token_fix did, so we drop it entirely.
    try:
        senter.disable_pipe("token_fix")
    except ValueError:
        pass
    # Suppress any sentence start that falls inside a parenthesis.  la_senter
    # treats '!'/'?' as hard terminators, so a '!' inside "(...)" would split
    # the host sentence and break paren extraction.  A single forward depth
    # pass is all we need (the unmatched-paren case is caught earlier by
    # check_paren_balance).
    senter.add_pipe("suppress_paren_starts", after="senter")
    senter.add_pipe("prevent_colon_split", after="suppress_paren_starts")
    return senter


@Language.component("suppress_paren_starts")
def suppress_paren_starts(doc: Any) -> Any:
    """Suppress sentence starts that occur inside parentheses.

    la_senter treats '!' and '?' as hard sentence terminators, so a
    parenthetical like "(infandum!)" would otherwise start a new sentence
    mid-paren, splitting the host and preventing clean paren extraction.  We
    walk the document once, tracking bracket depth, and clear ``is_sent_start``
    on any token reached while depth > 0.  Unmatched brackets are not handled
    here (they are rejected up front by :func:`check_paren_balance`).
    """
    depth = 0
    for token in doc:
        if token.text.startswith("("):
            depth += 1
        if depth > 0 and token.is_sent_start:
            token.is_sent_start = False
        if ")" in token.text:
            depth -= 1
    return doc


@Language.component("prevent_colon_split")
def prevent_colon_split(doc: Any) -> Any:
    """Prevent colons from being treated as sentence boundaries.

    A colon that closes a parenthetical (e.g. ``(...):``) is followed by the
    next real sentence, so its sentence-start suppression must NOT apply there
    — otherwise the next sentence gets swallowed into the previous one (a
    cascade that can merge thousands of tokens).  We therefore skip the
    suppression when the colon is the closing token of a parenthetical, i.e.
    when it is immediately followed by ``)`` or itself ends with ``)``.

    Args:
        doc: spaCy document to process.

    Returns:
        Modified document with colon handling applied.

    """
    for i, token in enumerate(doc[:-1]):
        if token.text == ":":
            nxt = doc[i + 1]
            # If the colon closes a parenthetical, the following token is the
            # start of the next sentence — do not suppress it.
            if nxt.text.startswith(")") or token.text.endswith(")"):
                continue
            nxt.is_sent_start = False
    return doc


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text by collapsing multiple whitespace chars.

    Replaces multiple newlines, tabs, and spaces with a single space.
    This handles files with irregular formatting.

    Args:
        text: Text to normalize.

    Returns:
        Text with normalized whitespace.

    """
    # Replace all whitespace sequences (including newlines, tabs) with single space
    # and strip leading/trailing whitespace
    return re.sub(r"\s+", " ", text).strip()


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


# Quote characters stripped before segmentation (same set as the old
# preprocess_text).
QUOTE_CHARS = (
    r"[\u0022\u0027\u00AB\u00BB\u2018\u2019\u201A\u201B\u201C\u201D"
    r"\u201E\u201F\u2039\u203A]"
)

# Weird punctuation characters stripped during cleaning (same set as the
# preprocessor.PUNCT_CHARS, minus the brackets which are handled separately).
WEIRD_PUNCT = r"<>{}†'\""

# A bracketed editorial passage may span at most this many lines.  Anything
# larger indicates malformed input (e.g. an unclosed "["), not a real note.
MAX_BRACKET_LINES = 100


def protect_eol_tokens(text: str) -> tuple[str, list[str]]:
    """Protect ``<EOL>`` (and any ``<...>`` token) with placeholders.

    The weird-punctuation strip below would otherwise delete the ``<`` / ``>``
    of an ``<EOL>`` token, so it is hidden behind a placeholder that survives
    normalization and is restored at the end.

    Args:
        text: Text possibly containing ``<EOL>`` / ``<...>`` tokens.

    Returns:
        Tuple of (text with tokens replaced by ``__PROT_n__`` placeholders,
        ordered list of the original tokens).

    """
    protected: list[str] = []

    def _protect(m: re.Match[str]) -> str:
        protected.append(m.group(0))
        return f"__PROT_{len(protected) - 1}__"

    text = re.sub(r"<\s*[A-Za-z][^>]*>", _protect, text)
    return text, protected


def restore_eol_tokens(text: str, protected: list[str]) -> str:
    """Restore ``__PROT_n__`` placeholders to their original tokens.

    Args:
        text: Text containing ``__PROT_n__`` placeholders.
        protected: Ordered list of original tokens from :func:`protect_eol_tokens`.

    Returns:
        Text with placeholders restored.

    """
    for i, tok in enumerate(protected):
        text = text.replace(f"__PROT_{i}__", tok)
    return text


def strip_quote_chars(text: str) -> str:
    """Remove unicode quotation marks from text.

    Args:
        text: Text possibly containing unicode quote characters.

    Returns:
        Text with quote characters removed.

    """
    return re.sub(QUOTE_CHARS, "", text)


def strip_dashes_after_terminators(text: str) -> str:
    """Remove dashes that immediately follow sentence-terminating punctuation.

    Args:
        text: Text possibly containing ``-`` after ``.``/``!``/``?``/``;``/``:``.

    Returns:
        Text with trailing dashes after terminators removed.

    """
    return re.sub(r"[.!?;:]+\s*-", "", text)


def normalize_dashes(text: str) -> str:
    """Replace em/en dashes with a space so la_senter keeps the adjacent word.

    la_senter's tokenizer silently drops a final token ending in a non-ASCII
    dash (e.g. ``meus!—``), losing both the word and its ``<EOL>``.  Replacing
    the dash with a space preserves the word; a dash that follows a terminator
    (``!–``) still yields a sentence boundary via the ``!``.  A dash already
    flanked by spaces (``word — word``) would otherwise produce two spaces in
    a row, so runs of multiple spaces are collapsed to one.
    """
    text = text.replace("\u2014", " ").replace("\u2013", " ")
    return re.sub(r" {2,}", " ", text)


def handle_brackets(text: str, mode: str = "strip") -> str:
    """Handle square-bracket editorial markers.

    Two modes are supported, controlled by *mode*:

    * ``"strip"`` (default): remove the ``[`` / ``]`` characters but keep the
      content.  This is the normal behaviour and matches the preprocessor.
    * ``"drop"``: remove entire bracketed passages (content and delimiters),
      defensively.  This recovers the old behaviour of discarding editorial
      notes.  Malformed input (unmatched/nested brackets, or a span longer
      than ``MAX_BRACKET_LINES``) raises ``ValueError``.

    Args:
        text: Text possibly containing ``[...]`` editorial markers.
        mode: ``"strip"`` or ``"drop"``.

    Returns:
        Text with brackets handled per *mode*.

    """
    if mode == "strip":
        return text.replace("[", "").replace("]", "")
    if mode == "drop":
        return _strip_bracket_passages(text)
    raise ValueError(f"unknown bracket mode: {mode!r} (expected 'strip' or 'drop')")


def _strip_bracket_passages(text: str) -> str:
    """Remove square-bracket editorial passages entirely, defensively.

    Bracket spans may run across multiple lines.  Defensive checks:
      * An unmatched "[" (no closing "]") is rejected as malformed input.
      * A nested "[" inside an open span is rejected.
      * A span longer than ``MAX_BRACKET_LINES`` lines is rejected.
    """
    out = []
    i = 0
    n = len(text)
    depth = 0
    start = -1
    while i < n:
        ch = text[i]
        if ch == "[":
            if depth == 0:
                out.append(text[start + 1 : i])
                start = i
                depth = 1
            else:
                raise ValueError(
                    f"nested '[' at offset {i}: bracketed passages must not "
                    f"contain further '['"
                )
        elif ch == "]":
            if depth == 0:
                raise ValueError(f"unmatched ']' at offset {i}: stray closing bracket")
            span = text[start : i + 1]
            if span.count("\n") > MAX_BRACKET_LINES:
                raise ValueError(
                    f"bracketed passage spans {span.count(chr(10))} lines "
                    f"(max {MAX_BRACKET_LINES}); likely an unclosed '['"
                )
            depth = 0
            start = i
        i += 1
    if depth != 0:
        raise ValueError(f"unmatched '[' at offset {start}: no closing ']' found")
    out.append(text[start + 1 :])
    return "".join(out)


def remove_macrons(text: str) -> str:
    """Remove macrons from Latin text (NFD decomposition).

    Args:
        text: Text possibly containing macronned vowels.

    Returns:
        Text with macron combining marks removed.

    """
    import unicodedata

    return unicodedata.normalize("NFD", text).replace("\u0304", "")


def strip_weird_punctuation(text: str) -> str:
    """Strip weird punctuation characters (``<>{}†'"``).

    Args:
        text: Text possibly containing weird punctuation.

    Returns:
        Text with weird punctuation removed.

    """
    return text.translate(str.maketrans("", "", WEIRD_PUNCT))


def clean_text(text: str, *, bracket_mode: str = "strip") -> str:
    """Fully clean the whole text BEFORE segmentation.

    Each step is an individual function so the pipeline is easy to maintain.
    Order:

      1. Protect ``<EOL>`` (and similar ``<...>`` tokens) with a placeholder.
      2. Strip unicode quotes.
      3. Strip dashes that follow sentence-terminating punctuation.
      4. Handle square brackets per *bracket_mode*.
      5. Normalize UV/IJ (v→u, j→i) and ch/h (michi→mihi, nichil→nihil).
      6. Remove macrons.
      7. Strip weird punctuation (``<>{}†'"``), then restore ``<EOL>``.

    Sentence punctuation (. ! ? ; :) is deliberately preserved so la_senter
    can still find boundaries.

    Args:
        text: Raw text to clean.
        bracket_mode: ``"strip"`` (default) or ``"drop"`` (see :func:`handle_brackets`).

    Returns:
        Cleaned text with ``<EOL>`` tokens restored.

    """
    text, protected = protect_eol_tokens(text)
    text = strip_quote_chars(text)
    text = strip_dashes_after_terminators(text)
    text = normalize_dashes(text)
    text = handle_brackets(text, mode=bracket_mode)
    text = normalize_text(text)
    text = remove_macrons(text)
    text = strip_weird_punctuation(text)
    text = restore_eol_tokens(text, protected)
    return text


def build_source(chunk: list[str]) -> tuple[list[tuple[str, int]], list[int]]:
    """Turn cleaned source lines into a flat (token, global_index) stream.

    Returns (tokens, line_token_counts) where *tokens* is the concatenation
    of every cleaned line's tokens (each paired with its 0-based global
    index) and *line_token_counts* is the number of tokens on each line.

    The caller passes content only (``<EOL>`` markers are stripped before
    this point), so every token here is real content.  Empty lines contribute
    no tokens and no <EOL>, keeping the global index stream aligned.
    """
    tokens: list[tuple[str, int]] = []
    line_token_counts: list[int] = []
    idx = 0
    for ln in chunk:
        toks = ln.split()
        if not toks:
            continue
        line_token_counts.append(len(toks))
        for t in toks:
            tokens.append((t, idx))
            idx += 1
    return tokens, line_token_counts


def _sent_char_span(sent) -> tuple[int | None, int | None]:
    try:
        return sent.start_char, sent.end_char
    except AttributeError:
        return None, None


def _char_span_to_tokens(
    toks: list[tuple[str, int]], cstart: int, cend: int
) -> tuple[int, int]:
    """Map a char span (within " ".join of *toks*) to token indices."""
    pos = 0
    offsets = []
    for t, _ in toks:
        offsets.append(pos)
        pos += len(t) + 1
    sa = 0
    for i, (t, _) in enumerate(toks):
        if offsets[i] >= cstart:
            sa = i
            break
    sb = 0
    for i, (t, _) in enumerate(toks):
        if offsets[i] + len(t) <= cend:
            sb = i + 1
        else:
            break
    return sa, sb


def _text_span_to_tokens(
    toks: list[tuple[str, int]], words: list[str], ptr: int
) -> tuple[int, int, int]:
    """Fallback: locate *words* as a contiguous run of *toks* at/after *ptr*."""
    n = len(toks)
    i = ptr
    while i + len(words) <= n:
        if [t for t, _ in toks[i : i + len(words)]] == words:
            return i, i + len(words), i + len(words)
        i += 1
    return ptr, ptr, ptr


def check_paren_balance(text: str) -> None:
    """Verify every ``(`` in *text* is matched by a ``)``.

    Runs as a cheap pre-flight check before segmentation.  An unmatched
    parenthesis would make the paren-suppression component treat the rest of
    the document as one run-on sentence, so we fail loudly with the exact
    location and a short context snippet instead of producing silently wrong
    output.

    Args:
        text: Raw text to validate (newline-separated verse lines or prose).

    Raises:
        ValueError: If an unmatched ``(`` or ``)`` is found, with the line
            number, the offending token, and surrounding context.

    """
    depth = 0
    # Track the first unmatched '(' (line, token) so we can report where it
    # opened.  Only meaningful while depth > 0.
    open_loc: tuple[int, str] | None = None
    for line_no, line in enumerate(text.split("\n"), start=1):
        for token in line.split():
            for ch in token:
                if ch == "(":
                    if depth == 0:
                        open_loc = (line_no, token)
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth < 0:
                        snippet = " ".join(line.split()[:8])
                        raise ValueError(
                            f"Unmatched ')' at line {line_no}: {token!r}. "
                            f"Context: {snippet}"
                        )
    if depth != 0:
        assert open_loc is not None  # depth > 0 implies an open '(' was seen
        open_line, open_token = open_loc
        snippet = " ".join((text.split("\n")[open_line - 1]).split()[:8])
        raise ValueError(
            f"Unmatched '(' opened at line {open_line}: {open_token!r}. "
            f"Context: {snippet}"
        )


def find_paren_spans(toks: list[tuple[str, int]]) -> list[tuple[int, int]]:
    """Find (open, close) token-index pairs for inline parens.

    Parens are attached to words (e.g. "(nam", "ore);"), so a token starting
    with "(" opens a paren and one containing ")" closes it (the close token
    may carry trailing punctuation, e.g. "ore);").  Non-nested parens are
    paired first-open with first-close.

    Args:
        toks: Flat (token, global_index) stream.

    Returns:
        List of (open_index, close_index) pairs.

    """
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    for i, (t, _) in enumerate(toks):
        if t.startswith("("):
            stack.append(i)
        if ")" in t:
            if stack:
                spans.append((stack.pop(), i))
    return spans


def split_parens(toks: list[tuple[str, int]], senter) -> list[list[tuple[str, int]]]:
    """Split one raw sentence's tokens into final sentences.

    The host (tokens outside any paren) is emitted first, then each paren's
    content is recursively re-segmented into subsentences placed after it.
    Paren delimiters are stripped from the emitted tokens.

    Args:
        toks: Flat (token, global_index) stream for one raw sentence.
        senter: Configured la_senter pipeline used for recursive re-segmentation.

    Returns:
        List of sentences, each a list of (token, global_index) tuples.

    """
    spans = find_paren_spans(toks)
    if not spans:
        return [toks] if any(re.search(r"[a-zA-ZÀ-ÿ]", t) for t, _ in toks) else []

    removed: set[int] = set()
    for o, c in spans:
        for i in range(o, c + 1):
            removed.add(i)

    host = [toks[i] for i in range(len(toks)) if i not in removed]

    result: list[list[tuple[str, int]]] = []
    if host and any(re.search(r"[a-zA-ZÀ-ÿ]", t) for t, _ in host):
        result.append(host)

    for o, c in spans:
        if o == c:
            # Single-token paren, e.g. "(infandum!)": strip both delimiters.
            inner = [(toks[o][0].lstrip("(").rstrip(")"), toks[o][1])]
        else:
            inner: list[tuple[str, int]] = []
            ot = toks[o][0].lstrip("(")
            if ot:
                inner.append((ot, toks[o][1]))
            inner.extend(toks[o + 1 : c])
            ct = toks[c][0].replace(")", "")
            if ct:
                inner.append((ct, toks[c][1]))
        if inner:
            result.extend(segment_tokens(inner, senter))
    return result


def segment_tokens(toks: list[tuple[str, int]], senter) -> list[list[tuple[str, int]]]:
    """Run la_senter over *toks* and recursively extract parens.

    Args:
        toks: Flat (token, global_index) stream to segment.
        senter: Configured la_senter pipeline.

    Returns:
        List of sentences, each a list of (token, global_index) tuples.

    """
    seg_text = " ".join(t for t, _ in toks)
    doc = senter(seg_text)
    out: list[list[tuple[str, int]]] = []
    ptr = 0
    for sent in doc.sents:
        stext = sent.text.strip()
        if not stext or not re.search(r"[a-zA-ZÀ-ÿ]", stext):
            continue
        words = stext.split()
        cstart, cend = _sent_char_span(sent)
        if cstart is not None and cend is not None:
            sa, sb = _char_span_to_tokens(toks, cstart, cend)
        else:
            sa, sb, ptr = _text_span_to_tokens(toks, words, ptr)
        out.extend(split_parens(toks[sa:sb], senter))
    return out


def place_eol(
    sentences: list[list[tuple[str, int]]], line_token_counts: list[int]
) -> list[list[tuple[str, int]]]:
    """Tag each line-ending source token with a trailing '<EOL>'.

    Every final sentence carries the global indices of its source tokens, so
    we locate each verse line's final token wherever it landed (host or a
    subsentence) and append '<EOL>' to it.

    Args:
        sentences: Segmented sentences, each a list of (token, global_index).
        line_token_counts: Number of tokens on each source verse line.

    Returns:
        The same sentences, with '<EOL>' appended to each line-ending token.

    """
    loc: dict[int, tuple[int, int]] = {}
    for si, sent in enumerate(sentences):
        for pos, (_, gidx) in enumerate(sent):
            loc[gidx] = (si, pos)

    cum = 0
    for n in line_token_counts:
        cum += n
        end_idx = cum - 1
        if end_idx in loc:
            si, pos = loc[end_idx]
            text, gidx = sentences[si][pos]
            sentences[si][pos] = (text + " <EOL>", gidx)
    return sentences


def cleanup_sentence(s: str) -> str:
    """Remove spaces directly before basic punctuation and strip.

    Args:
        s: Sentence string to clean up.

    Returns:
        Sentence with spaces before punctuation removed and surrounding
        whitespace stripped.

    """
    return re.sub(r"\s+([,.;:!?])", r"\1", s).strip()


def split_sentences(
    text: str,
    *,
    bracket_mode: str = "strip",
    preserve_eol: bool = True,
) -> list[str]:
    """Split text into sentences using la_senter (vendored 3.9.2).

    Pipeline (strictly: clean the whole text -> segment -> place EOLs):

    1. The raw text is split into verse lines on newlines (when
       *preserve_eol* is True) or treated as one block (when False).  ``<EOL>``
       is an *output-only* artifact and is never split on as input.
    2. The whole text is cleaned (quotes, dashes after terminators, brackets
       per *bracket_mode*, UV/IJ + ch/h normalization, macron removal, weird
       punctuation).  Sentence punctuation (. ! ? ; :) is preserved so la_senter
       can still find boundaries.
    3. The cleaned lines are turned into a flat token stream with a global
       index per token; we remember how many tokens fall on each verse line.
    4. la_senter segments the text; each raw sentence is recursively
       re-segmented so inline ``(...)`` content becomes subsentences placed
       AFTER the host.  Every produced sentence keeps the global indices of
       the source tokens it contains.
    5. Only AFTER segmentation do we place ``<EOL>`` tokens (when *preserve_eol*
       is True): for each verse line we tag its final source token (wherever
       it landed) with ``<EOL>``.  This is exact regardless of the reordering
       caused by paren extraction.  When *preserve_eol* is False (prose, where
       line breaks are meaningless) no ``<EOL>`` tokens are emitted.
    6. Spaces directly before basic punctuation are removed.

    Args:
        text: Raw text to split into sentences (newline-separated verse lines,
            or prose with arbitrary line breaks).
        bracket_mode: How to handle ``[...]`` editorial markers: ``"strip"``
            (default, keep content, drop delimiters) or ``"drop"`` (remove
            whole passages).
        preserve_eol: If True (default, verse), emit ``<EOL>`` at the word that
            ends each source line.  If False (prose), line breaks are treated
            as meaningless and no ``<EOL>`` tokens are produced.

    Returns:
        List of sentence strings.  With *preserve_eol* True each sentence
        carries ``<EOL>`` at the word ending its source verse line; with False
        the output is one sentence per line and no ``<EOL>`` tokens appear.

    """
    # Pre-flight: an unmatched parenthesis would make the paren-suppression
    # component treat the rest of the document as one run-on sentence, so fail
    # loudly with the exact location before doing any further work.
    check_paren_balance(text)

    if preserve_eol:
        # Split into verse lines on newlines.  EOL is NOT part of the input.
        raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    else:
        # Prose: line breaks are meaningless; treat the whole text as one block.
        raw_lines = [text]

    # Clean each line's CONTENT (no EOL in the stream).
    cleaned_lines = [
        clean_text(normalize_whitespace(ln), bracket_mode=bracket_mode)
        for ln in raw_lines
    ]
    cleaned_lines = [ln for ln in cleaned_lines if ln.strip()]

    # Token count of each cleaned (non-empty) source line, for EOL placement.
    line_token_counts = [len(ln.split()) for ln in cleaned_lines]

    # Build the global-index token stream from cleaned content only.
    tokens, _ = build_source(cleaned_lines)

    senter = get_senter()
    final = segment_tokens(tokens, senter)
    if preserve_eol:
        place_eol(final, line_token_counts)

    out: list[str] = []
    for sent in final:
        s = " ".join(t for t, _ in sent)
        out.append(cleanup_sentence(s))
    return out
