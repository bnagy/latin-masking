"""CoNLL-U response parsing for UDPipe output."""

from __future__ import annotations

import re
from typing import Iterator

import pandas as pd

from latin_masking.types import ConlluFrame


def _cap_diff(x: int) -> str:
    """Format difference value for tok_3 column.

    Args:
        x: Difference value.

    Returns:
        Formatted string: '+' for >5, '-' for <-5, otherwise the number.

    """
    if x > 5:
        return "+"
    elif x < -5:
        return "-"
    else:
        return str(x)


def _get_token(x: pd.Series) -> str:
    """Generate token string for tok_3 column.

    Args:
        x: DataFrame row with rel, parent_rel, diff columns.

    Returns:
        Formatted token string.

    """
    return f"{x['rel']}_{x['parent_rel']}({x['diff']})"


def _process_frame(
    this_frame: list[list[str]],
    frames: list[ConlluFrame],
    cols: list[str],
) -> None:
    """Process accumulated frame data into DataFrame and append to frames list.

    Args:
        this_frame: Accumulated row data.
        frames: List to append the processed DataFrame to.
        cols: Column names for the DataFrame.

    """
    if not this_frame:
        return
    df = pd.DataFrame(this_frame, columns=cols)
    # Filter out header words from clitic splits (parent == "_")
    df = df[df["parent"] != "_"]
    parent_vals = df["parent"].astype(str)
    df["parent_rel"] = [
        "ISAT" if int(x) == 0 else df.iloc[int(x) - 1]["rel"] for x in parent_vals
    ]
    df["diff"] = pd.Series(
        _cap_diff(x) for x in (df["idx"].astype(int) - df["parent"].astype(int))
    )
    df["tok_3"] = df.apply(_get_token, axis=1)
    df.drop(["POS2", "junk", "junk2"], inplace=True, axis=1)
    stripped_df = df[df["POS"] != "PUNCT"]
    # Sometimes you have a sentence that is just punctuation
    if len(stripped_df) > 0:
        frames.append(pd.DataFrame(stripped_df))
    this_frame.clear()


def parse_conllu(response: str) -> tuple[list[ConlluFrame], list[str]]:
    """Parse CoNLL-U response into DataFrames (one per sentence) + sentence texts.

    Filters out "header words" from clitic splits (rows where parent == "_").
    Computes parent_rel, diff, tok_3 columns.

    Args:
        response: Raw CoNLL-U response string from UDPipe.

    Returns:
        Tuple of (list of DataFrames, list of sentence texts).

    """
    frames: list[ConlluFrame] = []
    texts: list[str] = []
    gather = False
    this_frame: list[list[str]] = []
    cols = [
        "idx",
        "word",
        "lemma",
        "POS",
        "POS2",
        "Feats",
        "parent",
        "rel",
        "junk",
        "junk2",
    ]

    for line in response.splitlines():
        if line and not (line.startswith("#") or re.match(r"^\d+", line)):
            raise ValueError(f"PANIC: Unknown line type in response: {line}")
        if not line:
            continue
        if gather:
            if line.startswith("# text"):
                # Start of a new sentence - flush the previous one first
                _process_frame(this_frame, frames, cols)
                texts.append(line.split("=")[1].strip())
            elif not re.match(r"^\d+", line):
                gather = False
                _process_frame(this_frame, frames, cols)
            else:
                this_frame.append(line.split("\t"))
        else:
            if not line.startswith("# text"):
                continue
            texts.append(line.split("=")[1].strip())
            gather = True

    # Flush the final sentence if the response ended while still gathering
    _process_frame(this_frame, frames, cols)

    return (frames, texts)


def parse_conllu_light(response: str) -> list[dict[str, list[str]]]:
    """Lighter alternative returning list of dicts without pandas.

    Args:
        response: Raw CoNLL-U response string from UDPipe.

    Returns:
        List of dicts with 'words', 'pos', 'lemmas' keys.

    """
    result: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {"words": [], "pos": [], "lemmas": []}
    in_sentence = False

    for line in response.splitlines():
        if line.startswith("# text"):
            if in_sentence and current["words"]:
                result.append(current)
            current = {"words": [], "pos": [], "lemmas": []}
            in_sentence = True
        elif re.match(r"^\d+", line):
            parts = line.split("\t")
            if len(parts) >= 4:
                current["words"].append(parts[1])
                current["lemmas"].append(parts[2])
                current["pos"].append(parts[3])
        elif not line and in_sentence:
            if current["words"]:
                result.append(current)
            current = {"words": [], "pos": [], "lemmas": []}
            in_sentence = False

    # Don't forget the last sentence
    if current["words"]:
        result.append(current)

    return result


def iter_conllu_sentences(
    response: str,
) -> Iterator[tuple[list[str], list[str], str]]:
    """Generator yielding (words, pos_tags, text) per sentence.

    Most memory-efficient for large responses.

    Args:
        response: Raw CoNLL-U response string from UDPipe.

    Yields:
        Tuple of (words list, pos_tags list, sentence text) for each sentence.

    """
    words: list[str] = []
    pos_tags: list[str] = []
    text = ""
    in_sentence = False

    for line in response.splitlines():
        if line.startswith("# text"):
            if in_sentence and words:
                yield (words, pos_tags, text)
            words = []
            pos_tags = []
            text = line.split("=")[1].strip()
            in_sentence = True
        elif re.match(r"^\d+", line):
            parts = line.split("\t")
            if len(parts) >= 4:
                words.append(parts[1])
                pos_tags.append(parts[3])
        elif not line and in_sentence:
            if words:
                yield (words, pos_tags, text)
            words = []
            pos_tags = []
            text = ""
            in_sentence = False

    # Yield the final sentence
    if words:
        yield (words, pos_tags, text)
