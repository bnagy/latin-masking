"""Type definitions, dataclasses, and custom exceptions for latin-masking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd  # pyright: ignore[reportMissingImports]

# Type aliases
Token = str
POSTag = str
MaskedSentence = str
ConlluFrame = pd.DataFrame

# Tokens that are preserved as-is throughout the pipeline (not lowercased,
# not punctuation-stripped, not POS-masked). Add to this set as needed.
PROTECTED_TOKENS: set[str] = {"<EOL>"}


def protect_tokens(
    text: str, tokens: set[str] | None = None
) -> tuple[str, dict[str, str]]:
    """Replace protected tokens with placeholders before processing.

    Args:
        text: Input text.
        tokens: Set of tokens to protect. Defaults to PROTECTED_TOKENS.

    Returns:
        Tuple of (text with placeholders, mapping of placeholder → original token).
    """
    if tokens is None:
        tokens = PROTECTED_TOKENS
    mapping: dict[str, str] = {}
    for token in tokens:
        if token in text:
            placeholder = f"__PROTECTED_{len(mapping)}__"
            mapping[placeholder] = token
            text = text.replace(token, placeholder)
    return text, mapping


def restore_tokens(text: str, mapping: dict[str, str]) -> str:
    """Restore protected tokens from placeholders after processing.

    Args:
        text: Text with placeholders.
        mapping: Mapping of placeholder → original token.

    Returns:
        Text with original tokens restored.
    """
    for placeholder, token in mapping.items():
        text = text.replace(placeholder, token)
    return text


class UDPipeError(Exception):
    """Base exception for UDPipe-related errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Error message.
            original_error: The underlying exception that caused this error.

        """
        super().__init__(message)
        self.original_error = original_error


class UDPipeAPIError(UDPipeError):
    """HTTP or SSL error when communicating with UDPipe API."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Error message.
            status_code: HTTP status code if applicable.
            original_error: The underlying exception that caused this error.

        """
        super().__init__(message, original_error)
        self.status_code = status_code


class UDPipeParseError(UDPipeError):
    """Malformed CoNLL-U response from UDPipe."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Error message.
            original_error: The underlying exception that caused this error.

        """
        super().__init__(message, original_error)


class UDPipeInputError(UDPipeError):
    """Invalid input provided to UDPipe processing."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Error message.
            original_error: The underlying exception that caused this error.

        """
        super().__init__(message, original_error)


@dataclass
class MaskingConfig:
    """Configuration for the masking pipeline.

    Attributes:
        output_dir: Directory for all output files (sentences, masked, cache, adverbs).
        model: UDPipe model name (e.g., 'latin-ittb-ud-2.5-191005').
        adverb_threshold: Maximum number of adverbs to include in common adverbs set.
        common_adverbs_path: Path to file containing common adverbs list.
        cache_dir: Directory for caching UDPipe responses. Defaults to
            output_dir / "udpipe_cache".
        unsafe_certs_ok: Whether to accept self-signed SSL certificates.

    """

    output_dir: Path = field(default_factory=lambda: Path("output"))
    model: str = "latin-evalatin24-240520"
    adverb_threshold: int = 200
    common_adverbs_path: Path = field(
        default_factory=lambda: Path("common_adverbs.txt")
    )
    cache_dir: Path = field(default_factory=lambda: Path("udpipe_cache"))
    unsafe_certs_ok: bool = True

    def __post_init__(self) -> None:
        """Convert string paths to Path objects if needed."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.common_adverbs_path, str):
            self.common_adverbs_path = Path(self.common_adverbs_path)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        # Default cache_dir to output_dir / "udpipe_cache" if still at default
        if self.cache_dir == Path("udpipe_cache") and self.output_dir != Path("output"):
            self.cache_dir = self.output_dir / "udpipe_cache"
        # Default cache_dir to output_dir / "udpipe_cache"
        if self.cache_dir is None:
            self.cache_dir = self.output_dir / "udpipe_cache"


@dataclass
class PipelineResult:
    """Result from running the full pipeline.

    Attributes:
        output_files: List of paths to generated output files.
        sentences_processed: Number of sentences processed.
        uv_replacements: Number of UV/IJ replacements made.
        adverbs_found: Number of adverbs identified.
        cache_hits: Number of times cached responses were used.

    """

    output_files: list[Path] = field(default_factory=list)
    sentences_processed: int = 0
    uv_replacements: int = 0
    adverbs_found: int = 0
    cache_hits: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage1Result:
    """Result from running stage 1 of the pipeline.

    Attributes:
        adverb_counts: Aggregated adverb counts across all files.
        sentences_per_file: Mapping of input path to sentence count.
        common_adverbs_path: Path where the adverb list was written.
    """

    adverb_counts: Counter[str] = field(default_factory=Counter)
    sentences_per_file: dict[Path, int] = field(default_factory=dict)
    common_adverbs_path: Path = field(
        default_factory=lambda: Path("common_adverbs.txt")
    )


@dataclass
class Stage2Result:
    """Result from running stage 2 of the pipeline.

    Attributes:
        output_files: List of paths to generated masked output files.
        sentences_processed: Number of sentences processed.
        cache_hits: Number of times cached responses were used.
    """

    output_files: list[Path] = field(default_factory=list)
    sentences_processed: int = 0
    cache_hits: int = 0
