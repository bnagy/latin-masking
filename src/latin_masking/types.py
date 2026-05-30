"""Type definitions, dataclasses, and custom exceptions for latin-masking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Type aliases
Token = str
POSTag = str
MaskedSentence = str
ConlluFrame = pd.DataFrame


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
        model: UDPipe model name (e.g., 'latin-ittb-ud-2.5-191005').
        adverb_threshold: Maximum number of adverbs to include in common adverbs set.
        common_adverbs_path: Path to file containing common adverbs list.
        replacement_dict_path: Path to file containing UV/IJ replacement dictionary.
        cache_dir: Directory for caching UDPipe responses.
        presegmented: Whether input is already pre-segmented.
        strip_punct: Whether to strip punctuation from output.
        remove_macrons: Whether to remove macrons from input text.
        unsafe_certs_ok: Whether to accept self-signed SSL certificates.

    """

    model: str = "latin-evalatin24-240520"
    adverb_threshold: int = 200
    common_adverbs_path: Path = field(
        default_factory=lambda: Path("common_adverbs.txt")
    )
    replacement_dict_path: Path | None = None
    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "latin-masking"
    )
    presegmented: bool = False
    strip_punct: bool = True
    remove_macrons: bool = True
    unsafe_certs_ok: bool = True
    normalize: bool = True
    preserve_eol: bool = True

    def __post_init__(self) -> None:
        """Convert string paths to Path objects if needed."""
        if isinstance(self.common_adverbs_path, str):
            self.common_adverbs_path = Path(self.common_adverbs_path)
        if isinstance(self.replacement_dict_path, str):
            self.replacement_dict_path = Path(self.replacement_dict_path)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)


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
