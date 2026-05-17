"""latin-masking: A self-contained Python package for Latin text processing pipeline.

This package provides tools for:
- Sentence splitting (spaCy la_senter + NLTK fallback)
- UDPipe REST API client for Latin POS tagging
- CoNLL-U parsing
- UV/IJ normalization
- Adverb dictionary generation
- POS masking
"""

from udpipe_masking.types import (
    MaskingConfig,
    PipelineResult,
    UDPipeAPIError,
    UDPipeError,
    UDPipeInputError,
    UDPipeParseError,
)

__version__ = "0.1.0"
__all__ = [
    "MaskingConfig",
    "PipelineResult",
    "UDPipeAPIError",
    "UDPipeError",
    "UDPipeInputError",
    "UDPipeParseError",
]
