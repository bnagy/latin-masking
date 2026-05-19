"""latin-masking: A self-contained Python package for Latin text processing pipeline.

This package provides tools for:
- Sentence splitting (spaCy la_senter + NLTK fallback)
- UDPipe REST API client for Latin POS tagging
- CoNLL-U parsing
- UV/IJ normalization
- Adverb dictionary generation
- POS masking
"""

from latin_masking.clitics import (
    get_default_blacklist,
    load_que_blacklist,
    load_que_whitelist,
    load_que_words,
    split_que,
    split_que_blacklist,
)
from latin_masking.client import process_file_with_cache
from latin_masking.types import (
    MaskingConfig,
    PipelineResult,
    UDPipeAPIError,
    UDPipeError,
    UDPipeInputError,
    UDPipeParseError,
)

__version__ = "0.1.0"
__all__ = [
    "get_default_blacklist",
    "load_que_blacklist",
    "load_que_whitelist",
    "load_que_words",
    "split_que",
    "split_que_blacklist",
    "process_file_with_cache",
    "MaskingConfig",
    "PipelineResult",
    "UDPipeAPIError",
    "UDPipeError",
    "UDPipeInputError",
    "UDPipeParseError",
]
