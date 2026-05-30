"""latin-masking: A self-contained Python package for Latin text processing pipeline.

This package provides tools for:
- Text normalization (UV/IJ, ch/h)
- Sentence splitting (spaCy la_senter + NLTK fallback)
- UDPipe REST API client for Latin POS tagging
- CoNLL-U parsing
- Adverb dictionary generation
- POS masking
- Two-stage pipeline with -que splitting
"""

from latin_masking.clitics import (
    get_default_blacklist,
    load_que_blacklist,
    split_que_blacklist,
)
from latin_masking.client import process_file_with_cache
from latin_masking.normalize import normalize_text
from latin_masking.pipeline import run_pipeline_stage1, run_pipeline_stage2
from latin_masking.types import (
    MaskingConfig,
    Stage1Result,
    Stage2Result,
    UDPipeAPIError,
    UDPipeError,
    UDPipeInputError,
    UDPipeParseError,
)

__version__ = "0.2.0"
__all__ = [
    "get_default_blacklist",
    "load_que_blacklist",
    "split_que_blacklist",
    "process_file_with_cache",
    "normalize_text",
    "run_pipeline_stage1",
    "run_pipeline_stage2",
    "MaskingConfig",
    "Stage1Result",
    "Stage2Result",
    "UDPipeAPIError",
    "UDPipeError",
    "UDPipeInputError",
    "UDPipeParseError",
]
