# latin-masking

A self-contained Python package for Latin text processing pipeline:
plain text → sentence splitting → -que clitic splitting → UDPipe POS tagging →
CoNLL-U parsing → UV/IJ normalization → adverb dictionary generation → POS masking.

## Installation

### From GitHub (recommended)

```bash
pip install git+https://github.com/bnagy/latin-masking.git
```

### From source

```bash
git clone https://github.com/bnagy/latin-masking.git
cd latin-masking
pip install -e ".[dev]"
```

## Quick Start

See `Quickstart.ipynb` for a complete example notebook demonstrating the pipeline.

### Command Line Interface

```bash
# Process text through the full pipeline
latin-mask process input.txt --output ./output

# Split text into sentences
latin-mask split-sentences input.txt --output ./output

# Split -que enclitics
latin-mask split-que input.txt --que-words que_words.txt

# Generate adverb list from input
latin-mask generate-adverbs input.txt --output ./output --max 200

# Apply masking to pre-segmented input
latin-mask mask input.txt --adverbs adverbs.txt --output ./output
```

### Python API

```python
from pathlib import Path
from latin_masking import process_file_with_cache
from latin_masking.sentences import split_sentences
from latin_masking.conllu import parse_conllu
from latin_masking.adverbs import (
    collect_adverbs,
    normalize_adverb_counts,
    generate_adverb_list,
)
from latin_masking.clitics import split_que_blacklist
from latin_masking.mask import two_pass_mask

# Split text into sentences
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
sentences = split_sentences(text)

# Process with UDPipe and caching
cache_dir = Path("udpipe_cache")
response = process_file_with_cache(
    Path("sentences.txt"),
    "latin-evalatin24-240520",
    cache_dir=cache_dir,
    presegmented=True,
    raw=True,
)

# Parse and collect adverbs
frames, _ = parse_conllu(response)
adverbs = collect_adverbs(frames)
```

## Pipeline Stages

1. **Sentence Splitting**: Uses spaCy `la_senter` with colon handling, falls back to NLTK Punkt for Latin.

2. **-que Splitting**: Optional splitting of -que enclitics (e.g., "etiamque" → "etiam -que").

3. **UDPipe Processing**: Sends text to UDPipe REST API for POS tagging and parsing.

4. **CoNLL-U Parsing**: Parses UDPipe output into structured DataFrames.

5. **UV/IJ Normalization**: Normalizes variant spellings (v↔u, i↔j).

6. **Adverb Generation**: Collects and ranks adverbs by frequency.

7. **POS Masking**: Two-pass masking algorithm that preserves common adverbs and normalizes variants.

## Configuration

### Default Model

The default UDPipe model is `latin-evalatin24-240520`. You can specify a different model with the `--model` flag.

### Caching

The `process_file_with_cache` function automatically caches UDPipe responses:

```python
response = process_file_with_cache(
    input_path,
    "latin-evalatin24-240520",
    cache_dir=Path("cache"),
    force_refresh=False,  # Set True to bypass cache
    presegmented=True,
    raw=True,
)
```

- Cache files are named using a hash of the input path and model name
- Cache is invalidated when the input file is newer than the cache file
- Use `force_refresh=True` to re-process even with valid cache

## Output Format

The masked output format is one sentence per line, with tokens separated by spaces:

```
PROPN VERB ADP NOUN
ADV NOUN VERB ADV
```

- `NOUN`, `VERB`, `ADJ`, `PROPN`, `NUM`, `AUX` → replaced with POS tag
- `ADV` → lowercased word (if in common adverbs) or `ADV`
- Other tokens → lowercased normalized word

## Troubleshooting

### SSL Certificate Errors

If you encounter SSL certificate errors when connecting to the UDPipe API:

```bash
# The CLI accepts --unsafe-certs-ok by default
latin-mask process input.txt --output ./output --unsafe-certs-ok

# Or in Python, pass unsafe_certs_ok=True
response = process_file_with_cache(
    input_path,
    "latin-evalatin24-240520",
    unsafe_certs_ok=True,
    ...
)
```

### Model Not Found

If you get a "model not found" error, verify the model name:

```bash
# List available models
latin-mask generate-adverbs input.txt --model latin-evalatin24-240520
```

### Cache Issues

If you suspect cache corruption:

```bash
# Force regeneration
latin-mask process input.txt --output ./output --regenerate

# Or delete the cache directory
rm -rf udpipe_cache/
```

## Development

Run tests:

```bash
pytest tests/ -v --cov=latin_masking
```

Type checking:

```bash
mypy src/latin_masking
```

## License

MIT License