# udpipe-masking

A self-contained Python package for Latin text processing pipeline:
plain text → sentence splitting → -que clitic splitting → UDPipe POS tagging →
CoNLL-U parsing → UV/IJ normalization → adverb dictionary generation → POS masking.

## Installation

```bash
pip install udpipe-masking
```

Or install from source:

```bash
git clone https://github.com/yourname/udpipe-masking.git
cd udpipe-masking
pip install -e ".[dev]"
```

## Quick Start

### Command Line Interface

```bash
# Process text through the full pipeline
udpipe-mask process input.txt --output ./output

# Split text into sentences
udpipe-mask split-sentences input.txt --output ./output

# Split -que enclitics
udpipe-mask split-que input.txt --que-words que_words.txt

# Generate adverb list from input
udpipe-mask generate-adverbs input.txt --output ./output --max 200

# Apply masking to pre-segmented input
udpipe-mask mask input.txt --adverbs adverbs.txt --output ./output
```

### Python API

```python
from pathlib import Path
from udpipe_masking import MaskingConfig
from udpipe_masking.pipeline import run_pipeline

# Configure the pipeline
config = MaskingConfig(
    model="latin-ittb-ud-2.5-191005",
    adverb_threshold=200,
)

# Run the pipeline
result = run_pipeline(
    [Path("input.txt")],
    Path("output/"),
    config=config,
)

print(f"Processed {result.sentences_processed} sentences")
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

The `MaskingConfig` dataclass controls pipeline behavior:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `latin-ittb-ud-2.5-191005` | UDPipe model name |
| `adverb_threshold` | `200` | Max adverbs to include |
| `common_adverbs_path` | `None` | Path to pre-generated adverb list |
| `replacement_dict_path` | `None` | Path to UV/IJ replacement dictionary |
| `cache_dir` | `~/.cache/udpipe-masking` | Cache directory for API responses |
| `presegmented` | `False` | Input already has one sentence per line |
| `strip_punct` | `True` | Strip punctuation from output |
| `remove_macrons` | `True` | Remove macrons from input |

## Output Format

The masked output format is one sentence per line, with tokens separated by spaces:

```
PROPN VERB ADP NOUN
ADV NOUN VERB ADV
```

- `NOUN`, `VERB`, `ADJ`, `PROPN`, `NUM`, `AUX` → replaced with POS tag
- `ADV` → lowercased word (if in common adverbs) or `ADV`
- Other tokens → lowercased normalized word

## Development

Run tests:

```bash
pytest tests/ -v --cov=udpipe_masking
```

Type checking:

```bash
mypy src/udpipe_masking
```

## License

MIT License