# latin-masking

A self-contained Python package for Latin text processing: plain text → sentence splitting → adverb generation → -que clitic splitting → UDPipe POS tagging → CoNLL-U parsing → POS masking.

## Installation

```bash
pip install git+https://github.com/bnagy/latin-masking.git
```

Or from source:

```bash
git clone https://github.com/bnagy/latin-masking.git
cd latin-masking
pip install -e ".[dev]"
```

## Quick Start

### Command Line

The CLI tool is `udpipe-mask`. It provides four subcommands:

```bash
# Full pipeline: sentence split → adverb generation → -que split → POS tag → mask
udpipe-mask process input.txt --output ./output --quesplit

# Sentence splitting only
udpipe-mask split-sentences input.txt --output ./output

# Generate an adverb frequency list from a corpus
udpipe-mask generate-adverbs input1.txt input2.txt --output ./output --max 200

# Apply POS masking to pre-tagged text
udpipe-mask mask input.txt --adverbs adverbs.txt --output ./output
```

Use `--model` to specify a UDPipe model (default: `latin-evalatin24-240520`). Use `--regenerate` to bypass the cache.

### Python API

Below is a minimal walkthrough:

```python
from pathlib import Path
from collections import Counter

from latin_masking import process_file_with_cache
from latin_masking.sentences import split_sentences
from latin_masking.conllu import parse_conllu
from latin_masking.adverbs import (
    collect_adverbs,
    normalize_adverb_counts,
    generate_adverb_list,
    save_adverb_list,
)
from latin_masking.clitics import split_que_blacklist
from latin_masking.mask import two_pass_mask

INPUT_DIR = Path("./data")
MODEL = "latin-evalatin24-240520"
ADVERB_THRESHOLD = 200

# --- Step 1: Sentence splitting ---
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
sentences = split_sentences(text)

sentences_file = INPUT_DIR / "input_sentences.txt"
with open(sentences_file, "w", encoding="utf-8") as f:
    for sent in sentences:
        f.write(sent + "\n")

# --- Step 2: Generate common adverbs ---
cache_dir = INPUT_DIR / "udpipe_cache"
cache_dir.mkdir(exist_ok=True)

response = process_file_with_cache(
    sentences_file,
    MODEL,
    cache_dir=cache_dir,
    presegmented=True,
    raw=True,
    unsafe_certs_ok=True,
)
frames, _ = parse_conllu(response)
adverbs = collect_adverbs(frames)

# --- Step 3: Build and save the common adverbs list ---
normalized = normalize_adverb_counts(Counter(adverbs))
top_adverbs = generate_adverb_list(normalized, ADVERB_THRESHOLD)
save_adverb_list(top_adverbs, Path("common_adverbs.txt"))

# --- Step 4: -que splitting ---
from latin_masking.adverbs import load_adverb_list

common_adverbs = load_adverb_list(Path("common_adverbs.txt"), ADVERB_THRESHOLD)
quesplit_text, n = split_que_blacklist(text, common_adverbs=common_adverbs)

quesplit_file = INPUT_DIR / "input_sentences.quesplit.txt"
with open(quesplit_file, "w", encoding="utf-8") as f:
    f.write(quesplit_text)

# --- Step 5: POS tagging and masking ---
response = process_file_with_cache(
    quesplit_file,
    MODEL,
    cache_dir=cache_dir,
    presegmented=True,
    raw=True,
    unsafe_certs_ok=True,
)
frames, _ = parse_conllu(response)
masked = two_pass_mask(frames, common_adverbs=common_adverbs)

with open("output_masked.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(masked))
```

## Pipeline Stages

The pipeline runs in this order:

1. **Sentence Splitting** — Uses spaCy `la_senter` with colon handling; falls back to NLTK Punkt for Latin.
2. **Adverb Generation** — Processes sentences through UDPipe, collects adverbs by frequency, and builds a common-adverbs list. This list is needed in the next step.
3. **-que Splitting** — Splits -que enclitics (e.g. *efficiuntque* → *efficiunt -que*) using the common-adverbs list as a blacklist, so that words like *itaque*, *neque*, *quoque* are left intact.
4. **UDPipe Processing** — Sends the -que-split text to the [UDPipe REST API](https://lindat.mff.cuni.cz/services/udpipe/api) for POS tagging and lemmatization.
5. **CoNLL-U Parsing** — Parses UDPipe output into structured DataFrames.
6. **POS Masking** — Replaces tokens with their POS tags while preserving common adverbs.

## Caching

`process_file_with_cache` automatically caches UDPipe responses to disk as pickle files. The cache filename is derived from a **SHA-256 hash of the file content** and the model name, so the cache remains valid as long as the file content hasn't changed — regardless of modification time.

```python
response = process_file_with_cache(
    Path("input.txt"),
    "latin-evalatin24-240520",
    cache_dir=Path("udpipe_cache"),
    presegmented=True,
    raw=True,
    unsafe_certs_ok=True,
    force_refresh=False,  # set True to bypass cache
)
```

Cache files are stored in the specified `cache_dir` with names like `input_<hash>.pkl`. To invalidate, either delete the cache file or use `force_refresh=True`.

## Output Format

Masked output is one sentence per line, tokens separated by spaces. Real examples:

| Original | Masked |
|---|---|
| *Cum nequeat Ionathas iram mulcere paternam.* | `cum VERB PROPN NOUN VERB ADJ` |
| *Vnda tegit terram, tegit aera, sic elementa Hec tria miscentur efficiuntque chaos.* | `NOUN VERB NOUN VERB NOUN sic NOUN hec NUM VERB VERB -que NOUN` |


| Token type | Output |
|---|---|
| `NOUN`, `VERB`, `ADJ`, `PROPN`, `NUM`, `AUX` | POS tag (e.g. `NOUN`) |
| `ADV` (in common list) | lowercased word (e.g. `sic`, `unde`) |
| `ADV` (not in common list) | `ADV` |
| Other | lowercased normalized word |

UV/IJ normalization is applied universally (e.g. *Cum* → `cum`, *Deus* → `deus`).

## Development

```bash
pytest tests/ -v --cov=latin_masking
mypy src/latin_masking
```

## License

MIT
