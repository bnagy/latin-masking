# latin-masking

A self-contained Python package for Latin text processing: plain text → normalization → sentence splitting → UDPipe POS tagging → adverb collection → -que clitic splitting → POS masking.

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

The CLI tool is `latin-mask`. It provides four subcommands:

```bash
# Full two-stage pipeline (normalize → sentence-split → UDPipe → adverbs → quesplit → mask)
latin-mask process input.txt --output ./output

# Run only stage 1 (adverb collection) — review common_adverbs.txt before stage 2
latin-mask process input.txt --output ./output --stage1-only

# Generate an adverb frequency list from a corpus (stage 1 only)
latin-mask generate-adverbs input1.txt input2.txt --output ./output

# Apply -que splitting and masking (stage 2 only)
latin-mask mask input.txt --output ./output

# Sentence splitting only
latin-mask split-sentences input.txt --output ./output
```

Use `--model` to specify a UDPipe model (default: `latin-evalatin24-240520`). Use `--regenerate` to bypass the cache. Use `--no-preserve-eol` to disable `<EOL>` token insertion between verse lines.

### Python API

The pipeline has two stages with a manual review point between them:

```python
from pathlib import Path
from latin_masking import run_pipeline_stage1, run_pipeline_stage2
from latin_masking.types import MaskingConfig

INPUT_FILES = [Path("poem1.txt"), Path("poem2.txt")]
OUTPUT_DIR = Path("./output")
MODEL = "latin-evalatin24-240520"

config = MaskingConfig(model=MODEL, cache_dir=OUTPUT_DIR / "udpipe_cache")

# ── Stage 1: Normalize → sentence-split → UDPipe → collect adverbs ──
result1 = run_pipeline_stage1(INPUT_FILES, OUTPUT_DIR, config=config)
print(f"Collected {len(result1.adverb_counts)} unique adverbs")
print(f"Saved to: {result1.common_adverbs_path}")

# 🛑 Review common_adverbs.txt and que_blacklist.txt before proceeding

# ── Stage 2: -que split → UDPipe → mask ──
result2 = run_pipeline_stage2(INPUT_FILES, OUTPUT_DIR, config=config)
print(f"Processed {result2.sentences_processed} sentences")
print(f"Output files: {[f.name for f in result2.output_files]}")
```

### Low-Level API

For finer control, use the individual functions:

```python
from latin_masking import normalize_text, process_file_with_cache
from latin_masking.sentences import split_sentences
from latin_masking.conllu import parse_conllu
from latin_masking.mask import two_pass_mask
from latin_masking.clitics import split_que_blacklist

# Normalize text (UV/IJ + ch/h) — applied automatically by the pipeline
normalized = normalize_text("jam in horto")  # → "iam in horto"

# Sentence splitting
sentences = split_sentences(normalized)

# UDPipe processing with caching
response = process_file_with_cache(
    Path("input.txt"),
    "latin-evalatin24-240520",
    cache_dir=Path("udpipe_cache"),
    presegmented=True,
    raw=True,
)

# CoNLL-U parsing
frames, texts = parse_conllu(response)

# POS masking
masked = two_pass_mask(frames, common_adverbs={"itaque", "namque", "saepe"})
```

## Pipeline Stages

The pipeline runs in two stages with a manual review point:

### Stage 1 — UDPipe + Adverb Collection
1. **Text Normalization** — UV/IJ and ch/h normalization (e.g. *jam* → *iam*, *virumque* → *uirumque*)
2. **Sentence Splitting** — Uses spaCy `la_senter` with colon handling; falls back to NLTK Punkt for Latin
3. **UDPipe Processing** — Sends sentences to the [UDPipe REST API](https://lindat.mff.cuni.cz/services/udpipe/api) for POS tagging
4. **Adverb Collection** — Collects adverbs by frequency across all files, builds a common-adverbs list, writes `common_adverbs.txt`

### 🛑 Manual Review
Review `common_adverbs.txt` and `que_blacklist.txt` before proceeding to stage 2.

### Stage 2 — Quesplit + Masking
5. **-que Splitting** — Splits -que enclitics (e.g. *efficiuntque* → *efficiunt -que*) using the blacklist + common adverbs as protection
6. **UDPipe Processing** — Sends the -que-split text to UDPipe for POS tagging
7. **POS Masking** — Replaces tokens with their POS tags while preserving common adverbs

## Caching

`process_file_with_cache` automatically caches UDPipe responses to disk as pickle files. The cache filename is derived from a **SHA-256 hash of the file content** and the model name, so the cache remains valid as long as the file content hasn't changed — regardless of modification time.

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
