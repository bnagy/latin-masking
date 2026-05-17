"""Shared fixtures for latin-masking tests."""

from __future__ import annotations

from pathlib import Path

import pytest

# Sample CoNLL-U response for testing
SAMPLE_CONLLU = """# text = Marcus est in horto.
1	Marcus	Marcus	PROPN	_	_	2	nsubj	_	_
2	est	sum	VERB	_	_	0	root	_	_
3	in	in	ADP	_	_	2	case	_	_
4	horto	hortus	NOUN	_	_	2	nmod	_	_
5	.	.	PUNCT	_	_	2	punct	_	_

# text = Puella librum legit.
1	Puella	Puella	PROPN	_	_	2	nsubj	_	_
2	librum	liber	ADV	_	_	3	obj	_	_
3	legit	legere	VERB	_	_	0	root	_	_
4	.	.	PUNCT	_	_	3	punct	_	_

# text = -que test with clitic.
1	-que	-que	ADV	_	_	2	discourse	_	_
2	test	test	NOUN	_	_	3	nsubj	_	_
3	with	with	ADP	_	_	2	case	_	_
4	clitic	clitic	NOUN	_	_	3	nmod	_	_
5	.	.	PUNCT	_	_	3	punct	_	_
"""

SAMPLE_SENTENCES = [
    "Marcus est in horto.",
    "Puella librum legit.",
    "Terra est bona.",
    "Aqua et ignis sunt contraria.",
]

SAMPLE_QUE_WORDS = [
    "atque",
    "atque",
    "que",
    "neque",
    "seque",
    "utque",
]


@pytest.fixture
def sample_conllu() -> str:
    """Return sample CoNLL-U response."""
    return SAMPLE_CONLLU


@pytest.fixture
def sample_sentences() -> list[str]:
    """Return sample sentences for testing."""
    return SAMPLE_SENTENCES.copy()


@pytest.fixture
def sample_que_words() -> list[str]:
    """Return sample -que words for testing."""
    return SAMPLE_QUE_WORDS.copy()


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"
