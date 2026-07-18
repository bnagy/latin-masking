"""Shared fixtures for latin-masking tests."""

from __future__ import annotations

from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

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

SAMPLE_CONLLU_RICH_FEATS = """# text = Marcus est in horto.
1\tMarcus\tMarcus\tPROPN\t_\tCase=Nom|Gender=Masc|Number=Sing|InflClass=IndEurO\t2\tnsubj\t_\t_
2\test\tsum\tVERB\t_\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act\t0\troot\t_\t_
3\tin\tin\tADP\t_\t_\t2\tcase\t_\t_
4\thorto\thortus\tNOUN\t_\tCase=Abl|Gender=Masc|Number=Sing|InflClass=IndEurO\t2\tnmod\t_\t_
5\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_
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
def sample_conllu_rich_feats() -> str:
    """Return sample CoNLL-U response with rich morphological features."""
    return SAMPLE_CONLLU_RICH_FEATS


# CoNLL-U where a protected token (<EOL>) is tagged PUNCT by UDPipe. Without
# the fix, the PUNCT strip in parse_conllu would drop it and lose the
# verse-line marker. The <EOL> parent points at a real token (1) so the
# parent_rel computation stays in bounds.
SAMPLE_CONLLU_PROTECTED_PUNCT = """# text = Nondum solis equos declinis mitigat aestas. <EOL>
1	Nondum	Nondum	ADV	_	_	0	root	_	_
2	solis	sol	NOUN	_	_	1	nmod	_	_
3	equos	equus	NOUN	_	_	1	obj	_	_
4	declinis	declinis	ADJ	_	_	1	amod	_	_
5	mitigat	mitigo	VERB	_	_	1	advcl	_	_
6	aestas	aestas	NOUN	_	_	5	nsubj	_	_
7	.	.	PUNCT	_	_	1	punct	_	_
8	<EOL>	_	X	_	_	1	punct	_	_

# text = Quamvis et madidis incumbant prela racemis. <EOL>
1	Quamvis	quamvis	SCONJ	_	_	0	root	_	_
2	et	et	CCONJ	_	_	1	cc	_	_
3	madidis	madidus	ADJ	_	_	1	amod	_	_
4	incumbant	incumbo	VERB	_	_	1	advcl	_	_
5	prela	prelum	NOUN	_	_	4	nsubj	_	_
6	racemis	racemus	NOUN	_	_	4	obl	_	_
7	.	.	PUNCT	_	_	1	punct	_	_
8	<EOL>	_	PUNCT	_	_	1	punct	_	_
"""


@pytest.fixture
def sample_conllu_protected_punct() -> str:
    """Return CoNLL-U where a protected token is tagged PUNCT by UDPipe."""
    return SAMPLE_CONLLU_PROTECTED_PUNCT


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
