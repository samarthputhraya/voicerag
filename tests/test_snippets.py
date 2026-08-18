"""The citation cleaner, checked against passages captured from the live demo.

Every string in ``REAL_JUNK`` was rendered on screen in the Sources panel during
a real session -- these are regression tests for things a judge actually saw,
not invented fixtures.
"""

from __future__ import annotations

import pytest

from voicerag.snippets import clean_snippet, looks_like_junk

# (raw passage, substring that must survive, substring that must not)
REAL_JUNK = [
    pytest.param(
        "Browse all drug interactions of Gabapentin and Morphine a b c d e f g h "
        "i j k l m n o p q r s t u v w x y z On eHealthMe, Gabapentin "
        "(gabapentin) is often used to treat neuropathy peripheral.",
        "often used to treat neuropathy",
        "d e f g",
        id="alphabet-index-strip",
    ),
    pytest.param(
        "http://www.consumergenetics.com/caffeine_test.php. The lethal dose of "
        "caffeine is 2 grams of caffeine per day, that is, 80-100 cups of coffee.",
        "lethal dose of caffeine",
        "http://",
        id="leading-bare-url",
    ),
    pytest.param(
        "-How much of caffeine is excreted unchanged?-What enzyme(s) break(s) "
        "down caffeine?-What are the metabolites?",
        "What are the metabolites?",
        "?-What",
        id="flattened-question-rail",
    ),
]


@pytest.mark.parametrize("raw,keep,drop", REAL_JUNK)
def test_strips_furniture_but_keeps_the_evidence(raw: str, keep: str, drop: str) -> None:
    cleaned = clean_snippet(raw)
    assert keep in cleaned
    assert drop not in cleaned


def test_prose_is_left_alone() -> None:
    """The common case must be a no-op, or the cleaner is editing evidence."""
    prose = (
        "Gabapentin is in a class of medications called anticonvulsants. It "
        "treats seizures by decreasing excitement in the brain."
    )
    assert clean_snippet(prose) == prose


def test_never_empties_a_passage() -> None:
    """A passage that is *mostly* a URL keeps its original text.

    The guard matters because blanking a citation is a far worse failure than
    showing a scruffy one -- the ``[n]`` marker in the answer would point at an
    empty box.
    """
    mostly_url = "https://example.com/a/very/long/path/that/dominates/the/text x"
    assert clean_snippet(mostly_url).strip() != ""


def test_initials_are_not_mistaken_for_an_index_strip() -> None:
    """Three single letters is a name; six in a row is navigation."""
    kept = "The J. R. R. Tolkien estate confirmed the manuscript is authentic."
    assert clean_snippet(kept) == kept


def test_clip_lands_on_a_word_boundary() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    clipped = clean_snippet(text, max_chars=30)
    assert len(clipped) <= 31  # +1 for the ellipsis
    assert clipped.endswith("\u2026")
    assert not clipped[:-1].endswith(" ")


def test_looks_like_junk_rejects_nav_but_passes_short_prose() -> None:
    assert looks_like_junk("Home | About | Contact")
    assert looks_like_junk("a b c d e f g h")
    # Short but genuinely answers a question -- must survive.
    assert not looks_like_junk("The lethal dose of caffeine is 2 grams per day.")


def test_empty_input_is_safe() -> None:
    assert clean_snippet("") == ""
    assert looks_like_junk("")
