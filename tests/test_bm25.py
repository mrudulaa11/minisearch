import math

import pytest

from minisearch.index import InvertedIndex
from minisearch.scoring import bm25_scores

DOCS = [
    ("d1", "the cat sat on the mat"),   # cat sat mat  (3 tokens)
    ("d2", "the dog sat"),              # dog sat      (2 tokens)
    ("d3", "cat cat dog"),              # cat cat dog  (3 tokens)
]


def make_index() -> InvertedIndex:
    return InvertedIndex.build(DOCS)


def test_only_documents_containing_the_term_are_scored():
    assert set(bm25_scores(make_index(), "mat")) == {"d1"}


def test_score_value_matches_formula():
    # N=3, avgdl=8/3. For "mat" in d1: df=1, tf=1, doclen=3
    idf = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)
    norm = 1 - 0.75 + 0.75 * 3 / (8 / 3)
    expected = idf * 1 * (1.5 + 1) / (1 + 1.5 * norm)
    assert bm25_scores(make_index(), "mat")["d1"] == pytest.approx(expected)


def test_term_frequency_saturates():
    # equal-length docs with 1, 2, 3 occurrences of "cat"
    index = InvertedIndex.build([
        ("a", "cat w1 w2 w3 w4 w5"),
        ("b", "cat cat w1 w2 w3 w4"),
        ("c", "cat cat cat w1 w2 w3"),
        ("x", "other stuff here now ok yes"),
    ])
    s = bm25_scores(index, "cat")
    first_gain = s["b"] - s["a"]
    second_gain = s["c"] - s["b"]
    assert first_gain > second_gain > 0


def test_shorter_document_scores_higher_for_same_tf():
    index = InvertedIndex.build([
        ("short", "cat fox"),
        ("long", "cat fox a1 a2 a3 a4 a5 a6"),
        ("none", "zebra"),
    ])
    s = bm25_scores(index, "cat")
    assert s["short"] > s["long"]


def test_rare_term_outweighs_common_term():
    s = bm25_scores(make_index(), "cat mat")
    assert s["d1"] > s["d3"]


def test_unknown_term_is_ignored():
    assert bm25_scores(make_index(), "zebra") == {}
    assert set(bm25_scores(make_index(), "mat zebra")) == {"d1"}


def test_empty_and_stopword_only_queries():
    assert bm25_scores(make_index(), "") == {}
    assert bm25_scores(make_index(), "the and of") == {}


def test_repeated_query_terms_not_double_counted():
    index = make_index()
    assert bm25_scores(index, "cat cat") == bm25_scores(index, "cat")


def test_term_in_every_document_still_gets_positive_weight():
    index = InvertedIndex.build([("a", "apple banana"), ("b", "apple cherry")])
    scores = bm25_scores(index, "apple")
    assert all(v > 0 for v in scores.values())
