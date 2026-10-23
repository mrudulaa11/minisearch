import math

import pytest

from minisearch.index import InvertedIndex
from minisearch.scoring import tfidf_scores

DOCS = [
    ("d1", "the cat sat on the mat"),
    ("d2", "the dog sat"),
    ("d3", "cat cat dog"),
]


def make_index() -> InvertedIndex:
    return InvertedIndex.build(DOCS)


def test_only_documents_containing_the_term_are_scored():
    assert set(tfidf_scores(make_index(), "mat")) == {"d1"}


def test_score_value_matches_formula():
    # mat: tf=1, df=1, N=3  ->  (1 + ln 1) * ln(3/1) = ln 3
    scores = tfidf_scores(make_index(), "mat")
    assert scores["d1"] == pytest.approx(math.log(3))


def test_higher_term_frequency_scores_higher():
    scores = tfidf_scores(make_index(), "cat")   # d3 has cat twice, d1 once
    assert scores["d3"] > scores["d1"]


def test_rare_term_outweighs_common_term():
    # d1 matches cat + mat (mat is rare); d3 matches cat twice
    scores = tfidf_scores(make_index(), "cat mat")
    assert scores["d1"] > scores["d3"]
    assert scores["d1"] == pytest.approx(math.log(1.5) + math.log(3))


def test_unknown_term_is_ignored():
    assert tfidf_scores(make_index(), "zebra") == {}
    assert set(tfidf_scores(make_index(), "mat zebra")) == {"d1"}


def test_empty_and_stopword_only_queries():
    assert tfidf_scores(make_index(), "") == {}
    assert tfidf_scores(make_index(), "the and of") == {}


def test_repeated_query_terms_not_double_counted():
    index = make_index()
    assert tfidf_scores(index, "cat cat") == tfidf_scores(index, "cat")


def test_term_in_every_document_has_zero_weight():
    index = InvertedIndex.build([("a", "apple banana"), ("b", "apple cherry")])
    assert tfidf_scores(index, "apple") == {"a": 0.0, "b": 0.0}

