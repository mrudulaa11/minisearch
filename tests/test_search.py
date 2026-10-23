import random

from minisearch.index import InvertedIndex
from minisearch.scoring import tfidf_scores
from minisearch.search import search, top_k

DOCS = [
    ("d1", "the cat sat on the mat"),
    ("d2", "the dog sat"),
    ("d3", "cat cat dog"),
]


def test_top_k_matches_full_sort():
    rng = random.Random(0)
    scores = {f"d{i}": rng.random() for i in range(200)}
    expected = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
    assert top_k(scores, 10) == expected


def test_top_k_orders_best_first():
    scores = {"a": 5.0, "b": 9.0, "c": 1.0, "d": 7.0}
    assert top_k(scores, 2) == [("b", 9.0), ("d", 7.0)]


def test_top_k_larger_than_n_returns_everything_sorted():
    scores = {"a": 1.0, "b": 3.0, "c": 2.0}
    assert top_k(scores, 10) == [("b", 3.0), ("c", 2.0), ("a", 1.0)]


def test_top_k_zero_and_empty():
    assert top_k({"a": 1.0}, 0) == []
    assert top_k({}, 5) == []


def test_top_k_ties_still_return_k_items():
    result = top_k({"a": 1.0, "b": 1.0, "c": 1.0}, 2)
    assert len(result) == 2
    assert all(score == 1.0 for _, score in result)


def test_search_returns_best_document_first():
    index = InvertedIndex.build(DOCS)
    results = search(index, "cat mat", k=3)
    assert results[0][0] == "d1"


def test_search_respects_k():
    index = InvertedIndex.build(DOCS)
    assert len(search(index, "cat", k=1)) == 1


def test_search_accepts_other_scorer():
    index = InvertedIndex.build(DOCS)
    results = search(index, "cat mat", k=3, scorer=tfidf_scores)
    assert results[0][0] == "d1"
