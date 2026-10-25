import random

import pytest

from minisearch.index import InvertedIndex
from minisearch.naive import NaiveSearcher
from minisearch.scoring import bm25_scores

DOCS = [
    ("d1", "the cat sat on the mat"),
    ("d2", "the dog sat"),
    ("d3", "cat cat dog"),
]

def assert_same_scores(a: dict[str, float], b: dict[str, float]) -> None:
    assert set(a) == set(b)
    for doc_id in a:
        assert a[doc_id] == pytest.approx(b[doc_id])
        
def test_naive_matches_indexed_bm25_on_toy_corpus():
    index = InvertedIndex.build(DOCS)
    naive = NaiveSearcher(DOCS)
    for query in ["cat mat", "dog sat", "zebra", "mat zebra"]:
        assert_same_scores(naive.bm25_scores(query), bm25_scores(index, query))

def test_naive_matches_indexed_bm25_on_random_corpus():
    rng = random.Random(1)
    vocab = [f"w{i}" for i in range(50)]
    docs = [(f"d{i}", " ".join(rng.choices(vocab, k=rng.randint(5, 30)))) for i in range(100)]
    index = InvertedIndex.build(docs)
    naive = NaiveSearcher(docs)
    for _ in range(20):
        query = " ".join(rng.sample(vocab, 3))
        assert_same_scores(naive.bm25_scores(query), bm25_scores(index, query))
        
def test_naive_empty_and_stopword_queries():
    naive = NaiveSearcher(DOCS)
    assert naive.bm25_scores("") == {}
    assert naive.bm25_scores("the and of") == {}
    
    