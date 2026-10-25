"""Quick preview: indexed BM25 vs naive full scan on SciFact queries."""

import time

from minisearch.data import load_corpus, load_queries
from minisearch.index import InvertedIndex
from minisearch.naive import NaiveSearcher
from minisearch.scoring import bm25_scores

corpus = load_corpus("data/scifact")
queries = list(load_queries("data/scifact").values())[:200]

index = InvertedIndex.build(corpus.items())
naive = NaiveSearcher(corpus.items())

#sanity: both must return the same set of matching documents
sample = queries[0]
assert set(bm25_scores(index, sample)) == set(naive.bm25_scores(sample))

def ms_per_query(fn) -> float:
    start = time.perf_counter()
    for q in queries:
        fn(q)
    return(time.perf_counter() - start) / len(queries) * 1000

indexed_ms = ms_per_query(lambda q: bm25_scores(index, q))
naive_ms = ms_per_query(naive.bm25_scores)

print(f"queries: {len(queries)} | docs: {index.num_docs}")
print(f"indexed: {indexed_ms:.2f} ms/query")
print(f"naive: {naive_ms:.2f} ms/query")
print(f"speed-up: {naive_ms / indexed_ms:.1f}x")

