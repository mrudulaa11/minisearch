"""Build the index on SciFact and print basic statistics."""

import time

from minisearch.data import load_corpus
from minisearch.index import InvertedIndex

corpus = load_corpus("data/scifact")

start = time.perf_counter()
index = InvertedIndex.build(corpus.items())
elapsed = time.perf_counter() - start

print(f"docs: {index.num_docs}")
print(f"unique terms: {len(index.postings)}")
print(f"avg doc length: {index.avg_doc_length:.1f} tokens")
print(f"build time: {elapsed:.2f}s")

top = sorted(index.postings, key=index.doc_freq, reverse=True)[:10]
print("most common terms:", [(t, index.doc_freq(t)) for t in top])
