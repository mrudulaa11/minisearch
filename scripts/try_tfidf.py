"""Run a TF-IDF query on SciFact and print the top 5 results."""

import sys

from minisearch.data import load_corpus
from minisearch.index import InvertedIndex
from minisearch.scoring import tfidf_scores

query = " ".join(sys.argv[1:]) or "vitamin d deficiency"

corpus = load_corpus("data/scifact")
index = InvertedIndex.build(corpus.items())
scores = tfidf_scores(index, query)

#sorted() is fine for now; a heap replaces it in a later step
ranked = sorted(scores.items(), key = lambda kv: kv[1], reverse= True)[:5]

print(f"query: {query!r} | {len(scores)} candidate docs out of {index.num_docs}")
for doc_id, score in ranked:
	print(f"{score: 6.2f} {doc_id:>10} {corpus[doc_id][:90]}")
