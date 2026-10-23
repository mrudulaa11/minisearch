"""Print the top 5 results of TF-IDF and BM25 for the same query."""

import sys

from minisearch.data import load_corpus
from minisearch.index import InvertedIndex
from minisearch.scoring import bm25_scores, tfidf_scores

query = " ".join(sys.argv[1:]) or "vitamin d deficiency"

corpus = load_corpus("data/scifact")
index = InvertedIndex.build(corpus.items())

for name, scorer in [("TF-IDF", tfidf_scores), ("BM25", bm25_scores)]:
    scores = scorer(index, query)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print(f"\n{name}  (query: {query!r})")
    for doc_id, score in ranked:
        print(f"{score:6.2f}  {doc_id:>10}  {corpus[doc_id][:80]}")
