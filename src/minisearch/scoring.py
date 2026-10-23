"""Ranking functions: TF-IDF (baseline) and later, BM25."""

import math

from minisearch.index import InvertedIndex
from minisearch.tokenizer import tokenize

def tfidf_scores(index: InvertedIndex, query: str) -> dict[str, float]:         
	"""Score every document containing at least one query term.

       score(d) = sum over query terms t in d of (1 + ln(tf)) * ln(N/df)
	"""
	scores: dict[str, float] = {}
	n_docs = index.num_docs
	
	for term in set(tokenize(query)): 
		df = index.doc_freq(term)
		if df == 0:
			continue
		idf = math.log(n_docs / df)
		for doc_id, tf in index.get_postings(term).items():
			scores[doc_id] = scores.get(doc_id, 0.0) + (1 + math.log(tf)) * idf
	return scores
