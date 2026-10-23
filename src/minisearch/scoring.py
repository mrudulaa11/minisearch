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

def bm25_scores( index: InvertedIndex, query: str, k1: float = 1.5, b: float =0.75) -> dict[str, float]:
	"""Okapi BM25 over every document containing at least one query term.
	per term: idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doclen / avgd1))
	idf = ln((N - df + 0.5) / (df + 0.5) + 1) 
	"""
	scores: dict[str, float] = {}
	n_docs = index.num_docs
	avgdl = index.avg_doc_length 

	for term in set(tokenize(query)):
		df = index.doc_freq(term)
		if df == 0:
			continue
		idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
		for doc_id, tf in index.get_postings(term).items():
			norm = 1 - b + b * index.doc_length(doc_id) / avgdl
			scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (k1 + 1) / (tf + k1 * norm)
	return scores
