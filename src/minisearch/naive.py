"""Naive baseline: BM25 by scanning every document, with no inverted index."""

import math
from collections import Counter
from collections.abc import Iterable 

from minisearch.tokenizer import tokenize


class NaiveSearcher:
    def __init__(self, docs: Iterable[tuple[str,str]]) -> None:
        #doc_id -> Counter of its terms (pre-tokenized once, up front)
        self.doc_terms: dict[str, Counter[str]] = {}
        self.doc_lengths: dict[str,int] = {}
        for doc_id, text in docs:
            tokens= tokenize(text)
            self.doc_terms[doc_id] = Counter(tokens)
            self.doc_lengths[doc_id] = len(tokens)
        total = sum(self.doc_lengths.values())
        self.avg_doc_length = total / len(self.doc_lengths) if self.doc_lengths else 0.0
        
    def bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
        terms = set(tokenize(query))
        n_docs = len(self.doc_terms)
        
        #Pass 1: scan EVERY document to count how many contain each query term
        df = {term: 0 for term in terms}
        for counts in self.doc_terms.values():
            for term in terms:
                if term in counts:
                    df[term] += 1
                    
        #Pass 2: scan EVERY document again to score it
        scores: dict[str, float] = {}
        for doc_id, counts in self.doc_terms.items():
            norm = 1 - b + b * self.doc_lengths[doc_id] / self.avg_doc_length if self.avg_doc_length else 1.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5 ) + 1)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (k1 + 1) / (tf + k1 * norm)
                
        return scores
    