"""Inverted index: term-> {doc_id: term frequency}"""

from collections import Counter
from collections.abc import Iterable
from minisearch.tokenizer import tokenize

class InvertedIndex:
    def __init__(self):
        #term -> {doc_id: how many times the term appears in that doc}
        self.postings: dict[str, dict[str, int]] = {}
        #doc_id -> number of indexed terms in that doc
        self.doc_lengths: dict[str, int] = {}
        self._total_length = 0
        
    @classmethod
    def build(cls, docs: Iterable[tuple[str,str]]) -> "InvertedIndex":
        """Building an index from (doc_id, text) pairs"""
        index = cls()
        for doc_id, text in docs:
            index.add_document(doc_id, text)
        return index
        
    def add_document(self, doc_id: str, text: str) -> None:
        if doc_id in self.doc_lengths:
            raise ValueError(f"Duplicate document id: {doc_id!r}")
            
        tokens = tokenize(text)
        for term, tf in Counter(tokens).items():
            self.postings.setdefault(term, {})[doc_id]= tf
            
        self.doc_lengths[doc_id] = len(tokens)
        self._total_length += len(tokens)
            
    @property
    def num_docs(self) -> int: 
        return len(self.doc_lengths)
        
    @property 
    def avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return self._total_length / len(self.doc_lengths)
        
    def doc_freq(self, term: str) -> int:
        """Number of documents containing the 'term'."""
        return len(self.postings.get(term, {}))
        
    def get_postings(self, term: str) -> dict[str, int]:
        """{doc_id: tf} for `term`; empty dict if the term is unknown."""
        return self.postings.get(term, {})
        
    def doc_length(self, doc_id: str) -> int:
        return self.doc_lengths[doc_id]
        