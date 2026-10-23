"""Query interface: heap-based top-k retrieval and a small command-line search."""

import argparse
import heapq
from collections.abc import Callable

from minisearch.data import load_corpus
from minisearch.index import InvertedIndex
from minisearch.scoring import bm25_scores, tfidf_scores

Scorer = Callable[[InvertedIndex, str], dict[str, float]]


def top_k(scores: dict[str, float], k: int = 10) -> list[tuple[str, float]]:
    """Return the k highest-scoring (doc_id, score) pairs, best first.

    Keeps a min-heap of size k: O(n log k) time, O(k) extra memory.
    """
    if k <= 0:
        return []

    heap: list[tuple[float, str]] = []   # (score, doc_id); root = weakest of the top k
    for doc_id, score in scores.items():
        if len(heap) < k:
            heapq.heappush(heap, (score, doc_id))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, doc_id))   # pop root + push, in one step

    # best first; ties ordered by doc_id so output is deterministic
    ranked = sorted(heap, key=lambda item: (-item[0], item[1]))
    return [(doc_id, score) for score, doc_id in ranked]


def search(
    index: InvertedIndex, query: str, k: int = 10, scorer: Scorer = bm25_scores
) -> list[tuple[str, float]]:
    """Score all matching documents, then return the top k."""
    return top_k(scorer(index, query), k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the SciFact corpus.")
    parser.add_argument("query", nargs="+", help="query words")
    parser.add_argument("-k", type=int, default=10, help="number of results")
    parser.add_argument("--ranker", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("--data", default="data/scifact", help="dataset folder")
    args = parser.parse_args()

    corpus = load_corpus(args.data)
    index = InvertedIndex.build(corpus.items())
    scorer = bm25_scores if args.ranker == "bm25" else tfidf_scores

    query = " ".join(args.query)
    results = search(index, query, args.k, scorer)

    print(f"query: {query!r} | ranker: {args.ranker} | {len(results)} results")
    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"{rank:>2}. {score:6.2f}  {doc_id:>10}  {corpus[doc_id][:80]}")


if __name__ == "__main__":
    main()
