"""Evaluate rankers on a BEIR dataset: nDCG@10 and MRR@10."""

import argparse
import json
from pathlib import Path

from minisearch.data import load_corpus, load_qrels, load_queries
from minisearch.index import InvertedIndex
from minisearch.metrics import ndcg_at_k, reciprocal_rank
from minisearch.scoring import bm25_scores, tfidf_scores
from minisearch.search import search

K = 10


def evaluate(index, queries, qrels, scorer, k=K) -> dict[str, float]:
    ndcgs, rrs = [], []
    for query_id, relevant in qrels.items():
        ranked = [doc_id for doc_id, _ in search(index, queries[query_id], k, scorer)]
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
        rrs.append(reciprocal_rank(ranked, relevant, k))
    n = len(ndcgs)
    return {"queries": n, "ndcg@10": sum(ndcgs) / n, "mrr@10": sum(rrs) / n}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rankers with nDCG@10 and MRR@10.")
    parser.add_argument("--data", default="data/scifact")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    corpus = load_corpus(args.data)
    queries = load_queries(args.data)
    qrels = load_qrels(args.data, args.split)
    index = InvertedIndex.build(corpus.items())

    results = {}
    for name, scorer in [("tfidf", tfidf_scores), ("bm25", bm25_scores)]:
        results[name] = evaluate(index, queries, qrels, scorer)

    print(f"{'ranker':<8}{'queries':>8}{'nDCG@10':>10}{'MRR@10':>10}")
    for name, r in results.items():
        print(f"{name:<8}{r['queries']:>8}{r['ndcg@10']:>10.4f}{r['mrr@10']:>10.4f}")

    out = Path("results") / "evaluation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
