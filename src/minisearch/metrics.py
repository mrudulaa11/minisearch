"""Ranking qualtiy metrics: nDCG@k and reciprocal rank."""

import math

def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted cumulative gain: sum of rel / log2(rank + 1) over the first k."""
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances[:k], start = 1))

def ndcg_at_k(ranked_doc_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """nDCG@k for one query. `relevant` maps doc_id -> relevance label."""
    gains = [max(relevant.get(doc_id,0),0) for doc_id in ranked_doc_ids[:k]]
    ideal = sorted((rel for rel in relevant.values() if rel > 0), reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(gains, k) / ideal_dcg

def reciprocal_rank(ranked_doc_ids: list[str], relevant: dict[str, int], k: int=10) -> float:
    """1 / rank of the first relevant document within the top k, else 0."""
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0
    