"""Grid-search BM25 k1 and b on TRAIN qrels, then report once on TEST."""

import itertools
import json
from functools import partial
from pathlib import Path

from evaluate import evaluate #scripts/evaluate.py (same folder as this script)
from minisearch.data import load_corpus, load_qrels, load_queries
from minisearch.index import InvertedIndex
from minisearch.scoring import bm25_scores

DATA = "data/scifact"
K1_GRID = [0.8, 1.2, 1.5, 2.0]
B_GRID = [0.4, 0.6, 0.75, 0.9]

corpus = load_corpus(DATA)
queries = load_queries(DATA)
train_qrels = load_qrels(DATA, "train")
test_qrels = load_qrels(DATA, "test")
index = InvertedIndex.build(corpus.items())

rows= []
for k1, b in itertools.product(K1_GRID, B_GRID):
    scorer = partial(bm25_scores, k1 = k1, b=b)
    result = evaluate(index, queries, train_qrels, scorer)
    rows.append((result["ndcg@10"], k1, b))
rows.sort(reverse=True)

print(f"Top 5 settings on TRAIN ({len(train_qrels)} queries):")
for ndcg, k1, b in rows[:5]:
    print(f"k1 = {k1:<4} b ={b:<4} nDCG@10={ndcg:.4f}")

_, best_k1, best_b = rows[0]
default_test = evaluate(index, queries, test_qrels, bm25_scores)
tuned_test = evaluate(index, queries, test_qrels, partial(bm25_scores, k1=best_k1, b=best_b))

print(f"\nTEST ({default_test['queries']} queries), reported once:")
print(f" deafult k1=1.5 b=0.75 : nDCG@10={default_test['ndcg@10']:.4f} MRR@10={default_test['mrr@10']:.4f}")
print(f" tuned   k1={best_k1} b={best_b}: nDCG@10={tuned_test['ndcg@10']:.4f} MRR@10={tuned_test['mrr@10']:.4f}")

out = Path("results") / "tuning.json"
out.write_text(json.dumps({
    "grid_train_ndcg": [{"k1": k1, "b": b, "ndcg@10": n} for n, k1, b in rows],
    "best": {"k1": best_k1, "b": best_b},
    "test_default": default_test,
    "test_tuned": tuned_test,
}, indent=2))
print(f"\nsaved to {out}")
