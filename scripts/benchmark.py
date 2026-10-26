"""Benchmark minisearch: build time, memory, query latency (indexed vs naive), top-k selection."""

import heapq
import json
import math
import statistics
import time
import tracemalloc
from pathlib import Path

from minisearch.data import load_corpus, load_qrels, load_queries
from minisearch.index import InvertedIndex
from minisearch.naive import NaiveSearcher
from minisearch.scoring import bm25_scores
from minisearch.search import search, top_k

DATA = "data/scifact"
ROUNDS = 3
K = 10

def percentile(values: list[float], p:float) -> float:
    """Linear-interpolated percentile, p in [0,100]."""
    ordered = sorted(values)
    pos = (len(ordered)-1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

def latency_ms(fn, queries: list[str], rounds: int = ROUNDS) -> dict[str, float]: 
    """Per-query latemcy over several rounds, after one unrecorded warm-up round."""
    for q in queries:
        fn(q)
    times = []
    for _ in range(rounds):
        for q in queries:
            start = time.perf_counter()
            fn(q)
            times.append((time.perf_counter() - start) * 1000)
    return{
        "p50_ms": percentile(times, 50),
        "p95_ms": percentile(times, 95),
        "mean_ms": statistics.fmean(times),
    }
    
def build_time_s(corpus: dict[str, str], repeats: int = 3) -> float:
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        InvertedIndex.build(corpus.items())
        runs.append(time.perf_counter() - start)
    return statistics.median(runs)

def index_memory_mb(corpus: dict[str, str]) -> float:
    """Memory allocated by the index structures (the corpus itself is not counted)."""
    tracemalloc.start()
    index = InvertedIndex.build(corpus.items())
    current, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del index
    return current / 1e6

def selection_us(score_dicts: list[dict[str, float]], repeats: int = 20) -> dict[str, float]:
    """Mean microseconds per call to pick the top K from an already-scored dict."""
    candidates = {
        "top_k (my heap)": lambda s: top_k(s, K),
        "sorted()[:k]": lambda s: sorted(s.items(), key=lambda kv: kv[1], reverse=True)[:K],
        "heapq.nlargest": lambda s: heapq.nlargest(K, s.items(), key=lambda kv: kv[1]),
    }
    out = {}
    for name, fn in candidates.items():
        start = time.perf_counter()
        for _ in range(repeats):
            for scores in score_dicts:
                fn(scores)
            elapsed = time.perf_counter() - start
            out[name] = elapsed / (repeats * len(score_dicts)) * 1e6
    return out

def main() -> None:
    corpus = load_corpus(DATA)
    queries_by_id = load_queries(DATA)
    queries = [queries_by_id[qid] for qid in load_qrels(DATA, "test")]
    
    results: dict = {"docs": len(corpus), "queries": len(queries),"k":K}
    results["build_time_s"] = build_time_s(corpus)
    results["index_memory_mb"] = index_memory_mb(corpus)
    
    index = InvertedIndex.build(corpus.items())
    naive = NaiveSearcher(corpus.items())
    
    results["indexed"] = latency_ms(lambda q:search(index, q, K, bm25_scores), queries)
    results["naive"] = latency_ms(lambda q: top_k(naive.bm25_scores(q), K), queries)
    results["speedup_p50"] = results["naive"]["p50_ms"] / results["indexed"]["p50_ms"]
    results["speedup_mean"] = results["naive"]["mean_ms"] / results["indexed"]["mean_ms"]
    
    score_dicts = [bm25_scores(index, q) for q in queries]
    results["avg_candidates"] = statistics.fmean(len(s) for s in score_dicts)
    results["selection_us_per_call"] = selection_us(score_dicts)
    
    print(f"docs: {results['docs']} | queries: {results['queries']} | k: {K}")
    print(f"index build time (median of 3): {results['build_time_s']:.2f} s")
    print(f"index memory: {results['index_memory_mb']:.1f} MB")
    print()
    print(f"{'searcher':<10}{'p50 ms':>10}{'p95 ms':>10}{'mean ms':>10}")
    for name in ("indexed", "naive"):
        r = results[name]
        print(f"{name:<10}{r['p50_ms']:>10.2f}{r['p95_ms']:>10.2f}{r['mean_ms']:>10.2f}")
    print(f"speed-up: {results['speedup_p50']:.1f}x (p50), {results['speedup_mean']:.1f}x (mean)")
    print()
    print(f"top-{K} selection, avg {results['avg_candidates']:.0f} candidates per query (µs per call):")
    for name, us in results["selection_us_per_call"].items():
        print(f"  {name:<18}{us:>8.1f}")

    out = Path("results") / "benchmark.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()