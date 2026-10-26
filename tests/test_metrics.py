import math

import pytest

from minisearch.metrics import ndcg_at_k, reciprocal_rank

def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "x"], {"a": 1, "b":1}) == pytest.approx(1.0)
    
def test_ndcg_nothing_relevant_retrieved_is_zero():
    assert ndcg_at_k(["x", "y", "z"], {"a": 1}) == 0.0
    
def test_ndcg_relevant_at_rabk_two():
    #DCG = 1/log2(3); ideal DCG 1/log2(2) = 1
    assert ndcg_at_k(["x", "a"], {"a": 1}) == pytest.approx(1 / math.log2(3))
    
def test_ndcg_graded_relevance():
    #ranking [b,a] with labels a=2, b=1
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}) == pytest.approx(dcg / ideal)
    
def test_ndcg_ignores_results_beyond_k():
    ranked = [f"junk{i}" for i in range(10)] + ["a"]   #relevant doc at rank 11
    assert ndcg_at_k(ranked, {"a": 1}, k=10) == 0.0
    
def test_ndcg_with_no_relevant_documents_in_qrels_is_zero():
    assert ndcg_at_k(["a", "b"], {"a": 0}) == 0.0
    
def test_reciprocal_rank_first_relevant_at_rank_three():
    assert reciprocal_rank(["x", "y", "a"], {"a": 1}) == pytest.approx(1 / 3)
    
def test_reciprocal_rank_none_found():
    assert reciprocal_rank(["x", "y"], {"a":1}) == 0.0
    
def test_reciprocal_rank_respects_cutoff():
    ranked = [f"junk{i}" for i in range(10)] + ["a"]
    assert reciprocal_rank(ranked, {"a": 1}, k=10) == 0.0
    
def test_reciprocal_rank_ignores_zero_labels():
    #"a" is labeled but not relevant (0), so the first relevant is "b" at rank 2
    assert reciprocal_rank(["a", "b"], {"a": 0, "b": 1}) == pytest.approx(0.5)
    