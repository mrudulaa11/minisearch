import json

from minisearch.data import load_corpus, load_qrels, load_queries


def make_dataset(tmp_path):
    (tmp_path / "qrels").mkdir()
    with open(tmp_path / "corpus.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"_id": "d1", "title": "Cats", "text": "Cats sit."}) + "\n")
        f.write(json.dumps({"_id": "d2", "title": "", "text": "Dogs bark."}) + "\n")
    with open(tmp_path / "queries.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"_id": "q1", "text": "why cats sit"}) + "\n")
    with open(tmp_path / "qrels" / "test.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        f.write("q1\td1\t1\n")
        f.write("q1\td2\t0\n")
    return tmp_path


def test_load_corpus_joins_title_and_text(tmp_path):
    corpus = load_corpus(make_dataset(tmp_path))
    assert corpus == {"d1": "Cats Cats sit.", "d2": "Dogs bark."}


def test_load_queries(tmp_path):
    assert load_queries(make_dataset(tmp_path)) == {"q1": "why cats sit"}


def test_load_qrels(tmp_path):
    assert load_qrels(make_dataset(tmp_path)) == {"q1": {"d1": 1, "d2": 0}}
