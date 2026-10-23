"""Loaders for BEIR-format datasets."""

import csv
import json
from pathlib import Path

def load_corpus(dataset_dir: str | Path) -> dict[str, str]:
    """Return {doc_id: "title text"} from corpus.jsonl"""
    corpus : dict[str, str] = {} 
    with open(Path(dataset_dir) / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            corpus[row["_id"]] = f"{row.get('title', '')} {row.get('text', '')}".strip()
    return corpus
        
def load_queries(dataset_dir: str | Path) -> dict[str, str]:
    """Return {query_id: query text} from queries.jsonl"""
    queries: dict[str, str] = {}
    with open(Path(dataset_dir)/ "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            queries[row["_id"]] = row["text"]
    return queries

def load_qrels(dataset_dir: str | Path, split: str = "test") -> dict[str, dict[str, int]]:
    """Return {query_id: {doc_id: relevance}} from qrels/<split>.tsv"""
    qrels: dict[str, dict[str,int]] = {}
    path = Path(dataset_dir)/"qrels"/f"{split}.tsv"
    with open(path, encoding="utf-8", newline="")as f:
        reader= csv.reader(f, delimiter="\t")
        next(reader) #skip header row
        for query_id, doc_id, score in reader:
            qrels.setdefault(query_id, {})[doc_id] = int(score)
    return qrels

