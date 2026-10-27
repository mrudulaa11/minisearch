# minisearch

![tests](https://github.com/mrudulaa11/minisearch/actions/workflows/tests.yml/badge.svg)

A search engine built from scratch in Python: inverted index, TF-IDF and BM25 ranking, and heap-based top-k retrieval, evaluated on the BEIR SciFact benchmark.

## Results

BEIR SciFact: 5,183 documents, 300 test queries. Measured on MACHINE_PLACEHOLDER, Python 3.12, Ubuntu. Timings vary a few percent between runs.

### Ranking quality

| Ranker | nDCG@10 | MRR@10 |
|---|---|---|
| TF-IDF | 0.6408 | 0.6094 |
| BM25 (k1=1.5, b=0.75) | 0.6663 | 0.6353 |

I grid-searched k1 and b (16 settings) on the 809 train queries. The best setting gave no gain on the test queries (nDCG@10 0.6605), so the standard defaults are kept.

### Speed (single-threaded, pure Python, top 10)

| Searcher | p50 | p95 |
|---|---|---|
| Inverted index + BM25 | 1.05 ms | 2.5 ms |
| Full-scan baseline | 18.5 ms | 31.4 ms |

That is 17.6x lower median latency. The baseline computes identical BM25 scores (checked by tests) and keeps documents pre-tokenized in memory; it just scans every document for every query.

Index build time is under 1 s and the index uses about 22 MB.

Selecting the top 10 from about 2,000 scored candidates: custom heap 140 µs, `sorted()` 391 µs, `heapq.nlargest` 156 µs. This is specific to CPython and these candidate set sizes.

## How it works

1. **Tokenizer** lowercases text, splits on non-alphanumeric characters and drops a small stopword list.
2. **Inverted index** maps each term to `{doc_id: term frequency}`, and stores document lengths, so a query only touches documents containing its terms.
3. **Scoring**: TF-IDF `(1 + ln tf) * ln(N/df)` as a baseline, and Okapi BM25 (term-frequency saturation and document-length normalization).
4. **Top-k** keeps a size-k min-heap, O(n log k).
5. **Evaluation** computes nDCG@10 and MRR@10 against the BEIR relevance labels.

## Usage

```bash
git clone https://github.com/mrudulaa11/minisearch.git
cd minisearch
python3 -m venv .venv && source .venv/bin/activate
pip install -e . pytest

mkdir -p data && cd data
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
unzip scifact.zip && rm scifact.zip && cd ..

python -m minisearch.search vitamin d deficiency -k 5
python scripts/evaluate.py
python scripts/benchmark.py
pytest -q
```

## Project layout

- `src/minisearch/`: tokenizer, index, scoring, search, data loaders, naive baseline, metrics
- `tests/`: unit tests (none depend on the downloaded dataset)
- `scripts/`: evaluation, tuning and benchmark scripts
- `results/`: saved evaluation, tuning and benchmark outputs

## Limitations

- Postings are plain Python dicts, not compressed sorted lists, so memory use is high and phrase or boolean queries are not supported.
- No stemming ("vaccine" and "vaccines" are different terms); apostrophes split words; single-letter tokens are kept; the stopword list is small.
- Index lives in memory only, with no persistence, and queries are single-threaded.
- Keyword matching only, with no semantic understanding.
- Evaluated on one small dataset; the speed-up is against a baseline that is also pure Python.

## Ideas for improvement

Stemming, a persistent compressed index, dynamic-pruning algorithms such as WAND, evaluation on larger BEIR datasets, and a hybrid with dense retrieval.
EOF

CPU=$(lscpu | grep 'Model name' | sed 's/.*: *//')
RAM=$(free -g | awk '/Mem:/ {print $2}')
sed -i "s|MACHINE_PLACEHOLDER|${CPU}, ${RAM} GB RAM|" README.md
grep -n "Measured on" README.md     # check the machine line looks right

mkdir -p .github/workflows
cat > .github/workflows/tests.yml << 'EOF'
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e . pytest
      - run: pytest -q