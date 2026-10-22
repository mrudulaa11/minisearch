import pytest

from minisearch.index import InvertedIndex

DOCS= [
    ("d1", "the cat sat on the mat"),
    ("d2", "the dog sat"),
    ("d3", "cat cat dog"),
]

def make_index() -> InvertedIndex:
    return InvertedIndex.build(DOCS)

def test_add_single_document():
    index = InvertedIndex()
    index.add_document("d1", "cat sat")
    assert index.get_postings("cat") == {"d1": 1}
    assert index.doc_length("d1") == 2
    
def test_term_frequency_and_postings():
    index = make_index()
    assert index.get_postings("cat") == {"d1":1, "d3": 2}
    assert index.get_postings("sat") == {"d1":1, "d2": 1}
    
def test_doc_freq():
    index = make_index()
    assert index.doc_freq('cat') == 2
    assert index.doc_freq("mat") == 1
    
def test_num_docs_and_avg_length():
    index = make_index()
    assert index.num_docs == 3
    assert index.avg_doc_length == pytest.approx(8/3)

def test_missing_term():
    index = make_index()
    assert index.get_postings("zebra") == {}
    assert index.doc_freq("zebra") == 0
    
def test_duplicate_doc_id_raises():
    index = make_index()
    with pytest.raises(ValueError):
        index.add_document("d1", "anything")
        
def test_empty_document_still_counts():
    index = InvertedIndex()
    index.add_document("e", "the and") #all stopwords
    assert index.num_docs == 1
    assert index.doc_length("e") == 0 
    assert index.avg_doc_length == 0.0
    
def test_stopwords_not_indexed():
    index = make_index()
    assert index.doc_freq("the") == 0
    
def test_build_from_iterable():
    index = InvertedIndex.build(iter(DOCS)) #works with generators too
    assert index.num_docs == 3
    