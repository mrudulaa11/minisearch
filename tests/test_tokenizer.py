from minisearch.tokenizer import tokenize

def test_lowercase_and_splits_on_whitespace():
    assert tokenize("Hello World") == ["hello", "world"]

def test_strips_punctuation():
    assert tokenize("Hello, World!") == ["hello", "world"]
    
def test_removes_stopwords():
    assert tokenize("The quick brown fox jumps over the lazy dog") == ["quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    
def test_keeps_stopwords_when_disabled():
    assert tokenize("the cat", remove_stopwords=False) == ["the", "cat"]
    
def test_keeps_numbers():
    assert tokenize("COVID-19 vaccines") == ["covid", "19", "vaccines"]

def test_hypens_split_words():
    assert tokenize("state-of-the-art") == ["state", "art"]
    
def test_keeps_duplicates_for_term_frequency():
    assert tokenize("cat cat dog") == ["cat", "cat", "dog"]
    
def test_empty_and_degenrate_inputs():
    assert tokenize("") == []
    assert tokenize("!!! ... ???") == []
    assert tokenize("the and of") == []

def test_unicode_letters_survive():
    assert tokenize("Café naïve façade") == ["café", "naïve", "façade"]