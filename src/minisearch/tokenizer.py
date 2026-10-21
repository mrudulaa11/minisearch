"""Text  tokenization for minisearch."""

import re

#unicode aware, underscores and punctuation are separators.
_TOKEN_RE = re.compile(r"[^\W_]+")

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "this", "to", "was", "were", "will", "with",
})

def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lowercase 'text', split it into alphanumeric tokens, optionally drop stopwords.
    Duplicates are kept because term frequency is needed for ranking.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens