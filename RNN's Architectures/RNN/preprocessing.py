from functools import lru_cache
from typing import List

import spacy

import helper


@lru_cache(maxsize=1)
def _english_tokenizer():
    return spacy.blank("en").tokenizer


def clean_review(text: str) -> str:
    """Apply the same text cleanup used when preparing the training reviews."""
    if not isinstance(text, str):
        raise TypeError("Review text must be a string.")

    text = helper.remove_html_tags(text)
    text = helper.remove_urls(text)
    text = helper.replace_chatwords(text)
    return text.lower().strip()


def tokenize_review(text: str) -> List[str]:
    return [token.text for token in _english_tokenizer()(clean_review(text))]


