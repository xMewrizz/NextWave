"""Shared lexical admissibility policy for Evidence LLM passages and quotes."""

from __future__ import annotations

import unicodedata


def _tokens(value: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char if char.isalnum() else " " for char in folded).split()


def text_supports_matched_term(text: str, matched_term: str) -> bool:
    """Return whether text locally names enough of the reviewed term.

    A one-token term must occur verbatim after normalization. Longer terms
    require up to three unique matched tokens and one adjacent term bigram in
    its original order.
    """

    term_tokens = _tokens(matched_term)
    text_tokens = _tokens(text)
    if not term_tokens:
        return False
    if len(term_tokens) == 1:
        return term_tokens[0] in text_tokens
    required_matches = min(3, len(frozenset(term_tokens)))
    if len(frozenset(term_tokens) & frozenset(text_tokens)) < required_matches:
        return False
    text_pairs = set(zip(text_tokens, text_tokens[1:], strict=False))
    term_pairs = set(zip(term_tokens, term_tokens[1:], strict=False))
    return bool(text_pairs & term_pairs)
