"""Whitespace-and-punctuation tokenizer used for every prompt comparison.

Counts are additive across concatenations because whitespace is not a token.
That lets the harness budget tools without rebuilding giant strings.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    return _TOKEN.findall(text)


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def truncate_to_tokens(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    matches = list(_TOKEN.finditer(text))
    if len(matches) <= limit:
        return text
    return text[: matches[limit - 1].end()]
