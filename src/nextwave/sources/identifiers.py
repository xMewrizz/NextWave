"""Shared normalization for persistent source identifiers."""

from __future__ import annotations

import re
from urllib.parse import quote

_DOI = re.compile(r"10\.[0-9]{4,9}/\S+\Z", re.IGNORECASE)


def normalize_doi(value: object) -> str | None:
    """Return a lowercase bare DOI or None when a source did not provide one."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("doi must be a string or null")
    normalized = value.strip().casefold()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix).strip()
            break
    if not _DOI.fullmatch(normalized):
        raise ValueError("doi has an unsupported format")
    return normalized


def doi_url(doi: str) -> str:
    """Build the canonical HTTPS resolver URL for a normalized DOI."""

    normalized = normalize_doi(doi)
    if normalized is None:
        raise ValueError("doi must not be null")
    return f"https://doi.org/{quote(normalized, safe='/:;()._-')}"
