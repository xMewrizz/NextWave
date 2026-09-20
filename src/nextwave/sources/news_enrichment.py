"""Bounded retrieval and verbatim text extraction for discovered news pages."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from enum import Enum, StrEnum
from html.parser import HTMLParser
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from nextwave.contracts import SourceDocument

from .http import HttpResponse, HttpTransport

NEWS_CONTENT_ENRICHER_VERSION = "news-content-enricher-v1"
NEWS_CONNECTORS = frozenset({"mediacloud", "gdelt"})
MAX_NEWS_PAGE_BYTES = 2_000_000
MAX_NEWS_EXCERPT_CHARS = 12_000
MIN_NEWS_TEXT_CHARS = 60
DEFAULT_NEWS_ENRICHMENT_CONCURRENCY = 6


class NewsContentStatus(StrEnum):
    EXISTING_EXCERPT = "existing_excerpt"
    ARTICLE_TEXT = "article_text"
    META_DESCRIPTION = "meta_description"
    TITLE_ONLY = "title_only"


class NewsEnrichmentIssueCode(StrEnum):
    BLOCKED_URL = "blocked_url"
    NETWORK_ERROR = "network_error"
    HTTP_ERROR = "http_error"
    RESPONSE_TOO_LARGE = "response_too_large"
    UNSUPPORTED_CONTENT_TYPE = "unsupported_content_type"
    NO_USABLE_TEXT = "no_usable_text"


@dataclass(frozen=True, slots=True)
class NewsPageText:
    text: str | None
    status: NewsContentStatus
    source: str | None
    truncated: bool


@dataclass(frozen=True, slots=True)
class NewsDocumentEnrichment:
    document: SourceDocument
    status: NewsContentStatus
    issue_code: NewsEnrichmentIssueCode | None
    message: str | None
    content_sha256: str | None
    fetched_bytes: int
    excerpt_source: str | None
    excerpt_truncated: bool
    enricher_version: str = NEWS_CONTENT_ENRICHER_VERSION

    @property
    def candidate_text_available(self) -> bool:
        return self.status in {
            NewsContentStatus.EXISTING_EXCERPT,
            NewsContentStatus.ARTICLE_TEXT,
            NewsContentStatus.META_DESCRIPTION,
        }

    @property
    def evidence_text_available(self) -> bool:
        return self.status is NewsContentStatus.ARTICLE_TEXT

    def to_dict(self) -> dict[str, object]:
        value = _json_value(asdict(self))
        if not isinstance(value, dict):
            raise TypeError("news enrichment must serialize as an object")
        value["status"] = self.status.value
        value["issue_code"] = self.issue_code.value if self.issue_code else None
        return value


class BoundedNewsPageTransport:
    """Fetch HTML with a byte limit and public-address checks on redirects."""

    def __init__(
        self,
        resolve_host: Callable[[str], tuple[str, ...]],
    ) -> None:
        self._resolve_host = resolve_host

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        _validate_public_url(url, self._resolve_host)
        opener = build_opener(_PublicRedirectHandler(self._resolve_host))
        request = Request(url, headers=dict(headers), method="GET")
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                return HttpResponse(
                    status_code=response.status,
                    headers=dict(response.headers.items()),
                    body=response.read(MAX_NEWS_PAGE_BYTES + 1),
                )
        except HTTPError as error:
            return HttpResponse(
                status_code=error.code,
                headers=dict(error.headers.items()) if error.headers is not None else {},
                body=error.read(MAX_NEWS_PAGE_BYTES + 1),
            )


class _PublicRedirectHandler(HTTPRedirectHandler):
    def __init__(self, resolve_host: Callable[[str], tuple[str, ...]]) -> None:
        super().__init__()
        self._resolve_host = resolve_host

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_url(newurl, self._resolve_host)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class NewsDocumentEnricher:
    """Fetch a media URL and attach a bounded verbatim excerpt when possible."""

    def __init__(
        self,
        *,
        transport: HttpTransport | None = None,
        timeout_seconds: float = 10.0,
        resolve_host: Callable[[str], tuple[str, ...]] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._resolve_host = resolve_host or _resolve_host_addresses
        self._transport = transport or BoundedNewsPageTransport(self._resolve_host)
        self._timeout_seconds = timeout_seconds

    def enrich(self, document: SourceDocument) -> NewsDocumentEnrichment:
        if document.connector_id not in NEWS_CONNECTORS:
            raise ValueError("news enrichment accepts only Media Cloud or GDELT documents")
        if document.excerpt is not None:
            return NewsDocumentEnrichment(
                document=document,
                status=NewsContentStatus.EXISTING_EXCERPT,
                issue_code=None,
                message=None,
                content_sha256=None,
                fetched_bytes=0,
                excerpt_source="provider_excerpt",
                excerpt_truncated=False,
            )

        try:
            _validate_public_url(document.url, self._resolve_host)
        except ValueError as error:
            return self._title_only(
                document,
                NewsEnrichmentIssueCode.BLOCKED_URL,
                str(error),
            )
        except OSError as error:
            return self._title_only(
                document,
                NewsEnrichmentIssueCode.NETWORK_ERROR,
                str(error),
            )

        try:
            response = self._transport.get(
                document.url,
                headers={
                    "Accept": "text/html,application/xhtml+xml;q=0.9",
                    "Accept-Encoding": "identity",
                    "User-Agent": "NextWave/0.1 news-content-enricher",
                },
                timeout_seconds=self._timeout_seconds,
            )
        except ValueError as error:
            return self._title_only(
                document,
                NewsEnrichmentIssueCode.BLOCKED_URL,
                str(error),
            )
        except OSError as error:
            return self._title_only(
                document,
                NewsEnrichmentIssueCode.NETWORK_ERROR,
                str(error),
            )

        if response.status_code != 200:
            return self._title_only(
                document,
                NewsEnrichmentIssueCode.HTTP_ERROR,
                f"news page returned HTTP {response.status_code}",
                response=response,
            )
        if len(response.body) > MAX_NEWS_PAGE_BYTES:
            return self._title_only(
                document,
                NewsEnrichmentIssueCode.RESPONSE_TOO_LARGE,
                f"news page exceeds {MAX_NEWS_PAGE_BYTES} bytes",
                response=response,
            )
        content_type = _header(response.headers, "content-type")
        if content_type and not _is_html_content_type(content_type):
            return self._title_only(
                document,
                NewsEnrichmentIssueCode.UNSUPPORTED_CONTENT_TYPE,
                f"unsupported news page content type: {content_type}",
                response=response,
            )

        page_text = extract_news_page_text(response.body, content_type=content_type)
        content_sha256 = hashlib.sha256(response.body).hexdigest()
        if page_text.text is None:
            return self._title_only(
                document,
                NewsEnrichmentIssueCode.NO_USABLE_TEXT,
                "news page did not contain a usable article excerpt",
                response=response,
                content_sha256=content_sha256,
            )
        return NewsDocumentEnrichment(
            document=replace(document, excerpt=page_text.text),
            status=page_text.status,
            issue_code=None,
            message=None,
            content_sha256=content_sha256,
            fetched_bytes=len(response.body),
            excerpt_source=page_text.source,
            excerpt_truncated=page_text.truncated,
        )

    def enrich_many(
        self,
        documents: tuple[SourceDocument, ...],
        *,
        max_concurrency: int = DEFAULT_NEWS_ENRICHMENT_CONCURRENCY,
    ) -> tuple[NewsDocumentEnrichment, ...]:
        """Enrich a bounded document pool concurrently while preserving input order."""

        if not documents:
            return ()
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        document_ids = [document.document_id for document in documents]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("documents must contain unique document_id values")
        if any(document.connector_id not in NEWS_CONNECTORS for document in documents):
            raise ValueError("news enrichment accepts only Media Cloud or GDELT documents")

        with ThreadPoolExecutor(max_workers=min(max_concurrency, len(documents))) as pool:
            futures = [pool.submit(self.enrich, document) for document in documents]
            return tuple(future.result() for future in futures)

    def _title_only(
        self,
        document: SourceDocument,
        code: NewsEnrichmentIssueCode,
        message: str,
        *,
        response: HttpResponse | None = None,
        content_sha256: str | None = None,
    ) -> NewsDocumentEnrichment:
        body = response.body if response is not None else b""
        return NewsDocumentEnrichment(
            document=document,
            status=NewsContentStatus.TITLE_ONLY,
            issue_code=code,
            message=message,
            content_sha256=content_sha256,
            fetched_bytes=len(body),
            excerpt_source=None,
            excerpt_truncated=False,
        )


def extract_news_page_text(
    payload: bytes,
    *,
    content_type: str | None = None,
) -> NewsPageText:
    """Extract JSON-LD article body, article paragraphs, or a meta description."""

    html = _decode_html(payload, content_type)
    parser = _NewsHTMLParser()
    parser.feed(html)
    parser.close()

    article_bodies = [
        _normalize_text(text)
        for script in parser.json_ld_scripts
        for text in _json_ld_article_bodies(script)
    ]
    article_bodies = [text for text in article_bodies if len(text) >= MIN_NEWS_TEXT_CHARS]
    if article_bodies:
        return _bounded_page_text(
            max(article_bodies, key=len),
            status=NewsContentStatus.ARTICLE_TEXT,
            source="json_ld.articleBody",
        )

    article_paragraphs = _deduplicate_texts(
        normalized
        for text, inside_article in parser.paragraphs
        if inside_article
        and len(normalized := _normalize_text(text)) >= MIN_NEWS_TEXT_CHARS
    )
    if article_paragraphs:
        return _bounded_page_text(
            "\n\n".join(article_paragraphs),
            status=NewsContentStatus.ARTICLE_TEXT,
            source="html.article_paragraphs",
        )

    paragraphs = _deduplicate_texts(
        normalized
        for text, _ in parser.paragraphs
        if len(normalized := _normalize_text(text)) >= MIN_NEWS_TEXT_CHARS
    )
    paragraph_text = "\n\n".join(paragraphs)
    if len(paragraphs) >= 2 and len(paragraph_text) >= 240:
        return _bounded_page_text(
            paragraph_text,
            status=NewsContentStatus.ARTICLE_TEXT,
            source="html.paragraphs",
        )

    descriptions = _deduplicate_texts(
        _normalize_text(text)
        for text in parser.meta_descriptions
        if len(_normalize_text(text)) >= MIN_NEWS_TEXT_CHARS
    )
    if descriptions:
        return _bounded_page_text(
            max(descriptions, key=len),
            status=NewsContentStatus.META_DESCRIPTION,
            source="html.meta_description",
        )
    return NewsPageText(
        text=None,
        status=NewsContentStatus.TITLE_ONLY,
        source=None,
        truncated=False,
    )


class _NewsHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta_descriptions: list[str] = []
        self.paragraphs: list[tuple[str, bool]] = []
        self.json_ld_scripts: list[str] = []
        self._paragraph: list[str] | None = None
        self._paragraph_in_article = False
        self._json_ld: list[str] | None = None
        self._ignored_depth = 0
        self._article_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = {key.casefold(): value for key, value in attrs}
        normalized_tag = tag.casefold()
        if normalized_tag in {"style", "noscript", "nav", "footer", "aside"}:
            self._ignored_depth += 1
        if normalized_tag == "article":
            self._article_depth += 1
        if normalized_tag == "meta":
            name = (attributes.get("name") or attributes.get("property") or "").casefold()
            content = attributes.get("content")
            if name in {"description", "og:description", "twitter:description"} and content:
                self.meta_descriptions.append(content)
        if normalized_tag == "script":
            script_type = (attributes.get("type") or "").split(";", 1)[0].strip().casefold()
            if script_type == "application/ld+json":
                self._json_ld = []
        elif normalized_tag == "p" and self._ignored_depth == 0:
            self._paragraph = []
            self._paragraph_in_article = self._article_depth > 0

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag == "p" and self._paragraph is not None:
            self.paragraphs.append(("".join(self._paragraph), self._paragraph_in_article))
            self._paragraph = None
        elif normalized_tag == "script" and self._json_ld is not None:
            self.json_ld_scripts.append("".join(self._json_ld))
            self._json_ld = None
        if normalized_tag in {"style", "noscript", "nav", "footer", "aside"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        if normalized_tag == "article":
            self._article_depth = max(0, self._article_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._json_ld is not None:
            self._json_ld.append(data)
        elif self._paragraph is not None and self._ignored_depth == 0:
            self._paragraph.append(data)


def _json_ld_article_bodies(script: str) -> tuple[str, ...]:
    try:
        value = json.loads(script)
    except (json.JSONDecodeError, TypeError):
        return ()
    result: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, dict):
            body = item.get("articleBody")
            if isinstance(body, str) and body.strip():
                result.append(body)
            for nested in item.values():
                if isinstance(nested, dict | list):
                    visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(result)


def _bounded_page_text(
    text: str,
    *,
    status: NewsContentStatus,
    source: str,
) -> NewsPageText:
    limited = text[:MAX_NEWS_EXCERPT_CHARS].rstrip()
    return NewsPageText(
        text=limited,
        status=status,
        source=source,
        truncated=len(limited) < len(text),
    )


def _decode_html(payload: bytes, content_type: str | None) -> str:
    declared = _charset_from_content_type(content_type)
    if declared is None:
        match = re.search(br"charset\s*=\s*['\"]?([a-zA-Z0-9._-]+)", payload[:4096], re.I)
        declared = match.group(1).decode("ascii", errors="ignore") if match else None
    encodings = tuple(
        dict.fromkeys(
            value for value in (declared, "utf-8", "windows-1251") if value
        )
    )
    for encoding in encodings:
        try:
            return payload.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("utf-8", errors="replace")


def _charset_from_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None
    match = re.search(r"charset\s*=\s*([^;\s]+)", content_type, re.I)
    return match.group(1).strip("'\"") if match else None


def _is_html_content_type(content_type: str) -> bool:
    media_type = content_type.split(";", 1)[0].strip().casefold()
    return media_type in {"text/html", "application/xhtml+xml"}


def _header(headers: Mapping[str, str], name: str) -> str | None:
    normalized_name = name.casefold()
    for key, value in headers.items():
        if key.casefold() == normalized_name:
            return value
    return None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _deduplicate_texts(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.casefold()
        if normalized not in seen:
            seen.add(normalized)
            result.append(value)
    return tuple(result)


def _validate_public_url(
    url: str,
    resolve_host: Callable[[str], tuple[str, ...]],
) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("news URL must be absolute HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("news URL must not contain credentials")
    addresses = resolve_host(parsed.hostname)
    if not addresses:
        raise ValueError("news URL hostname did not resolve")
    for address in addresses:
        if not ipaddress.ip_address(address).is_global:
            raise ValueError("news URL resolves to a non-public address")


def _resolve_host_addresses(hostname: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                item[4][0]
                for item in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
            }
        )
    )


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value
