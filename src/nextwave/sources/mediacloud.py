"""Media Cloud request planning and bounded retrieval."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from math import isfinite
from urllib.parse import urlencode

from .contracts import (
    ConnectorError,
    ConnectorId,
    ConnectorRequest,
    ConnectorRun,
    ConnectorStatus,
    QueryParameter,
    RawResponseArtifact,
    RetrievalChannel,
    SourceQuery,
)
from .http import HttpResponse, HttpTransport, UrllibHttpTransport
from .snapshots import SnapshotWriter

MEDIACLOUD_STORY_LIST_ENDPOINT = (
    "https://search.mediacloud.org/api/search/story-list"
)
MEDIACLOUD_COUNT_OVER_TIME_ENDPOINT = (
    "https://search.mediacloud.org/api/search/count-over-time"
)

_CONTENT_TERM = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)
MEDIACLOUD_PLATFORM = "onlinenews-mediacloud"


def _request_digest(
    query: SourceQuery,
    endpoint: str,
    parameters: list[QueryParameter],
    page_index: int,
    attempt: int,
) -> str:
    serialized_parameters = "&".join(
        f"{parameter.name}={parameter.value}" for parameter in parameters
    )
    identity = "|".join(
        (query.query_id, endpoint, serialized_parameters, str(page_index), str(attempt))
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _collection_parameter(collection_ids: tuple[int, ...]) -> str | None:
    if not collection_ids:
        return None
    if any(isinstance(item, bool) or item <= 0 for item in collection_ids):
        raise ValueError("Media Cloud collection_ids must be positive integers")
    if len(set(collection_ids)) != len(collection_ids):
        raise ValueError("Media Cloud collection_ids must be unique")
    return ",".join(str(item) for item in collection_ids)


def _search_expression(
    query: SourceQuery,
    search_text: str,
    languages: tuple[str, ...] | None = None,
    *,
    safe_syntax: bool = False,
) -> str:
    if search_text not in query.search_texts:
        raise ValueError("search_text must be one of SourceQuery.search_texts")
    normalized = search_text.strip()
    if not normalized or '"' in normalized or "\\" in normalized:
        raise ValueError("Media Cloud search_text must be a plain non-blank phrase")
    words = (
        _CONTENT_TERM.findall(normalized)
        if safe_syntax
        else normalized.split()
    )
    if len(words) <= 2:
        # Short phrases occur verbatim in news; proven live ("fintech solutions").
        subject = f'"{normalized}"'
    else:
        # Long phrases never occur verbatim (proven live: 0 hits); AND the
        # content words instead and let the gate filter downstream. Single
        # characters are prepositions/particles ("в", "и"), never content.
        terms = [word for word in words if len(word) >= 2]
        if not terms:
            raise ValueError("Media Cloud search_text has no content words")
        subject = " AND ".join(terms)
    selected_languages = _selected_languages(query, languages)
    filters = [
        f"language:{language.split('-', 1)[0]}" for language in selected_languages
    ]
    if len(filters) == 1:
        return f"{subject} AND {filters[0]}"
    return f"{subject} AND ({' OR '.join(filters)})"


def _selected_languages(
    query: SourceQuery,
    languages: tuple[str, ...] | None,
) -> tuple[str, ...]:
    selected = query.languages if languages is None else languages
    if not selected:
        raise ValueError("Media Cloud languages must not be empty")
    if len(set(selected)) != len(selected):
        raise ValueError("Media Cloud languages must be unique")
    if any(language not in query.languages for language in selected):
        raise ValueError("Media Cloud languages must come from SourceQuery.languages")
    return selected


def build_mediacloud_story_request(
    query: SourceQuery,
    *,
    search_text: str,
    collection_ids: tuple[int, ...],
    page_size: int = 100,
    pagination_token: str | None = None,
    page_index: int = 1,
    attempt: int = 1,
    languages: tuple[str, ...] | None = None,
    safe_syntax: bool = False,
) -> ConnectorRequest:
    """Build one deterministic page request against selected news collections."""

    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")
    if pagination_token is not None and not pagination_token.strip():
        raise ValueError("pagination_token must not be blank")
    parameters = [
        QueryParameter("end", query.published_until.isoformat()),
        QueryParameter("page_size", str(page_size)),
        QueryParameter("platform", MEDIACLOUD_PLATFORM),
        QueryParameter(
            "q",
            _search_expression(
                query,
                search_text,
                languages,
                safe_syntax=safe_syntax,
            ),
        ),
        QueryParameter("sort_order", "desc"),
        QueryParameter("start", query.published_from.isoformat()),
    ]
    collections = _collection_parameter(collection_ids)
    if collections is not None:
        parameters.append(QueryParameter("cs", collections))
    if pagination_token is not None:
        parameters.append(QueryParameter("pagination_token", pagination_token.strip()))
    parameters.sort(key=lambda parameter: parameter.name)
    digest = _request_digest(
        query,
        MEDIACLOUD_STORY_LIST_ENDPOINT,
        parameters,
        page_index,
        attempt,
    )
    return ConnectorRequest(
        request_id=f"request-mediacloud-{digest}",
        query_id=query.query_id,
        connector_id=ConnectorId.MEDIACLOUD,
        channel=RetrievalChannel.TEXT,
        endpoint=MEDIACLOUD_STORY_LIST_ENDPOINT,
        parameters=tuple(parameters),
        page_index=page_index,
        attempt=attempt,
    )


def build_mediacloud_timeline_request(
    query: SourceQuery,
    *,
    search_text: str,
    collection_ids: tuple[int, ...],
    attempt: int = 1,
    languages: tuple[str, ...] | None = None,
) -> ConnectorRequest:
    """Build a normalized attention request independent from story retrieval."""

    parameters = [
        QueryParameter("end", query.published_until.isoformat()),
        QueryParameter("platform", MEDIACLOUD_PLATFORM),
        QueryParameter("q", _search_expression(query, search_text, languages)),
        QueryParameter("start", query.published_from.isoformat()),
    ]
    collections = _collection_parameter(collection_ids)
    if collections is not None:
        parameters.append(QueryParameter("cs", collections))
    parameters.sort(key=lambda parameter: parameter.name)
    digest = _request_digest(
        query,
        MEDIACLOUD_COUNT_OVER_TIME_ENDPOINT,
        parameters,
        1,
        attempt,
    )
    return ConnectorRequest(
        request_id=f"request-mediacloud-{digest}",
        query_id=query.query_id,
        connector_id=ConnectorId.MEDIACLOUD,
        channel=RetrievalChannel.TEXT,
        endpoint=MEDIACLOUD_COUNT_OVER_TIME_ENDPOINT,
        parameters=tuple(parameters),
        page_index=1,
        attempt=attempt,
    )


def mediacloud_request_url(request: ConnectorRequest) -> str:
    if request.connector_id is not ConnectorId.MEDIACLOUD:
        raise ValueError("request must target Media Cloud")
    parameters = [(item.name, item.value) for item in request.parameters]
    return f"{request.endpoint}?{urlencode(parameters)}"


def _media_type(headers: Mapping[str, str]) -> str:
    for name, value in headers.items():
        if name.casefold() == "content-type":
            return value
    return "application/octet-stream"


def _record_count(response: HttpResponse, request: ConnectorRequest) -> int:
    payload = json.loads(response.body.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Media Cloud response must be an object")
    if request.endpoint == MEDIACLOUD_STORY_LIST_ENDPOINT:
        stories = payload.get("stories")
        if not isinstance(stories, list):
            raise ValueError("Media Cloud story response must contain a stories list")
        token = payload.get("pagination_token")
        if token is not None and not isinstance(token, str):
            raise ValueError("Media Cloud pagination_token must be a string or null")
        return len(stories)
    if request.endpoint == MEDIACLOUD_COUNT_OVER_TIME_ENDPOINT:
        timeline = payload.get("count_over_time")
        if not isinstance(timeline, dict) or not isinstance(timeline.get("counts"), list):
            raise ValueError("Media Cloud timeline response must contain a counts list")
        return len(timeline["counts"])
    raise ValueError("unsupported Media Cloud endpoint")


class MediaCloudConnector:
    def __init__(
        self,
        *,
        api_key: str,
        transport: HttpTransport | None = None,
        timeout_seconds: float = 60.0,
        min_interval_seconds: float = 30.0,
        clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Media Cloud api_key must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        self._api_key = api_key.strip()
        self._transport = transport or UrllibHttpTransport()
        self._timeout_seconds = timeout_seconds
        self._min_interval_seconds = min_interval_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._last_request_started_at: float | None = None

    def run_page(
        self,
        query: SourceQuery,
        writer: SnapshotWriter,
        *,
        search_text: str,
        collection_ids: tuple[int, ...],
        page_size: int = 100,
        pagination_token: str | None = None,
        page_index: int = 1,
        attempt: int = 1,
        languages: tuple[str, ...] | None = None,
        safe_syntax: bool = False,
    ) -> ConnectorRun:
        request = build_mediacloud_story_request(
            query,
            search_text=search_text,
            collection_ids=collection_ids,
            page_size=page_size,
            pagination_token=pagination_token,
            page_index=page_index,
            attempt=attempt,
            languages=languages,
            safe_syntax=safe_syntax,
        )
        return self._run_request(request, writer)

    def run_timeline(
        self,
        query: SourceQuery,
        writer: SnapshotWriter,
        *,
        search_text: str,
        collection_ids: tuple[int, ...],
        attempt: int = 1,
        languages: tuple[str, ...] | None = None,
    ) -> ConnectorRun:
        request = build_mediacloud_timeline_request(
            query,
            search_text=search_text,
            collection_ids=collection_ids,
            attempt=attempt,
            languages=languages,
        )
        return self._run_request(request, writer)

    def _run_request(
        self,
        request: ConnectorRequest,
        writer: SnapshotWriter,
    ) -> ConnectorRun:
        started_at = self._clock()
        headers = {
            "Accept": "application/json",
            "Authorization": f"Token {self._api_key}",
            "User-Agent": "NextWave/0.1",
        }
        self._respect_rate_limit()
        try:
            response = self._transport.get(
                mediacloud_request_url(request),
                headers=headers,
                timeout_seconds=self._timeout_seconds,
            )
        except Exception as error:
            finished_at = self._clock()
            return ConnectorRun(
                run_id=f"run-{request.request_id.removeprefix('request-')}",
                request=request,
                status=ConnectorStatus.FAILED,
                started_at=started_at,
                finished_at=finished_at,
                http_status=None,
                returned_records=None,
                artifact=None,
                error=ConnectorError(
                    code="network_error",
                    message=f"{type(error).__name__}: {error}",
                    retryable=True,
                ),
            )

        finished_at = self._clock()
        artifact = writer.write_response(
            request,
            response.body,
            _media_type(response.headers),
            finished_at,
        )
        if not 200 <= response.status_code <= 299:
            return self._failed_http_run(request, response, artifact, started_at, finished_at)

        try:
            returned_records = _record_count(response, request)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            return ConnectorRun(
                run_id=f"run-{request.request_id.removeprefix('request-')}",
                request=request,
                status=ConnectorStatus.FAILED,
                started_at=started_at,
                finished_at=finished_at,
                http_status=response.status_code,
                returned_records=None,
                artifact=artifact,
                error=ConnectorError(
                    code="invalid_response",
                    message=str(error),
                    retryable=False,
                ),
            )

        return ConnectorRun(
            run_id=f"run-{request.request_id.removeprefix('request-')}",
            request=request,
            status=ConnectorStatus.SUCCESS,
            started_at=started_at,
            finished_at=finished_at,
            http_status=response.status_code,
            returned_records=returned_records,
            artifact=artifact,
        )

    @staticmethod
    def _failed_http_run(
        request: ConnectorRequest,
        response: HttpResponse,
        artifact: RawResponseArtifact,
        started_at: datetime,
        finished_at: datetime,
    ) -> ConnectorRun:
        status_code = response.status_code
        return ConnectorRun(
            run_id=f"run-{request.request_id.removeprefix('request-')}",
            request=request,
            status=ConnectorStatus.FAILED,
            started_at=started_at,
            finished_at=finished_at,
            http_status=status_code,
            returned_records=None,
            artifact=artifact,
            error=ConnectorError(
                code=f"http_{status_code}",
                message=f"Media Cloud returned HTTP {status_code}",
                retryable=status_code in {408, 425, 429} or status_code >= 500,
                retry_after_seconds=_retry_after_seconds(response),
            ),
        )

    def _respect_rate_limit(self) -> None:
        now = self._monotonic_clock()
        if self._last_request_started_at is not None:
            elapsed = now - self._last_request_started_at
            remaining = self._min_interval_seconds - elapsed
            if remaining > 0:
                self._sleeper(remaining)
                now = self._monotonic_clock()
        self._last_request_started_at = now


def _retry_after_seconds(response: HttpResponse) -> float | None:
    """Honor the server's Retry-After header; ignore garbage instead of guessing."""

    for name, value in response.headers.items():
        if name.casefold() != "retry-after":
            continue
        try:
            seconds = float(value.strip())
        except (AttributeError, ValueError):
            return None
        return seconds if isfinite(seconds) and seconds >= 0 else None
    return None
