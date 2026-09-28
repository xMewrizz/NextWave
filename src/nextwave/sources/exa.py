"""Deterministic Exa news search with immutable raw snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from math import isfinite
from typing import Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen

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
from .http import HttpResponse
from .snapshots import SnapshotWriter

EXA_SEARCH_ENDPOINT = "https://api.exa.ai/search"


class ExaTransport(Protocol):
    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> HttpResponse: ...


class UrllibExaTransport:
    """Small POST boundary; HTTP error bodies remain available for snapshots."""

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> HttpResponse:
        request = Request(url, headers=dict(headers), data=body, method="POST")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return HttpResponse(
                    status_code=response.status,
                    headers=dict(response.headers.items()),
                    body=response.read(),
                )
        except HTTPError as error:
            return HttpResponse(
                status_code=error.code,
                headers=dict(error.headers.items()) if error.headers is not None else {},
                body=error.read(),
            )


def normalize_exa_api_key(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _timestamp(day, *, end: bool) -> str:
    suffix = "23:59:59.999Z" if end else "00:00:00.000Z"
    return f"{day.isoformat()}T{suffix}"


def build_exa_news_request(
    query: SourceQuery,
    *,
    search_text: str,
    num_results: int = 10,
    attempt: int = 1,
) -> ConnectorRequest:
    """Build a stable Exa news request without credentials in its identity."""

    if search_text not in query.search_texts:
        raise ValueError("search_text must be one of SourceQuery.search_texts")
    if not 1 <= num_results <= 100:
        raise ValueError("num_results must be between 1 and 100")
    parameters = [
        QueryParameter("category", "news"),
        QueryParameter("contents.highlights", "true"),
        QueryParameter("endPublishedDate", _timestamp(query.published_until, end=True)),
        QueryParameter("numResults", str(num_results)),
        QueryParameter("query", search_text.strip()),
        QueryParameter("startPublishedDate", _timestamp(query.published_from, end=False)),
        QueryParameter("type", "auto"),
    ]
    parameters.sort(key=lambda item: item.name)
    identity = "|".join(
        (
            query.query_id,
            EXA_SEARCH_ENDPOINT,
            "&".join(f"{item.name}={item.value}" for item in parameters),
            str(attempt),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return ConnectorRequest(
        request_id=f"request-{digest}",
        query_id=query.query_id,
        connector_id=ConnectorId.EXA,
        channel=RetrievalChannel.TEXT,
        endpoint=EXA_SEARCH_ENDPOINT,
        parameters=tuple(parameters),
        attempt=attempt,
    )


def exa_request_body(request: ConnectorRequest) -> bytes:
    if request.connector_id is not ConnectorId.EXA:
        raise ValueError("request must target Exa")
    values = {item.name: item.value for item in request.parameters}
    expected = {
        "category",
        "contents.highlights",
        "endPublishedDate",
        "numResults",
        "query",
        "startPublishedDate",
        "type",
    }
    if set(values) != expected:
        raise ValueError("Exa request parameters do not match the news contract")
    payload = {
        "query": values["query"],
        "type": values["type"],
        "category": values["category"],
        "numResults": int(values["numResults"]),
        "startPublishedDate": values["startPublishedDate"],
        "endPublishedDate": values["endPublishedDate"],
        "contents": {"highlights": values["contents.highlights"] == "true"},
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class ExaConnector:
    def __init__(
        self,
        *,
        api_key: str,
        transport: ExaTransport | None = None,
        timeout_seconds: float = 30.0,
        clock=None,
    ) -> None:
        normalized = normalize_exa_api_key(api_key)
        if normalized is None:
            raise ValueError("Exa api_key must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._api_key = normalized
        self._transport = transport or UrllibExaTransport()
        self._timeout_seconds = timeout_seconds
        self._clock = clock or (lambda: datetime.now(UTC))

    def run_search(
        self,
        query: SourceQuery,
        writer: SnapshotWriter,
        *,
        search_text: str,
        num_results: int = 10,
        attempt: int = 1,
    ) -> ConnectorRun:
        request = build_exa_news_request(
            query,
            search_text=search_text,
            num_results=num_results,
            attempt=attempt,
        )
        started_at = self._clock()
        try:
            response = self._transport.post_json(
                request.endpoint,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "NextWave/0.1",
                    "x-api-key": self._api_key,
                },
                body=exa_request_body(request),
                timeout_seconds=self._timeout_seconds,
            )
        except Exception as error:
            return self._failed_network_run(request, started_at, type(error).__name__)

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
            returned_records = _record_count(response.body)
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
                    message=f"Exa response is invalid: {type(error).__name__}",
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

    def _failed_network_run(
        self,
        request: ConnectorRequest,
        started_at: datetime,
        error_type: str,
    ) -> ConnectorRun:
        return ConnectorRun(
            run_id=f"run-{request.request_id.removeprefix('request-')}",
            request=request,
            status=ConnectorStatus.FAILED,
            started_at=started_at,
            finished_at=self._clock(),
            http_status=None,
            returned_records=None,
            artifact=None,
            error=ConnectorError(
                code="network_error",
                message=f"Exa request failed ({error_type})",
                retryable=True,
            ),
        )

    @staticmethod
    def _failed_http_run(
        request: ConnectorRequest,
        response: HttpResponse,
        artifact: RawResponseArtifact,
        started_at: datetime,
        finished_at: datetime,
    ) -> ConnectorRun:
        status = response.status_code
        return ConnectorRun(
            run_id=f"run-{request.request_id.removeprefix('request-')}",
            request=request,
            status=ConnectorStatus.FAILED,
            started_at=started_at,
            finished_at=finished_at,
            http_status=status,
            returned_records=None,
            artifact=artifact,
            error=ConnectorError(
                code=f"http_{status}",
                message=f"Exa returned HTTP {status}",
                retryable=status in {408, 425, 429} or status >= 500,
                retry_after_seconds=_retry_after_seconds(response.headers),
            ),
        )


def _record_count(payload: bytes) -> int:
    decoded = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(decoded, dict) or not isinstance(decoded.get("results"), list):
        raise ValueError("Exa response must contain a results list")
    return len(decoded["results"])


def _media_type(headers: Mapping[str, str]) -> str:
    for name, value in headers.items():
        if name.casefold() == "content-type" and value.strip():
            return value.strip()
    return "application/octet-stream"


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    for name, value in headers.items():
        if name.casefold() != "retry-after":
            continue
        try:
            seconds = float(value.strip())
        except (AttributeError, ValueError):
            return None
        return seconds if isfinite(seconds) and seconds >= 0 else None
    return None
