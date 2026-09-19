"""GDELT DOC request planning and one-window retrieval."""

from __future__ import annotations

import calendar
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
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

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_LANGUAGE_NAMES = {
    "ar": "arabic",
    "de": "german",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "pt": "portuguese",
    "ru": "russian",
    "zh": "chinese",
}


def _request_digest(
    query: SourceQuery,
    parameters: list[QueryParameter],
    attempt: int,
) -> str:
    serialized_parameters = "&".join(
        f"{parameter.name}={parameter.value}" for parameter in parameters
    )
    identity = "|".join((query.query_id, serialized_parameters, str(attempt)))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def build_gdelt_request(
    query: SourceQuery,
    *,
    search_text: str,
    max_records: int = 250,
    attempt: int = 1,
) -> ConnectorRequest:
    """Build one bounded GDELT article-list request for an observation window."""

    if search_text not in query.search_texts:
        raise ValueError("search_text must be one of SourceQuery.search_texts")
    if '"' in search_text or not search_text.strip():
        raise ValueError("GDELT search_text must be a non-blank phrase without quotes")
    if not 1 <= max_records <= 250:
        raise ValueError("max_records must be between 1 and 250")
    if query.published_until > _add_months(query.published_from, 3):
        raise ValueError("GDELT article-list window must not exceed three calendar months")

    gdelt_query = f'"{search_text.strip()}" {_language_filter(query.languages)}'
    parameters = [
        QueryParameter("enddatetime", f"{query.published_until:%Y%m%d}235959"),
        QueryParameter("format", "json"),
        QueryParameter("maxrecords", str(max_records)),
        QueryParameter("mode", "artlist"),
        QueryParameter("query", gdelt_query),
        QueryParameter("sort", "hybridrel"),
        QueryParameter("startdatetime", f"{query.published_from:%Y%m%d}000000"),
    ]
    parameters.sort(key=lambda parameter: parameter.name)
    digest = _request_digest(query, parameters, attempt)
    return ConnectorRequest(
        request_id=f"request-gdelt-{digest}",
        query_id=query.query_id,
        connector_id=ConnectorId.GDELT,
        channel=RetrievalChannel.TEXT,
        endpoint=GDELT_DOC_ENDPOINT,
        parameters=tuple(parameters),
        page_index=1,
        attempt=attempt,
    )


def gdelt_request_url(request: ConnectorRequest) -> str:
    if request.connector_id is not ConnectorId.GDELT:
        raise ValueError("request must target GDELT")
    parameters = [(item.name, item.value) for item in request.parameters]
    return f"{request.endpoint}?{urlencode(parameters)}"


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _language_filter(languages: tuple[str, ...]) -> str:
    names: list[str] = []
    for language in languages:
        base_language = language.split("-", 1)[0]
        try:
            names.append(_GDELT_LANGUAGE_NAMES[base_language])
        except KeyError as error:
            raise ValueError(f"GDELT language is not mapped: {language}") from error
    if len(names) == 1:
        return f"sourcelang:{names[0]}"
    options = " OR ".join(f"sourcelang:{name}" for name in names)
    return f"({options})"


def _media_type(headers: Mapping[str, str]) -> str:
    for name, value in headers.items():
        if name.casefold() == "content-type":
            return value
    return "application/octet-stream"


def _record_count(response: HttpResponse) -> int:
    payload = json.loads(response.body.decode("utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("articles"), list):
        raise ValueError("GDELT response must contain an articles list")
    return len(payload["articles"])


class GdeltConnector:
    def __init__(
        self,
        *,
        transport: HttpTransport | None = None,
        timeout_seconds: float = 30.0,
        min_interval_seconds: float = 5.0,
        clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        self._transport = transport or UrllibHttpTransport()
        self._timeout_seconds = timeout_seconds
        self._min_interval_seconds = min_interval_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._last_request_started_at: float | None = None

    def run_window(
        self,
        query: SourceQuery,
        writer: SnapshotWriter,
        *,
        search_text: str,
        max_records: int = 250,
        attempt: int = 1,
    ) -> ConnectorRun:
        request = build_gdelt_request(
            query,
            search_text=search_text,
            max_records=max_records,
            attempt=attempt,
        )
        started_at = self._clock()
        headers = {
            "Accept": "application/json",
            "User-Agent": "NextWave/0.1",
        }
        self._respect_rate_limit()
        try:
            response = self._transport.get(
                gdelt_request_url(request),
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
            returned_records = _record_count(response)
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
                message=f"GDELT returned HTTP {status_code}",
                retryable=status_code in {408, 425, 429} or status_code >= 500,
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
