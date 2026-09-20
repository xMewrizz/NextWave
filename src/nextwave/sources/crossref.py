"""Crossref request planning and one-page retrieval."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
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
from .identifiers import normalize_doi
from .snapshots import SnapshotWriter

CROSSREF_WORKS_ENDPOINT = "https://api.crossref.org/v1/works"
CROSSREF_SELECT_FIELDS = (
    "DOI",
    "title",
    "subtitle",
    "URL",
    "published",
    "published-print",
    "published-online",
    "issued",
    "type",
    "author",
    "publisher",
    "container-title",
    "abstract",
    "subject",
    "resource",
)


def _request_digest(
    query: SourceQuery,
    channel: RetrievalChannel,
    parameters: list[QueryParameter],
    attempt: int,
) -> str:
    serialized_parameters = "&".join(
        f"{parameter.name}={parameter.value}" for parameter in parameters
    )
    identity = "|".join((query.query_id, channel.value, serialized_parameters, str(attempt)))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def build_crossref_request(
    query: SourceQuery,
    *,
    channel: RetrievalChannel,
    search_text: str | None = None,
    doi: str | None = None,
    cursor: str = "*",
    page_index: int = 1,
    rows: int = 100,
    attempt: int = 1,
    contact_email: str | None = None,
) -> ConnectorRequest:
    """Translate a provider-independent query into one exact Crossref request."""

    if not 1 <= rows <= 1000:
        raise ValueError("rows must be between 1 and 1000")
    if channel is RetrievalChannel.TEXT:
        if search_text is None or search_text not in query.search_texts:
            raise ValueError("search_text must be one of SourceQuery.search_texts")
        if doi is not None:
            raise ValueError("text retrieval must not contain doi")
        if not cursor.strip():
            raise ValueError("cursor must not be blank")
        parameters = [
            QueryParameter("cursor", cursor),
            QueryParameter(
                "filter",
                ",".join(
                    (
                        f"from-pub-date:{query.published_from.isoformat()}",
                        f"until-pub-date:{query.published_until.isoformat()}",
                    )
                ),
            ),
            QueryParameter("query.bibliographic", search_text),
            QueryParameter("rows", str(rows)),
            QueryParameter("select", ",".join(CROSSREF_SELECT_FIELDS)),
        ]
    elif channel is RetrievalChannel.IDENTIFIER:
        if search_text is not None:
            raise ValueError("identifier retrieval must not contain search_text")
        normalized_doi = normalize_doi(doi)
        if normalized_doi is None:
            raise ValueError("identifier retrieval requires doi")
        parameters = [
            QueryParameter("filter", f"doi:{normalized_doi}"),
            QueryParameter("rows", "1"),
            QueryParameter("select", ",".join(CROSSREF_SELECT_FIELDS)),
        ]
    else:
        raise ValueError("Crossref supports only text and identifier retrieval")

    if contact_email is not None:
        if not contact_email.strip() or "@" not in contact_email:
            raise ValueError("contact_email must be a non-blank email address")
        parameters.append(QueryParameter("mailto", contact_email.strip()))
    parameters.sort(key=lambda parameter: parameter.name)

    digest = _request_digest(query, channel, parameters, attempt)
    return ConnectorRequest(
        request_id=f"request-crossref-{digest}",
        query_id=query.query_id,
        connector_id=ConnectorId.CROSSREF,
        channel=channel,
        endpoint=CROSSREF_WORKS_ENDPOINT,
        parameters=tuple(parameters),
        page_index=page_index,
        attempt=attempt,
    )


def crossref_request_url(request: ConnectorRequest) -> str:
    if request.connector_id is not ConnectorId.CROSSREF:
        raise ValueError("request must target Crossref")
    parameters = [(item.name, item.value) for item in request.parameters]
    return f"{request.endpoint}?{urlencode(parameters)}"


def _media_type(headers: Mapping[str, str]) -> str:
    for name, value in headers.items():
        if name.casefold() == "content-type":
            return value
    return "application/octet-stream"


def _record_count(response: HttpResponse) -> int:
    payload = json.loads(response.body.decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
        raise ValueError("Crossref response must contain a message object")
    items = payload["message"].get("items")
    if not isinstance(items, list):
        raise ValueError("Crossref response message must contain an items list")
    return len(items)


class CrossrefConnector:
    def __init__(
        self,
        *,
        transport: HttpTransport | None = None,
        api_token: str | None = None,
        contact_email: str | None = None,
        timeout_seconds: float = 20.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._transport = transport or UrllibHttpTransport()
        self._api_token = api_token
        self._contact_email = contact_email
        self._timeout_seconds = timeout_seconds
        self._clock = clock or (lambda: datetime.now(UTC))

    def run_page(
        self,
        query: SourceQuery,
        writer: SnapshotWriter,
        *,
        channel: RetrievalChannel,
        search_text: str | None = None,
        doi: str | None = None,
        cursor: str = "*",
        page_index: int = 1,
        rows: int = 100,
        attempt: int = 1,
    ) -> ConnectorRun:
        request = build_crossref_request(
            query,
            channel=channel,
            search_text=search_text,
            doi=doi,
            cursor=cursor,
            page_index=page_index,
            rows=rows,
            attempt=attempt,
            contact_email=self._contact_email,
        )
        started_at = self._clock()
        headers = {
            "Accept": "application/json",
            "User-Agent": "NextWave/0.1",
        }
        if self._api_token:
            headers["Crossref-Plus-API-Token"] = f"Bearer {self._api_token}"

        try:
            response = self._transport.get(
                crossref_request_url(request),
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
                message=f"Crossref returned HTTP {status_code}",
                retryable=status_code in {408, 425, 429} or status_code >= 500,
            ),
        )
