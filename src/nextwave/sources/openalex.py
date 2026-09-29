"""OpenAlex request planning and one-page retrieval."""

from __future__ import annotations

import hashlib
import json
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

OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"
OPENALEX_SELECT_FIELDS = (
    "id",
    "doi",
    "title",
    "display_name",
    "publication_date",
    "publication_year",
    "language",
    "type",
    "authorships",
    "primary_topic",
    "topics",
    "keywords",
    "abstract_inverted_index",
    "primary_location",
    "locations",
    "cited_by_count",
    "referenced_works",
)


def normalize_openalex_api_key(value: str | None) -> str | None:
    """Normalize an optional OpenAlex API key without logging or storing blanks.

    ``None`` or a blank string means anonymous access; a non-blank value is
    stripped and kept only in a private field, never in URLs or artifacts.
    """

    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


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


def build_openalex_request(
    query: SourceQuery,
    *,
    channel: RetrievalChannel,
    search_text: str | None = None,
    page_index: int = 1,
    per_page: int = 100,
    attempt: int = 1,
    contact_email: str | None = None,
) -> ConnectorRequest:
    """Translate a provider-independent query into one exact OpenAlex request."""

    if not 1 <= per_page <= 100:
        raise ValueError("per_page must be between 1 and 100")
    if channel in {RetrievalChannel.TEXT, RetrievalChannel.SEMANTIC}:
        if search_text is None or search_text not in query.search_texts:
            raise ValueError("search_text must be one of SourceQuery.search_texts")
    elif channel is RetrievalChannel.TAXONOMY:
        if search_text is not None:
            raise ValueError("taxonomy retrieval must not contain search_text")
        if not query.topic_ids and not query.subfield_ids:
            raise ValueError(
                "taxonomy retrieval requires SourceQuery.topic_ids or subfield_ids"
            )
    else:
        raise ValueError("identifier retrieval is not supported by the works search endpoint")

    filters = [
        f"from_publication_date:{query.published_from.isoformat()}",
        f"to_publication_date:{query.published_until.isoformat()}",
        "has_abstract:true",
        f"language:{'|'.join(query.languages)}",
    ]
    if channel is RetrievalChannel.TAXONOMY:
        taxonomy_filter = "topics.id"
        taxonomy_ids = query.topic_ids
        if query.subfield_ids:
            taxonomy_filter = "topics.subfield.id"
            taxonomy_ids = query.subfield_ids
        filters.append(f"{taxonomy_filter}:{'|'.join(taxonomy_ids)}")

    parameters = [
        QueryParameter("filter", ",".join(filters)),
        QueryParameter("page", str(page_index)),
        QueryParameter("per_page", str(per_page)),
        QueryParameter("select", ",".join(OPENALEX_SELECT_FIELDS)),
    ]
    if channel is RetrievalChannel.TEXT:
        parameters.append(QueryParameter("search", search_text or ""))
    elif channel is RetrievalChannel.SEMANTIC:
        parameters.append(QueryParameter("search.semantic", search_text or ""))
    if contact_email is not None:
        if not contact_email.strip() or "@" not in contact_email:
            raise ValueError("contact_email must be a non-blank email address")
        parameters.append(QueryParameter("mailto", contact_email.strip()))
    parameters.sort(key=lambda parameter: parameter.name)

    digest = _request_digest(query, channel, parameters, attempt)
    return ConnectorRequest(
        request_id=f"request-openalex-{digest}",
        query_id=query.query_id,
        connector_id=ConnectorId.OPENALEX,
        channel=channel,
        endpoint=OPENALEX_WORKS_ENDPOINT,
        parameters=tuple(parameters),
        page_index=page_index,
        attempt=attempt,
    )


def openalex_request_url(request: ConnectorRequest) -> str:
    if request.connector_id is not ConnectorId.OPENALEX:
        raise ValueError("request must target OpenAlex")
    parameters = [(item.name, item.value) for item in request.parameters]
    return f"{request.endpoint}?{urlencode(parameters)}"


def _media_type(headers: Mapping[str, str]) -> str:
    for name, value in headers.items():
        if name.casefold() == "content-type":
            return value
    return "application/octet-stream"


def _record_count(response: HttpResponse) -> int:
    payload = json.loads(response.body.decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("OpenAlex response must contain a results list")
    return len(payload["results"])


class OpenAlexConnector:
    def __init__(
        self,
        *,
        transport: HttpTransport | None = None,
        contact_email: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._transport = transport or UrllibHttpTransport()
        self._contact_email = contact_email
        self._api_key = normalize_openalex_api_key(api_key)
        self._timeout_seconds = timeout_seconds
        self._clock = clock or (lambda: datetime.now(UTC))

    def run_page(
        self,
        query: SourceQuery,
        writer: SnapshotWriter,
        *,
        channel: RetrievalChannel,
        search_text: str | None = None,
        page_index: int = 1,
        per_page: int = 100,
        attempt: int = 1,
    ) -> ConnectorRun:
        request = build_openalex_request(
            query,
            channel=channel,
            search_text=search_text,
            page_index=page_index,
            per_page=per_page,
            attempt=attempt,
            contact_email=self._contact_email,
        )
        started_at = self._clock()
        headers = {
            "Accept": "application/json",
            "User-Agent": "NextWave/0.1",
        }
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = self._transport.get(
                openalex_request_url(request),
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
                    message=f"{type(error).__name__}: OpenAlex request failed",
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
                message=f"OpenAlex returned HTTP {status_code}",
                retryable=_is_retryable(status_code),
                retry_after_seconds=_retry_after_seconds(response),
            ),
        )


def _is_retryable(status_code: int) -> bool:
    return status_code in {408, 425, 429} or status_code >= 500


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
