"""Source connector contracts and snapshot support."""

from .contracts import (
    CONNECTOR_REQUEST_SCHEMA_VERSION,
    SNAPSHOT_MANIFEST_SCHEMA_VERSION,
    SOURCE_QUERY_SCHEMA_VERSION,
    ConnectorError,
    ConnectorId,
    ConnectorRequest,
    ConnectorRun,
    ConnectorStatus,
    QueryParameter,
    QueryPurpose,
    RawResponseArtifact,
    RetrievalChannel,
    SnapshotManifest,
    SnapshotStatus,
    SourceQuery,
)
from .crossref import (
    CROSSREF_SELECT_FIELDS,
    CROSSREF_WORKS_ENDPOINT,
    CrossrefConnector,
    build_crossref_request,
    crossref_request_url,
)
from .crossref_parser import (
    CrossrefParseIssue,
    CrossrefParseResult,
    parse_crossref_response,
    parse_crossref_work,
)
from .http import HttpResponse, HttpTransport, UrllibHttpTransport
from .identifiers import doi_url, normalize_doi
from .openalex import (
    OPENALEX_SELECT_FIELDS,
    OPENALEX_WORKS_ENDPOINT,
    OpenAlexConnector,
    build_openalex_request,
    openalex_request_url,
)
from .openalex_parser import (
    OpenAlexParseIssue,
    OpenAlexParseResult,
    parse_openalex_response,
    parse_openalex_work,
    reconstruct_openalex_abstract,
)
from .snapshots import SnapshotWriter

__all__ = [
    "CONNECTOR_REQUEST_SCHEMA_VERSION",
    "CROSSREF_SELECT_FIELDS",
    "CROSSREF_WORKS_ENDPOINT",
    "SNAPSHOT_MANIFEST_SCHEMA_VERSION",
    "SOURCE_QUERY_SCHEMA_VERSION",
    "ConnectorError",
    "ConnectorId",
    "ConnectorRequest",
    "ConnectorRun",
    "ConnectorStatus",
    "CrossrefConnector",
    "CrossrefParseIssue",
    "CrossrefParseResult",
    "HttpResponse",
    "HttpTransport",
    "OPENALEX_SELECT_FIELDS",
    "OPENALEX_WORKS_ENDPOINT",
    "OpenAlexConnector",
    "OpenAlexParseIssue",
    "OpenAlexParseResult",
    "QueryParameter",
    "QueryPurpose",
    "RawResponseArtifact",
    "RetrievalChannel",
    "SnapshotManifest",
    "SnapshotStatus",
    "SnapshotWriter",
    "SourceQuery",
    "UrllibHttpTransport",
    "build_openalex_request",
    "build_crossref_request",
    "crossref_request_url",
    "doi_url",
    "normalize_doi",
    "openalex_request_url",
    "parse_openalex_response",
    "parse_openalex_work",
    "parse_crossref_response",
    "parse_crossref_work",
    "reconstruct_openalex_abstract",
]
