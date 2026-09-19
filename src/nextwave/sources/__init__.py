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

__all__ = [
    "CONNECTOR_REQUEST_SCHEMA_VERSION",
    "SNAPSHOT_MANIFEST_SCHEMA_VERSION",
    "SOURCE_QUERY_SCHEMA_VERSION",
    "ConnectorError",
    "ConnectorId",
    "ConnectorRequest",
    "ConnectorRun",
    "ConnectorStatus",
    "QueryParameter",
    "QueryPurpose",
    "RawResponseArtifact",
    "RetrievalChannel",
    "SnapshotManifest",
    "SnapshotStatus",
    "SourceQuery",
]
