"""Persistent analysis repository for SQLite locally and PostgreSQL in Compose."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, String, Text, create_engine, inspect, select, text, update
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Analysis, AnalysisSummary, Trend


def _database_url() -> str:
    value = os.getenv("NEXTWAVE_DATABASE_URL", "sqlite+pysqlite:///./nextwave.db")
    if value.startswith("postgresql://"):
        return value.replace("postgresql://", "postgresql+psycopg://", 1)
    return value


DATABASE_URL = _database_url()
engine_options: dict = {"pool_pre_ping": True}
if DATABASE_URL == "sqlite+pysqlite:///:memory:":
    engine_options.update(connect_args={"check_same_thread": False}, poolclass=StaticPool)
elif DATABASE_URL.startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_options)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    query: Mapped[str] = mapped_column(String(200), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    stage: Mapped[str | None] = mapped_column(String(200), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0)
    notice: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    corpus_version: Mapped[str] = mapped_column(String(100))
    method_version: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trends: Mapped[list[dict]] = mapped_column(JSON, default=list)


def init_database() -> None:
    """Create fresh storage or add V1-10 columns without rewriting existing results."""
    with engine.begin() as connection:
        if connection.dialect.name == "postgresql":
            connection.execute(text("SELECT pg_advisory_xact_lock(1010)"))
        Base.metadata.create_all(connection)
        columns = {column["name"] for column in inspect(connection).get_columns("analyses")}
        additions = {
            "started_at": "TIMESTAMP WITH TIME ZONE",
            "model_version": "TEXT",
            "feature_version": "TEXT",
            "error_code": "VARCHAR(64)",
        }
        for name, sql_type in additions.items():
            if name not in columns:
                connection.execute(text(f"ALTER TABLE analyses ADD COLUMN {name} {sql_type}"))


def reset_database() -> None:
    """Test helper; never called by application startup."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _record_to_analysis(record: AnalysisRecord) -> Analysis:
    return Analysis(
        id=record.id,
        query=record.query,
        status=record.status,
        stage=record.stage,
        progress=record.progress,
        notice=record.notice,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        corpus_version=record.corpus_version,
        method_version=record.method_version,
        model_version=record.model_version,
        feature_version=record.feature_version,
        error_code=record.error_code,
        trends=[Trend.model_validate(item) for item in record.trends],
    )


def save_analysis(analysis: Analysis) -> None:
    payload = {
        "query": analysis.query,
        "status": analysis.status,
        "stage": analysis.stage,
        "progress": analysis.progress,
        "notice": analysis.notice,
        "created_at": analysis.created_at,
        "started_at": analysis.started_at,
        "finished_at": analysis.finished_at,
        "corpus_version": analysis.corpus_version,
        "method_version": analysis.method_version,
        "model_version": analysis.model_version,
        "feature_version": analysis.feature_version,
        "error_code": analysis.error_code,
        "trends": [item.model_dump(mode="json") for item in analysis.trends],
    }
    with session_scope() as session:
        record = session.get(AnalysisRecord, analysis.id)
        if record is None:
            session.add(AnalysisRecord(id=analysis.id, **payload))
            return
        for key, value in payload.items():
            setattr(record, key, value)


def get_analysis(analysis_id: str) -> Analysis | None:
    with session_scope() as session:
        record = session.get(AnalysisRecord, analysis_id)
        return _record_to_analysis(record) if record else None


def list_analyses() -> list[AnalysisSummary]:
    with session_scope() as session:
        statement = select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc())
        records = session.scalars(statement).all()
        return [
            AnalysisSummary(
                id=record.id,
                query=record.query,
                status=record.status,
                created_at=record.created_at,
                trend_count=len(record.trends),
            )
            for record in records
        ]


def mark_interrupted_analyses() -> None:
    with session_scope() as session:
        session.execute(
            update(AnalysisRecord)
            .where(AnalysisRecord.status.in_(("pending", "running")))
            .values(
                status="error",
                stage=None,
                progress=1.0,
                finished_at=datetime.now(UTC),
                error_code="interrupted",
                notice="Анализ был прерван перезапуском сервера. Запустите его повторно.",
            )
        )
