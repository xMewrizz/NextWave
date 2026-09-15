"""Persistent analysis repository for SQLite locally and PostgreSQL in Compose."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, String, Text, create_engine, select, update
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
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    corpus_version: Mapped[str] = mapped_column(String(100))
    method_version: Mapped[str] = mapped_column(String(100))
    trends: Mapped[list[dict]] = mapped_column(JSON, default=list)


def init_database() -> None:
    Base.metadata.create_all(engine)


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
        finished_at=record.finished_at,
        corpus_version=record.corpus_version,
        method_version=record.method_version,
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
        "finished_at": analysis.finished_at,
        "corpus_version": analysis.corpus_version,
        "method_version": analysis.method_version,
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
                notice="Анализ был прерван перезапуском сервера. Запустите его повторно.",
            )
        )
