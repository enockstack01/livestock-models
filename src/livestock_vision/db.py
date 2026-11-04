"""Detection and feedback records (SQLite for the pilot; set DATABASE_URL to PostgreSQL in production)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


def _now():
    return datetime.now(timezone.utc)


class Detection(Base):
    __tablename__ = "detections"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)          # the diagnosis_id the LLM API reads
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    species: Mapped[str] = mapped_column(String(32), default="")
    district: Mapped[str] = mapped_column(String(128), default="")
    animal_id: Mapped[str] = mapped_column(String(128), default="", index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    image_path: Mapped[str] = mapped_column(String(512))
    annotated_path: Mapped[str] = mapped_column(String(512))
    detections: Mapped[str] = mapped_column(Text)                          # JSON
    summary: Mapped[str] = mapped_column(Text)                             # JSON
    flags: Mapped[str] = mapped_column(Text)                               # JSON
    latency_ms: Mapped[float] = mapped_column(Float)
    offline_model_version: Mapped[str | None] = mapped_column(String(64))
    offline_detections: Mapped[str | None] = mapped_column(Text)           # JSON
    disagreement: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    diagnosis_id: Mapped[str] = mapped_column(ForeignKey("detections.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16))                        # confirmed | corrected
    findings: Mapped[str] = mapped_column(Text)                            # JSON: the corrected truth
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


def make_session_factory(url: str):
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, future=True, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
