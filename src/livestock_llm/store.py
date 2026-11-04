"""Generated reports are stored against the detection (and so against the animal's health record)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from livestock_vision.db import Base          # same declarative base: one create_all makes the vision AND llm tables


class LlmReport(Base):
    __tablename__ = "llm_reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    diagnosis_id: Mapped[str] = mapped_column(ForeignKey("detections.id"), index=True)
    key: Mapped[str] = mapped_column(String(40), index=True)               # hash of (diagnosis, language, animal context)
    user_id: Mapped[str] = mapped_column(String(128))
    language: Mapped[str] = mapped_column(String(8))
    source: Mapped[str] = mapped_column(String(16))                        # llm | advice_pack
    model_version: Mapped[str] = mapped_column(String(96))
    animal: Mapped[str] = mapped_column(Text, default="{}")
    report: Mapped[str] = mapped_column(Text)                              # JSON
    adjustments: Mapped[str] = mapped_column(Text, default="[]")           # JSON: what the safety layer changed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
