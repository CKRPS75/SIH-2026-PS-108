from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Standard(Base):
    __tablename__ = "standards"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    standard_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    canonical_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    base_code: Mapped[str] = mapped_column(String(64), index=True)
    part: Mapped[str | None] = mapped_column(String(64))
    section: Mapped[str | None] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(1000))
    scope_text: Mapped[str | None] = mapped_column(Text)
    division: Mapped[str | None] = mapped_column(String(128))
    publication_year: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE", index=True)
    supersedes_id: Mapped[UUID | None] = mapped_column(Uuid, ForeignKey("standards.id"))
    replaced_by_id: Mapped[UUID | None] = mapped_column(Uuid, ForeignKey("standards.id"))
    amendments_count: Mapped[int] = mapped_column(Integer, default=0)
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_url: Mapped[str | None] = mapped_column(Text)
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class StandardAmendment(Base):
    __tablename__ = "standard_amendments"
    __table_args__ = (
        UniqueConstraint("standard_id", "amendment_no", name="uq_standard_amendment"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    standard_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("standards.id"), index=True)
    amendment_no: Mapped[str] = mapped_column(String(64))
    publication_date: Mapped[date | None] = mapped_column(Date)
    summary: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)


class QcoRule(Base):
    __tablename__ = "qco_rules"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    standard_id: Mapped[UUID | None] = mapped_column(Uuid, ForeignKey("standards.id"), index=True)
    product_category: Mapped[str | None] = mapped_column(String(256), index=True)
    hs_code: Mapped[str | None] = mapped_column(String(32), index=True)
    scheme: Mapped[str] = mapped_column(String(64))
    notifying_ministry: Mapped[str] = mapped_column(String(256))
    order_ref: Mapped[str] = mapped_column(String(256), unique=True)
    effective_from: Mapped[date | None] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE")
    source_url: Mapped[str | None] = mapped_column(Text)


class Tender(Base):
    __tablename__ = "tenders"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    filename: Mapped[str] = mapped_column(String(512))
    file_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="UPLOADED")
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    language: Mapped[str | None] = mapped_column(String(32))
    source_hash: Mapped[str] = mapped_column(String(128), unique=True)


class TenderItem(Base):
    __tablename__ = "tender_items"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    tender_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("tenders.id"), index=True)
    item_no: Mapped[str | None] = mapped_column(String(64))
    raw_text: Mapped[str] = mapped_column(Text)
    page_or_row_ref: Mapped[str] = mapped_column(String(128))
    normalized_text: Mapped[str | None] = mapped_column(Text)
    extracted_specs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    tender_item_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("tender_items.id"), index=True
    )
    query_text: Mapped[str] = mapped_column(Text)
    normalized_query: Mapped[str | None] = mapped_column(Text)
    primary_results: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    allied_results: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    compliance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    trace: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    analysis_result_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("analysis_results.id"), index=True
    )
    selected_standard_id: Mapped[UUID | None] = mapped_column(Uuid, ForeignKey("standards.id"))
    rating: Mapped[int | None] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
