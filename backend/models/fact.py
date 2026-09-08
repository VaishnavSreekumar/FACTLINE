"""Canonical Fact and Provenance Schemas for FACTLINE."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class EpistemicStatus(str, Enum):
    REPORTED = "reported"
    ESTIMATED = "estimated"
    PROJECTED = "projected"
    TARGET = "target"
    AUDITED = "audited"


class Provenance(BaseModel):
    document_id: str
    document_date: Optional[str] = None
    page_number: int
    supporting_text: str


class TimePeriod(BaseModel):
    label: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class FactRecord(BaseModel):
    fact_id: str

    entity: str
    metric: str

    value_raw: str
    value_numeric: Optional[float] = None
    unit: Optional[str] = None

    time_period: TimePeriod

    scope: Optional[str] = None
    geography: Optional[str] = None

    epistemic_status: EpistemicStatus = EpistemicStatus.REPORTED

    data_vintage: Optional[str] = None

    provenance: Provenance

    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
