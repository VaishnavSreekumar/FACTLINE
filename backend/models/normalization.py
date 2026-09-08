"""Canonical Normalization Schemas for FACTLINE."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from backend.models.fact import FactRecord, TimePeriod


class NormalizationStatus(str, Enum):
    """Indicates whether a fact's derived normalization succeeded completely, partially, or failed."""
    NORMALIZED = "normalized"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"


class NormalizedValue(BaseModel):
    """Represents a deterministically normalized numeric and unit value."""

    numeric_value: Optional[float] = Field(default=None, description="Base canonical numeric value in standard unit")
    canonical_unit: Optional[str] = Field(default=None, description="Standardized canonical unit (e.g. INR, percent, count, tonnes)")
    scale: Optional[str] = Field(default=None, description="Original scale denominator (e.g. crore, million, lakh, billion)")
    value_qualifier: Optional[str] = Field(default=None, description="Optional inequality qualifier (e.g. >, <, >=, <=, =)")
    currency: Optional[str] = Field(default=None, description="Currency code (e.g. INR, USD, EUR) if monetary")
    original_value_raw: str = Field(..., description="Immutable original raw value string from FactRecord")
    normalization_status: NormalizationStatus = Field(default=NormalizationStatus.UNRESOLVED, description="Status of value normalization")
    normalization_notes: Optional[str] = Field(default=None, description="Diagnostic notes or reason for partial/unresolved status")


class NormalizedFact(BaseModel):
    """Derived representation of a FactRecord with normalized dimensions."""

    fact: FactRecord = Field(..., description="Immutable source FactRecord preserving raw evidence")
    normalized_value: NormalizedValue = Field(..., description="Derived canonical numerical and unit representation")
    canonical_entity: Optional[str] = Field(default=None, description="Cleaned, presentation-ready canonical entity name")
    canonical_metric: Optional[str] = Field(default=None, description="Standardized metric string")
    normalized_time_period: Optional[TimePeriod] = Field(default=None, description="Bounded ISO time period if safely resolvable")
    normalization_warnings: List[str] = Field(default_factory=list, description="Audit warnings recorded during normalization")
