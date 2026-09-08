"""Relationship and Comparability Schemas for FACTLINE."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from backend.models.fact import Provenance


class ComparabilityStatus(str, Enum):
    """Status emitted by the Comparability Gate."""
    COMPARABLE = "comparable"
    NON_COMPARABLE = "non_comparable"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class RelationshipType(str, Enum):
    """Locked relationship vocabulary for FACTLINE reasoning."""
    CORROBORATES = "CORROBORATES"
    CONTRADICTS = "CONTRADICTS"
    CONTEXT_RESOLVES = "CONTEXT_RESOLVES"
    EVOLVES_FROM = "EVOLVES_FROM"
    SUPERSEDES = "SUPERSEDES"
    UNRESOLVED = "UNRESOLVED"


class CandidatePair(BaseModel):
    """Plausible candidate fact pair generated for comparability evaluation."""

    fact_a_id: str = Field(..., description="ID of the first candidate fact")
    fact_b_id: str = Field(..., description="ID of the second candidate fact")
    entity_match: bool = Field(..., description="Whether entity alignment was established")
    metric_match: bool = Field(..., description="Whether metric signal/similarity was established")
    candidate_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Deterministic candidate score")
    reasons: List[str] = Field(default_factory=list, description="Deterministic signals supporting candidate pairing")


class ComparabilityResult(BaseModel):
    """Detailed evaluation result emitted by the Comparability Gate."""

    status: ComparabilityStatus = Field(..., description="Overall comparability decision")
    reason_codes: List[str] = Field(default_factory=list, description="Machine-readable decision reason codes")
    compared_dimensions: Dict[str, str] = Field(default_factory=dict, description="Per-dimension outcome descriptions")
    notes: List[str] = Field(default_factory=list, description="Diagnostic explanatory notes")

    @property
    def reasons(self) -> List[str]:
        return self.reason_codes


class RelationshipResult(BaseModel):
    """Result of relationship classification between facts with grounded evidence and explanation."""

    relationship_id: str = Field(..., description="Deterministic unique identifier for this relationship")
    fact_a_id: str = Field(..., description="ID of the first fact")
    fact_b_id: str = Field(..., description="ID of the second fact")
    relationship_type: RelationshipType = Field(..., description="Classified relationship category")
    reason_codes: List[str] = Field(default_factory=list, description="Machine-readable reasoning codes")
    explanation: str = Field(..., description="Deterministic grounded natural-language explanation")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Deterministic confidence score based on evidence quality")
    evidence_a: Optional[Provenance] = Field(default=None, description="Grounded source provenance for Fact A")
    evidence_b: Optional[Provenance] = Field(default=None, description="Grounded source provenance for Fact B")
    contextual_factors: List[str] = Field(default_factory=list, description="Key contextual dimensions evaluated")
