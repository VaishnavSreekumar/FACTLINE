"""Reasoning package for FACTLINE."""

from backend.reasoning.comparability import ComparabilityGate
from backend.reasoning.matcher import CandidateMatcher
from backend.reasoning.relationships import RelationshipEngine
from backend.reasoning.surfacing import RelationshipSurfacingFilter, RelationshipSurfacingDecision

__all__ = [
    "CandidateMatcher",
    "ComparabilityGate",
    "RelationshipEngine",
    "RelationshipSurfacingFilter",
    "RelationshipSurfacingDecision",
]
