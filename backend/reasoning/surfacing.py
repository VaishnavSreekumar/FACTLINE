"""
Deterministic Relationship Surfacing Filter.

Evaluates whether a candidate fact pair that has passed candidate matching
and comparability evaluation represents a meaningful relationship worth surfacing.

Core Principle:
'Never compare two values before establishing that the claims are comparable.'
And distinguish between:
1. A candidate that is worth investigating / surfacing.
2. A pair that is simply not a meaningful comparison.
"""

import re
from typing import List, Optional
from pydantic import BaseModel, Field

from backend.models.normalization import NormalizedFact
from backend.models.relationship import (
    CandidatePair,
    ComparabilityResult,
    ComparabilityStatus,
)


class RelationshipSurfacingDecision(BaseModel):
    """Deterministic surfacing decision for a candidate fact pair."""

    should_surface: bool = Field(..., description="Whether the pair represents a meaningful relationship to surface")
    reason_codes: List[str] = Field(default_factory=list, description="Machine-readable decision codes")
    explanation: str = Field(..., description="Deterministic natural-language explanation of surfacing decision")


class RelationshipSurfacingFilter:
    """Filters candidate pairs after ComparabilityGate and before RelationshipEngine.

    Ensures that UNRESOLVED relationships are only emitted for genuinely meaningful
    comparisons (e.g. same period with missing context, or explicit temporal comparisons),
    rather than every disparate cross-period candidate pair.
    """

    # Deterministic comparative / temporal link indicators
    TEMPORAL_COMPARISON_PATTERNS = [
        re.compile(r"\b(?:up|down|increased|decreased|rose|fell|grown|declined)\s+from\b", re.IGNORECASE),
        re.compile(r"\bcompared\s+(?:with|to)\b", re.IGNORECASE),
        re.compile(r"\b(?:versus|vs\.?)\b", re.IGNORECASE),
        re.compile(r"\b(?:previously|earlier)\b", re.IGNORECASE),
        re.compile(r"\bfrom\s+(?:[\$\€\£\¥\₹]?\d+(?:\.\d+)?%?|\d{4})\s+to\s+(?:[\$\€\£\¥\₹]?\d+(?:\.\d+)?%?|\d{4})\b", re.IGNORECASE),
        re.compile(r"\bwas\s+.+?\s+in\s+\d{4}\s+and\s+.+?\s+in\s+\d{4}\b", re.IGNORECASE),
        re.compile(r"\b(?:year[- ]on[- ]year|yoy|annualized|quarter[- ]on[- ]quarter|qoq)\b", re.IGNORECASE),
    ]

    TEMPORAL_PHRASES = [
        "up from", "down from", "increased from", "decreased from",
        "rose from", "fell from", "compared with", "compared to",
        "versus", "vs.", "vs", "previously", "earlier",
    ]

    @classmethod
    def _has_explicit_temporal_language(
        cls,
        fact_a: NormalizedFact,
        fact_b: NormalizedFact,
    ) -> bool:
        """Determines if source evidence explicitly connects claims with comparative language."""
        text_a = (fact_a.fact.provenance.supporting_text if fact_a.fact.provenance else "").strip().lower()
        text_b = (fact_b.fact.provenance.supporting_text if fact_b.fact.provenance else "").strip().lower()
        combined = f"{text_a} {text_b}"

        # 1. Fast substring checks
        for phrase in cls.TEMPORAL_PHRASES:
            if phrase in text_a or phrase in text_b:
                return True

        # 2. Regex patterns
        for pattern in cls.TEMPORAL_COMPARISON_PATTERNS:
            if pattern.search(combined):
                return True

        return False

    def evaluate(
        self,
        fact_a: NormalizedFact,
        fact_b: NormalizedFact,
        comparability: ComparabilityResult,
        candidate_pair: Optional[CandidatePair] = None,
    ) -> RelationshipSurfacingDecision:
        """Evaluates whether to surface a candidate pair as a primary relationship.

        Relies strictly on existing CandidatePair and ComparabilityResult signals.
        """
        # 1. Duplicate / Self-comparison Check
        if fact_a.fact.fact_id == fact_b.fact.fact_id:
            return RelationshipSurfacingDecision(
                should_surface=False,
                reason_codes=["SAME_DOCUMENT_DUPLICATE"],
                explanation="Duplicate or self-comparison of the same extracted fact.",
            )

        # 2. Fully Comparable Claims -> Always Surface
        if comparability.status == ComparabilityStatus.COMPARABLE:
            return RelationshipSurfacingDecision(
                should_surface=True,
                reason_codes=["MEANINGFUL_COMPARISON"],
                explanation="Claims are comparable across all verified dimensions.",
            )

        # 3. Non-Comparable Claims
        if comparability.status == ComparabilityStatus.NON_COMPARABLE:
            # Case 3A: Metric Mismatch
            if "METRIC_MISMATCH" in comparability.reason_codes:
                return RelationshipSurfacingDecision(
                    should_surface=False,
                    reason_codes=["METRIC_MISMATCH_SUPPRESSED", "NON_COMPARABLE_NOT_MEANINGFUL"],
                    explanation="Claims represent different metrics/measurements.",
                )

            # Case 3B: Time Period Mismatch
            if "TIME_MISMATCH" in comparability.reason_codes:
                if self._has_explicit_temporal_language(fact_a, fact_b):
                    return RelationshipSurfacingDecision(
                        should_surface=True,
                        reason_codes=["EXPLICIT_TEMPORAL_COMPARISON"],
                        explanation="Source evidence explicitly contains temporal comparison language connecting the periods.",
                    )
                else:
                    return RelationshipSurfacingDecision(
                        should_surface=False,
                        reason_codes=["DIFFERENT_TIME_PERIOD_NO_LINK", "NON_COMPARABLE_NOT_MEANINGFUL"],
                        explanation="Claims describe different bounded time periods without explicit comparative language in source text.",
                    )

            # Case 3C: Other Hard Non-Comparable Incompatibilities (e.g. Entity Mismatch, Unit Mismatch)
            return RelationshipSurfacingDecision(
                should_surface=False,
                reason_codes=["NON_COMPARABLE_NOT_MEANINGFUL"] + comparability.reason_codes,
                explanation=f"Claims are fundamentally non-comparable ({', '.join(comparability.reason_codes)}).",
            )

        # 4. Insufficient Context Claims
        if comparability.status == ComparabilityStatus.INSUFFICIENT_CONTEXT:
            # Surface only when candidate is a strong deterministic candidate under existing matcher
            # and there is no hard entity, metric, or time mismatch
            is_strong_candidate = candidate_pair is not None and candidate_pair.metric_match is True
            has_hard_conflict = any(
                code in comparability.reason_codes
                for code in ["ENTITY_MISMATCH", "METRIC_MISMATCH", "TIME_MISMATCH", "UNIT_MISMATCH"]
            )

            if is_strong_candidate and not has_hard_conflict:
                return RelationshipSurfacingDecision(
                    should_surface=True,
                    reason_codes=["INSUFFICIENT_CONTEXT_MEANINGFUL"],
                    explanation="Meaningful claim comparison with unresolved missing contextual dimensions.",
                )
            else:
                return RelationshipSurfacingDecision(
                    should_surface=False,
                    reason_codes=["WEAK_SEMANTIC_CANDIDATE", "NON_COMPARABLE_NOT_MEANINGFUL"],
                    explanation="Candidate pair lacks sufficient core alignment to surface as an unresolved relationship.",
                )

        # Fallback default (safe conservative suppression)
        return RelationshipSurfacingDecision(
            should_surface=False,
            reason_codes=["NON_COMPARABLE_NOT_MEANINGFUL"],
            explanation="Candidate not eligible for primary relationship surfacing.",
        )
