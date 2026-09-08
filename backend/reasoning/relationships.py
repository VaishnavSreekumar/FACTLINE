"""Deterministic Relationship Inference Engine."""

import hashlib
import re
from decimal import Decimal
from typing import List, Optional, Tuple

from backend.models.normalization import NormalizedFact
from backend.models.relationship import (
    CandidatePair,
    ComparabilityResult,
    ComparabilityStatus,
    RelationshipResult,
    RelationshipType,
)
from backend.models.fact import EpistemicStatus
from backend.normalization.units import UnitNormalizer


class RelationshipEngine:
    """Classifies relationships between facts with grounded contextual explanations."""

    # Keywords signaling explicit revision, restatement, or supersession
    SUPERSEDES_KEYWORDS = {
        "superseded", "supersedes", "restated", "restatement",
        "revised actuals", "subsequently restated", "superseding",
    }

    VINTAGE_REVISION_PAIRS = [
        ("first advance estimate", "second advance estimate"),
        ("first advance estimate", "provisional actuals"),
        ("first advance estimate", "revised estimate"),
        ("second advance estimate", "provisional actuals"),
        ("second advance estimate", "revised estimate"),
        ("provisional", "revised"),
        ("advance estimate", "revised estimate"),
        ("interim", "final"),
    ]

    @staticmethod
    def _generate_relationship_id(fact_a_id: str, fact_b_id: str, rel_type: RelationshipType) -> str:
        """Derives a deterministic unique ID for a relationship result."""
        raw_key = f"{fact_a_id}|{fact_b_id}|{rel_type.value}"
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
        return f"rel-{digest}"

    @classmethod
    def _derive_display_resolution(cls, value_raw: str, scale: Optional[str]) -> Decimal:
        """Derives the implied unit-aware display resolution from raw value representation.

        Example:
            '₹8,142 crore' -> 0 decimals in crore scale = 1 * 10,000,000 = 10,000,000 INR
            '₹81,415.38 million' -> 2 decimals in million scale = 0.01 * 1,000,000 = 10,000 INR
            '₹12.3 crore' -> 1 decimal in crore scale = 0.1 * 10,000,000 = 1,000,000 INR
        """
        # Extract numeric string
        cleaned = re.sub(r"[^\d.,\-+]", "", value_raw.replace(",", "")).lstrip(">+~")
        decimals = 0
        if "." in cleaned:
            decimals = len(cleaned.split(".")[1])

        decimal_factor = Decimal("10") ** (-decimals)

        scale_mult = Decimal("1")
        if scale and scale.lower() in UnitNormalizer.SCALES:
            scale_mult, _ = UnitNormalizer.SCALES[scale.lower()]

        return decimal_factor * scale_mult

    def determine_relationship(
        self,
        fact_a: NormalizedFact,
        fact_b: NormalizedFact,
        comparability: ComparabilityResult,
        candidate_pair: Optional[CandidatePair] = None,
    ) -> RelationshipResult:
        """Determines the grounded relationship between two normalized facts.

        Mandatory Invariant: Respects Comparability Gate status before comparing values.

        Args:
            fact_a: First normalized fact.
            fact_b: Second normalized fact.
            comparability: Output from ComparabilityGate.evaluate().
            candidate_pair: Optional CandidatePair record.

        Returns:
            Structured RelationshipResult with dual source provenance and explanation.
        """
        fact_a_id = fact_a.fact.fact_id
        fact_b_id = fact_b.fact.fact_id

        # -------------------------------------------------------------------
        # 1. Mandatory Comparability Gate Check
        # -------------------------------------------------------------------
        if comparability.status == ComparabilityStatus.NON_COMPARABLE:
            rel_id = self._generate_relationship_id(fact_a_id, fact_b_id, RelationshipType.UNRESOLVED)
            reasons = ["NON_COMPARABLE_CLAIMS"] + comparability.reason_codes
            expl = (
                f"Claims cannot be compared because the Comparability Gate identified incompatibility "
                f"({', '.join(comparability.reason_codes)}). "
                f"Fact A: '{fact_a.fact.metric}' ({fact_a.fact.time_period.label}), "
                f"Fact B: '{fact_b.fact.metric}' ({fact_b.fact.time_period.label})."
            )
            return RelationshipResult(
                relationship_id=rel_id,
                fact_a_id=fact_a_id,
                fact_b_id=fact_b_id,
                relationship_type=RelationshipType.UNRESOLVED,
                reason_codes=reasons,
                explanation=expl,
                confidence=0.0,
                evidence_a=fact_a.fact.provenance,
                evidence_b=fact_b.fact.provenance,
                contextual_factors=list(comparability.compared_dimensions.keys()),
            )

        if comparability.status == ComparabilityStatus.INSUFFICIENT_CONTEXT:
            rel_id = self._generate_relationship_id(fact_a_id, fact_b_id, RelationshipType.UNRESOLVED)
            reasons = ["INSUFFICIENT_CONTEXT"] + comparability.reason_codes
            expl = (
                f"Relationship cannot be safely established due to missing context "
                f"({', '.join(comparability.reason_codes)})."
            )
            return RelationshipResult(
                relationship_id=rel_id,
                fact_a_id=fact_a_id,
                fact_b_id=fact_b_id,
                relationship_type=RelationshipType.UNRESOLVED,
                reason_codes=reasons,
                explanation=expl,
                confidence=0.0,
                evidence_a=fact_a.fact.provenance,
                evidence_b=fact_b.fact.provenance,
                contextual_factors=list(comparability.compared_dimensions.keys()),
            )

        # -------------------------------------------------------------------
        # 2. Epistemic Incompatibility (Target/Projected vs Reported)
        # -------------------------------------------------------------------
        st_a = fact_a.fact.epistemic_status
        st_b = fact_b.fact.epistemic_status

        target_projected = {EpistemicStatus.TARGET, EpistemicStatus.PROJECTED}
        reported_audited = {EpistemicStatus.REPORTED, EpistemicStatus.AUDITED}

        if (st_a in target_projected and st_b in reported_audited) or (st_b in target_projected and st_a in reported_audited):
            rel_id = self._generate_relationship_id(fact_a_id, fact_b_id, RelationshipType.UNRESOLVED)
            expl = (
                f"Fact A ({st_a.value}) and Fact B ({st_b.value}) represent distinct epistemic categories "
                f"(target/projection vs observed actuals) and are not contradictory."
            )
            return RelationshipResult(
                relationship_id=rel_id,
                fact_a_id=fact_a_id,
                fact_b_id=fact_b_id,
                relationship_type=RelationshipType.UNRESOLVED,
                reason_codes=["EPISTEMIC_STATUS_INCOMPATIBLE", "EPISTEMIC_STATUS_DIFFERENCE"],
                explanation=expl,
                confidence=0.9,
                evidence_a=fact_a.fact.provenance,
                evidence_b=fact_b.fact.provenance,
                contextual_factors=["epistemic_status"],
            )

        # -------------------------------------------------------------------
        # 3. Explicit Supersession / Restatement Check
        # -------------------------------------------------------------------
        text_context_a = f"{fact_a.fact.data_vintage or ''} {fact_a.fact.provenance.supporting_text}".lower()
        text_context_b = f"{fact_b.fact.data_vintage or ''} {fact_b.fact.provenance.supporting_text}".lower()

        supersedes_a = any(k in text_context_a for k in self.SUPERSEDES_KEYWORDS)
        supersedes_b = any(k in text_context_b for k in self.SUPERSEDES_KEYWORDS)

        if supersedes_b or supersedes_a:
            rel_id = self._generate_relationship_id(fact_a_id, fact_b_id, RelationshipType.SUPERSEDES)
            superseding_fact = "Fact B" if supersedes_b else "Fact A"
            superseded_fact = "Fact A" if supersedes_b else "Fact B"
            expl = f"{superseding_fact} explicitly supersedes/restates the earlier claim in {superseded_fact} based on official revision evidence."
            return RelationshipResult(
                relationship_id=rel_id,
                fact_a_id=fact_a_id,
                fact_b_id=fact_b_id,
                relationship_type=RelationshipType.SUPERSEDES,
                reason_codes=["EXPLICIT_SUPERSEDED_CLAIM"],
                explanation=expl,
                confidence=0.95,
                evidence_a=fact_a.fact.provenance,
                evidence_b=fact_b.fact.provenance,
                contextual_factors=["supersession_evidence"],
            )

        # -------------------------------------------------------------------
        # 4. Data Vintage Evolution (EVOLVES_FROM)
        # -------------------------------------------------------------------
        dv_a = (fact_a.fact.data_vintage or "").strip().lower()
        dv_b = (fact_b.fact.data_vintage or "").strip().lower()

        if dv_a and dv_b and dv_a != dv_b:
            # Check if pair matches established sequential revision progressions
            is_sequential_evolution = False
            for v1, v2 in self.VINTAGE_REVISION_PAIRS:
                if (dv_a == v1 and dv_b == v2) or (dv_b == v1 and dv_a == v2):
                    is_sequential_evolution = True
                    break

            if is_sequential_evolution:
                rel_id = self._generate_relationship_id(fact_a_id, fact_b_id, RelationshipType.EVOLVES_FROM)
                expl = (
                    f"Fact B ('{fact_b.fact.data_vintage}') represents a subsequent data vintage revision "
                    f"of the claim in Fact A ('{fact_a.fact.data_vintage}') for {fact_a.canonical_metric} "
                    f"in {fact_a.fact.time_period.label}."
                )
                return RelationshipResult(
                    relationship_id=rel_id,
                    fact_a_id=fact_a_id,
                    fact_b_id=fact_b_id,
                    relationship_type=RelationshipType.EVOLVES_FROM,
                    reason_codes=["DATA_VINTAGE_EVOLUTION", "LATER_DATA_VINTAGE"],
                    explanation=expl,
                    confidence=0.95,
                    evidence_a=fact_a.fact.provenance,
                    evidence_b=fact_b.fact.provenance,
                    contextual_factors=["data_vintage"],
                )
            else:
                # Vintage differs but revision semantics are not explicitly established
                rel_id = self._generate_relationship_id(fact_a_id, fact_b_id, RelationshipType.UNRESOLVED)
                expl = (
                    f"Facts have different data vintages ('{fact_a.fact.data_vintage}' vs '{fact_b.fact.data_vintage}'), "
                    f"but sequential revision relationship is not explicitly established."
                )
                return RelationshipResult(
                    relationship_id=rel_id,
                    fact_a_id=fact_a_id,
                    fact_b_id=fact_b_id,
                    relationship_type=RelationshipType.UNRESOLVED,
                    reason_codes=["UNRESOLVED_VINTAGE_RELATIONSHIP", "DATA_VINTAGE_DIFFERENCE"],
                    explanation=expl,
                    confidence=0.8,
                    evidence_a=fact_a.fact.provenance,
                    evidence_b=fact_b.fact.provenance,
                    contextual_factors=["data_vintage"],
                )

        # -------------------------------------------------------------------
        # 5. Deterministic Numeric Comparison
        # -------------------------------------------------------------------
        val_a = fact_a.normalized_value.numeric_value
        val_b = fact_b.normalized_value.numeric_value

        if val_a is None or val_b is None:
            rel_id = self._generate_relationship_id(fact_a_id, fact_b_id, RelationshipType.UNRESOLVED)
            return RelationshipResult(
                relationship_id=rel_id,
                fact_a_id=fact_a_id,
                fact_b_id=fact_b_id,
                relationship_type=RelationshipType.UNRESOLVED,
                reason_codes=["NON_NUMERIC_COMPARISON"],
                explanation="One or both facts lack normalized numerical values for mathematical comparison.",
                confidence=0.5,
                evidence_a=fact_a.fact.provenance,
                evidence_b=fact_b.fact.provenance,
            )

        unit_label = fact_a.normalized_value.canonical_unit or ""

        # Exact Numeric Equality -> CORROBORATES
        if val_a == val_b:
            rel_id = self._generate_relationship_id(fact_a_id, fact_b_id, RelationshipType.CORROBORATES)
            expl = (
                f"Both sources independently corroborate the identical canonical value of "
                f"{val_a:,.2f} {unit_label} for {fact_a.canonical_entity} '{fact_a.canonical_metric}' "
                f"in {fact_a.fact.time_period.label}."
            )
            return RelationshipResult(
                relationship_id=rel_id,
                fact_a_id=fact_a_id,
                fact_b_id=fact_b_id,
                relationship_type=RelationshipType.CORROBORATES,
                reason_codes=["EQUAL_NORMALIZED_VALUE"],
                explanation=expl,
                confidence=1.0,
                evidence_a=fact_a.fact.provenance,
                evidence_b=fact_b.fact.provenance,
                contextual_factors=["normalized_value"],
            )

        # -------------------------------------------------------------------
        # 6. Display Rounding / Precision Reconciliation -> CONTEXT_RESOLVES
        # -------------------------------------------------------------------
        res_a = self._derive_display_resolution(fact_a.fact.value_raw, fact_a.normalized_value.scale)
        res_b = self._derive_display_resolution(fact_b.fact.value_raw, fact_b.normalized_value.scale)

        max_resolution = max(res_a, res_b)
        allowed_rounding_delta = float(max_resolution * Decimal("0.5"))
        actual_delta = abs(val_a - val_b)

        if actual_delta <= allowed_rounding_delta:
            rel_id = self._generate_relationship_id(fact_a_id, fact_b_id, RelationshipType.CONTEXT_RESOLVES)
            expl = (
                f"The claims ({fact_a.fact.value_raw} vs {fact_b.fact.value_raw}) appear different at surface level, "
                f"but are mathematically consistent with display rounding within the implied "
                f"{float(max_resolution):,.0f} {unit_label} resolution."
            )
            return RelationshipResult(
                relationship_id=rel_id,
                fact_a_id=fact_a_id,
                fact_b_id=fact_b_id,
                relationship_type=RelationshipType.CONTEXT_RESOLVES,
                reason_codes=["ROUNDING_DIFFERENCE", "CONTEXTUAL_RESOLUTION"],
                explanation=expl,
                confidence=0.95,
                evidence_a=fact_a.fact.provenance,
                evidence_b=fact_b.fact.provenance,
                contextual_factors=["display_rounding", "scale_resolution"],
            )

        # -------------------------------------------------------------------
        # 7. Genuine Value Conflict -> CONTRADICTS
        # -------------------------------------------------------------------
        rel_id = self._generate_relationship_id(fact_a_id, fact_b_id, RelationshipType.CONTRADICTS)
        expl = (
            f"Both sources report materially conflicting values for {fact_a.canonical_entity} "
            f"'{fact_a.canonical_metric}' in {fact_a.fact.time_period.label} "
            f"({fact_a.fact.value_raw} [{val_a:,.2f} {unit_label}] vs "
            f"{fact_b.fact.value_raw} [{val_b:,.2f} {unit_label}]) "
            f"exceeding display rounding precision without contextual justification."
        )
        return RelationshipResult(
            relationship_id=rel_id,
            fact_a_id=fact_a_id,
            fact_b_id=fact_b_id,
            relationship_type=RelationshipType.CONTRADICTS,
            reason_codes=["VALUE_CONFLICT"],
            explanation=expl,
            confidence=0.95,
            evidence_a=fact_a.fact.provenance,
            evidence_b=fact_b.fact.provenance,
            contextual_factors=["value_conflict"],
        )
