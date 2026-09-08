"""Deterministic Comparability Gate Implementation."""

from typing import Dict, List, Optional
from backend.models.normalization import NormalizedFact
from backend.models.relationship import ComparabilityResult, ComparabilityStatus


class ComparabilityGate:
    """Evaluates whether two facts are genuinely comparable across 8 essential dimensions.
    
    Principle: 'Never compare two values before establishing that the claims are comparable.'
    """

    def __init__(self):
        pass

    def evaluate(
        self,
        fact_a: NormalizedFact,
        fact_b: NormalizedFact,
    ) -> ComparabilityResult:
        """Determines if fact_a and fact_b are COMPARABLE, NON_COMPARABLE, or INSUFFICIENT_CONTEXT.

        Args:
            fact_a: First normalized fact.
            fact_b: Second normalized fact.

        Returns:
            ComparabilityResult containing status, reason codes, compared dimensions, and diagnostic notes.
        """
        reason_codes: List[str] = []
        compared_dimensions: Dict[str, str] = {}
        notes: List[str] = []

        incompatible = False
        missing_context = False

        # -------------------------------------------------------------------
        # 1. Entity Dimension
        # -------------------------------------------------------------------
        ent_a = (fact_a.canonical_entity or fact_a.fact.entity or "").strip().lower()
        ent_b = (fact_b.canonical_entity or fact_b.fact.entity or "").strip().lower()

        if not ent_a or not ent_b:
            missing_context = True
            reason_codes.append("MISSING_ENTITY")
            compared_dimensions["entity"] = "unresolved (missing entity)"
        elif ent_a != ent_b:
            incompatible = True
            reason_codes.append("ENTITY_MISMATCH")
            compared_dimensions["entity"] = f"mismatch ('{ent_a}' != '{ent_b}')"
            notes.append(f"Different entities: '{ent_a}' vs '{ent_b}'.")
        else:
            compared_dimensions["entity"] = f"compatible ('{ent_a}')"

        # -------------------------------------------------------------------
        # 2. Metric Dimension (No Semantic Equivalence Conflation)
        # -------------------------------------------------------------------
        met_a = (fact_a.canonical_metric or fact_a.fact.metric or "").strip().lower()
        met_b = (fact_b.canonical_metric or fact_b.fact.metric or "").strip().lower()

        if not met_a or not met_b:
            missing_context = True
            reason_codes.append("MISSING_METRIC")
            compared_dimensions["metric"] = "unresolved (missing metric)"
        elif met_a != met_b:
            incompatible = True
            reason_codes.append("METRIC_MISMATCH")
            compared_dimensions["metric"] = f"mismatch ('{met_a}' != '{met_b}')"
            notes.append(f"Different metrics: '{met_a}' vs '{met_b}'.")
        else:
            compared_dimensions["metric"] = f"compatible ('{met_a}')"

        # -------------------------------------------------------------------
        # 3. Unit / Semantic Quantity Dimension
        # -------------------------------------------------------------------
        u_a = fact_a.normalized_value.canonical_unit
        u_b = fact_b.normalized_value.canonical_unit

        if not u_a or not u_b:
            missing_context = True
            reason_codes.append("MISSING_UNIT")
            compared_dimensions["unit"] = f"unresolved unit (fact_a='{u_a}', fact_b='{u_b}')"
            notes.append("Unit/currency could not be safely resolved on one or both facts.")
        elif u_a != u_b:
            incompatible = True
            reason_codes.append("UNIT_MISMATCH")
            compared_dimensions["unit"] = f"mismatch ('{u_a}' != '{u_b}')"
            notes.append(f"Incompatible units: '{u_a}' vs '{u_b}'.")
        else:
            compared_dimensions["unit"] = f"compatible ('{u_a}')"

        # -------------------------------------------------------------------
        # 4. Time Period Dimension (Interval Equality vs Granularity)
        # -------------------------------------------------------------------
        tp_a = fact_a.normalized_time_period
        tp_b = fact_b.normalized_time_period

        if not tp_a or not tp_b or not tp_a.start_date or not tp_b.start_date:
            missing_context = True
            reason_codes.append("MISSING_TIME_PERIOD")
            compared_dimensions["time_period"] = "unresolved interval"
            notes.append("Time period interval not safely bounded on one or both facts.")
        else:
            # Check exact interval match
            if tp_a.start_date == tp_b.start_date and tp_a.end_date == tp_b.end_date:
                compared_dimensions["time_period"] = f"compatible ({tp_a.start_date} to {tp_a.end_date})"
            else:
                # Invariant: Different measurement intervals (e.g. FY24 vs Q4 FY24) are non-comparable
                incompatible = True
                reason_codes.append("TIME_MISMATCH")
                compared_dimensions["time_period"] = f"mismatch ({tp_a.start_date}..{tp_a.end_date} vs {tp_b.start_date}..{tp_b.end_date})"
                notes.append(
                    f"Different measurement periods: [{tp_a.start_date} to {tp_a.end_date}] vs [{tp_b.start_date} to {tp_b.end_date}]."
                )

        # -------------------------------------------------------------------
        # 5. Scope Dimension
        # -------------------------------------------------------------------
        sc_a = (fact_a.fact.scope or "").strip().lower() if fact_a.fact.scope else None
        sc_b = (fact_b.fact.scope or "").strip().lower() if fact_b.fact.scope else None

        if sc_a and sc_b:
            if sc_a == sc_b:
                compared_dimensions["scope"] = f"compatible ('{sc_a}')"
            else:
                incompatible = True
                reason_codes.append("SCOPE_MISMATCH")
                compared_dimensions["scope"] = f"mismatch ('{sc_a}' != '{sc_b}')"
                notes.append(f"Conflicting scopes: '{sc_a}' vs '{sc_b}'.")
        elif sc_a != sc_b:
            # One specified and one unspecified: missing is not same
            missing_context = True
            reason_codes.append("MISSING_SCOPE")
            compared_dimensions["scope"] = f"unspecified on one claim (fact_a='{sc_a}', fact_b='{sc_b}')"
            notes.append("Scope is specified on one claim but unspecified on the other.")
        else:
            compared_dimensions["scope"] = "compatible (both unspecified)"

        # -------------------------------------------------------------------
        # 6. Geography Dimension
        # -------------------------------------------------------------------
        geo_a = (fact_a.fact.geography or "").strip().lower() if fact_a.fact.geography else None
        geo_b = (fact_b.fact.geography or "").strip().lower() if fact_b.fact.geography else None

        if geo_a and geo_b:
            if geo_a == geo_b:
                compared_dimensions["geography"] = f"compatible ('{geo_a}')"
            else:
                incompatible = True
                reason_codes.append("GEOGRAPHY_MISMATCH")
                compared_dimensions["geography"] = f"mismatch ('{geo_a}' != '{geo_b}')"
                notes.append(f"Conflicting geographies: '{geo_a}' vs '{geo_b}'.")
        elif geo_a != geo_b:
            missing_context = True
            reason_codes.append("MISSING_GEOGRAPHY")
            compared_dimensions["geography"] = f"unspecified on one claim (fact_a='{geo_a}', fact_b='{geo_b}')"
            notes.append("Geography is specified on one claim but unspecified on the other.")
        else:
            compared_dimensions["geography"] = "compatible (both unspecified)"

        # -------------------------------------------------------------------
        # 7. Epistemic Status Dimension (Preserved as diagnostic)
        # -------------------------------------------------------------------
        ep_a = fact_a.fact.epistemic_status
        ep_b = fact_b.fact.epistemic_status

        if ep_a != ep_b:
            reason_codes.append("EPISTEMIC_STATUS_DIFFERENCE")
            compared_dimensions["epistemic_status"] = f"differs ('{ep_a.value}' vs '{ep_b.value}')"
            notes.append(f"Epistemic status differs: '{ep_a.value}' vs '{ep_b.value}'.")
        else:
            compared_dimensions["epistemic_status"] = f"matching ('{ep_a.value}')"

        # -------------------------------------------------------------------
        # 8. Data Vintage Dimension (Preserved as diagnostic)
        # -------------------------------------------------------------------
        dv_a = (fact_a.fact.data_vintage or "").strip().lower() if fact_a.fact.data_vintage else None
        dv_b = (fact_b.fact.data_vintage or "").strip().lower() if fact_b.fact.data_vintage else None

        if dv_a and dv_b and dv_a != dv_b:
            reason_codes.append("DATA_VINTAGE_DIFFERENCE")
            compared_dimensions["data_vintage"] = f"differs ('{dv_a}' vs '{dv_b}')"
            notes.append(f"Data vintage differs: '{dv_a}' vs '{dv_b}'.")
        else:
            compared_dimensions["data_vintage"] = "compatible / unspecified"

        # -------------------------------------------------------------------
        # 9. Final Decision Determination
        # -------------------------------------------------------------------
        if incompatible:
            status = ComparabilityStatus.NON_COMPARABLE
        elif missing_context:
            status = ComparabilityStatus.INSUFFICIENT_CONTEXT
        else:
            status = ComparabilityStatus.COMPARABLE

        return ComparabilityResult(
            status=status,
            reason_codes=reason_codes,
            compared_dimensions=compared_dimensions,
            notes=notes,
        )
