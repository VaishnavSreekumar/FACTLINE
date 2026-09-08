"""Deterministic Relationship Engine test suite."""

import pytest
from backend.models.fact import EpistemicStatus, FactRecord, Provenance, TimePeriod
from backend.models.normalization import NormalizationStatus, NormalizedFact, NormalizedValue
from backend.models.relationship import ComparabilityResult, ComparabilityStatus, RelationshipType
from backend.reasoning.comparability import ComparabilityGate
from backend.reasoning.matcher import CandidateMatcher
from backend.reasoning.relationships import RelationshipEngine


@pytest.fixture
def gate():
    return ComparabilityGate()


@pytest.fixture
def engine():
    return RelationshipEngine()


@pytest.fixture
def matcher():
    return CandidateMatcher()


def _make_norm_fact(
    fact_id: str,
    entity: str,
    metric: str,
    value_raw: str,
    canonical_unit: str = "INR",
    numeric_value: float = 1000.0,
    scale: str = None,
    start_date: str = "2023-04-01",
    end_date: str = "2024-03-31",
    scope: str = None,
    geography: str = None,
    epistemic_status: EpistemicStatus = EpistemicStatus.REPORTED,
    data_vintage: str = None,
    supporting_text: str = "Evidence text from page",
    has_time_period: bool = True,
) -> NormalizedFact:
    """Helper to construct NormalizedFact instances for relationship tests."""
    prov = Provenance(document_id=f"doc-{fact_id}", page_number=1, supporting_text=supporting_text)
    tp_label = "FY 2023-24" if has_time_period else "None"

    source_fact = FactRecord(
        fact_id=fact_id,
        entity=entity,
        metric=metric,
        value_raw=value_raw,
        value_numeric=numeric_value,
        time_period=TimePeriod(label=tp_label),
        scope=scope,
        geography=geography,
        epistemic_status=epistemic_status,
        data_vintage=data_vintage,
        provenance=prov,
    )

    norm_val = NormalizedValue(
        numeric_value=numeric_value,
        canonical_unit=canonical_unit,
        currency=canonical_unit if canonical_unit in ["INR", "USD", "EUR", "GBP"] else None,
        scale=scale,
        original_value_raw=value_raw,
        normalization_status=NormalizationStatus.NORMALIZED,
    )

    norm_tp = TimePeriod(label=tp_label, start_date=start_date, end_date=end_date) if has_time_period else None

    canonical_ent = entity.replace(" Limited", "").replace(" Ltd.", "").strip()
    canonical_met = metric.strip().lower()

    return NormalizedFact(
        fact=source_fact,
        normalized_value=norm_val,
        canonical_entity=canonical_ent,
        canonical_metric=canonical_met,
        normalized_time_period=norm_tp,
    )


# ---------------------------------------------------------------------------
# 1. Test 1: Equal Values -> CORROBORATES
# ---------------------------------------------------------------------------

def test_equal_values_corroborates(gate, engine):
    """Test 1: Identical normalized values on comparable claims -> CORROBORATES."""
    f1 = _make_norm_fact("f1", "Delhivery", "Revenue from operations", "₹81,415.38 million", numeric_value=81415380000.0)
    f2 = _make_norm_fact("f2", "Delhivery", "Revenue from operations", "₹81,415.38 million", numeric_value=81415380000.0)

    comp = gate.evaluate(f1, f2)
    assert comp.status == ComparabilityStatus.COMPARABLE

    rel = engine.determine_relationship(f1, f2, comp)
    assert rel.relationship_type == RelationshipType.CORROBORATES
    assert "EQUAL_NORMALIZED_VALUE" in rel.reason_codes
    assert rel.confidence == 1.0
    assert rel.evidence_a == f1.fact.provenance
    assert rel.evidence_b == f2.fact.provenance


# ---------------------------------------------------------------------------
# 2. Test 2: Genuine Conflict -> CONTRADICTS
# ---------------------------------------------------------------------------

def test_genuine_value_conflict_contradicts(gate, engine):
    """Test 2: Distinct numbers beyond rounding without vintage/context difference -> CONTRADICTS."""
    f1 = _make_norm_fact("f1", "Delhivery", "Revenue from operations", "₹80,000 million", numeric_value=80000000000.0, scale="million")
    f2 = _make_norm_fact("f2", "Delhivery", "Revenue from operations", "₹95,000 million", numeric_value=95000000000.0, scale="million")

    comp = gate.evaluate(f1, f2)
    assert comp.status == ComparabilityStatus.COMPARABLE

    rel = engine.determine_relationship(f1, f2, comp)
    assert rel.relationship_type == RelationshipType.CONTRADICTS
    assert "VALUE_CONFLICT" in rel.reason_codes


# ---------------------------------------------------------------------------
# 3. Test 3: Non-Comparable Time -> UNRESOLVED (No Numeric Comparison)
# ---------------------------------------------------------------------------

def test_non_comparable_time_unresolved(gate, engine):
    """Test 3: FY24 vs Q4 FY24 -> UNRESOLVED with gate failure preserved (no value comparison)."""
    f_year = _make_norm_fact("f1", "Delhivery", "Revenue", "₹80,000 million", numeric_value=80000000000.0, start_date="2023-04-01", end_date="2024-03-31")
    f_qtr = _make_norm_fact("f2", "Delhivery", "Revenue", "₹20,000 million", numeric_value=20000000000.0, start_date="2024-01-01", end_date="2024-03-31")

    comp = gate.evaluate(f_year, f_qtr)
    assert comp.status == ComparabilityStatus.NON_COMPARABLE

    rel = engine.determine_relationship(f_year, f_qtr, comp)
    assert rel.relationship_type == RelationshipType.UNRESOLVED
    assert "NON_COMPARABLE_CLAIMS" in rel.reason_codes
    assert "TIME_MISMATCH" in rel.reason_codes


# ---------------------------------------------------------------------------
# 4. Test 4: Missing Context -> UNRESOLVED
# ---------------------------------------------------------------------------

def test_missing_context_unresolved(gate, engine):
    """Test 4: Missing time period -> UNRESOLVED."""
    f1 = _make_norm_fact("f1", "India", "Real GDP Growth", "6.4%", canonical_unit="percent", numeric_value=6.4, has_time_period=True)
    f2 = _make_norm_fact("f2", "India", "Real GDP Growth", "6.5%", canonical_unit="percent", numeric_value=6.5, has_time_period=False)

    comp = gate.evaluate(f1, f2)
    assert comp.status == ComparabilityStatus.INSUFFICIENT_CONTEXT

    rel = engine.determine_relationship(f1, f2, comp)
    assert rel.relationship_type == RelationshipType.UNRESOLVED
    assert "INSUFFICIENT_CONTEXT" in rel.reason_codes
    assert "MISSING_TIME_PERIOD" in rel.reason_codes


# ---------------------------------------------------------------------------
# 5. Test 5: Rounding Reconciliation -> CONTEXT_RESOLVES
# ---------------------------------------------------------------------------

def test_generic_rounding_reconciliation_context_resolves(gate, engine):
    """Test 5: Generic synthetic values consistent within implied display precision -> CONTEXT_RESOLVES."""
    # Fact A: ₹123.45 million (123,450,000 INR, display resolution 0.01 million = 10,000 INR)
    # Fact B: ₹12.3 crore (123,000,000 INR, display resolution 0.1 crore = 1,000,000 INR)
    # Delta: 450,000 INR <= 500,000 INR (half of 1,000,000)
    f_a = _make_norm_fact("fa", "Company", "Revenue", "₹123.45 million", numeric_value=123450000.0, scale="million")
    f_b = _make_norm_fact("fb", "Company", "Revenue", "₹12.3 crore", numeric_value=123000000.0, scale="crore")

    comp = gate.evaluate(f_a, f_b)
    assert comp.status == ComparabilityStatus.COMPARABLE

    rel = engine.determine_relationship(f_a, f_b, comp)
    assert rel.relationship_type == RelationshipType.CONTEXT_RESOLVES
    assert "ROUNDING_DIFFERENCE" in rel.reason_codes
    assert "display rounding" in rel.explanation.lower()


# ---------------------------------------------------------------------------
# 6. Test 6: Vintage Evolution -> EVOLVES_FROM
# ---------------------------------------------------------------------------

def test_vintage_evolution_evolves_from(gate, engine):
    """Test 6: First Advance Estimate vs Second Advance Estimate -> EVOLVES_FROM."""
    f1 = _make_norm_fact(
        "f1", "India", "Real GDP Growth", "6.4%", canonical_unit="percent",
        numeric_value=6.4, data_vintage="First Advance Estimate"
    )
    f2 = _make_norm_fact(
        "f2", "India", "Real GDP Growth", "6.5%", canonical_unit="percent",
        numeric_value=6.5, data_vintage="Second Advance Estimate"
    )

    comp = gate.evaluate(f1, f2)
    assert comp.status == ComparabilityStatus.COMPARABLE

    rel = engine.determine_relationship(f1, f2, comp)
    assert rel.relationship_type == RelationshipType.EVOLVES_FROM
    assert "DATA_VINTAGE_EVOLUTION" in rel.reason_codes


# ---------------------------------------------------------------------------
# 7. Test 7: Target vs Reported -> UNRESOLVED (Not Contradiction)
# ---------------------------------------------------------------------------

def test_target_vs_reported_unresolved(gate, engine):
    """Test 7: Target (120) vs Reported (100) must NOT become CONTRADICTS."""
    f_rep = _make_norm_fact("f1", "Company", "Revenue", "100", numeric_value=100.0, epistemic_status=EpistemicStatus.REPORTED)
    f_tgt = _make_norm_fact("f2", "Company", "Revenue", "120", numeric_value=120.0, epistemic_status=EpistemicStatus.TARGET)

    comp = gate.evaluate(f_rep, f_tgt)
    assert comp.status == ComparabilityStatus.COMPARABLE

    rel = engine.determine_relationship(f_rep, f_tgt, comp)
    assert rel.relationship_type == RelationshipType.UNRESOLVED
    assert "EPISTEMIC_STATUS_INCOMPATIBLE" in rel.reason_codes
    assert rel.relationship_type != RelationshipType.CONTRADICTS


# ---------------------------------------------------------------------------
# 8. Test 8: Supersession with Explicit Evidence -> SUPERSEDES
# ---------------------------------------------------------------------------

def test_explicit_supersession_supersedes(gate, engine):
    """Test 8: Explicit restatement / supersession context -> SUPERSEDES."""
    f1 = _make_norm_fact("f1", "Company", "Revenue", "100", numeric_value=100.0)
    f2 = _make_norm_fact(
        "f2", "Company", "Revenue", "105", numeric_value=105.0,
        supporting_text="Subsequently restated revenue from operations to ₹105 million"
    )

    comp = gate.evaluate(f1, f2)
    assert comp.status == ComparabilityStatus.COMPARABLE

    rel = engine.determine_relationship(f1, f2, comp)
    assert rel.relationship_type == RelationshipType.SUPERSEDES
    assert "EXPLICIT_SUPERSEDED_CLAIM" in rel.reason_codes


# ---------------------------------------------------------------------------
# 9. Test 9: Later Document Without Revision Evidence
# ---------------------------------------------------------------------------

def test_different_vintage_without_revision_context_unresolved(gate, engine):
    """Test 9: Different vintage labels without sequential revision context -> UNRESOLVED."""
    f1 = _make_norm_fact("f1", "Gov", "Metric", "10", numeric_value=10.0, data_vintage="Custom Vintage A")
    f2 = _make_norm_fact("f2", "Gov", "Metric", "12", numeric_value=12.0, data_vintage="Custom Vintage B")

    comp = gate.evaluate(f1, f2)
    assert comp.status == ComparabilityStatus.COMPARABLE

    rel = engine.determine_relationship(f1, f2, comp)
    assert rel.relationship_type == RelationshipType.UNRESOLVED
    assert "UNRESOLVED_VINTAGE_RELATIONSHIP" in rel.reason_codes


# ---------------------------------------------------------------------------
# 10. Test 10: Evidence Preservation
# ---------------------------------------------------------------------------

def test_evidence_preservation_in_relationship_result(gate, engine):
    """Test 10: Verify both provenance objects are retained in RelationshipResult."""
    f1 = _make_norm_fact("f1", "Delhivery", "Revenue", "₹81,415.38 million", numeric_value=81415380000.0, supporting_text="Revenue on page 12")
    f2 = _make_norm_fact("f2", "Delhivery", "Revenue", "₹81,415.38 million", numeric_value=81415380000.0, supporting_text="Revenue on page 15")

    comp = gate.evaluate(f1, f2)
    rel = engine.determine_relationship(f1, f2, comp)

    assert rel.evidence_a is not None
    assert rel.evidence_a.supporting_text == "Revenue on page 12"
    assert rel.evidence_b is not None
    assert rel.evidence_b.supporting_text == "Revenue on page 15"


# ---------------------------------------------------------------------------
# 11. Test 11: Lexical candidate does not imply semantic equivalence
# ---------------------------------------------------------------------------

def test_lexical_candidate_does_not_imply_semantic_equivalence(matcher, gate, engine):
    """Test 11: Metric A = Employee Count, Metric B = Employee Cost.

    Matcher surfaces candidate due to token 'employee', Gate rejects with METRIC_MISMATCH,
    Engine returns UNRESOLVED.
    """
    f_count = _make_norm_fact("f1", "Company", "Employee Count", "50,000", canonical_unit="count", numeric_value=50000.0)
    f_cost = _make_norm_fact("f2", "Company", "Employee Cost", "₹5,000 million", canonical_unit="INR", numeric_value=5000000000.0)

    # 1. CandidateMatcher surfaces lexical candidate
    cand = matcher.match_pair(f_count, f_cost)
    assert cand is not None
    assert cand.entity_match is True
    assert cand.metric_match is True

    # 2. ComparabilityGate rejects with METRIC_MISMATCH
    comp = gate.evaluate(f_count, f_cost)
    assert comp.status == ComparabilityStatus.NON_COMPARABLE
    assert "METRIC_MISMATCH" in comp.reason_codes

    # 3. RelationshipEngine strictly returns UNRESOLVED
    rel = engine.determine_relationship(f_count, f_cost, comp, candidate_pair=cand)
    assert rel.relationship_type == RelationshipType.UNRESOLVED
    assert "NON_COMPARABLE_CLAIMS" in rel.reason_codes
    assert "METRIC_MISMATCH" in rel.reason_codes


# ---------------------------------------------------------------------------
# 12. Test 12: Determinism & No LLM Dependency
# ---------------------------------------------------------------------------

def test_determinism_and_no_llm(gate, engine):
    """Test 12: Repeated executions yield identical relationship output."""
    f1 = _make_norm_fact("f1", "Delhivery", "Revenue", "₹81,415.38 million", numeric_value=81415380000.0)
    f2 = _make_norm_fact("f2", "Delhivery", "Revenue", "₹8,142 crore", numeric_value=81420000000.0, scale="crore")

    comp = gate.evaluate(f1, f2)
    rel1 = engine.determine_relationship(f1, f2, comp)
    rel2 = engine.determine_relationship(f1, f2, comp)

    assert rel1.model_dump() == rel2.model_dump()
    assert rel1.relationship_id == rel2.relationship_id
