"""Candidate Fact Matching and Comparability Gate test suite."""

import pytest
from backend.models.fact import EpistemicStatus, FactRecord, Provenance, TimePeriod
from backend.models.normalization import NormalizationStatus, NormalizedFact, NormalizedValue
from backend.models.relationship import CandidatePair, ComparabilityResult, ComparabilityStatus
from backend.normalization.normalizer import FactNormalizer
from backend.reasoning.comparability import ComparabilityGate
from backend.reasoning.matcher import CandidateMatcher


@pytest.fixture
def gate():
    return ComparabilityGate()


@pytest.fixture
def matcher():
    return CandidateMatcher()


@pytest.fixture
def fact_normalizer():
    return FactNormalizer()


def _make_norm_fact(
    fact_id: str,
    entity: str,
    metric: str,
    value_raw: str,
    canonical_unit: str = "INR",
    numeric_value: float = 1000.0,
    start_date: str = "2023-04-01",
    end_date: str = "2024-03-31",
    scope: str = None,
    geography: str = None,
    epistemic_status: EpistemicStatus = EpistemicStatus.REPORTED,
    data_vintage: str = None,
    has_time_period: bool = True,
) -> NormalizedFact:
    """Helper to construct NormalizedFact instances for comparability tests."""
    prov = Provenance(document_id="doc-test", page_number=1, supporting_text="Sample text")
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
        original_value_raw=value_raw,
        normalization_status=NormalizationStatus.NORMALIZED,
    )

    norm_tp = TimePeriod(label=tp_label, start_date=start_date, end_date=end_date) if has_time_period else None

    # Canonicalize entity & metric cleanly
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
# 1. Candidate Matcher Tests
# ---------------------------------------------------------------------------

def test_candidate_matcher_exact_and_token_overlap(matcher):
    """Test candidate matching on exact metric match and overlapping tokens."""
    f1 = _make_norm_fact("f1", "Delhivery Limited", "Revenue from Operations", "₹81,415.38 million")
    f2 = _make_norm_fact("f2", "Delhivery Ltd.", "Revenue from Operations", "₹8,142 Cr")
    f3 = _make_norm_fact("f3", "Delhivery", "Revenue", "₹81,415.38 million")
    f_diff_ent = _make_norm_fact("f4", "Blue Dart", "Revenue from Operations", "₹5,000 Cr")
    f_diff_met = _make_norm_fact("f5", "Delhivery", "Active Customers", "33,278", canonical_unit="count")

    # Exact entity + exact metric -> Candidate (score 1.0)
    cand_exact = matcher.match_pair(f1, f2)
    assert cand_exact is not None
    assert cand_exact.entity_match is True
    assert cand_exact.metric_match is True
    assert cand_exact.candidate_score == 1.0

    # Exact entity + subset overlap ('revenue') -> Candidate (score 0.85)
    cand_overlap = matcher.match_pair(f1, f3)
    assert cand_overlap is not None
    assert cand_overlap.entity_match is True
    assert cand_overlap.candidate_score >= 0.75

    # Different entity -> None
    assert matcher.match_pair(f1, f_diff_ent) is None

    # Completely disjoint metric -> None
    assert matcher.match_pair(f1, f_diff_met) is None


def test_candidate_matcher_rejects_unrelated_metric_phrases(matcher):
    """Phase 9 Test: Reject candidate generation for noisy/unrelated phrases sharing generic words."""
    # Example 1: 'Part-truckload freight delivered since inception' vs 'Freight service centers'
    f_ptl_freight = _make_norm_fact(
        "f_ptl", "Delhivery", "Part-truckload freight delivered since inception",
        ">4.8Mn tonnes", canonical_unit="tonnes", numeric_value=4800000.0
    )
    f_service_centers = _make_norm_fact(
        "f_sc", "Delhivery", "Freight service centers",
        "150", canonical_unit="count", numeric_value=150.0
    )
    assert matcher.match_pair(f_ptl_freight, f_service_centers) is None

    # Example 2: 'Express parcel shipments delivered since inception' vs 'Part-truckload freight delivered since inception'
    f_exp_shipments = _make_norm_fact(
        "f_exp", "Delhivery", "Express parcel shipments delivered since inception",
        ">2.8Bn", canonical_unit="shipments", numeric_value=2800000000.0
    )
    assert matcher.match_pair(f_exp_shipments, f_ptl_freight) is None


def test_candidate_matcher_discovers_legitimate_variants(matcher):
    """Phase 9 Test: Discover legitimate semantic metric variants (subset/modifier/synonym)."""
    # Revenue vs Revenue from operations
    f_rev = _make_norm_fact("f_rev", "Acme", "Revenue", "100 million USD", canonical_unit="USD")
    f_rev_ops = _make_norm_fact("f_rev_ops", "Acme", "Revenue from operations", "100 million USD", canonical_unit="USD")
    match_rev = matcher.match_pair(f_rev, f_rev_ops)
    assert match_rev is not None
    assert match_rev.candidate_score >= 0.75

    # Employee count vs Number of employees
    f_emp1 = _make_norm_fact("f_emp1", "Acme", "Employee count", "1200", canonical_unit="count")
    f_emp2 = _make_norm_fact("f_emp2", "Acme", "Number of employees", "1200", canonical_unit="count")
    match_emp = matcher.match_pair(f_emp1, f_emp2)
    assert match_emp is not None
    assert match_emp.candidate_score >= 0.75


# ---------------------------------------------------------------------------
# 2. Case A: Comparable Facts (Delhivery Revenue)
# ---------------------------------------------------------------------------

def test_case_a_delhivery_revenue_comparable(gate):
    """Case A: Same entity, metric, unit (INR), and period (FY24) -> COMPARABLE."""
    fact_a = _make_norm_fact(
        fact_id="fact-delhivery-a",
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value_raw="₹81,415.38 million",
        canonical_unit="INR",
        numeric_value=81415380000.0,
        start_date="2023-04-01",
        end_date="2024-03-31",
    )
    fact_b = _make_norm_fact(
        fact_id="fact-delhivery-b",
        entity="Delhivery Ltd.",
        metric="Revenue from Operations",
        value_raw="₹8,142 Cr",
        canonical_unit="INR",
        numeric_value=81420000000.0,
        start_date="2023-04-01",
        end_date="2024-03-31",
    )

    result = gate.evaluate(fact_a, fact_b)
    assert result.status == ComparabilityStatus.COMPARABLE
    assert result.compared_dimensions["entity"].startswith("compatible")
    assert result.compared_dimensions["metric"].startswith("compatible")
    assert result.compared_dimensions["unit"].startswith("compatible")
    assert result.compared_dimensions["time_period"].startswith("compatible")


# ---------------------------------------------------------------------------
# 3. Case B: Non-Comparable Metric (Revenue vs Employees)
# ---------------------------------------------------------------------------

def test_case_b_metric_mismatch_non_comparable(gate):
    """Case B: Different metrics (Revenue vs Employees) -> NON_COMPARABLE with METRIC_MISMATCH."""
    fact_rev = _make_norm_fact("f-rev", "Delhivery", "Revenue", "₹81,415.38 million", canonical_unit="INR")
    fact_emp = _make_norm_fact("f-emp", "Delhivery", "Employees", "50,000", canonical_unit="count")

    result = gate.evaluate(fact_rev, fact_emp)
    assert result.status == ComparabilityStatus.NON_COMPARABLE
    assert "METRIC_MISMATCH" in result.reason_codes
    assert "UNIT_MISMATCH" in result.reason_codes


# ---------------------------------------------------------------------------
# 4. Case C: Insufficient Context (Missing Time Period)
# ---------------------------------------------------------------------------

def test_case_c_missing_time_period_insufficient_context(gate):
    """Case C: India GDP growth FY25 vs no time period -> INSUFFICIENT_CONTEXT."""
    fact_a = _make_norm_fact(
        fact_id="f-gdp-a",
        entity="India",
        metric="Real GDP Growth",
        value_raw="6.4%",
        canonical_unit="percent",
        numeric_value=6.4,
        start_date="2024-04-01",
        end_date="2025-03-31",
        has_time_period=True,
    )
    fact_b = _make_norm_fact(
        fact_id="f-gdp-b",
        entity="India",
        metric="Real GDP Growth",
        value_raw="6.5%",
        canonical_unit="percent",
        numeric_value=6.5,
        has_time_period=False,
    )

    result = gate.evaluate(fact_a, fact_b)
    assert result.status == ComparabilityStatus.INSUFFICIENT_CONTEXT
    assert "MISSING_TIME_PERIOD" in result.reason_codes


# ---------------------------------------------------------------------------
# 5. Case D: Time Granularity Mismatch (FY24 vs Q4 FY24)
# ---------------------------------------------------------------------------

def test_case_d_time_granularity_mismatch_non_comparable(gate):
    """Case D: FY24 full year vs Q4 FY24 quarter -> NON_COMPARABLE (containment is not equality)."""
    fact_annual = _make_norm_fact(
        fact_id="f-annual",
        entity="Delhivery",
        metric="Revenue from Operations",
        value_raw="₹81,415.38 million",
        start_date="2023-04-01",
        end_date="2024-03-31",
    )
    fact_quarterly = _make_norm_fact(
        fact_id="f-q4",
        entity="Delhivery",
        metric="Revenue from Operations",
        value_raw="₹20,000 million",
        start_date="2024-01-01",
        end_date="2024-03-31",
    )

    result = gate.evaluate(fact_annual, fact_quarterly)
    assert result.status == ComparabilityStatus.NON_COMPARABLE
    assert "TIME_MISMATCH" in result.reason_codes


# ---------------------------------------------------------------------------
# 6. Case E: Data Vintage Difference (First vs Second Advance Estimate)
# ---------------------------------------------------------------------------

def test_case_e_data_vintage_difference_comparable(gate):
    """Case E: First vs Second Advance Estimate -> COMPARABLE with DATA_VINTAGE_DIFFERENCE diagnostic."""
    fact_ae1 = _make_norm_fact(
        fact_id="f-ae1",
        entity="India",
        metric="Real GDP Growth",
        value_raw="6.4%",
        canonical_unit="percent",
        numeric_value=6.4,
        start_date="2024-04-01",
        end_date="2025-03-31",
        data_vintage="First Advance Estimate",
    )
    fact_ae2 = _make_norm_fact(
        fact_id="f-ae2",
        entity="India",
        metric="Real GDP Growth",
        value_raw="6.5%",
        canonical_unit="percent",
        numeric_value=6.5,
        start_date="2024-04-01",
        end_date="2025-03-31",
        data_vintage="Second Advance Estimate",
    )

    result = gate.evaluate(fact_ae1, fact_ae2)
    assert result.status == ComparabilityStatus.COMPARABLE
    assert "DATA_VINTAGE_DIFFERENCE" in result.reason_codes


# ---------------------------------------------------------------------------
# 7. Epistemic Status & Unit Mismatch Tests
# ---------------------------------------------------------------------------

def test_epistemic_status_difference_preserved_as_diagnostic(gate):
    """Estimated vs Reported claims remain COMPARABLE with diagnostic code."""
    fact_est = _make_norm_fact(
        "f-est", "India", "Real GDP Growth", "6.4%", canonical_unit="percent",
        epistemic_status=EpistemicStatus.ESTIMATED
    )
    fact_rep = _make_norm_fact(
        "f-rep", "India", "Real GDP Growth", "6.5%", canonical_unit="percent",
        epistemic_status=EpistemicStatus.REPORTED
    )

    result = gate.evaluate(fact_est, fact_rep)
    assert result.status == ComparabilityStatus.COMPARABLE
    assert "EPISTEMIC_STATUS_DIFFERENCE" in result.reason_codes


def test_unit_mismatch_non_comparable(gate):
    """INR vs percent -> NON_COMPARABLE."""
    fact_curr = _make_norm_fact("f-curr", "Entity", "Metric", "100", canonical_unit="INR")
    fact_pct = _make_norm_fact("f-pct", "Entity", "Metric", "10%", canonical_unit="percent")

    result = gate.evaluate(fact_curr, fact_pct)
    assert result.status == ComparabilityStatus.NON_COMPARABLE
    assert "UNIT_MISMATCH" in result.reason_codes


# ---------------------------------------------------------------------------
# 8. Scope & Geography Dimension Tests
# ---------------------------------------------------------------------------

def test_scope_conflict_and_missing(gate):
    """Conflicting scopes -> NON_COMPARABLE; specified vs None -> INSUFFICIENT_CONTEXT."""
    f_cons = _make_norm_fact("f-cons", "Delhivery", "Revenue", "100", scope="Consolidated")
    f_stand = _make_norm_fact("f-stand", "Delhivery", "Revenue", "80", scope="Standalone")
    f_none = _make_norm_fact("f-none", "Delhivery", "Revenue", "100", scope=None)

    # Consolidated vs Standalone -> NON_COMPARABLE
    res_conflict = gate.evaluate(f_cons, f_stand)
    assert res_conflict.status == ComparabilityStatus.NON_COMPARABLE
    assert "SCOPE_MISMATCH" in res_conflict.reason_codes

    # Consolidated vs None -> INSUFFICIENT_CONTEXT (missing is not same)
    res_missing = gate.evaluate(f_cons, f_none)
    assert res_missing.status == ComparabilityStatus.INSUFFICIENT_CONTEXT
    assert "MISSING_SCOPE" in res_missing.reason_codes


def test_geography_conflict_and_missing(gate):
    """Conflicting geography -> NON_COMPARABLE; specified vs None -> INSUFFICIENT_CONTEXT."""
    f_india = _make_norm_fact("f-ind", "Gov", "GDP", "6.4%", canonical_unit="percent", geography="India")
    f_global = _make_norm_fact("f-glob", "Gov", "GDP", "3.2%", canonical_unit="percent", geography="Global")
    f_none = _make_norm_fact("f-none-geo", "Gov", "GDP", "6.4%", canonical_unit="percent", geography=None)

    # India vs Global -> NON_COMPARABLE
    res_conflict = gate.evaluate(f_india, f_global)
    assert res_conflict.status == ComparabilityStatus.NON_COMPARABLE
    assert "GEOGRAPHY_MISMATCH" in res_conflict.reason_codes

    # India vs None -> INSUFFICIENT_CONTEXT
    res_missing = gate.evaluate(f_india, f_none)
    assert res_missing.status == ComparabilityStatus.INSUFFICIENT_CONTEXT
    assert "MISSING_GEOGRAPHY" in res_missing.reason_codes


# ---------------------------------------------------------------------------
# 9. Numeric Independence & Determinism Tests
# ---------------------------------------------------------------------------

def test_numeric_independence(gate):
    """Numerical magnitude difference does NOT cause non-comparability or alter candidate decision."""
    fact_small = _make_norm_fact("f-sm", "Company", "Revenue", "10", canonical_unit="INR", numeric_value=10.0)
    fact_huge = _make_norm_fact("f-hg", "Company", "Revenue", "1000000000", canonical_unit="INR", numeric_value=1000000000.0)

    result = gate.evaluate(fact_small, fact_huge)
    assert result.status == ComparabilityStatus.COMPARABLE


def test_determinism(gate):
    """Identical inputs produce identical ComparabilityResult."""
    f1 = _make_norm_fact("f1", "Delhivery", "Revenue from Operations", "₹81,415.38 million")
    f2 = _make_norm_fact("f2", "Delhivery", "Revenue from Operations", "₹8,142 Cr")

    res1 = gate.evaluate(f1, f2)
    res2 = gate.evaluate(f1, f2)

    assert res1.model_dump() == res2.model_dump()
