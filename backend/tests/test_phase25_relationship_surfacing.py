"""
Phase 25: Relationship Quality & Candidate Surfacing Unit Tests.

Tests:
1. Türkiye GDP (2025=2.7% vs 2026=3.2%) -> NOT surfaced (DIFFERENT_TIME_PERIOD_NO_LINK).
2. US Debt (2024=121% vs 2030=130%) -> NOT surfaced (DIFFERENT_TIME_PERIOD_NO_LINK).
3. Global growth (2025=2.8% vs past few years=3%) without comparative language -> NOT surfaced.
4. Explicit temporal language ("increased from 2.5 percent in 2024 to 3.0 percent in 2025") -> Surfaced (EXPLICIT_TEMPORAL_COMPARISON).
5. Same-period conflicting values (2025=2.8% vs 2025=3.2%) -> Surfaced (MEANINGFUL_COMPARISON -> CONTRADICTS).
6. Same-period identical values (2025=2.8% vs 2025=2.8%) -> Surfaced (MEANINGFUL_COMPARISON -> CORROBORATES).
7. Employee Count vs Employee Cost -> NOT surfaced (METRIC_MISMATCH_SUPPRESSED).
8. GDP Growth vs GDP Level -> NOT surfaced (METRIC_MISMATCH_SUPPRESSED).
9. Missing scope with otherwise aligned claim -> Surfaced as UNRESOLVED (INSUFFICIENT_CONTEXT_MEANINGFUL).
10. Cross-document same-period comparison -> Surfaced.
11. No numeric proximity -> Close values across different periods are NOT surfaced.
12. Evidence preservation -> Surfaced relationships retain both provenance records.
"""

import pytest
from backend.models.document import ParsedDocument, PageText
from backend.models.fact import FactRecord, Provenance, TimePeriod, EpistemicStatus
from backend.models.normalization import NormalizedFact, NormalizedValue, NormalizationStatus
from backend.models.relationship import ComparabilityStatus, RelationshipType
from backend.reasoning.matcher import CandidateMatcher
from backend.reasoning.comparability import ComparabilityGate
from backend.reasoning.surfacing import RelationshipSurfacingFilter
from backend.reasoning.relationships import RelationshipEngine
from backend.services.analysis import AnalysisService


def _create_normalized_fact(
    fact_id: str,
    entity: str,
    metric: str,
    value_raw: str,
    value_numeric: float,
    unit: str,
    time_label: str,
    start_date: str,
    end_date: str,
    supporting_text: str,
    scope: str = None,
    geography: str = None,
    doc_id: str = "doc1",
    page_num: int = 1,
    epistemic_status: EpistemicStatus = EpistemicStatus.REPORTED,
) -> NormalizedFact:
    prov = Provenance(
        document_id=doc_id,
        page_number=page_num,
        supporting_text=supporting_text,
    )
    tp = TimePeriod(label=time_label, start_date=start_date, end_date=end_date)
    fact = FactRecord(
        fact_id=fact_id,
        entity=entity,
        metric=metric,
        value_raw=value_raw,
        value_numeric=value_numeric,
        unit=unit,
        time_period=tp,
        scope=scope,
        geography=geography,
        epistemic_status=epistemic_status,
        provenance=prov,
        extraction_confidence=1.0,
    )
    norm_val = NormalizedValue(
        original_value_raw=value_raw,
        numeric_value=value_numeric,
        canonical_unit=unit,
        normalization_status=NormalizationStatus.NORMALIZED,
    )
    return NormalizedFact(
        fact=fact,
        canonical_entity=entity.lower(),
        canonical_metric=metric.lower(),
        normalized_value=norm_val,
        normalized_time_period=tp,
    )


# Test 1: Türkiye GDP: 2025 = 2.7%, 2026 = 3.2% -> NOT surfaced
def test_turkiye_gdp_different_years_suppressed():
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()

    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="Türkiye",
        metric="GDP Growth",
        value_raw="2.7%",
        value_numeric=2.7,
        unit="percent",
        time_label="2025",
        start_date="2025-01-01",
        end_date="2025-12-31",
        supporting_text="Türkiye's GDP growth is projected at 2.7 percent in 2025.",
    )
    f2 = _create_normalized_fact(
        fact_id="f2",
        entity="Türkiye",
        metric="GDP Growth",
        value_raw="3.2%",
        value_numeric=3.2,
        unit="percent",
        time_label="2026",
        start_date="2026-01-01",
        end_date="2026-12-31",
        supporting_text="Türkiye's GDP growth is projected at 3.2 percent in 2026.",
    )

    cand = matcher.match_pair(f1, f2)
    assert cand is not None
    comp = gate.evaluate(f1, f2)
    assert comp.status == ComparabilityStatus.NON_COMPARABLE
    assert "TIME_MISMATCH" in comp.reason_codes

    decision = surfacing.evaluate(f1, f2, comp, cand)
    assert decision.should_surface is False
    assert "DIFFERENT_TIME_PERIOD_NO_LINK" in decision.reason_codes


# Test 2: US debt: 2024 = 121%, 2030 = 130% -> NOT surfaced
def test_us_debt_different_years_suppressed():
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()

    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="US",
        metric="Public Debt",
        value_raw="121% of GDP",
        value_numeric=121.0,
        unit="percent",
        time_label="2024",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="US general government gross debt reached 121 percent of GDP in 2024.",
    )
    f2 = _create_normalized_fact(
        fact_id="f2",
        entity="US",
        metric="Public Debt",
        value_raw="130% of GDP",
        value_numeric=130.0,
        unit="percent",
        time_label="2030",
        start_date="2030-01-01",
        end_date="2030-12-31",
        supporting_text="US debt is projected to exceed 130 percent of GDP by 2030.",
    )

    cand = matcher.match_pair(f1, f2)
    comp = gate.evaluate(f1, f2)
    decision = surfacing.evaluate(f1, f2, comp, cand)
    assert decision.should_surface is False
    assert "DIFFERENT_TIME_PERIOD_NO_LINK" in decision.reason_codes


# Test 3: Global growth: 2025 = 2.8% vs past few years = around 3% -> NOT surfaced
def test_global_growth_vague_past_years_suppressed_without_explicit_link():
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()

    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="Global",
        metric="GDP Growth",
        value_raw="2.8%",
        value_numeric=2.8,
        unit="percent",
        time_label="2025",
        start_date="2025-01-01",
        end_date="2025-12-31",
        supporting_text="Global growth is projected to decelerate to 2.8 percent in 2025.",
    )
    f2 = _create_normalized_fact(
        fact_id="f2",
        entity="Global",
        metric="GDP Growth",
        value_raw="around 3%",
        value_numeric=3.0,
        unit="percent",
        time_label="2021-2023",
        start_date="2021-01-01",
        end_date="2023-12-31",
        supporting_text="The global economy grew by around 3 percent in the past few years.",
    )

    cand = matcher.match_pair(f1, f2)
    comp = gate.evaluate(f1, f2)
    decision = surfacing.evaluate(f1, f2, comp, cand)
    assert decision.should_surface is False
    assert "DIFFERENT_TIME_PERIOD_NO_LINK" in decision.reason_codes


# Test 4: Explicit temporal language -> Surfaced
def test_explicit_temporal_comparison_language_surfaced():
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()

    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="Global",
        metric="GDP Growth",
        value_raw="2.5%",
        value_numeric=2.5,
        unit="percent",
        time_label="2024",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="Growth increased from 2.5 percent in 2024 to 3.0 percent in 2025.",
    )
    f2 = _create_normalized_fact(
        fact_id="f2",
        entity="Global",
        metric="GDP Growth",
        value_raw="3.0%",
        value_numeric=3.0,
        unit="percent",
        time_label="2025",
        start_date="2025-01-01",
        end_date="2025-12-31",
        supporting_text="Growth increased from 2.5 percent in 2024 to 3.0 percent in 2025.",
    )

    cand = matcher.match_pair(f1, f2)
    comp = gate.evaluate(f1, f2)
    decision = surfacing.evaluate(f1, f2, comp, cand)
    assert decision.should_surface is True
    assert "EXPLICIT_TEMPORAL_COMPARISON" in decision.reason_codes


# Test 5: Same-period conflicting values -> Surfaced (CONTRADICTS)
def test_same_period_conflicting_values_surfaced():
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()
    rel_engine = RelationshipEngine()

    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="Acme Corp",
        metric="Revenue",
        value_raw="$500M",
        value_numeric=500.0,
        unit="usd",
        time_label="FY24",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="Revenue was $500M in FY24.",
    )
    f2 = _create_normalized_fact(
        fact_id="f2",
        entity="Acme Corp",
        metric="Revenue",
        value_raw="$550M",
        value_numeric=550.0,
        unit="usd",
        time_label="FY24",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="Revenue reached $550M for FY24.",
    )

    cand = matcher.match_pair(f1, f2)
    comp = gate.evaluate(f1, f2)
    assert comp.status == ComparabilityStatus.COMPARABLE

    decision = surfacing.evaluate(f1, f2, comp, cand)
    assert decision.should_surface is True
    assert "MEANINGFUL_COMPARISON" in decision.reason_codes

    rel = rel_engine.determine_relationship(f1, f2, comp, cand)
    assert rel.relationship_type == RelationshipType.CONTRADICTS


# Test 6: Same-period identical values -> Surfaced (CORROBORATES)
def test_same_period_identical_values_surfaced():
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()
    rel_engine = RelationshipEngine()

    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="Global",
        metric="GDP Growth",
        value_raw="2.8%",
        value_numeric=2.8,
        unit="percent",
        time_label="2025",
        start_date="2025-01-01",
        end_date="2025-12-31",
        supporting_text="Global growth is projected at 2.8 percent in 2025.",
    )
    f2 = _create_normalized_fact(
        fact_id="f2",
        entity="Global",
        metric="GDP Growth",
        value_raw="2.8%",
        value_numeric=2.8,
        unit="percent",
        time_label="2025",
        start_date="2025-01-01",
        end_date="2025-12-31",
        supporting_text="2025 world GDP expansion is expected to be 2.8 percent.",
    )

    cand = matcher.match_pair(f1, f2)
    comp = gate.evaluate(f1, f2)
    decision = surfacing.evaluate(f1, f2, comp, cand)
    assert decision.should_surface is True
    assert "MEANINGFUL_COMPARISON" in decision.reason_codes

    rel = rel_engine.determine_relationship(f1, f2, comp, cand)
    assert rel.relationship_type == RelationshipType.CORROBORATES


# Test 7: Employee Count vs Employee Cost -> NOT surfaced
def test_employee_count_vs_employee_cost_suppressed():
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()

    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="Acme Corp",
        metric="Employee Count",
        value_raw="15,000",
        value_numeric=15000.0,
        unit="count",
        time_label="2024",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="Total employee count stood at 15,000 employees.",
    )
    f2 = _create_normalized_fact(
        fact_id="f2",
        entity="Acme Corp",
        metric="Employee Cost",
        value_raw="$150M",
        value_numeric=150.0,
        unit="usd",
        time_label="2024",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="Total employee cost was $150M in 2024.",
    )

    comp = gate.evaluate(f1, f2)
    assert comp.status == ComparabilityStatus.NON_COMPARABLE
    assert "METRIC_MISMATCH" in comp.reason_codes

    decision = surfacing.evaluate(f1, f2, comp, None)
    assert decision.should_surface is False
    assert "METRIC_MISMATCH_SUPPRESSED" in decision.reason_codes


# Test 8: GDP Growth vs GDP Level -> NOT surfaced
def test_gdp_growth_vs_gdp_level_suppressed():
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()

    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="India",
        metric="GDP Growth",
        value_raw="7.0%",
        value_numeric=7.0,
        unit="percent",
        time_label="2024",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="India's real GDP growth reached 7.0 percent in 2024.",
    )
    f2 = _create_normalized_fact(
        fact_id="f2",
        entity="India",
        metric="GDP Level",
        value_raw="$3.75T",
        value_numeric=3.75,
        unit="usd",
        time_label="2024",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="India's nominal GDP level was $3.75 trillion in 2024.",
    )

    comp = gate.evaluate(f1, f2)
    assert comp.status == ComparabilityStatus.NON_COMPARABLE
    assert "METRIC_MISMATCH" in comp.reason_codes

    decision = surfacing.evaluate(f1, f2, comp, None)
    assert decision.should_surface is False
    assert "METRIC_MISMATCH_SUPPRESSED" in decision.reason_codes


# Test 9: Missing scope with otherwise aligned claim -> Surfaced as UNRESOLVED
def test_missing_scope_otherwise_aligned_surfaced_as_unresolved():
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()
    rel_engine = RelationshipEngine()

    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="Acme Corp",
        metric="Revenue",
        value_raw="$500M",
        value_numeric=500.0,
        unit="usd",
        time_label="FY24",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="Consolidated revenue reached $500M in FY24.",
        scope="consolidated",
    )
    f2 = _create_normalized_fact(
        fact_id="f2",
        entity="Acme Corp",
        metric="Revenue",
        value_raw="$520M",
        value_numeric=520.0,
        unit="usd",
        time_label="FY24",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="Total revenue was reported at $520M in FY24.",
        scope=None,  # Unspecified scope -> INSUFFICIENT_CONTEXT
    )

    cand = matcher.match_pair(f1, f2)
    assert cand is not None
    comp = gate.evaluate(f1, f2)
    assert comp.status == ComparabilityStatus.INSUFFICIENT_CONTEXT
    assert "MISSING_SCOPE" in comp.reason_codes

    decision = surfacing.evaluate(f1, f2, comp, cand)
    assert decision.should_surface is True
    assert "INSUFFICIENT_CONTEXT_MEANINGFUL" in decision.reason_codes

    rel = rel_engine.determine_relationship(f1, f2, comp, cand)
    assert rel.relationship_type == RelationshipType.UNRESOLVED


# Test 10: Cross-document same-period comparison -> Surfaced
def test_cross_document_same_period_comparison_surfaced():
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()

    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="India",
        metric="GDP Growth",
        value_raw="6.8%",
        value_numeric=6.8,
        unit="percent",
        time_label="2024",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="IMF estimated India's 2024 GDP growth at 6.8 percent.",
        doc_id="doc_imf.pdf",
    )
    f2 = _create_normalized_fact(
        fact_id="f2",
        entity="India",
        metric="GDP Growth",
        value_raw="7.0%",
        value_numeric=7.0,
        unit="percent",
        time_label="2024",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="World Bank projected India's 2024 GDP growth at 7.0 percent.",
        doc_id="doc_wb.pdf",
    )

    cand = matcher.match_pair(f1, f2)
    assert cand is not None
    comp = gate.evaluate(f1, f2)
    assert comp.status == ComparabilityStatus.COMPARABLE

    decision = surfacing.evaluate(f1, f2, comp, cand)
    assert decision.should_surface is True
    assert "MEANINGFUL_COMPARISON" in decision.reason_codes


# Test 11: No numeric proximity dependency
def test_no_numeric_proximity_dependency():
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()

    # 2.7% and 2.71% are extremely close numerically, but across different years
    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="Türkiye",
        metric="GDP Growth",
        value_raw="2.7%",
        value_numeric=2.7,
        unit="percent",
        time_label="2024",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="Türkiye's GDP growth was 2.7 percent in 2024.",
    )
    f2 = _create_normalized_fact(
        fact_id="f2",
        entity="Türkiye",
        metric="GDP Growth",
        value_raw="2.71%",
        value_numeric=2.71,
        unit="percent",
        time_label="2025",
        start_date="2025-01-01",
        end_date="2025-12-31",
        supporting_text="Türkiye's GDP growth is forecast at 2.71 percent in 2025.",
    )

    cand = matcher.match_pair(f1, f2)
    comp = gate.evaluate(f1, f2)
    decision = surfacing.evaluate(f1, f2, comp, cand)
    # Proximity MUST NOT make it surfacable
    assert decision.should_surface is False
    assert "DIFFERENT_TIME_PERIOD_NO_LINK" in decision.reason_codes


# Test 12: Dual evidence preservation on surfaced relationships
def test_surfaced_relationship_preserves_dual_evidence():
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()
    rel_engine = RelationshipEngine()

    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="Global",
        metric="GDP Growth",
        value_raw="2.8%",
        value_numeric=2.8,
        unit="percent",
        time_label="2025",
        start_date="2025-01-01",
        end_date="2025-12-31",
        supporting_text="Global growth is projected at 2.8 percent in 2025.",
        doc_id="report_a.pdf",
        page_num=4,
    )
    f2 = _create_normalized_fact(
        fact_id="f2",
        entity="Global",
        metric="GDP Growth",
        value_raw="3.2%",
        value_numeric=3.2,
        unit="percent",
        time_label="2025",
        start_date="2025-01-01",
        end_date="2025-12-31",
        supporting_text="World output expansion is estimated at 3.2 percent in 2025.",
        doc_id="report_b.pdf",
        page_num=9,
    )

    cand = matcher.match_pair(f1, f2)
    comp = gate.evaluate(f1, f2)
    decision = surfacing.evaluate(f1, f2, comp, cand)
    assert decision.should_surface is True

    rel = rel_engine.determine_relationship(f1, f2, comp, cand)
    assert rel.evidence_a is not None
    assert rel.evidence_a.document_id == "report_a.pdf"
    assert rel.evidence_a.page_number == 4
    assert rel.evidence_a.supporting_text == "Global growth is projected at 2.8 percent in 2025."

    assert rel.evidence_b is not None
    assert rel.evidence_b.document_id == "report_b.pdf"
    assert rel.evidence_b.page_number == 9
    assert rel.evidence_b.supporting_text == "World output expansion is estimated at 3.2 percent in 2025."


# Regression Tests for Phase 25 Constraints

def test_same_fact_duplicate_self_comparison_suppressed():
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()

    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="Acme Corp",
        metric="Revenue",
        value_raw="$500M",
        value_numeric=500.0,
        unit="usd",
        time_label="2024",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="Revenue was $500M in 2024.",
    )

    comp = gate.evaluate(f1, f1)
    decision = surfacing.evaluate(f1, f1, comp, None)
    assert decision.should_surface is False
    assert "SAME_DOCUMENT_DUPLICATE" in decision.reason_codes


@pytest.mark.parametrize(
    "phrase",
    [
        "down from",
        "decreased from",
        "rose from",
        "fell from",
        "compared with",
        "compared to",
        "versus",
        "vs.",
        "previously",
        "earlier",
    ],
)
def test_all_temporal_comparison_phrases_surfaced(phrase: str):
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()

    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="Global",
        metric="GDP Growth",
        value_raw="3.0%",
        value_numeric=3.0,
        unit="percent",
        time_label="2024",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text=f"Growth was 2.5% in 2025, {phrase} 3.0% recorded in 2024.",
    )
    f2 = _create_normalized_fact(
        fact_id="f2",
        entity="Global",
        metric="GDP Growth",
        value_raw="2.5%",
        value_numeric=2.5,
        unit="percent",
        time_label="2025",
        start_date="2025-01-01",
        end_date="2025-12-31",
        supporting_text=f"Growth was 2.5% in 2025, {phrase} 3.0% recorded in 2024.",
    )

    cand = matcher.match_pair(f1, f2)
    comp = gate.evaluate(f1, f2)
    decision = surfacing.evaluate(f1, f2, comp, cand)
    assert decision.should_surface is True
    assert "EXPLICIT_TEMPORAL_COMPARISON" in decision.reason_codes


def test_insufficient_context_weak_candidate_suppressed():
    gate = ComparabilityGate()
    surfacing = RelationshipSurfacingFilter()

    f1 = _create_normalized_fact(
        fact_id="f1",
        entity="Acme Corp",
        metric="Revenue",
        value_raw="$500M",
        value_numeric=500.0,
        unit="usd",
        time_label="2024",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="Revenue was $500M in 2024.",
        scope="consolidated",
    )
    f2 = _create_normalized_fact(
        fact_id="f2",
        entity="Acme Corp",
        metric="Revenue",
        value_raw="$520M",
        value_numeric=520.0,
        unit="usd",
        time_label="2024",
        start_date="2024-01-01",
        end_date="2024-12-31",
        supporting_text="Revenue was $520M in 2024.",
        scope=None,
    )

    comp = gate.evaluate(f1, f2)
    assert comp.status == ComparabilityStatus.INSUFFICIENT_CONTEXT

    # If candidate_pair is None or metric_match is False
    decision = surfacing.evaluate(f1, f2, comp, candidate_pair=None)
    assert decision.should_surface is False
    assert "WEAK_SEMANTIC_CANDIDATE" in decision.reason_codes


def test_analysis_service_filters_unrelated_cross_period_relationships(tmp_path):
    """End-to-end integration test verifying that AnalysisService properly applies surfacing filter."""
    from unittest.mock import MagicMock
    from backend.db.database import DatabaseRepository
    from backend.extraction.fact_extractor import FactExtractor
    from backend.extraction.pdf_parser import PDFParser
    from backend.normalization.normalizer import FactNormalizer

    # Mock parser & extractor to return our test facts
    mock_parser = MagicMock(spec=PDFParser)
    mock_doc = ParsedDocument(
        document_id="doc1",
        document_name="sample.pdf",
        total_pages=1,
        pages=[PageText(page_number=1, text="Sample text", char_count=11, has_text=True)],
    )
    mock_parser.parse_bytes.return_value = mock_doc

    f_tr_2025 = _create_normalized_fact(
        fact_id="f_tr_2025",
        entity="Türkiye",
        metric="GDP Growth",
        value_raw="2.7%",
        value_numeric=2.7,
        unit="percent",
        time_label="2025",
        start_date="2025-01-01",
        end_date="2025-12-31",
        supporting_text="Türkiye's GDP growth is projected at 2.7 percent in 2025.",
    ).fact

    f_tr_2026 = _create_normalized_fact(
        fact_id="f_tr_2026",
        entity="Türkiye",
        metric="GDP Growth",
        value_raw="3.2%",
        value_numeric=3.2,
        unit="percent",
        time_label="2026",
        start_date="2026-01-01",
        end_date="2026-12-31",
        supporting_text="Türkiye's GDP growth is projected at 3.2 percent in 2026.",
    ).fact

    f_world_a = _create_normalized_fact(
        fact_id="f_world_a",
        entity="Global",
        metric="GDP Growth",
        value_raw="2.8%",
        value_numeric=2.8,
        unit="percent",
        time_label="2025",
        start_date="2025-01-01",
        end_date="2025-12-31",
        supporting_text="Global growth is projected at 2.8 percent in 2025.",
    ).fact

    f_world_b = _create_normalized_fact(
        fact_id="f_world_b",
        entity="Global",
        metric="GDP Growth",
        value_raw="3.0%",
        value_numeric=3.0,
        unit="percent",
        time_label="2025",
        start_date="2025-01-01",
        end_date="2025-12-31",
        supporting_text="Global growth is projected at 3.0 percent in 2025.",
    ).fact

    mock_extractor = MagicMock(spec=FactExtractor)
    mock_extractor.extract_from_document.return_value = [f_tr_2025, f_tr_2026, f_world_a, f_world_b]

    db_path = str(tmp_path / "test_analysis.db")
    repo = DatabaseRepository(db_path=db_path)

    service = AnalysisService(
        parser=mock_parser,
        extractor=mock_extractor,
        normalizer=FactNormalizer(),
        matcher=CandidateMatcher(),
        gate=ComparabilityGate(),
        surfacing_filter=RelationshipSurfacingFilter(),
        rel_engine=RelationshipEngine(),
        repository=repo,
    )

    result = service.analyze_documents([("sample.pdf", b"%PDF-1.4 dummy")])
    summary = result["summary"]

    # Candidate pairs were found for both (Türkiye 2025 vs 2026, and Global 2025 vs 2025)
    assert summary["candidate_pairs"] >= 2

    # But only the same-period Global comparison (2025 vs 2025) surfaced as a relationship!
    # The Türkiye cross-period comparison (2025 vs 2026) was suppressed by RelationshipSurfacingFilter.
    assert summary["relationships_created"] == 1
    analysis_record = service.get_analysis(result["analysis_id"])
    assert len(analysis_record["relationships"]) == 1
    surfaced_rel = analysis_record["relationships"][0]
    assert surfaced_rel["fact_a_id"] in ["f_world_a", "f_world_b"]
    assert surfaced_rel["fact_b_id"] in ["f_world_a", "f_world_b"]

