"""Comprehensive Integration Tests for Phase 14 Production Extraction Pipeline.

Validates:
1. Single-page extraction interface.
2. Multi-page batching (batch sizes 3 and 2).
3. Cross-page contamination rejection via EvidenceVerifier.
4. Cross-document provenance preservation.
5. Strict evidence verification against authoritative original page text.
6. Page filter disabled fallback.
7. Context selector disabled fallback.
8. Partial quota execution and deferred page planning.
9. HTTP 429 immediate halt and verified fact retention.
10. HTTP 503 conservative retry with backoff.
11. COMPLETE status resolution when budget >= eligible pages.
12. Arbitrary unseen PDF document extraction.
13. Downstream reasoning invariant preservation (Normalization -> Matcher -> Gate -> Relationships).
14. No numeric-only candidate matching invariant.
15. Full AnalysisService end-to-end integration with extraction status persistence.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from backend.db.database import DatabaseRepository
from backend.context_selector.selector import ContextSelector
from backend.extraction.evidence import EvidenceVerifier
from backend.extraction.fact_extractor import (
    BatchExtractionResult,
    ExtractionError,
    ExtractionQuotaError,
    ExtractionStatus,
    FactExtractor,
)
from backend.page_filter.relevance import PageRelevanceFilter
from backend.quota_experiment.planner import QuotaPlanner
from backend.models.document import PageText, ParsedDocument
from backend.models.fact import EpistemicStatus, FactRecord, Provenance, TimePeriod
from backend.models.normalization import NormalizedFact
from backend.models.relationship import ComparabilityStatus, RelationshipType
from backend.normalization.normalizer import FactNormalizer
from backend.reasoning.comparability import ComparabilityGate
from backend.reasoning.matcher import CandidateMatcher
from backend.reasoning.relationships import RelationshipEngine
from backend.services.analysis import AnalysisService


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pages():
    t1 = "Alpha Corp Annual Report FY24.\nTotal Revenue from operations reached $500M in FY24."
    t2 = "Alpha Corp Financial Highlights.\nOperating profit was $120M in FY24 across all business units."
    t3 = "Alpha Corp Operational Metrics.\nTotal full-time employee count stood at 15,000 employees as of March 31, 2024."
    p1 = PageText(
        page_number=1,
        text=t1,
        char_count=len(t1),
        has_text=True,
    )
    p2 = PageText(
        page_number=2,
        text=t2,
        char_count=len(t2),
        has_text=True,
    )
    p3 = PageText(
        page_number=3,
        text=t3,
        char_count=len(t3),
        has_text=True,
    )
    return [p1, p2, p3]


@pytest.fixture
def mock_parsed_document(mock_pages):
    return ParsedDocument(
        document_id="doc-alpha-fy24",
        document_name="Alpha_FY24_Report.pdf",
        total_pages=3,
        pages=mock_pages,
    )


# ---------------------------------------------------------------------------
# 1. Single Page Extraction
# ---------------------------------------------------------------------------

def test_single_page_extraction(mock_pages):
    """Verify single page extraction delegates properly and extracts verified fact."""
    extractor = FactExtractor(page_filter_enabled=False, context_selector_enabled=False)

    gemini_resp = {
        "facts": [
            {
                "page_number": 1,
                "entity": "Alpha Corp",
                "metric": "Revenue from operations",
                "value_raw": "$500M",
                "value_numeric": 500.0,
                "unit": "million USD",
                "time_period": {"label": "FY24"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.95,
                "supporting_text": "Total Revenue from operations reached $500M in FY24.",
            }
        ]
    }

    with patch.object(extractor, "_call_gemini", return_value=gemini_resp):
        facts = extractor.extract_from_page(
            page=mock_pages[0],
            document_id="doc-alpha-fy24",
            document_name="Alpha_FY24_Report.pdf",
            document_date="2024-03-31",
        )

    assert len(facts) == 1
    f = facts[0]
    assert f.entity == "Alpha Corp"
    assert f.metric == "Revenue from operations"
    assert f.value_raw == "$500M"
    assert f.provenance.page_number == 1
    assert f.provenance.supporting_text == "Total Revenue from operations reached $500M in FY24."
    assert f.fact_id.startswith("fact-")


# ---------------------------------------------------------------------------
# 2. Multi-Page Batch Extraction (Batch Size 3)
# ---------------------------------------------------------------------------

def test_multi_page_batch_size_3(mock_pages):
    """Verify batch of 3 pages is processed in a single Gemini call with facts correctly attributed."""
    extractor = FactExtractor(
        batch_size=3,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    gemini_resp = {
        "facts": [
            {
                "page_number": 1,
                "entity": "Alpha Corp",
                "metric": "Revenue from operations",
                "value_raw": "$500M",
                "value_numeric": 500.0,
                "unit": "million USD",
                "time_period": {"label": "FY24"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.95,
                "supporting_text": "Total Revenue from operations reached $500M in FY24.",
            },
            {
                "page_number": 2,
                "entity": "Alpha Corp",
                "metric": "Operating profit",
                "value_raw": "$120M",
                "value_numeric": 120.0,
                "unit": "million USD",
                "time_period": {"label": "FY24"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.92,
                "supporting_text": "Operating profit was $120M in FY24 across all business units.",
            },
            {
                "page_number": 3,
                "entity": "Alpha Corp",
                "metric": "Employee count",
                "value_raw": "15,000 employees",
                "value_numeric": 15000.0,
                "unit": "employees",
                "time_period": {"label": "FY24"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.90,
                "supporting_text": "Total full-time employee count stood at 15,000 employees as of March 31, 2024.",
            },
        ]
    }

    with patch.object(extractor, "_call_gemini", return_value=gemini_resp) as mock_call:
        facts, rejected = extractor.extract_batch(
            batch_pages=mock_pages,
            document_id="doc-alpha-fy24",
            document_name="Alpha_FY24_Report.pdf",
        )

    assert mock_call.call_count == 1
    assert len(facts) == 3
    assert rejected == 0
    assert [f.provenance.page_number for f in facts] == [1, 2, 3]


# ---------------------------------------------------------------------------
# 3. Multi-Page Batch Extraction (Batch Size 2)
# ---------------------------------------------------------------------------

def test_multi_page_batch_size_2(mock_parsed_document):
    """Verify batch size 2 splits 3 pages into 2 batches."""
    extractor = FactExtractor(
        batch_size=2,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    batch1_resp = {
        "facts": [
            {
                "page_number": 1,
                "entity": "Alpha Corp",
                "metric": "Revenue",
                "value_raw": "$500M",
                "value_numeric": 500.0,
                "unit": "million USD",
                "time_period": {"label": "FY24"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.95,
                "supporting_text": "Total Revenue from operations reached $500M in FY24.",
            }
        ]
    }
    batch2_resp = {
        "facts": [
            {
                "page_number": 3,
                "entity": "Alpha Corp",
                "metric": "Employees",
                "value_raw": "15,000 employees",
                "value_numeric": 15000.0,
                "unit": "employees",
                "time_period": {"label": "FY24"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.90,
                "supporting_text": "Total full-time employee count stood at 15,000 employees as of March 31, 2024.",
            }
        ]
    }

    with patch.object(extractor, "_call_gemini", side_effect=[batch1_resp, batch2_resp]) as mock_call:
        res = extractor.extract_document_result(mock_parsed_document)

    assert mock_call.call_count == 2
    assert res.status == ExtractionStatus.COMPLETE
    assert res.requests_used == 2
    assert len(res.facts) == 2


# ---------------------------------------------------------------------------
# 4. Cross-Page Contamination Detection / Rejection
# ---------------------------------------------------------------------------

def test_cross_page_contamination_rejection(mock_pages):
    """Verify that if Gemini hallucinates attribution to Page 2 for text existing only on Page 1, it is dropped."""
    extractor = FactExtractor(
        batch_size=2,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    # Gemini mistakenly claims the page 1 revenue text is on page 2
    contaminated_resp = {
        "facts": [
            {
                "page_number": 2,  # Wrong page! Snippet is actually on page 1
                "entity": "Alpha Corp",
                "metric": "Revenue",
                "value_raw": "$500M",
                "value_numeric": 500.0,
                "unit": "million USD",
                "time_period": {"label": "FY24"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.95,
                "supporting_text": "Total Revenue from operations reached $500M in FY24.",
            }
        ]
    }

    with patch.object(extractor, "_call_gemini", return_value=contaminated_resp):
        facts, rejected = extractor.extract_batch(
            batch_pages=mock_pages[:2],
            document_id="doc-alpha-fy24",
            document_name="Alpha_FY24_Report.pdf",
        )

    # Must reject the candidate because snippet cannot be verified on page 2
    assert len(facts) == 0
    assert rejected == 1


# ---------------------------------------------------------------------------
# 5. Cross-Document Provenance Preservation
# ---------------------------------------------------------------------------

def test_cross_document_provenance_preservation():
    """Verify facts extracted across multiple distinct documents retain distinct document IDs and provenance."""
    t_a = "Company A reported revenue of $100M in 2023."
    t_b = "Company B reported revenue of $200M in 2023."
    doc_a = ParsedDocument(
        document_id="doc-a-hash",
        document_name="Doc_A.pdf",
        total_pages=1,
        pages=[PageText(page_number=1, text=t_a, char_count=len(t_a), has_text=True)],
    )
    doc_b = ParsedDocument(
        document_id="doc-b-hash",
        document_name="Doc_B.pdf",
        total_pages=1,
        pages=[PageText(page_number=1, text=t_b, char_count=len(t_b), has_text=True)],
    )

    extractor = FactExtractor(page_filter_enabled=False, context_selector_enabled=False)

    resp_a = {
        "facts": [
            {
                "page_number": 1,
                "entity": "Company A",
                "metric": "Revenue",
                "value_raw": "$100M",
                "value_numeric": 100.0,
                "unit": "million USD",
                "time_period": {"label": "2023"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.95,
                "supporting_text": "Company A reported revenue of $100M in 2023.",
            }
        ]
    }
    resp_b = {
        "facts": [
            {
                "page_number": 1,
                "entity": "Company B",
                "metric": "Revenue",
                "value_raw": "$200M",
                "value_numeric": 200.0,
                "unit": "million USD",
                "time_period": {"label": "2023"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.95,
                "supporting_text": "Company B reported revenue of $200M in 2023.",
            }
        ]
    }

    with patch.object(extractor, "_call_gemini", side_effect=[resp_a, resp_b]):
        facts_a = extractor.extract_from_document(doc_a)
        facts_b = extractor.extract_from_document(doc_b)

    assert len(facts_a) == 1
    assert len(facts_b) == 1
    assert facts_a[0].provenance.document_id == "doc-a-hash"
    assert facts_b[0].provenance.document_id == "doc-b-hash"
    assert facts_a[0].fact_id != facts_b[0].fact_id


# ---------------------------------------------------------------------------
# 6. Strict Evidence Verification Against Authoritative Original Page Text
# ---------------------------------------------------------------------------

def test_authoritative_original_page_evidence_verification():
    """Verify that even when context selector reduces text for the prompt, evidence verification checks full original page."""
    full_text = (
        "Header boilerplate.\n"
        "Some irrelevant filler sentence 1.\n"
        "Some irrelevant filler sentence 2.\n"
        "Total Revenue was $750M in FY24.\n"
        "Footer boilerplate.\n"
    )
    p = PageText(page_number=1, text=full_text, char_count=len(full_text), has_text=True)

    extractor = FactExtractor(
        page_filter_enabled=False,
        context_selector_enabled=True,
        context_radius=1,
    )

    gemini_resp = {
        "facts": [
            {
                "page_number": 1,
                "entity": "Test Entity",
                "metric": "Revenue",
                "value_raw": "$750M",
                "value_numeric": 750.0,
                "unit": "million USD",
                "time_period": {"label": "FY24"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.95,
                "supporting_text": "Total Revenue was $750M in FY24.",
            }
        ]
    }

    with patch.object(extractor, "_call_gemini", return_value=gemini_resp):
        facts, rejected = extractor.extract_batch([p], document_id="d1", document_name="d.pdf")

    assert len(facts) == 1
    assert rejected == 0
    assert facts[0].provenance.supporting_text == "Total Revenue was $750M in FY24."


# ---------------------------------------------------------------------------
# 7. Page Filter Disabled Fallback
# ---------------------------------------------------------------------------

def test_page_filter_disabled_fallback(mock_parsed_document):
    """Verify that with page_filter_enabled=False, all text pages are processed."""
    extractor = FactExtractor(
        batch_size=3,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    with patch.object(extractor, "_call_gemini", return_value={"facts": []}):
        res = extractor.extract_document_result(mock_parsed_document)

    assert res.total_eligible_pages == 3
    assert res.processed_pages == [1, 2, 3]


# ---------------------------------------------------------------------------
# 8. Context Selector Disabled Fallback
# ---------------------------------------------------------------------------

def test_context_selector_disabled_fallback(mock_pages):
    """Verify that with context_selector_enabled=False, raw text is sent without compression."""
    extractor = FactExtractor(
        batch_size=1,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    with patch.object(extractor, "_call_gemini", return_value={"facts": []}) as mock_call:
        extractor.extract_batch([mock_pages[0]], document_id="d1", document_name="d.pdf")

    payload = mock_call.call_args[0][0]
    assert "Total Revenue from operations reached $500M in FY24." in payload
    assert "===== DOCUMENT PAGE 1 START =====" in payload


# ---------------------------------------------------------------------------
# 9. Partial Quota Execution and Deferred Pages
# ---------------------------------------------------------------------------

def test_partial_quota_execution(mock_parsed_document):
    """Verify that if request budget (max_pages) is smaller than total batches, status is PARTIAL_QUOTA."""
    extractor = FactExtractor(
        batch_size=1,
        max_pages=1,  # Budget allows only 1 request
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    batch1_resp = {
        "facts": [
            {
                "page_number": 1,
                "entity": "Alpha Corp",
                "metric": "Revenue",
                "value_raw": "$500M",
                "value_numeric": 500.0,
                "unit": "million USD",
                "time_period": {"label": "FY24"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.95,
                "supporting_text": "Total Revenue from operations reached $500M in FY24.",
            }
        ]
    }

    with patch.object(extractor, "_call_gemini", return_value=batch1_resp):
        res = extractor.extract_document_result(mock_parsed_document)

    assert res.status == ExtractionStatus.PARTIAL_QUOTA
    assert res.requests_used == 1
    assert res.processed_pages == [1]
    assert set(res.unprocessed_pages) == {2, 3}
    assert len(res.facts) == 1
    assert "partially completed" in res.user_message.lower()


# ---------------------------------------------------------------------------
# 10. HTTP 429 Immediate Halt & Fact Retention
# ---------------------------------------------------------------------------

def test_http_429_immediate_halt_and_fact_retention(mock_parsed_document):
    """Verify that HTTP 429 stops immediately without wasting retries and retains facts from earlier batches."""
    extractor = FactExtractor(
        batch_size=1,
        max_pages=3,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    batch1_resp = {
        "facts": [
            {
                "page_number": 1,
                "entity": "Alpha Corp",
                "metric": "Revenue",
                "value_raw": "$500M",
                "value_numeric": 500.0,
                "unit": "million USD",
                "time_period": {"label": "FY24"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.95,
                "supporting_text": "Total Revenue from operations reached $500M in FY24.",
            }
        ]
    }

    # Batch 1 succeeds, Batch 2 throws 429
    quota_error = ExtractionQuotaError("HTTP 429: Resource has been exhausted (e.g. check quota).")

    with patch.object(extractor, "_call_gemini", side_effect=[batch1_resp, quota_error]):
        res = extractor.extract_document_result(mock_parsed_document)

    assert res.status == ExtractionStatus.PARTIAL_QUOTA
    assert res.requests_used == 1
    assert res.processed_pages == [1]
    assert len(res.facts) == 1
    assert "quota limit" in res.user_message.lower()


# ---------------------------------------------------------------------------
# 11. HTTP 503 Retry with Exponential Backoff
# ---------------------------------------------------------------------------

def test_http_503_retry_success(mock_pages):
    """Verify HTTP 503 retries once with backoff and succeeds on retry."""
    extractor = FactExtractor(
        api_key="mock-key",
        batch_size=1,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    success_data = {
        "facts": [
            {
                "page_number": 1,
                "entity": "Alpha Corp",
                "metric": "Revenue",
                "value_raw": "$500M",
                "value_numeric": 500.0,
                "unit": "million USD",
                "time_period": {"label": "FY24"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.95,
                "supporting_text": "Total Revenue from operations reached $500M in FY24.",
            }
        ]
    }

    mock_resp_503 = MagicMock()
    mock_resp_503.status_code = 503
    mock_resp_503.text = "Service Unavailable"

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps(success_data)}]
                }
            }
        ]
    }

    with patch("backend.extraction.fact_extractor.httpx.Client") as mock_client_cls, \
         patch("backend.extraction.fact_extractor.time.sleep") as mock_sleep:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = [mock_resp_503, mock_resp_200]
        mock_client_cls.return_value = mock_client

        facts, rejected = extractor.extract_batch([mock_pages[0]], document_id="d1", document_name="d.pdf")

    assert len(facts) == 1
    assert rejected == 0
    assert mock_sleep.call_count == 1  # Retried after backoff


# ---------------------------------------------------------------------------
# 12. COMPLETE Status When Budget >= Eligible Pages
# ---------------------------------------------------------------------------

def test_complete_status_when_budget_sufficient(mock_parsed_document):
    """Verify COMPLETE status is returned when all eligible pages are processed."""
    extractor = FactExtractor(
        batch_size=3,
        max_pages=5,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    with patch.object(extractor, "_call_gemini", return_value={"facts": []}):
        res = extractor.extract_document_result(mock_parsed_document)

    assert res.status == ExtractionStatus.COMPLETE
    assert res.total_eligible_pages == 3
    assert res.processed_pages == [1, 2, 3]
    assert res.unprocessed_pages == []
    assert "completed successfully" in res.user_message.lower()


# ---------------------------------------------------------------------------
# 13. Arbitrary Unseen PDF Document Extraction
# ---------------------------------------------------------------------------

def test_arbitrary_unseen_pdf_extraction():
    """Verify arbitrary unseen document layout with 4 pages extracts without hardcoded corpus assumptions."""
    t1 = "Global Logistics Inc. Investor Presentation Q3 2025."
    t2 = "Fleet Size: 4,500 active delivery vehicles in Europe."
    t3 = "Operating margin improved to 14.2% in Q3 2025."
    t4 = "Disclaimer: Forward-looking statements."
    unseen_pages = [
        PageText(page_number=1, text=t1, char_count=len(t1), has_text=True),
        PageText(page_number=2, text=t2, char_count=len(t2), has_text=True),
        PageText(page_number=3, text=t3, char_count=len(t3), has_text=True),
        PageText(page_number=4, text=t4, char_count=len(t4), has_text=True),
    ]
    unseen_doc = ParsedDocument(
        document_id="doc-unseen-99",
        document_name="Unseen_Q3_2025.pdf",
        total_pages=4,
        pages=unseen_pages,
    )

    extractor = FactExtractor(
        batch_size=2,
        page_filter_enabled=True,
        page_relevance_threshold=0.25,
        context_selector_enabled=True,
    )

    gemini_resp = {
        "facts": [
            {
                "page_number": 2,
                "entity": "Global Logistics Inc.",
                "metric": "Fleet Size",
                "value_raw": "4,500 active delivery vehicles",
                "value_numeric": 4500.0,
                "unit": "vehicles",
                "time_period": {"label": "Q3 2025"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.94,
                "supporting_text": "Fleet Size: 4,500 active delivery vehicles in Europe.",
            },
            {
                "page_number": 3,
                "entity": "Global Logistics Inc.",
                "metric": "Operating margin",
                "value_raw": "14.2%",
                "value_numeric": 14.2,
                "unit": "%",
                "time_period": {"label": "Q3 2025"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.93,
                "supporting_text": "Operating margin improved to 14.2% in Q3 2025.",
            },
        ]
    }

    with patch.object(extractor, "_call_gemini", return_value=gemini_resp):
        res = extractor.extract_document_result(unseen_doc)

    assert len(res.facts) == 2
    assert res.facts[0].metric == "Fleet Size"
    assert res.facts[1].metric == "Operating margin"


# ---------------------------------------------------------------------------
# 14. Downstream Reasoning Invariant Preservation
# ---------------------------------------------------------------------------

def test_downstream_reasoning_invariant_preservation():
    """Verify facts extracted flow cleanly through normalizer, candidate matcher, comparability gate, and relationship engine."""
    prov_a = Provenance(document_id="doc-1", page_number=1, supporting_text="FY24 revenue was $100M.")
    prov_b = Provenance(document_id="doc-2", page_number=1, supporting_text="FY24 revenue was $100M.")

    fact_a = FactRecord(
        fact_id="fact-a",
        entity="Acme Corp",
        metric="Revenue",
        value_raw="$100M",
        value_numeric=100.0,
        unit="million USD",
        time_period=TimePeriod(label="FY24", start_date="2023-04-01", end_date="2024-03-31"),
        epistemic_status=EpistemicStatus.REPORTED,
        provenance=prov_a,
    )

    fact_b = FactRecord(
        fact_id="fact-b",
        entity="Acme Corp",
        metric="Revenue",
        value_raw="$100M",
        value_numeric=100.0,
        unit="million USD",
        time_period=TimePeriod(label="FY24", start_date="2023-04-01", end_date="2024-03-31"),
        epistemic_status=EpistemicStatus.REPORTED,
        provenance=prov_b,
    )

    normalizer = FactNormalizer()
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    rel_engine = RelationshipEngine()

    norm_facts = normalizer.normalize_batch([fact_a, fact_b])
    assert len(norm_facts) == 2
    assert norm_facts[0].normalized_value.numeric_value == 100_000_000.0
    assert norm_facts[1].normalized_value.numeric_value == 100_000_000.0

    pairs = matcher.find_candidates(norm_facts)
    assert len(pairs) == 1

    candidate_pair = matcher.match_pair(pairs[0][0], pairs[0][1])
    assert candidate_pair is not None

    comp = gate.evaluate(pairs[0][0], pairs[0][1])
    assert comp.status == ComparabilityStatus.COMPARABLE

    rel = rel_engine.determine_relationship(
        fact_a=pairs[0][0],
        fact_b=pairs[0][1],
        comparability=comp,
        candidate_pair=candidate_pair,
    )
    assert rel.relationship_type == RelationshipType.CORROBORATES


# ---------------------------------------------------------------------------
# 15. No Numeric-Only Candidate Matching Invariant
# ---------------------------------------------------------------------------

def test_no_numeric_only_candidate_matching_invariant():
    """Verify that dissimilar metrics with identical numbers are rejected by CandidateMatcher."""
    prov = Provenance(document_id="d1", page_number=1, supporting_text="Text")

    f1 = FactRecord(
        fact_id="f1",
        entity="Company",
        metric="Total Express Shipments",
        value_raw="50,000",
        value_numeric=50000.0,
        unit="shipments",
        time_period=TimePeriod(label="FY24"),
        provenance=prov,
    )

    f2 = FactRecord(
        fact_id="f2",
        entity="Company",
        metric="Total Service Centers",
        value_raw="50,000",
        value_numeric=50000.0,
        unit="centers",
        time_period=TimePeriod(label="FY24"),
        provenance=prov,
    )

    normalizer = FactNormalizer()
    matcher = CandidateMatcher()

    norm_facts = normalizer.normalize_batch([f1, f2])
    candidates = matcher.find_candidates(norm_facts)

    # Must be rejected because express shipments != service centers
    assert len(candidates) == 0


# ---------------------------------------------------------------------------
# 16. AnalysisService End-to-End Extraction Status Persistence Test
# ---------------------------------------------------------------------------

def test_analysis_service_e2e_extraction_status_persisted(tmp_path):
    """Verify AnalysisService runs end-to-end and persists structured extraction_status in analysis summary."""
    db_file = tmp_path / "test_factline.db"
    repo = DatabaseRepository(db_path=str(db_file))

    extractor = FactExtractor(
        batch_size=2,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    gemini_resp = {
        "facts": [
            {
                "page_number": 1,
                "entity": "Test Co",
                "metric": "Revenue",
                "value_raw": "$100M",
                "value_numeric": 100.0,
                "unit": "million USD",
                "time_period": {"label": "FY24"},
                "epistemic_status": "reported",
                "extraction_confidence": 0.95,
                "supporting_text": "Total Revenue was $100M in FY24.",
            }
        ]
    }

    mock_parser = MagicMock()
    t = "Total Revenue was $100M in FY24."
    p = PageText(page_number=1, text=t, char_count=len(t), has_text=True)
    doc = ParsedDocument(document_id="doc-test-1", document_name="Test.pdf", total_pages=1, pages=[p])
    mock_parser.parse_bytes.return_value = doc

    service = AnalysisService(
        parser=mock_parser,
        extractor=extractor,
        repository=repo,
    )

    with patch.object(extractor, "_call_gemini", return_value=gemini_resp):
        persisted = service.analyze_documents([("Test.pdf", b"%PDF-1.4 dummy content")])

    assert persisted is not None
    assert "analysis_id" in persisted
    analysis_id = persisted["analysis_id"]

    # Verify retrieval
    retrieved = service.get_analysis(analysis_id)
    assert retrieved is not None
    assert "summary" in retrieved
    summary = retrieved["summary"]
    assert "extraction_status" in summary
    ext_status = summary["extraction_status"]
    assert ext_status["status"] == "COMPLETE"
    assert ext_status["total_eligible_pages"] == 1
    assert ext_status["total_processed_pages"] == 1
    assert ext_status["requests_used"] == 1
    assert len(retrieved["facts"]) == 1
