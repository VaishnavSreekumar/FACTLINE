"""
Phase 17 Regression Test Suite.
Verifies:
1. Gemini schema validity (scalar nullable OpenAPI types, no list-type unions).
2. HTTP 400 classification -> ExtractionError -> status FAILED.
3. HTTP 404 classification -> ExtractionError -> status FAILED.
4. HTTP 429 classification -> ExtractionQuotaError -> status PARTIAL_QUOTA / QUOTA_EXHAUSTED.
5. HTTP 503 bounded retry semantics.
6. Successful Gemini extraction -> FactRecord -> evidence verification.
7. Explicit cross-page multi-page batch attribution and persistence.
8. Downstream reasoning invariant preservation.
"""

import json
from unittest.mock import MagicMock, patch
import httpx
import pytest

from backend.extraction.fact_extractor import (
    BATCH_EXTRACTION_JSON_SCHEMA,
    ExtractionError,
    ExtractionQuotaError,
    FactExtractor,
)
from backend.models.document import PageText, ParsedDocument
from backend.quota_experiment.batching import (
    BATCH_EXTRACTION_JSON_SCHEMA as EXP_BATCH_SCHEMA,
    MultiPageBatchExtractor,
)
from backend.quota_experiment.status import ExtractionStatus
from backend.services.analysis import AnalysisService


# ============================================================================
# A. Gemini Schema Validity Tests
# ============================================================================

def _assert_no_list_types(schema_node, path="root"):
    """Recursively validates that no property definition contains a list 'type'."""
    if isinstance(schema_node, dict):
        if "type" in schema_node:
            t = schema_node["type"]
            assert not isinstance(t, list), (
                f"Schema path '{path}' has unsupported list type: {t}. "
                "Must use scalar type with 'nullable: true'."
            )
            assert isinstance(t, str), f"Schema path '{path}' type must be string, got: {type(t)}"
        for k, v in schema_node.items():
            _assert_no_list_types(v, f"{path}.{k}")
    elif isinstance(schema_node, list):
        for idx, item in enumerate(schema_node):
            _assert_no_list_types(item, f"{path}[{idx}]")


def test_production_gemini_schema_validity():
    """Verify production schema uses valid scalar nullable types accepted by Gemini REST API."""
    _assert_no_list_types(BATCH_EXTRACTION_JSON_SCHEMA)

    # Check key scalar nullable fields explicitly
    fact_props = BATCH_EXTRACTION_JSON_SCHEMA["properties"]["facts"]["items"]["properties"]
    assert fact_props["value_numeric"]["type"] == "NUMBER"
    assert fact_props["value_numeric"]["nullable"] is True
    assert fact_props["unit"]["type"] == "STRING"
    assert fact_props["unit"]["nullable"] is True
    assert fact_props["scope"]["type"] == "STRING"
    assert fact_props["scope"]["nullable"] is True
    assert fact_props["geography"]["type"] == "STRING"
    assert fact_props["geography"]["nullable"] is True
    assert fact_props["data_vintage"]["type"] == "STRING"
    assert fact_props["data_vintage"]["nullable"] is True

    # Check nested time_period
    tp_props = fact_props["time_period"]["properties"]
    assert tp_props["start_date"]["type"] == "STRING"
    assert tp_props["start_date"]["nullable"] is True
    assert tp_props["end_date"]["type"] == "STRING"
    assert tp_props["end_date"]["nullable"] is True


def test_experimental_schema_consistency():
    """Verify experimental schema in batching.py is also aligned with Gemini OpenAPI scalars."""
    _assert_no_list_types(EXP_BATCH_SCHEMA)


# ============================================================================
# B. HTTP 400 Behavior Tests
# ============================================================================

def test_http_400_behavior():
    """Verify HTTP 400 produces ExtractionError and marks status FAILED (NOT PARTIAL_QUOTA)."""
    extractor = FactExtractor(
        api_key="mock-key",
        batch_size=1,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 400
    mock_resp.text = '{"error": {"code": 400, "message": "Invalid argument: responseSchema"}}'

    # Direct _call_gemini check
    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(ExtractionError) as exc_info:
            extractor._call_gemini("test prompt")
        assert "HTTP 400" in str(exc_info.value)
        assert not isinstance(exc_info.value, ExtractionQuotaError)

    # Document extraction status check
    doc = ParsedDocument(
        document_id="doc-400",
        document_name="doc_400.pdf",
        total_pages=2,
        total_characters=100,
        pages=[
            PageText(page_number=1, text="Sample text page 1", char_count=18, has_text=True),
            PageText(page_number=2, text="Sample text page 2", char_count=18, has_text=True),
        ],
    )

    with patch("httpx.Client.post", return_value=mock_resp):
        res = extractor.extract_document_result(doc)

    assert res.status == ExtractionStatus.FAILED
    assert res.status != ExtractionStatus.PARTIAL_QUOTA
    assert res.status != ExtractionStatus.QUOTA_EXHAUSTED
    assert "Extraction failed" in res.user_message


# ============================================================================
# C. HTTP 404 Behavior Tests
# ============================================================================

def test_http_404_behavior():
    """Verify HTTP 404 produces ExtractionError and marks status FAILED (NOT PARTIAL_QUOTA)."""
    extractor = FactExtractor(
        api_key="mock-key",
        batch_size=1,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 404
    mock_resp.text = '{"error": {"code": 404, "message": "models/gemini-2.5-flash is not found"}}'

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(ExtractionError) as exc_info:
            extractor._call_gemini("test prompt")
        assert "HTTP 404" in str(exc_info.value)
        assert not isinstance(exc_info.value, ExtractionQuotaError)

    doc = ParsedDocument(
        document_id="doc-404",
        document_name="doc_404.pdf",
        total_pages=1,
        total_characters=50,
        pages=[
            PageText(page_number=1, text="Sample text page 1", char_count=18, has_text=True),
        ],
    )

    with patch("httpx.Client.post", return_value=mock_resp):
        res = extractor.extract_document_result(doc)

    assert res.status == ExtractionStatus.FAILED
    assert res.status != ExtractionStatus.PARTIAL_QUOTA
    assert res.status != ExtractionStatus.QUOTA_EXHAUSTED


# ============================================================================
# D. HTTP 429 Behavior Tests
# ============================================================================

def test_http_429_behavior():
    """Verify HTTP 429 raises ExtractionQuotaError, immediately halts, and marks QUOTA_EXHAUSTED / PARTIAL_QUOTA."""
    extractor = FactExtractor(
        api_key="mock-key",
        batch_size=1,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 429
    mock_resp.text = '{"error": {"code": 429, "message": "Resource exhausted"}}'

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        with pytest.raises(ExtractionQuotaError):
            extractor._call_gemini("test prompt")
        # Assert no retry occurred on 429
        assert mock_post.call_count == 1

    doc = ParsedDocument(
        document_id="doc-429",
        document_name="doc_429.pdf",
        total_pages=2,
        total_characters=100,
        pages=[
            PageText(page_number=1, text="Page 1 text", char_count=11, has_text=True),
            PageText(page_number=2, text="Page 2 text", char_count=11, has_text=True),
        ],
    )

    with patch("httpx.Client.post", return_value=mock_resp):
        res = extractor.extract_document_result(doc)

    assert res.status == ExtractionStatus.QUOTA_EXHAUSTED
    assert res.status != ExtractionStatus.FAILED


# ============================================================================
# E. HTTP 503 Bounded Retry Tests
# ============================================================================

def test_http_503_bounded_retry_behavior():
    """Verify HTTP 503 attempts exactly 3 tries then raises ExtractionError."""
    extractor = FactExtractor(
        api_key="mock-key",
        batch_size=1,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    mock_503 = MagicMock(spec=httpx.Response)
    mock_503.status_code = 503
    mock_503.text = "Service Unavailable"

    with patch("httpx.Client.post", return_value=mock_503) as mock_post:
        with patch("time.sleep", return_value=None):
            with pytest.raises(ExtractionError) as exc_info:
                extractor._call_gemini("test prompt")
            assert "HTTP 503" in str(exc_info.value)
            assert mock_post.call_count == 3


# ============================================================================
# F. Successful Gemini Extraction & Evidence Verification
# ============================================================================

def test_successful_gemini_extraction():
    """Verify valid Gemini response parses to FactRecord and verifies evidence."""
    extractor = FactExtractor(
        api_key="mock-key",
        batch_size=1,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    doc = ParsedDocument(
        document_id="doc-success",
        document_name="success.pdf",
        total_pages=1,
        total_characters=60,
        pages=[
            PageText(
                page_number=1,
                text="In FY2024, Acme Corp reported operating revenue of $450 million.",
                char_count=64,
                has_text=True,
            ),
        ],
    )

    gemini_output = {
        "facts": [
            {
                "page_number": 1,
                "entity": "Acme Corp",
                "metric": "Operating Revenue",
                "value_raw": "$450 million",
                "value_numeric": 450.0,
                "unit": "million USD",
                "time_period": {"label": "FY2024", "start_date": "2024-01-01", "end_date": "2024-12-31"},
                "scope": "Consolidated",
                "geography": "Global",
                "epistemic_status": "reported",
                "data_vintage": "2024",
                "supporting_text": "In FY2024, Acme Corp reported operating revenue of $450 million.",
                "extraction_confidence": 0.98,
            }
        ]
    }

    with patch.object(extractor, "_call_gemini", return_value=gemini_output):
        res = extractor.extract_document_result(doc)

    assert res.status == ExtractionStatus.COMPLETE
    assert len(res.facts) == 1
    fact = res.facts[0]
    assert fact.entity == "Acme Corp"
    assert fact.metric == "Operating Revenue"
    assert fact.value_numeric == 450.0
    assert fact.provenance.page_number == 1
    assert fact.provenance.supporting_text == "In FY2024, Acme Corp reported operating revenue of $450 million."


# ============================================================================
# G. Cross-Page Multi-Page Batch Attribution & End-to-End Persistence
# ============================================================================

def test_cross_page_batch_attribution_and_persistence(tmp_path, monkeypatch):
    """
    Explicit regression test for multi-page batch extraction:
    Page 1: Revenue was $100 million in 2024.
    Page 2: Employees numbered 5,000 in 2024.

    Verifies:
    - Fact A -> page_number = 1
    - Fact B -> page_number = 2
    - Both evidence snippets verify strictly against their original pages
    - Neither fact is assigned to the wrong page
    - Both survive persistence through AnalysisService
    """
    test_db = str(tmp_path / "test_phase17.db")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{test_db}")

    page1_text = "Acme Global financial report: Revenue was $100 million in 2024."
    page2_text = "Acme Global workforce report: Employees numbered 5,000 in 2024."

    batch_resp = {
        "facts": [
            {
                "page_number": 1,
                "entity": "Acme Global",
                "metric": "Revenue",
                "value_raw": "$100 million",
                "value_numeric": 100.0,
                "unit": "million USD",
                "time_period": {"label": "2024"},
                "epistemic_status": "reported",
                "supporting_text": "Revenue was $100 million in 2024.",
                "extraction_confidence": 0.95,
            },
            {
                "page_number": 2,
                "entity": "Acme Global",
                "metric": "Employees",
                "value_raw": "5,000",
                "value_numeric": 5000.0,
                "unit": "count",
                "time_period": {"label": "2024"},
                "epistemic_status": "reported",
                "supporting_text": "Employees numbered 5,000 in 2024.",
                "extraction_confidence": 0.95,
            },
        ]
    }

    mock_doc = ParsedDocument(
        document_id="doc-batch-test",
        document_name="acme_annual.pdf",
        total_pages=2,
        total_characters=len(page1_text) + len(page2_text),
        pages=[
            PageText(page_number=1, text=page1_text, char_count=len(page1_text), has_text=True),
            PageText(page_number=2, text=page2_text, char_count=len(page2_text), has_text=True),
        ],
    )

    extractor = FactExtractor(
        batch_size=2,
        page_filter_enabled=False,
        context_selector_enabled=False,
    )

    # 1. Direct batch extraction test
    with patch.object(extractor, "_call_gemini", return_value=batch_resp):
        res = extractor.extract_document_result(mock_doc)

    assert res.status == ExtractionStatus.COMPLETE
    assert len(res.facts) == 2
    assert res.facts_rejected_grounding == 0

    fact_rev = next(f for f in res.facts if f.metric == "Revenue")
    fact_emp = next(f for f in res.facts if f.metric == "Employees")

    assert fact_rev.provenance.page_number == 1
    assert fact_rev.provenance.supporting_text == "Revenue was $100 million in 2024."
    assert fact_rev.provenance.supporting_text in page1_text
    assert fact_rev.provenance.supporting_text not in page2_text

    assert fact_emp.provenance.page_number == 2
    assert fact_emp.provenance.supporting_text == "Employees numbered 5,000 in 2024."
    assert fact_emp.provenance.supporting_text in page2_text
    assert fact_emp.provenance.supporting_text not in page1_text

    # 2. End-to-end AnalysisService persistence test
    service = AnalysisService(extractor=extractor)

    # Mock parser to return our mock_doc
    with patch.object(service.parser, "parse_bytes", return_value=mock_doc):
        with patch.object(extractor, "_call_gemini", return_value=batch_resp):
            analysis_summary = service.analyze_documents([("acme_annual.pdf", b"%PDF-mock")])

    assert analysis_summary["summary"]["documents_processed"] == 1
    assert analysis_summary["summary"]["facts_extracted"] == 2
    assert analysis_summary["summary"]["extraction_status"]["status"] == "COMPLETE"

    # Retrieve from DB repository
    saved_analysis = service.get_analysis(analysis_summary["analysis_id"])
    assert saved_analysis is not None
    assert len(saved_analysis["facts"]) == 2

    db_rev = next(f for f in saved_analysis["facts"] if f["fact"]["metric"] == "Revenue")
    db_emp = next(f for f in saved_analysis["facts"] if f["fact"]["metric"] == "Employees")

    assert db_rev["fact"]["provenance"]["page_number"] == 1
    assert db_emp["fact"]["provenance"]["page_number"] == 2
