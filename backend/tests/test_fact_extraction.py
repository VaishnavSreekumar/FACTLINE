"""Unit and integration tests for Phase 2 Fact Extraction and Evidence Invariant."""

import os
from pathlib import Path
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from backend.main import app
from backend.extraction.fact_extractor import FactExtractor, ExtractionError
from backend.extraction.evidence import EvidenceVerifier, create_evidence
from backend.models.document import PageText, ParsedDocument
from backend.models.fact import EpistemicStatus, FactRecord, Provenance, TimePeriod


DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "sample-data" / "starter-datasets"
DELHIVERY_DIR = DATASET_DIR / "delhivery"


# ---------------------------------------------------------------------------
# 1. Schema Tests
# ---------------------------------------------------------------------------

def test_fact_record_valid_schema():
    """Verify FactRecord creation with complete valid schema fields."""
    provenance = Provenance(
        document_id="doc-delhivery-fy24",
        document_date="2024-05-17",
        page_number=12,
        supporting_text="Revenue from operations was ₹81,415.38 million in FY24.",
    )
    time_period = TimePeriod(label="FY24", start_date="2023-04-01", end_date="2024-03-31")
    fact = FactRecord(
        fact_id="fact-123456",
        entity="Delhivery Limited",
        metric="Revenue from operations",
        value_raw="₹81,415.38 million",
        value_numeric=81415.38,
        unit="million INR",
        time_period=time_period,
        scope="Consolidated",
        geography="India",
        epistemic_status=EpistemicStatus.REPORTED,
        data_vintage=None,
        provenance=provenance,
        extraction_confidence=0.98,
    )
    assert fact.fact_id == "fact-123456"
    assert fact.epistemic_status == EpistemicStatus.REPORTED
    assert fact.value_numeric == 81415.38
    assert fact.extraction_confidence == 0.98


def test_fact_record_epistemic_status_variants():
    """Verify supported epistemic statuses (reported, estimated, projected, target, audited)."""
    prov = Provenance(document_id="d1", page_number=1, supporting_text="Text")
    tp = TimePeriod(label="FY25")

    for status in [
        EpistemicStatus.REPORTED,
        EpistemicStatus.ESTIMATED,
        EpistemicStatus.PROJECTED,
        EpistemicStatus.TARGET,
        EpistemicStatus.AUDITED,
    ]:
        fact = FactRecord(
            fact_id=f"f-{status.value}",
            entity="India",
            metric="Real GDP Growth",
            value_raw="6.5%",
            value_numeric=6.5,
            time_period=tp,
            epistemic_status=status,
            provenance=prov,
            extraction_confidence=0.9,
        )
        assert fact.epistemic_status == status


def test_fact_record_invalid_confidence_rejected():
    """Verify confidence score must be within [0.0, 1.0]."""
    prov = Provenance(document_id="d1", page_number=1, supporting_text="Text")
    tp = TimePeriod(label="FY24")

    with pytest.raises(ValidationError):
        FactRecord(
            fact_id="f-bad-conf",
            entity="Company",
            metric="Metric",
            value_raw="100",
            time_period=tp,
            provenance=prov,
            extraction_confidence=1.5,  # Invalid: > 1.0
        )

    with pytest.raises(ValidationError):
        FactRecord(
            fact_id="f-bad-conf-neg",
            entity="Company",
            metric="Metric",
            value_raw="100",
            time_period=tp,
            provenance=prov,
            extraction_confidence=-0.1,  # Invalid: < 0.0
        )


def test_fact_record_missing_required_fields():
    """Verify missing required fields raises ValidationError."""
    with pytest.raises(ValidationError):
        FactRecord(
            fact_id="f-missing",
            # missing entity, metric, value_raw, time_period, provenance
        )


# ---------------------------------------------------------------------------
# 2. Evidence Verification Tests
# ---------------------------------------------------------------------------

def test_evidence_verification_exact_and_multiline():
    """Verify EvidenceVerifier matches exact and multi-line page text."""
    page_text = "In FY24, the Company achieved Revenue from operations of ₹81,415.38 million compared to previous year."
    page = PageText(page_number=5, text=page_text, char_count=len(page_text), has_text=True)

    # Exact snippet
    prov_exact = create_evidence("doc-1", 5, "Revenue from operations of ₹81,415.38 million")
    assert EvidenceVerifier.verify_provenance(prov_exact, page) is True

    # Multi-line snippet with line breaks
    multiline_page_text = "Revenue from\noperations of ₹81,415.38\nmillion in FY24."
    multiline_page = PageText(
        page_number=5,
        text=multiline_page_text,
        char_count=len(multiline_page_text),
        has_text=True,
    )
    prov_multiline = create_evidence("doc-1", 5, "Revenue from operations of ₹81,415.38 million")
    assert EvidenceVerifier.verify_provenance(prov_multiline, multiline_page) is True


def test_evidence_verification_rejection_of_hallucinated_text():
    """Verify ungrounded supporting text is rejected."""
    page_text = "Revenue was ₹500 crore in FY24."
    page = PageText(page_number=1, text=page_text, char_count=len(page_text), has_text=True)

    fake_prov = create_evidence("doc-1", 1, "Net profit was ₹100 crore in FY24.")
    assert EvidenceVerifier.verify_provenance(fake_prov, page) is False

    empty_prov = create_evidence("doc-1", 1, "   ")
    assert EvidenceVerifier.verify_provenance(empty_prov, page) is False


# ---------------------------------------------------------------------------
# 3. FactExtractor Mocked Tests
# ---------------------------------------------------------------------------

def test_extract_single_numerical_fact():
    """Verify FactExtractor parses a single numerical fact with verified evidence."""
    sample_text = "Delhivery recorded consolidated Revenue from operations of ₹81,415.38 million in FY24."
    page = PageText(page_number=1, text=sample_text, char_count=len(sample_text), has_text=True)

    mock_llm_response = {
        "facts": [
            {
                "entity": "Delhivery",
                "metric": "Revenue from operations",
                "value_raw": "₹81,415.38 million",
                "value_numeric": 81415.38,
                "unit": "million INR",
                "time_period": {"label": "FY24", "start_date": None, "end_date": None},
                "scope": "Consolidated",
                "geography": None,
                "epistemic_status": "reported",
                "data_vintage": None,
                "supporting_text": "consolidated Revenue from operations of ₹81,415.38 million in FY24",
                "extraction_confidence": 0.95,
            }
        ]
    }

    extractor = FactExtractor(llm_caller=lambda p: mock_llm_response)
    facts = extractor.extract_from_page(page, document_id="delhivery-fy24", document_name="annual_report.pdf")

    assert len(facts) == 1
    f = facts[0]
    assert f.entity == "Delhivery"
    assert f.metric == "Revenue from operations"
    assert f.value_raw == "₹81,415.38 million"
    assert f.value_numeric == 81415.38
    assert f.unit == "million INR"
    assert f.time_period.label == "FY24"
    assert f.scope == "Consolidated"
    assert f.geography is None
    assert f.data_vintage is None
    assert f.epistemic_status == EpistemicStatus.REPORTED
    assert f.provenance.page_number == 1
    assert f.provenance.document_id == "delhivery-fy24"
    assert f.extraction_confidence == 0.95


def test_extract_multiple_facts_and_reject_ungrounded():
    """Verify multiple facts from one page, where ungrounded fact is rejected."""
    sample_text = (
        "Delhivery had 33,278 active customers in Q4 FY24. "
        "Total express parcel shipment volume was 740 million."
    )
    page = PageText(page_number=3, text=sample_text, char_count=len(sample_text), has_text=True)

    mock_llm_response = {
        "facts": [
            {
                "entity": "Delhivery",
                "metric": "Active customers",
                "value_raw": "33,278",
                "value_numeric": 33278,
                "unit": "count",
                "time_period": {"label": "Q4 FY24"},
                "epistemic_status": "reported",
                "supporting_text": "33,278 active customers in Q4 FY24",
                "extraction_confidence": 0.99,
            },
            {
                "entity": "Delhivery",
                "metric": "Express parcel shipment volume",
                "value_raw": "740 million",
                "value_numeric": 740,
                "unit": "million shipments",
                "time_period": {"label": "FY24"},
                "epistemic_status": "reported",
                "supporting_text": "Total express parcel shipment volume was 740 million",
                "extraction_confidence": 0.95,
            },
            {
                "entity": "Delhivery",
                "metric": "Hallucinated Metric",
                "value_raw": "₹500 crore",
                "time_period": {"label": "FY24"},
                "epistemic_status": "reported",
                "supporting_text": "This hallucinated text does not exist anywhere on the page",
                "extraction_confidence": 0.90,
            },
        ]
    }

    extractor = FactExtractor(llm_caller=lambda p: mock_llm_response)
    facts = extractor.extract_from_page(page, document_id="doc-q4", document_name="presentation.pdf")

    # 2 grounded facts accepted, 1 ungrounded fact rejected
    assert len(facts) == 2
    metrics = [f.metric for f in facts]
    assert "Active customers" in metrics
    assert "Express parcel shipment volume" in metrics
    assert "Hallucinated Metric" not in metrics


def test_extract_empty_page_zero_llm_calls():
    """Verify empty pages return zero facts without calling LLM."""
    page_empty = PageText(page_number=4, text="", char_count=0, has_text=False)

    called = False
    def mock_caller(prompt):
        nonlocal called
        called = True
        return {"facts": []}

    extractor = FactExtractor(llm_caller=mock_caller)
    facts = extractor.extract_from_page(page_empty, document_id="doc-empty", document_name="doc.pdf")

    assert facts == []
    assert called is False


def test_extract_estimated_and_projected_facts():
    """Verify estimated and projected claims preserve epistemic status."""
    sample_text = (
        "Real GDP growth is estimated at 6.4% in FY25 according to the First Advance Estimate. "
        "Real GDP growth is projected at 7.0% for FY26."
    )
    page = PageText(page_number=2, text=sample_text, char_count=len(sample_text), has_text=True)

    mock_llm_response = {
        "facts": [
            {
                "entity": "India",
                "metric": "Real GDP Growth",
                "value_raw": "6.4%",
                "value_numeric": 6.4,
                "unit": "percent",
                "time_period": {"label": "FY25"},
                "epistemic_status": "estimated",
                "data_vintage": "First Advance Estimate",
                "supporting_text": "Real GDP growth is estimated at 6.4% in FY25",
                "extraction_confidence": 0.92,
            },
            {
                "entity": "India",
                "metric": "Real GDP Growth",
                "value_raw": "7.0%",
                "value_numeric": 7.0,
                "unit": "percent",
                "time_period": {"label": "FY26"},
                "epistemic_status": "projected",
                "supporting_text": "Real GDP growth is projected at 7.0% for FY26",
                "extraction_confidence": 0.88,
            },
        ]
    }

    extractor = FactExtractor(llm_caller=lambda p: mock_llm_response)
    facts = extractor.extract_from_page(page, document_id="doc-macro", document_name="economic_survey.pdf")

    assert len(facts) == 2
    f_est = next(f for f in facts if f.time_period.label == "FY25")
    assert f_est.epistemic_status == EpistemicStatus.ESTIMATED
    assert f_est.data_vintage == "First Advance Estimate"

    f_proj = next(f for f in facts if f.time_period.label == "FY26")
    assert f_proj.epistemic_status == EpistemicStatus.PROJECTED


def test_deterministic_fact_id_repeatability():
    """Verify fact_id generation is deterministic across repeated runs."""
    id1 = FactExtractor.generate_fact_id(
        document_id="doc-delhivery",
        page_number=15,
        entity="Delhivery Limited",
        metric="Revenue from operations",
        value_raw="₹81,415.38 million",
        time_period_label="FY24",
    )
    id2 = FactExtractor.generate_fact_id(
        document_id="doc-delhivery",
        page_number=15,
        entity="Delhivery Limited",
        metric="Revenue from operations",
        value_raw="₹81,415.38 million",
        time_period_label="FY24",
    )
    id_different = FactExtractor.generate_fact_id(
        document_id="doc-delhivery",
        page_number=16,  # different page
        entity="Delhivery Limited",
        metric="Revenue from operations",
        value_raw="₹81,415.38 million",
        time_period_label="FY24",
    )
    assert id1 == id2
    assert id1 != id_different
    assert id1.startswith("fact-")


def test_extract_facts_api_endpoint(monkeypatch):
    """Test POST /documents/extract-facts endpoint with mocked LLM extractor."""
    client = TestClient(app)
    sample_pdf = DELHIVERY_DIR / "01-delhivery-prospectus-2022-excerpt.pdf"
    assert sample_pdf.exists()

    # Mock extract_from_document on the global extractor instance in routes
    def mock_extract(doc, document_date=None):
        return [
            FactRecord(
                fact_id="fact-mock-1",
                entity="Delhivery Limited",
                metric="Express Parcel Pin Codes",
                value_raw="17,000+",
                time_period=TimePeriod(label="FY22"),
                epistemic_status=EpistemicStatus.REPORTED,
                provenance=Provenance(
                    document_id=doc.document_id,
                    page_number=1,
                    supporting_text="Delhivery covers over 17,000 pin codes",
                ),
                extraction_confidence=0.95,
            )
        ]

    from backend.api import routes
    monkeypatch.setattr(routes.extractor, "extract_from_document", mock_extract)

    with open(sample_pdf, "rb") as f:
        response = client.post(
            "/documents/extract-facts",
            files={"file": (sample_pdf.name, f, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert "total_pages" in data
    assert "total_facts" in data
    assert data["total_facts"] == 1
    assert data["facts"][0]["entity"] == "Delhivery Limited"
    assert data["facts"][0]["metric"] == "Express Parcel Pin Codes"
    assert data["facts"][0]["provenance"]["page_number"] == 1


# ---------------------------------------------------------------------------
# 4. Optional Live Gemini Test (Disabled by default)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.getenv("RUN_LIVE_GEMINI_TEST", "false").lower() != "true" or not os.getenv("GEMINI_API_KEY"),
    reason="Live Gemini tests disabled by default. Set RUN_LIVE_GEMINI_TEST=true and GEMINI_API_KEY.",
)
def test_live_gemini_extraction():
    """Opt-in live Gemini test for single page extraction."""
    extractor = FactExtractor()
    page = PageText(
        page_number=1,
        text="India's real GDP growth for FY2024-25 is estimated at 6.4 percent in the First Advance Estimates.",
        char_count=100,
        has_text=True,
    )
    facts = extractor.extract_from_page(page, "doc-live-test", "live_test.pdf")
    assert len(facts) >= 1
    assert facts[0].epistemic_status in [EpistemicStatus.ESTIMATED, EpistemicStatus.REPORTED]
    assert facts[0].provenance.supporting_text in page.text
