"""Comprehensive Test Suite for FACTLINE Phase 6: Orchestration & Persistence."""

import json
import os
import sqlite3
import tempfile
from unittest.mock import MagicMock

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

from backend.db.database import DatabaseRepository, get_db_connection, init_db
from backend.extraction.fact_extractor import ExtractionError, FactExtractor
from backend.extraction.pdf_parser import PDFParser
from backend.main import app
from backend.models.document import PageText, ParsedDocument
from backend.models.fact import EpistemicStatus, FactRecord, Provenance, TimePeriod
from backend.models.normalization import NormalizationStatus, NormalizedFact, NormalizedValue
from backend.models.relationship import RelationshipResult, RelationshipType
from backend.normalization.normalizer import FactNormalizer
from backend.reasoning.comparability import ComparabilityGate
from backend.reasoning.matcher import CandidateMatcher
from backend.reasoning.relationships import RelationshipEngine
from backend.services.analysis import AnalysisService


def create_in_memory_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate a valid PDF byte string in memory."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def temp_db_path():
    """Provides an isolated temporary SQLite database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    init_db(path)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def test_single_document_analysis(temp_db_path):
    """Test 1: Single document parse -> extract -> normalize -> persist."""
    repo = DatabaseRepository(db_path=temp_db_path)
    parser = PDFParser()
    normalizer = FactNormalizer()
    matcher = CandidateMatcher()
    gate = ComparabilityGate()
    rel_engine = RelationshipEngine()

    # Create synthetic PDF
    pdf_bytes = create_in_memory_pdf(["Delhivery reported revenue of 8142 crore for FY24."])

    # Mock extractor
    def extract_fn(doc: ParsedDocument):
        return [
            FactRecord(
                fact_id="fact_delhivery_1",
                entity="Delhivery",
                metric="Revenue",
                value_raw="8142 crore",
                value_numeric=8142.0,
                unit="crore",
                time_period=TimePeriod(label="FY24", start_date="2023-04-01", end_date="2024-03-31"),
                provenance=Provenance(
                    document_id=doc.document_id,
                    page_number=1,
                    supporting_text="Delhivery reported revenue of 8142 crore for FY24.",
                ),
                extraction_confidence=0.95,
            )
        ]

    mock_extractor = MagicMock(spec=FactExtractor)
    mock_extractor.extract_from_document.side_effect = extract_fn

    service = AnalysisService(
        parser=parser,
        extractor=mock_extractor,
        normalizer=normalizer,
        matcher=matcher,
        gate=gate,
        rel_engine=rel_engine,
        repository=repo,
    )

    result = service.analyze_documents([("delhivery_drhp.pdf", pdf_bytes)])

    assert "analysis_id" in result
    assert result["summary"]["documents_processed"] == 1
    assert result["summary"]["facts_extracted"] == 1

    # Verify retrieval
    analysis_data = service.get_analysis(result["analysis_id"])
    assert analysis_data is not None
    assert len(analysis_data["documents"]) == 1
    assert len(analysis_data["facts"]) == 1
    assert analysis_data["facts"][0]["fact"]["entity"] == "Delhivery"
    assert analysis_data["facts"][0]["normalized_value"]["numeric_value"] == 81420000000.0


def test_two_document_analysis(temp_db_path):
    """Test 2: Two-document analysis with cross-document candidates and relationship."""
    repo = DatabaseRepository(db_path=temp_db_path)
    pdf1 = create_in_memory_pdf(["Delhivery Revenue was Rs 8142 crore in FY24."])
    pdf2 = create_in_memory_pdf(["Delhivery reported FY24 Revenue of Rs 81415.38 million."])

    mock_extractor = MagicMock(spec=FactExtractor)

    def extract_side_effect(doc: ParsedDocument):
        if "doc1" in doc.document_name:
            return [
                FactRecord(
                    fact_id="fact_doc1",
                    entity="Delhivery",
                    metric="Revenue",
                    value_raw="₹8,142 crore",
                    value_numeric=8142.0,
                    unit="INR",
                    time_period=TimePeriod(label="FY24", start_date="2023-04-01", end_date="2024-03-31"),
                    provenance=Provenance(
                        document_id=doc.document_id,
                        page_number=1,
                        supporting_text="Delhivery Revenue was Rs 8142 crore in FY24.",
                    ),
                    extraction_confidence=0.95,
                )
            ]
        else:
            return [
                FactRecord(
                    fact_id="fact_doc2",
                    entity="Delhivery",
                    metric="Revenue",
                    value_raw="₹81,415.38 million",
                    value_numeric=81415.38,
                    unit="INR",
                    time_period=TimePeriod(label="FY24", start_date="2023-04-01", end_date="2024-03-31"),
                    provenance=Provenance(
                        document_id=doc.document_id,
                        page_number=1,
                        supporting_text="Delhivery reported FY24 Revenue of Rs 81415.38 million.",
                    ),
                    extraction_confidence=0.95,
                )
            ]

    mock_extractor.extract_from_document.side_effect = extract_side_effect

    service = AnalysisService(
        extractor=mock_extractor,
        repository=repo,
    )

    result = service.analyze_documents([("doc1.pdf", pdf1), ("doc2.pdf", pdf2)])

    assert result["summary"]["documents_processed"] == 2
    assert result["summary"]["facts_extracted"] == 2
    assert result["summary"]["candidate_pairs"] == 1
    assert result["summary"]["relationships_created"] == 1

    detail = service.get_analysis(result["analysis_id"])
    assert detail is not None
    rel = detail["relationships"][0]
    assert rel["relationship_type"] == "CONTEXT_RESOLVES"
    assert "ROUNDING_DIFFERENCE" in rel["reason_codes"]


def test_evidence_persistence(temp_db_path):
    """Test 3: Verify page_number, supporting_text, document_id survive persistence."""
    repo = DatabaseRepository(db_path=temp_db_path)
    pdf = create_in_memory_pdf(["Page one text", "Page two: Exact supporting snippet for Fact X."])

    mock_extractor = MagicMock(spec=FactExtractor)

    def extract_fn(doc: ParsedDocument):
        return [
            FactRecord(
                fact_id="fact_evidence_test",
                entity="EntityX",
                metric="MetricY",
                value_raw="500",
                value_numeric=500.0,
                unit="count",
                time_period=TimePeriod(label="2024", start_date="2024-01-01", end_date="2024-12-31"),
                provenance=Provenance(
                    document_id=doc.document_id,
                    page_number=2,
                    supporting_text="Page two: Exact supporting snippet for Fact X.",
                    document_date="2024-05-01",
                ),
                extraction_confidence=1.0,
            )
        ]

    mock_extractor.extract_from_document.side_effect = extract_fn

    service = AnalysisService(extractor=mock_extractor, repository=repo)
    result = service.analyze_documents([("doc_ev.pdf", pdf)])

    detail = service.get_analysis(result["analysis_id"])
    persisted_fact = detail["facts"][0]["fact"]

    assert persisted_fact["provenance"]["page_number"] == 2
    assert persisted_fact["provenance"]["supporting_text"] == "Page two: Exact supporting snippet for Fact X."
    assert persisted_fact["provenance"]["document_date"] == "2024-05-01"


def test_relationship_persistence(temp_db_path):
    """Test 4: Verify relationship type, reasons, explanation, and evidence survive persistence."""
    repo = DatabaseRepository(db_path=temp_db_path)
    pdf1 = create_in_memory_pdf(["Claim A"])
    pdf2 = create_in_memory_pdf(["Claim B"])

    mock_extractor = MagicMock(spec=FactExtractor)

    def extract_fn(doc: ParsedDocument):
        return [
            FactRecord(
                fact_id=f"fact_{doc.document_name}",
                entity="CompanyA",
                metric="Profit",
                value_raw="100 INR",
                value_numeric=100.0,
                unit="INR",
                time_period=TimePeriod(label="2024", start_date="2024-01-01", end_date="2024-12-31"),
                provenance=Provenance(document_id=doc.document_id, page_number=1, supporting_text=f"Claim in {doc.document_name}"),
                extraction_confidence=1.0,
            )
        ]

    mock_extractor.extract_from_document.side_effect = extract_fn

    service = AnalysisService(extractor=mock_extractor, repository=repo)
    result = service.analyze_documents([("doc1.pdf", pdf1), ("doc2.pdf", pdf2)])

    detail = service.get_analysis(result["analysis_id"])
    rel = detail["relationships"][0]

    assert rel["relationship_type"] == "CORROBORATES"
    assert "EQUAL_NORMALIZED_VALUE" in rel["reason_codes"]
    assert len(rel["explanation"]) > 0
    assert "Claim in doc1.pdf" in rel["evidence_a"]["supporting_text"]
    assert "Claim in doc2.pdf" in rel["evidence_b"]["supporting_text"]


def test_idempotency(temp_db_path):
    """Test 5: Analyzing same deterministic inputs twice does not multiply DB core entities."""
    repo = DatabaseRepository(db_path=temp_db_path)
    pdf = create_in_memory_pdf(["Static content for idempotent test."])

    mock_extractor = MagicMock(spec=FactExtractor)

    def extract_fn(doc: ParsedDocument):
        return [
            FactRecord(
                fact_id="fact_static_100",
                entity="StaticCo",
                metric="Revenue",
                value_raw="50",
                value_numeric=50.0,
                unit="USD",
                time_period=TimePeriod(label="2024", start_date="2024-01-01", end_date="2024-12-31"),
                provenance=Provenance(document_id=doc.document_id, page_number=1, supporting_text="Static content"),
                extraction_confidence=1.0,
            )
        ]

    mock_extractor.extract_from_document.side_effect = extract_fn

    service = AnalysisService(extractor=mock_extractor, repository=repo)

    res1 = service.analyze_documents([("static.pdf", pdf)])
    res2 = service.analyze_documents([("static.pdf", pdf)])

    assert res1["analysis_id"] != res2["analysis_id"]  # Different execution UUIDs

    # Query raw database tables directly to verify no duplicate rows created
    conn = get_db_connection(temp_db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents;")
    doc_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM facts;")
    fact_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM analyses;")
    analysis_count = cur.fetchone()[0]
    conn.close()

    assert doc_count == 1
    assert fact_count == 1
    assert analysis_count == 2


def test_empty_or_image_only_document(temp_db_path):
    """Test 6: Gracefully completes with 0 facts and 0 relationships on empty pages."""
    repo = DatabaseRepository(db_path=temp_db_path)
    # Empty PDF with 1 blank page
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()

    service = AnalysisService(repository=repo)
    result = service.analyze_documents([("empty.pdf", pdf_bytes)])

    assert result["summary"]["documents_processed"] == 1
    assert result["summary"]["facts_extracted"] == 0
    assert result["summary"]["candidate_pairs"] == 0
    assert result["summary"]["relationships_created"] == 0


def test_extraction_failure_handling(temp_db_path):
    """Test 7: Extractor failure surfaces error cleanly without partial success."""
    repo = DatabaseRepository(db_path=temp_db_path)
    pdf = create_in_memory_pdf(["Some text"])

    mock_extractor = MagicMock(spec=FactExtractor)
    mock_extractor.extract_from_document.side_effect = ExtractionError("Simulated Gemini API quota exceeded")

    service = AnalysisService(extractor=mock_extractor, repository=repo)

    with pytest.raises(ExtractionError, match="Simulated Gemini API quota exceeded"):
        service.analyze_documents([("doc.pdf", pdf)])

    # Confirm no orphaned analysis was committed
    conn = get_db_connection(temp_db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM analyses;")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 0


def test_database_rollback(temp_db_path):
    """Test 8: Database transaction rollback on persistence failure."""
    mock_repo = MagicMock(spec=DatabaseRepository)
    mock_repo.save_analysis.side_effect = sqlite3.OperationalError("Disk I/O error or constraint violation")

    pdf = create_in_memory_pdf(["Some text"])
    mock_extractor = MagicMock(spec=FactExtractor)
    mock_extractor.extract_from_document.return_value = []

    service = AnalysisService(extractor=mock_extractor, repository=mock_repo)

    with pytest.raises(sqlite3.OperationalError, match="Disk I/O error"):
        service.analyze_documents([("doc.pdf", pdf)])


def test_n_document_support(temp_db_path):
    """Test 9: Supports arbitrary N documents (e.g. 3 documents)."""
    repo = DatabaseRepository(db_path=temp_db_path)
    pdf1 = create_in_memory_pdf(["Doc 1 text"])
    pdf2 = create_in_memory_pdf(["Doc 2 text"])
    pdf3 = create_in_memory_pdf(["Doc 3 text"])

    mock_extractor = MagicMock(spec=FactExtractor)

    def extract_fn(doc: ParsedDocument):
        return [
            FactRecord(
                fact_id=f"fact_{doc.document_name}",
                entity="GlobalCorp",
                metric="Revenue",
                value_raw="1000",
                value_numeric=1000.0,
                unit="USD",
                time_period=TimePeriod(label="2024", start_date="2024-01-01", end_date="2024-12-31"),
                provenance=Provenance(document_id=doc.document_id, page_number=1, supporting_text="Text"),
                extraction_confidence=1.0,
            )
        ]

    mock_extractor.extract_from_document.side_effect = extract_fn

    service = AnalysisService(extractor=mock_extractor, repository=repo)
    result = service.analyze_documents([("doc1.pdf", pdf1), ("doc2.pdf", pdf2), ("doc3.pdf", pdf3)])

    assert result["summary"]["documents_processed"] == 3
    assert result["summary"]["facts_extracted"] == 3
    # 3 facts of same entity/metric -> 3 choose 2 = 3 pairs
    assert result["summary"]["candidate_pairs"] == 3
    assert result["summary"]["relationships_created"] == 3


def test_no_llm_in_reasoning(temp_db_path):
    """Test 10: Matcher, Gate, and RelationshipEngine execute deterministically with zero LLM calls."""
    repo = DatabaseRepository(db_path=temp_db_path)
    pdf1 = create_in_memory_pdf(["Doc A"])
    pdf2 = create_in_memory_pdf(["Doc B"])

    mock_extractor = MagicMock(spec=FactExtractor)

    def extract_fn(doc: ParsedDocument):
        if "d1" in doc.document_name:
            return [
                FactRecord(
                    fact_id="f1",
                    entity="Apex",
                    metric="EBITDA",
                    value_raw="200",
                    value_numeric=200.0,
                    unit="USD",
                    time_period=TimePeriod(label="2024", start_date="2024-01-01", end_date="2024-12-31"),
                    provenance=Provenance(document_id=doc.document_id, page_number=1, supporting_text="t1"),
                    extraction_confidence=1.0,
                )
            ]
        else:
            return [
                FactRecord(
                    fact_id="f2",
                    entity="Apex",
                    metric="EBITDA",
                    value_raw="250",
                    value_numeric=250.0,
                    unit="USD",
                    time_period=TimePeriod(label="2024", start_date="2024-01-01", end_date="2024-12-31"),
                    provenance=Provenance(document_id=doc.document_id, page_number=1, supporting_text="t2"),
                    extraction_confidence=1.0,
                )
            ]

    mock_extractor.extract_from_document.side_effect = extract_fn

    service = AnalysisService(extractor=mock_extractor, repository=repo)
    result = service.analyze_documents([("d1.pdf", pdf1), ("d2.pdf", pdf2)])

    # Extractor called exactly once per document during extraction step
    assert mock_extractor.extract_from_document.call_count == 2

    # Relationship reasoning executed deterministically (conflicting EBITDA values -> CONTRADICTS)
    detail = service.get_analysis(result["analysis_id"])
    assert detail["relationships"][0]["relationship_type"] == "CONTRADICTS"
    assert "VALUE_CONFLICT" in detail["relationships"][0]["reason_codes"]


def test_cross_document_numeric_isolation(temp_db_path):
    """Test 11 (MANDATORY): Cross-document numeric isolation.

    Demonstrates: 'Numbers never create candidates.'
    Doc A: Revenue = 100
    Doc B: Employees = 100
    Doc C: Revenue = 100
    Verify that numeric equality alone never creates a candidate or relationship between Revenue and Employees.
    """
    repo = DatabaseRepository(db_path=temp_db_path)
    pdf_a = create_in_memory_pdf(["Doc A text"])
    pdf_b = create_in_memory_pdf(["Doc B text"])
    pdf_c = create_in_memory_pdf(["Doc C text"])

    mock_extractor = MagicMock(spec=FactExtractor)

    def extract_mock(doc: ParsedDocument):
        if "doc_a" in doc.document_name:
            return [
                FactRecord(
                    fact_id="fact_rev_a",
                    entity="OmniCorp",
                    metric="Revenue",
                    value_raw="100",
                    value_numeric=100.0,
                    unit="INR",
                    time_period=TimePeriod(label="2024", start_date="2024-01-01", end_date="2024-12-31"),
                    provenance=Provenance(document_id=doc.document_id, page_number=1, supporting_text="A"),
                    extraction_confidence=1.0,
                )
            ]
        elif "doc_b" in doc.document_name:
            return [
                FactRecord(
                    fact_id="fact_emp_b",
                    entity="OmniCorp",
                    metric="Employees",
                    value_raw="100",
                    value_numeric=100.0,
                    unit="count",
                    time_period=TimePeriod(label="2024", start_date="2024-01-01", end_date="2024-12-31"),
                    provenance=Provenance(document_id=doc.document_id, page_number=1, supporting_text="B"),
                    extraction_confidence=1.0,
                )
            ]
        else:
            return [
                FactRecord(
                    fact_id="fact_rev_c",
                    entity="OmniCorp",
                    metric="Revenue",
                    value_raw="100",
                    value_numeric=100.0,
                    unit="INR",
                    time_period=TimePeriod(label="2024", start_date="2024-01-01", end_date="2024-12-31"),
                    provenance=Provenance(document_id=doc.document_id, page_number=1, supporting_text="C"),
                    extraction_confidence=1.0,
                )
            ]

    mock_extractor.extract_from_document.side_effect = extract_mock

    service = AnalysisService(extractor=mock_extractor, repository=repo)
    result = service.analyze_documents([("doc_a.pdf", pdf_a), ("doc_b.pdf", pdf_b), ("doc_c.pdf", pdf_c)])

    assert result["summary"]["documents_processed"] == 3
    assert result["summary"]["facts_extracted"] == 3

    # Only Rev A and Rev C should be paired. Emp B has no candidate pairing despite having numeric_value=100.
    assert result["summary"]["candidate_pairs"] == 1
    assert result["summary"]["relationships_created"] == 1

    detail = service.get_analysis(result["analysis_id"])
    rel = detail["relationships"][0]
    assert {rel["fact_a_id"], rel["fact_b_id"]} == {"fact_rev_a", "fact_rev_c"}
    assert rel["relationship_type"] == "CORROBORATES"


def test_api_analysis_endpoints(monkeypatch, temp_db_path):
    """Test 12: API smoke test for POST /analysis and GET /analysis/{analysis_id}."""
    monkeypatch.setenv("DATABASE_PATH", temp_db_path)

    # Patch FactExtractor.extract_from_document so the test is hermetic and doesn't call Gemini
    def mock_extract(self, doc: ParsedDocument):
        return [
            FactRecord(
                fact_id="fact_api_1",
                entity="ApiEntity",
                metric="Revenue",
                value_raw="5000",
                value_numeric=5000.0,
                unit="USD",
                time_period=TimePeriod(label="2024", start_date="2024-01-01", end_date="2024-12-31"),
                provenance=Provenance(document_id=doc.document_id, page_number=1, supporting_text="Sample invoice"),
                extraction_confidence=1.0,
            )
        ]

    monkeypatch.setattr(FactExtractor, "extract_from_document", mock_extract)

    client = TestClient(app)
    pdf_bytes = create_in_memory_pdf(["Sample invoice for testing API"])

    # POST /analysis with mock extractor attached
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        tmp_pdf.write(pdf_bytes)
        tmp_pdf_path = tmp_pdf.name

    try:
        with open(tmp_pdf_path, "rb") as f:
            resp = client.post(
                "/analysis",
                files=[("files", ("test_doc.pdf", f, "application/pdf"))],
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "analysis_id" in data
        assert "summary" in data
        assert data["summary"]["documents_processed"] == 1
        assert data["summary"]["facts_extracted"] == 1

        analysis_id = data["analysis_id"]

        # GET /analysis/{analysis_id}
        get_resp = client.get(f"/analysis/{analysis_id}")
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert get_data["analysis_id"] == analysis_id
        assert len(get_data["documents"]) == 1
        assert len(get_data["facts"]) == 1

        # GET /analysis/non_existent_id
        not_found_resp = client.get("/analysis/non_existent_uuid")
        assert not_found_resp.status_code == 404
    finally:
        if os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)
