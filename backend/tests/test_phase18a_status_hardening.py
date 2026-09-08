"""Tests for Phase 18A: Quota Status UI and Aggregate Count Hardening."""

from unittest.mock import MagicMock, patch
import pytest

from backend.db.database import DatabaseRepository
from backend.extraction.fact_extractor import (
    FactExtractor,
    ExtractionQuotaError,
    ExtractionError,
)
from backend.models.document import PageText, ParsedDocument
from backend.quota_experiment.status import BatchExtractionResult, ExtractionStatus
from backend.services.analysis import AnalysisService


def _create_dummy_parsed_doc(doc_id: str, doc_name: str, page_numbers: list[int]) -> ParsedDocument:
    pages = [
        PageText(
            page_number=p,
            text=f"Sample revenue fact on page {p}: Revenue is ${p * 10}M in FY24.",
            char_count=50,
            has_text=True,
        )
        for p in page_numbers
    ]
    return ParsedDocument(
        document_id=doc_id,
        document_name=doc_name,
        total_pages=len(page_numbers),
        pages=pages,
    )


def test_multi_document_unprocessed_page_aggregation(tmp_path):
    """Verify AnalysisService correctly accumulates unprocessed pages across multiple documents."""
    db_file = tmp_path / "test_aggregation.db"
    repo = DatabaseRepository(db_path=str(db_file))

    extractor = FactExtractor(batch_size=2, page_filter_enabled=False, context_selector_enabled=False)

    doc_a = _create_dummy_parsed_doc("doc-a", "DocA.pdf", [1, 2, 3, 4])
    doc_b = _create_dummy_parsed_doc("doc-b", "DocB.pdf", [1, 2, 3, 4, 5, 6])

    mock_parser = MagicMock()
    mock_parser.parse_bytes.side_effect = [doc_a, doc_b]

    # Mock extract_from_document with last_extraction_result on extractor
    call_count = 0

    def mock_extract(doc, document_date=None):
        nonlocal call_count
        call_count += 1
        if doc.document_id == "doc-a":
            # 2 processed, 2 unprocessed
            res = BatchExtractionResult(
                status=ExtractionStatus.PARTIAL_QUOTA,
                document_id="doc-a",
                total_eligible_pages=4,
                processed_pages=[1, 2],
                unprocessed_pages=[3, 4],
                requests_used=1,
                requests_available=None,
                batch_size=2,
                facts=[],
                facts_rejected_grounding=0,
                user_message="Partial",
            )
        else:
            # 0 processed, 6 unprocessed
            res = BatchExtractionResult(
                status=ExtractionStatus.QUOTA_EXHAUSTED,
                document_id="doc-b",
                total_eligible_pages=6,
                processed_pages=[],
                unprocessed_pages=[1, 2, 3, 4, 5, 6],
                requests_used=0,
                requests_available=None,
                batch_size=2,
                facts=[],
                facts_rejected_grounding=0,
                user_message="Quota exhausted",
            )
        extractor.last_extraction_result = res
        return res.facts

    with patch.object(extractor, "extract_from_document", side_effect=mock_extract):
        service = AnalysisService(
            parser=mock_parser,
            extractor=extractor,
            repository=repo,
        )
        persisted = service.analyze_documents([
            ("DocA.pdf", b"%PDF-1.4 a"),
            ("DocB.pdf", b"%PDF-1.4 b"),
        ])

    assert persisted is not None
    summary = service.get_analysis(persisted["analysis_id"])["summary"]
    ext_status = summary["extraction_status"]

    # Verify overall aggregation across documents
    assert ext_status["status"] == "QUOTA_EXHAUSTED"
    assert ext_status["total_eligible_pages"] == 10  # 4 + 6
    assert ext_status["total_processed_pages"] == 2   # 2 + 0
    assert ext_status["total_unprocessed_pages"] == 8 # 2 + 6
    assert ext_status["requests_used"] == 1           # 1 + 0

    # Verify per-document statuses are preserved
    docs = ext_status["documents"]
    assert len(docs) == 2
    assert docs[0]["document_name"] == "DocA.pdf"
    assert docs[0]["status"] == "PARTIAL_QUOTA"
    assert docs[0]["processed_pages"] == [1, 2]
    assert docs[0]["unprocessed_pages"] == [3, 4]

    assert docs[1]["document_name"] == "DocB.pdf"
    assert docs[1]["status"] == "QUOTA_EXHAUSTED"
    assert docs[1]["processed_pages"] == []
    assert docs[1]["unprocessed_pages"] == [1, 2, 3, 4, 5, 6]


def test_january_success_and_april_quota_exhaustion_scenario(tmp_path):
    """Simulates real-world scenario: Jan partially succeeds (6/10 pages), April hits 429 immediately (0/184 pages)."""
    db_file = tmp_path / "test_jan_apr.db"
    repo = DatabaseRepository(db_path=str(db_file))

    extractor = FactExtractor(batch_size=3, page_filter_enabled=False, context_selector_enabled=False)

    jan_doc = _create_dummy_parsed_doc("jan-id", "JAN IMF.pdf", list(range(1, 12)))
    apr_doc = _create_dummy_parsed_doc("apr-id", "APRIL IMF.pdf", list(range(1, 191)))

    mock_parser = MagicMock()
    mock_parser.parse_bytes.side_effect = [jan_doc, apr_doc]

    def mock_extract(doc, document_date=None):
        if doc.document_id == "jan-id":
            res = BatchExtractionResult(
                status=ExtractionStatus.PARTIAL_QUOTA,
                document_id="jan-id",
                total_eligible_pages=10,
                processed_pages=[2, 3, 4, 5, 6, 7],
                unprocessed_pages=[8, 9, 10, 11],
                requests_used=2,
                requests_available=None,
                batch_size=3,
                facts=[],
                facts_rejected_grounding=0,
                user_message="Analysis halted due to quota limit.",
                error_message="Gemini API quota is currently exhausted for this project/model.",
            )
        else:
            res = BatchExtractionResult(
                status=ExtractionStatus.QUOTA_EXHAUSTED,
                document_id="apr-id",
                total_eligible_pages=184,
                processed_pages=[],
                unprocessed_pages=list(range(4, 188)),
                requests_used=0,
                requests_available=None,
                batch_size=3,
                facts=[],
                facts_rejected_grounding=0,
                user_message="Analysis halted due to quota limit. 0 of 184 eligible pages were processed.",
                error_message="Gemini API quota is currently exhausted for this project/model.",
            )
        extractor.last_extraction_result = res
        return res.facts

    with patch.object(extractor, "extract_from_document", side_effect=mock_extract):
        service = AnalysisService(
            parser=mock_parser,
            extractor=extractor,
            repository=repo,
        )
        persisted = service.analyze_documents([
            ("JAN IMF.pdf", b"%PDF-1.4 jan"),
            ("APRIL IMF.pdf", b"%PDF-1.4 apr"),
        ])

    assert persisted is not None
    summary = service.get_analysis(persisted["analysis_id"])["summary"]
    ext_status = summary["extraction_status"]

    assert ext_status["status"] == "QUOTA_EXHAUSTED"
    assert ext_status["total_eligible_pages"] == 194
    assert ext_status["total_processed_pages"] == 6
    assert ext_status["total_unprocessed_pages"] == 188  # 4 from Jan + 184 from April
    assert ext_status["requests_used"] == 2


@pytest.mark.parametrize("status_enum,expected_status_str", [
    (ExtractionStatus.COMPLETE, "COMPLETE"),
    (ExtractionStatus.PARTIAL_QUOTA, "PARTIAL_QUOTA"),
    (ExtractionStatus.QUOTA_EXHAUSTED, "QUOTA_EXHAUSTED"),
    (ExtractionStatus.FAILED, "FAILED"),
])
def test_all_extraction_statuses_persisted(tmp_path, status_enum, expected_status_str):
    """Verify that all four extraction statuses are correctly propagated and saved."""
    db_file = tmp_path / f"test_{expected_status_str.lower()}.db"
    repo = DatabaseRepository(db_path=str(db_file))

    extractor = FactExtractor(batch_size=2, page_filter_enabled=False, context_selector_enabled=False)
    doc = _create_dummy_parsed_doc("doc-1", "Single.pdf", [1, 2])

    mock_parser = MagicMock()
    mock_parser.parse_bytes.return_value = doc

    processed = [1, 2] if status_enum == ExtractionStatus.COMPLETE else ([1] if status_enum == ExtractionStatus.PARTIAL_QUOTA else [])
    unprocessed = [] if status_enum == ExtractionStatus.COMPLETE else ([2] if status_enum == ExtractionStatus.PARTIAL_QUOTA else [1, 2])

    def mock_extract(doc, document_date=None):
        res = BatchExtractionResult(
            status=status_enum,
            document_id="doc-1",
            total_eligible_pages=2,
            processed_pages=processed,
            unprocessed_pages=unprocessed,
            requests_used=1 if processed else 0,
            requests_available=None,
            batch_size=2,
            facts=[],
            facts_rejected_grounding=0,
            user_message=f"Status is {expected_status_str}",
            error_message="Error" if status_enum == ExtractionStatus.FAILED else None,
        )
        extractor.last_extraction_result = res
        return res.facts

    with patch.object(extractor, "extract_from_document", side_effect=mock_extract):
        service = AnalysisService(
            parser=mock_parser,
            extractor=extractor,
            repository=repo,
        )
        persisted = service.analyze_documents([("Single.pdf", b"%PDF-1.4 single")])

    summary = service.get_analysis(persisted["analysis_id"])["summary"]
    ext_status = summary["extraction_status"]
    assert ext_status["status"] == expected_status_str
    assert ext_status["total_unprocessed_pages"] == len(unprocessed)
    assert ext_status["total_processed_pages"] == len(processed)
