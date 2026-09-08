"""Unit and integration tests for PDF extraction and evidence provenance."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.extraction.pdf_parser import PDFParser, PDFParsingError
from backend.extraction.evidence import create_evidence, EvidenceVerifier
from backend.models.document import ParsedDocument, PageText


DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "sample-data" / "starter-datasets"
DELHIVERY_DIR = DATASET_DIR / "delhivery"
MACRO_DIR = DATASET_DIR / "india-macroeconomy"


@pytest.fixture
def parser():
    return PDFParser()


@pytest.fixture
def client():
    return TestClient(app)


def test_page_numbering_one_indexed(parser):
    """Test 1: Confirm that page numbers are strictly 1-indexed."""
    sample_pdf = next(DELHIVERY_DIR.glob("*.pdf"), None)
    assert sample_pdf is not None, "No Delhivery PDF found in sample-data"

    doc = parser.parse(sample_pdf)
    assert doc.total_pages > 0
    assert doc.pages[0].page_number == 1

    for idx, page in enumerate(doc.pages):
        assert page.page_number == idx + 1


def test_text_extraction_fidelity(parser):
    """Test 2: Confirm text extraction returns content and calculates character counts."""
    sample_pdf = next(DELHIVERY_DIR.glob("*.pdf"), None)
    assert sample_pdf is not None

    doc = parser.parse(sample_pdf)
    has_any_text = any(p.has_text for p in doc.pages)
    assert has_any_text, "Expected at least one page to contain extractable text"

    first_text_page = next(p for p in doc.pages if p.has_text)
    assert len(first_text_page.text) > 0
    assert first_text_page.char_count == len(first_text_page.text)


def test_multi_page_extraction(parser):
    """Test 3: Confirm multi-page PDFs produce discrete, sequential page records."""
    sample_pdf = next(MACRO_DIR.glob("*.pdf"), None)
    assert sample_pdf is not None, "No Macroeconomics PDF found in sample-data"

    doc = parser.parse(sample_pdf)
    assert doc.total_pages >= 2
    assert len(doc.pages) == doc.total_pages

    # Ensure page numbers are unique and strictly monotonic
    page_numbers = [p.page_number for p in doc.pages]
    assert page_numbers == list(range(1, doc.total_pages + 1))


def test_evidence_provenance_and_verification(parser):
    """Test 4: Confirm evidence objects capture document_id, page_number, and supporting text."""
    sample_pdf = next(DELHIVERY_DIR.glob("*.pdf"), None)
    doc = parser.parse(sample_pdf)

    # Pick a page with text
    text_page = next(p for p in doc.pages if p.has_text and len(p.text) > 50)
    # Extract an actual snippet from that page
    snippet = text_page.text[:40].strip()

    evidence = create_evidence(
        document_id=doc.document_id,
        page_number=text_page.page_number,
        supporting_text=snippet,
    )

    assert evidence.document_id == doc.document_id
    assert evidence.page_number == text_page.page_number
    assert evidence.supporting_text == snippet

    # Verify provenance against the page
    assert EvidenceVerifier.verify_provenance(evidence, text_page) is True

    # Verify non-matching text fails provenance
    fake_evidence = create_evidence(
        document_id=doc.document_id,
        page_number=text_page.page_number,
        supporting_text="This synthetic sentence definitely does not exist on page.",
    )
    assert EvidenceVerifier.verify_provenance(fake_evidence, text_page) is False


def test_evidence_one_indexed_validation():
    """Ensure evidence creation enforces 1-indexed page constraint."""
    with pytest.raises(ValueError, match="page_number must be >= 1"):
        create_evidence(
            document_id="doc-123",
            page_number=0,
            supporting_text="Invalid 0-indexed page",
        )


def test_invalid_file_handling(parser):
    """Test 5: Confirm nonexistent or invalid PDF input produces clear errors."""
    with pytest.raises(FileNotFoundError):
        parser.parse("nonexistent_file_xyz_123.pdf")

    with pytest.raises(PDFParsingError):
        parser.parse_bytes(b"not a valid pdf content", document_name="corrupted.pdf")

    with pytest.raises(PDFParsingError):
        parser.parse_bytes(b"", document_name="empty.pdf")


def test_real_dataset_delhivery_and_macro_parsing(parser):
    """Validate parsing across actual Delhivery and India macroeconomy starter PDFs."""
    delhivery_pdf = DELHIVERY_DIR / "01-delhivery-prospectus-2022-excerpt.pdf"
    macro_pdf = MACRO_DIR / "01-india-economic-survey-2024-25-excerpt.pdf"

    assert delhivery_pdf.exists(), f"Delhivery test PDF missing: {delhivery_pdf}"
    assert macro_pdf.exists(), f"Macroeconomics test PDF missing: {macro_pdf}"

    delhivery_doc = parser.parse(delhivery_pdf)
    assert isinstance(delhivery_doc, ParsedDocument)
    assert delhivery_doc.total_pages > 0
    assert len(delhivery_doc.pages) == delhivery_doc.total_pages

    macro_doc = parser.parse(macro_pdf)
    assert isinstance(macro_doc, ParsedDocument)
    assert macro_doc.total_pages > 0
    assert len(macro_doc.pages) == macro_doc.total_pages


def test_api_parse_endpoint(client):
    """Test POST /documents/parse endpoint with a real PDF file upload."""
    sample_pdf = DELHIVERY_DIR / "01-delhivery-prospectus-2022-excerpt.pdf"
    assert sample_pdf.exists()

    with open(sample_pdf, "rb") as f:
        response = client.post(
            "/documents/parse",
            files={"file": (sample_pdf.name, f, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert "total_pages" in data
    assert "pages" in data
    assert len(data["pages"]) == data["total_pages"]
    assert data["pages"][0]["page_number"] == 1
