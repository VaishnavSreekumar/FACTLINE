"""Tests for Phase 8: Quota-Safe LLM Extraction Controls."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import httpx

from backend.api.routes import router
from backend.extraction.fact_extractor import (
    ExtractionError,
    ExtractionQuotaError,
    FactExtractor,
)
from backend.models.document import PageText, ParsedDocument
from backend.models.fact import FactRecord


@pytest.fixture
def sample_document() -> ParsedDocument:
    """Creates a sample 5-page document for page control testing."""
    pages = [
        PageText(page_number=1, text="Page 1: Acme revenue was 100 million USD.", char_count=41, has_text=True),
        PageText(page_number=2, text="Page 2: Acme profit was 20 million USD.", char_count=39, has_text=True),
        PageText(page_number=3, text="", char_count=0, has_text=False),  # Empty page
        PageText(page_number=4, text="Page 4: Acme assets were 500 million USD.", char_count=41, has_text=True),
        PageText(page_number=5, text="Page 5: Acme employees count was 1200.", char_count=37, has_text=True),
    ]
    return ParsedDocument(
        document_id="doc-quota-test",
        document_name="sample.pdf",
        total_pages=5,
        total_characters=sum(p.char_count for p in pages),
        pages=pages,
    )


def test_http_429_raises_extraction_quota_error():
    """Test 1: HTTP 429 raises the typed ExtractionQuotaError."""
    extractor = FactExtractor(api_key="dummy-key-12345")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.json.return_value = {
        "error": {
            "code": 429,
            "message": "Quota exceeded for quota metric 'GenerateRequestsPerDayPerModel-FreeTier'",
            "status": "RESOURCE_EXHAUSTED",
        }
    }

    with patch("httpx.Client.post", return_value=mock_response):
        with pytest.raises(ExtractionQuotaError) as exc_info:
            extractor._call_gemini("Test prompt")

        assert "Gemini API quota is currently exhausted" in str(exc_info.value)
        assert "dummy-key-12345" not in str(exc_info.value)


def test_http_429_does_not_trigger_fallback_models():
    """Test 2: HTTP 429 immediately stops and does not query fallback models."""
    extractor = FactExtractor(api_key="dummy-key-12345")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 429

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        with pytest.raises(ExtractionQuotaError):
            extractor._call_gemini("Test prompt")

        # Must have attempted exactly ONE request and stopped immediately
        assert mock_post.call_count == 1


def test_http_503_retains_conservative_retry_behavior():
    """Test 3: HTTP 503 retains retry behavior before falling through or raising."""
    extractor = FactExtractor(api_key="dummy-key-12345", model="gemini-3.6-flash")

    mock_503 = MagicMock(spec=httpx.Response)
    mock_503.status_code = 503

    mock_200 = MagicMock(spec=httpx.Response)
    mock_200.status_code = 200
    mock_200.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"facts": []}'}]}}]
    }

    # First call returns 503, second call succeeds with 200
    with patch("httpx.Client.post", side_effect=[mock_503, mock_200]) as mock_post:
        with patch("time.sleep", return_value=None):
            result = extractor._call_gemini("Test prompt")
            assert result == {"facts": []}
            assert mock_post.call_count == 2


def test_default_configuration_processes_all_eligible_pages(sample_document, monkeypatch):
    """Test 4: Default configuration (no limits) returns all pages for processing."""
    monkeypatch.delenv("FACTLINE_EXTRACTION_PAGE_SELECTION", raising=False)
    monkeypatch.delenv("FACTLINE_EXTRACTION_MAX_PAGES", raising=False)
    extractor = FactExtractor(api_key="dummy-key")
    filtered = extractor.filter_pages(sample_document.pages)
    assert len(filtered) == 5
    assert [p.page_number for p in filtered] == [1, 2, 3, 4, 5]


def test_max_pages_configuration_limits_extraction(sample_document):
    """Test 5: max_pages limits extraction to the first N eligible text pages."""
    extractor = FactExtractor(api_key="dummy-key", max_pages=2)
    filtered = extractor.filter_pages(sample_document.pages)
    # Only first 2 non-empty text pages: page 1 and page 2
    assert len(filtered) == 2
    assert [p.page_number for p in filtered] == [1, 2]


def test_explicit_page_selection(sample_document):
    """Test 6: Explicit page selection filters precisely to requested pages."""
    extractor = FactExtractor(api_key="dummy-key", page_selection=[2, 4])
    filtered = extractor.filter_pages(sample_document.pages)
    assert len(filtered) == 2
    assert [p.page_number for p in filtered] == [2, 4]


def test_selected_pages_retain_original_one_indexed_provenance(sample_document):
    """Test 7: Extracted facts from selected pages retain original 1-indexed PDF page number."""
    mock_caller = MagicMock(return_value={
        "facts": [
            {
                "entity": "Acme",
                "metric": "assets",
                "value_raw": "500 million USD",
                "value_numeric": 500.0,
                "unit": "million USD",
                "time_period": {"label": "FY24"},
                "supporting_text": "Page 4: Acme assets were 500 million USD.",
                "extraction_confidence": 0.95,
            }
        ]
    })
    extractor = FactExtractor(
        api_key="dummy-key",
        page_selection=[4],
        llm_caller=mock_caller,
    )
    facts = extractor.extract_from_document(sample_document)
    assert len(facts) == 1
    assert facts[0].provenance.page_number == 4
    assert facts[0].provenance.document_id == "doc-quota-test"


def test_invalid_out_of_range_selected_pages_safely_ignored(sample_document):
    """Test 8: Out-of-range page selection produces an empty list without error."""
    extractor = FactExtractor(api_key="dummy-key", page_selection=[99, 100])
    filtered = extractor.filter_pages(sample_document.pages)
    assert len(filtered) == 0

    facts = extractor.extract_from_document(sample_document)
    assert facts == []


def test_invalid_max_pages_value_validation():
    """Test 9: Non-positive max_pages raises ValueError upon initialization."""
    with pytest.raises(ValueError) as exc_0:
        FactExtractor(api_key="dummy-key", max_pages=0)
    assert "positive integer" in str(exc_0.value)

    with pytest.raises(ValueError) as exc_neg:
        FactExtractor(api_key="dummy-key", max_pages=-5)
    assert "positive integer" in str(exc_neg.value)


def test_quota_error_reaches_api_as_http_429(sample_document):
    """Test 10: Quota error reaches API endpoints with HTTP status 429."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("backend.api.routes.parser.parse_bytes", return_value=sample_document):
        with patch("backend.api.routes.extractor.extract_from_document") as mock_extract:
            mock_extract.side_effect = ExtractionQuotaError(
                "Gemini API quota is currently exhausted for this project/model. "
                "No further extraction requests were attempted."
            )

            response = client.post(
                "/documents/extract-facts",
                files={"file": ("sample.pdf", b"%PDF-dummy", "application/pdf")},
            )

            assert response.status_code == 429
            data = response.json()
            assert "Gemini API quota is currently exhausted" in data["detail"]


def test_no_api_key_in_user_facing_error(sample_document):
    """Test 11: API error output never exposes API key."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    secret_key = "AIzaSySecretKey12345"
    with patch("backend.api.routes.extractor.api_key", secret_key):
        with patch("backend.api.routes.parser.parse_bytes", return_value=sample_document):
            with patch("backend.api.routes.extractor.extract_from_document") as mock_extract:
                mock_extract.side_effect = ExtractionQuotaError(
                    "Gemini API quota is currently exhausted for this project/model."
                )

                response = client.post(
                    "/documents/extract-facts",
                    files={"file": ("sample.pdf", b"%PDF-dummy", "application/pdf")},
                )

                assert response.status_code == 429
                assert secret_key not in response.text
