"""
Phase 18C Regression Tests: Numeric Sign Preservation + Transient 503 Hardening.
"""

import json
from unittest.mock import MagicMock, patch
import httpx
import pytest

from backend.extraction.fact_extractor import (
    ExtractionError,
    ExtractionQuotaError,
    FactExtractor,
)
from backend.models.fact import (
    EpistemicStatus,
    FactRecord,
    Provenance,
    TimePeriod,
)
from backend.models.normalization import NormalizationStatus
from backend.normalization.normalizer import FactNormalizer
from backend.normalization.units import UnitNormalizer


# ============================================================================
# 1. FIX 1 TESTS: NUMERIC SIGN PRESERVATION & DIRECTIONAL WORDING
# ============================================================================

@pytest.fixture
def unit_normalizer():
    return UnitNormalizer()


@pytest.fixture
def fact_normalizer():
    return FactNormalizer()


def test_unicode_minus_and_dashes_sign_preservation(unit_normalizer):
    """Verify Unicode dashes/hyphens are normalized and negative signs preserved."""
    # U+2014 EM DASH —
    res1 = unit_normalizer.normalize("—5.3%")
    assert res1.numeric_value == -5.3
    assert res1.canonical_unit == "percent"
    assert res1.normalization_status == NormalizationStatus.NORMALIZED

    # U+2013 EN DASH –
    res2 = unit_normalizer.normalize("–7.2%")
    assert res2.numeric_value == -7.2
    assert res2.canonical_unit == "percent"

    # U+2212 MINUS SIGN −
    res3 = unit_normalizer.normalize("−1.1%")
    assert res3.numeric_value == -1.1
    assert res3.canonical_unit == "percent"

    # U+2014 EM DASH —
    res4 = unit_normalizer.normalize("—4.1%")
    assert res4.numeric_value == -4.1
    assert res4.canonical_unit == "percent"

    # U+2010 HYPHEN ‐
    res5 = unit_normalizer.normalize("\u20102.5%")
    assert res5.numeric_value == -2.5

    # U+2011 NON-BREAKING HYPHEN ‑
    res6 = unit_normalizer.normalize("\u20113.8%")
    assert res6.numeric_value == -3.8

    # U+2012 FIGURE DASH ‒
    res7 = unit_normalizer.normalize("\u20126.9%")
    assert res7.numeric_value == -6.9

    # U+2015 HORIZONTAL BAR ―
    res8 = unit_normalizer.normalize("\u20150.5%")
    assert res8.numeric_value == -0.5


def test_directional_wording_negative_sign(unit_normalizer):
    """Verify deterministic negative direction terms produce negative normalized values."""
    # 15.5 percent decline -> -15.5
    res1 = unit_normalizer.normalize("15.5 percent decline")
    assert res1.numeric_value == -15.5
    assert res1.canonical_unit == "percent"

    # 15.8 percent drop -> -15.8
    res2 = unit_normalizer.normalize("15.8 percent drop")
    assert res2.numeric_value == -15.8
    assert res2.canonical_unit == "percent"

    # 2 percent decrease -> -2
    res3 = unit_normalizer.normalize("2 percent decrease")
    assert res3.numeric_value == -2.0
    assert res3.canonical_unit == "percent"

    # 4 percent reduction -> -4
    res4 = unit_normalizer.normalize("4 percent reduction")
    assert res4.numeric_value == -4.0
    assert res4.canonical_unit == "percent"


def test_positive_values_unchanged(unit_normalizer):
    """Verify positive numbers and positive directional wording remain positive."""
    res1 = unit_normalizer.normalize("5.3%")
    assert res1.numeric_value == 5.3
    assert res1.canonical_unit == "percent"

    res2 = unit_normalizer.normalize("15.5 percent increase")
    assert res2.numeric_value == 15.5
    assert res2.canonical_unit == "percent"

    res3 = unit_normalizer.normalize("12.4 percent growth")
    assert res3.numeric_value == 12.4
    assert res3.canonical_unit == "percent"


def test_fact_record_negative_value_never_inverted(fact_normalizer):
    """Verify FactRecord with negative value_numeric preserves negative sign through FactNormalizer."""
    fact = FactRecord(
        fact_id="test-fact-neg-1",
        entity="China",
        metric="Real GDP prepandemic trend gap",
        value_raw="—5.3%",
        value_numeric=-5.3,
        unit="percent",
        time_period=TimePeriod(label="2024"),
        epistemic_status=EpistemicStatus.REPORTED,
        provenance=Provenance(
            document_id="doc1",
            page_number=23,
            supporting_text="China gap was —5.3%",
        ),
    )
    normalized = fact_normalizer.normalize(fact)
    assert normalized.normalized_value.numeric_value == -5.3
    assert normalized.normalized_value.canonical_unit == "percent"
    assert normalized.fact.value_raw == "—5.3%"


def test_qualifiers_preservation(unit_normalizer):
    """Verify qualifiers such as >, <, approximately, around, more than are preserved."""
    assert unit_normalizer.normalize(">500").value_qualifier == ">"
    assert unit_normalizer.normalize("<200").value_qualifier == "<"
    assert unit_normalizer.normalize("approximately 15%").value_qualifier == "approximately"
    assert unit_normalizer.normalize("around 25 million").value_qualifier == "around"
    assert unit_normalizer.normalize("more than 100").value_qualifier == ">"
    assert unit_normalizer.normalize("less than 50").value_qualifier == "<"


# ============================================================================
# 2. FIX 2 TESTS: BOUNDED RETRY FOR HTTP 503 & ERROR HANDLING
# ============================================================================

def test_503_succeeds_on_initial_request():
    """1. 503 succeeds on initial request -> 1 request."""
    extractor = FactExtractor(api_key="mock-key")
    mock_resp_200 = MagicMock(spec=httpx.Response)
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"facts": []}'}]}}]
    }

    with patch("httpx.Client.post", return_value=mock_resp_200) as mock_post:
        res = extractor._call_gemini("test prompt")
        assert res == {"facts": []}
        assert mock_post.call_count == 1


def test_503_once_then_success():
    """2. 503 once, then success -> 2 requests."""
    extractor = FactExtractor(api_key="mock-key")
    mock_503 = MagicMock(spec=httpx.Response)
    mock_503.status_code = 503

    mock_200 = MagicMock(spec=httpx.Response)
    mock_200.status_code = 200
    mock_200.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"facts": []}'}]}}]
    }

    with patch("httpx.Client.post", side_effect=[mock_503, mock_200]) as mock_post:
        with patch("time.sleep") as mock_sleep:
            res = extractor._call_gemini("test prompt")
            assert res == {"facts": []}
            assert mock_post.call_count == 2
            mock_sleep.assert_called_once_with(1.0)


def test_503_twice_then_success():
    """3. 503 twice, then success -> 3 requests with exponential backoff."""
    extractor = FactExtractor(api_key="mock-key")
    mock_503_1 = MagicMock(spec=httpx.Response)
    mock_503_1.status_code = 503
    mock_503_2 = MagicMock(spec=httpx.Response)
    mock_503_2.status_code = 503

    mock_200 = MagicMock(spec=httpx.Response)
    mock_200.status_code = 200
    mock_200.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"facts": []}'}]}}]
    }

    with patch("httpx.Client.post", side_effect=[mock_503_1, mock_503_2, mock_200]) as mock_post:
        with patch("time.sleep") as mock_sleep:
            res = extractor._call_gemini("test prompt")
            assert res == {"facts": []}
            assert mock_post.call_count == 3
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1.0)
            mock_sleep.assert_any_call(2.0)


def test_503_three_times_exhausts_and_raises():
    """4. 503 three times -> ExtractionError -> exactly 3 attempts."""
    extractor = FactExtractor(api_key="mock-key")
    mock_503 = MagicMock(spec=httpx.Response)
    mock_503.status_code = 503
    mock_503.text = "Service Unavailable"

    with patch("httpx.Client.post", return_value=mock_503) as mock_post:
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ExtractionError) as exc_info:
                extractor._call_gemini("test prompt")
            assert "HTTP 503" in str(exc_info.value)
            assert "3 attempts" in str(exc_info.value)
            assert mock_post.call_count == 3
            # Backoff before attempt 1 (1.0s) and attempt 2 (2.0s), no sleep after final attempt 2
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1.0)
            mock_sleep.assert_any_call(2.0)


def test_429_immediately_raises_no_retry():
    """5. 429 -> ExtractionQuotaError -> exactly 1 attempt -> no retry."""
    extractor = FactExtractor(api_key="mock-key")
    mock_429 = MagicMock(spec=httpx.Response)
    mock_429.status_code = 429
    mock_429.text = "Quota exceeded"

    with patch("httpx.Client.post", return_value=mock_429) as mock_post:
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ExtractionQuotaError):
                extractor._call_gemini("test prompt")
            assert mock_post.call_count == 1
            mock_sleep.assert_not_called()


def test_400_bad_request_raises_no_retry():
    """6. 400 -> ExtractionError -> exactly 1 attempt."""
    extractor = FactExtractor(api_key="mock-key")
    mock_400 = MagicMock(spec=httpx.Response)
    mock_400.status_code = 400
    mock_400.text = "Bad Request"

    with patch("httpx.Client.post", return_value=mock_400) as mock_post:
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ExtractionError) as exc_info:
                extractor._call_gemini("test prompt")
            assert "HTTP 400" in str(exc_info.value)
            assert mock_post.call_count == 1
            mock_sleep.assert_not_called()


def test_404_not_found_raises_no_retry():
    """7. 404 -> ExtractionError -> exactly 1 attempt."""
    extractor = FactExtractor(api_key="mock-key")
    mock_404 = MagicMock(spec=httpx.Response)
    mock_404.status_code = 404
    mock_404.text = "Model Not Found"

    with patch("httpx.Client.post", return_value=mock_404) as mock_post:
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ExtractionError) as exc_info:
                extractor._call_gemini("test prompt")
            assert "HTTP 404" in str(exc_info.value)
            assert mock_post.call_count == 1
            mock_sleep.assert_not_called()
