"""
Phase 20 Tests: Production Batch Size Optimization (Batch Size 5).

Verifies:
1. Production default batch size is 5.
2. 184 eligible pages produce ceil(184 / 5) = 37 planned batches.
3. The final batch correctly handles the remainder (36 batches of 5 + 1 batch of 4).
4. Page delimiters format correctly in multi-page prompts.
5. Page attribution maps facts strictly to their source pages.
6. EvidenceVerifier validates supporting text against the authoritative original page.
7. Cross-page contamination is detected and rejected.
8. HTTP 429 quota exhaustion stops immediately with no retries.
9. HTTP 503 retry maintains 3-attempt bounded backoff.
10. End-to-end downstream normalization, matching, and relationship generation work seamlessly.
"""

import json
from unittest.mock import MagicMock, patch
import httpx
import pytest

from backend.extraction.fact_extractor import (
    FactExtractor,
    ExtractionError,
    ExtractionQuotaError,
)
from backend.extraction.evidence import EvidenceVerifier, create_evidence
from backend.models.document import PageText, ParsedDocument
from backend.models.fact import (
    EpistemicStatus,
    FactRecord,
    Provenance,
    TimePeriod,
)
from backend.quota_experiment.planner import QuotaPlanner
from backend.quota_experiment.status import ExtractionStatus
from backend.normalization.normalizer import FactNormalizer
from backend.reasoning.matcher import CandidateMatcher
from backend.reasoning.comparability import ComparabilityGate
from backend.reasoning.relationships import RelationshipEngine


def test_01_production_default_batch_size_is_5(monkeypatch):
    """1. Production default batch size is 5 when unconfigured or default."""
    monkeypatch.delenv("FACTLINE_EXTRACTION_BATCH_SIZE", raising=False)
    extractor = FactExtractor(api_key="mock-key")
    assert extractor.batch_size == 5
    assert extractor.planner.default_batch_size == 5


def test_02_184_pages_produces_37_planned_batches():
    """2. 184 eligible pages produce ceil(184 / 5) = 37 planned batches."""
    planner = QuotaPlanner(default_batch_size=5)
    eligible_pages = list(range(1, 185))  # 184 pages
    plan = planner.calculate_plan(eligible_pages=eligible_pages, batch_size=5)
    assert len(plan.batches) == 37
    assert plan.estimated_requests == 37
    assert plan.eligible_pages_count == 184


def test_03_final_batch_remainder_handling():
    """3. The final batch correctly handles the remainder: 36 batches of 5 + 1 batch of 4."""
    planner = QuotaPlanner(default_batch_size=5)
    eligible_pages = list(range(1, 185))  # 184 pages
    plan = planner.calculate_plan(eligible_pages=eligible_pages, batch_size=5)
    
    # First 36 batches have size 5
    for b in plan.batches[:36]:
        assert len(b) == 5
    
    # 37th batch has remaining 4 pages
    assert len(plan.batches[36]) == 4
    assert plan.batches[36] == [181, 182, 183, 184]
    
    # Total pages across all batches
    all_pages = [p for b in plan.batches for p in b]
    assert all_pages == eligible_pages


def test_04_page_delimiters_formatting():
    """4. Page delimiters format cleanly in 5-page batch prompt."""
    extractor = FactExtractor(api_key="mock-key", batch_size=5)
    prompt_items = [
        (1, "Page 1 content"),
        (2, "Page 2 content"),
        (3, "Page 3 content"),
        (4, "Page 4 content"),
        (5, "Page 5 content"),
    ]
    prompt = extractor.format_batch_prompt(prompt_items, "test.pdf")
    
    for p_num in range(1, 6):
        assert f"===== DOCUMENT PAGE {p_num} START =====" in prompt
        assert f"Page {p_num} content" in prompt
        assert f"===== DOCUMENT PAGE {p_num} END =====" in prompt


def test_05_page_attribution_in_batch():
    """5. Page attribution correctly maps facts across a 5-page batch."""
    pages = [
        PageText(page_number=i, text=f"Company Page {i} revenue is ${i*10}M in FY24.", char_count=40, has_text=True)
        for i in range(1, 6)
    ]
    
    mock_facts = [
        {
            "page_number": i,
            "entity": f"Company {i}",
            "metric": "Revenue",
            "value_raw": f"${i*10}M",
            "value_numeric": float(i * 10),
            "unit": "million USD",
            "time_period": {"label": "FY24"},
            "supporting_text": f"Company Page {i} revenue is ${i*10}M in FY24.",
        }
        for i in range(1, 6)
    ]
    
    extractor = FactExtractor(api_key="mock-key", batch_size=5, page_filter_enabled=False, context_selector_enabled=False)
    extractor._llm_caller = lambda prompt: {"facts": mock_facts}
    
    facts, rejected = extractor.extract_batch(pages, document_id="doc-test", document_name="test.pdf")
    assert len(facts) == 5
    assert rejected == 0
    for idx, f in enumerate(facts):
        assert f.provenance.page_number == idx + 1
        assert f.entity == f"Company {idx + 1}"


def test_06_evidence_verifier_against_original_page():
    """6. EvidenceVerifier validates supporting text against the authoritative original page."""
    page = PageText(
        page_number=4,
        text="India real GDP projected at 6.5% in FY25.",
        char_count=41,
        has_text=True,
    )
    valid_prov = create_evidence("doc1", 4, "India real GDP projected at 6.5%")
    assert EvidenceVerifier.verify_provenance(valid_prov, page) is True
    
    invalid_prov = create_evidence("doc1", 4, "China real GDP projected at 4.5%")
    assert EvidenceVerifier.verify_provenance(invalid_prov, page) is False


def test_07_cross_page_contamination_rejected():
    """7. Facts claiming evidence from Page X but actually on Page Y are rejected."""
    pages = [
        PageText(page_number=1, text="Alpha Corp revenue was $100M.", char_count=30, has_text=True),
        PageText(page_number=2, text="Beta Corp revenue was $200M.", char_count=30, has_text=True),
    ]
    
    # Hallucinated attribution: claims Page 1 but supporting text is on Page 2
    hallucinated_candidate = {
        "page_number": 1,
        "entity": "Beta Corp",
        "metric": "Revenue",
        "value_raw": "$200M",
        "value_numeric": 200.0,
        "unit": "USD",
        "time_period": {"label": "FY24"},
        "supporting_text": "Beta Corp revenue was $200M.",  # NOT on page 1
    }
    
    extractor = FactExtractor(api_key="mock-key", batch_size=5, page_filter_enabled=False, context_selector_enabled=False)
    extractor._llm_caller = lambda prompt: {"facts": [hallucinated_candidate]}
    
    facts, rejected = extractor.extract_batch(pages, document_id="doc1", document_name="test.pdf")
    assert len(facts) == 0
    assert rejected == 1


def test_08_429_quota_exhaustion_halts_immediately():
    """8. HTTP 429 stops immediately without retrying and preserves partial extraction status."""
    extractor = FactExtractor(api_key="mock-key", batch_size=5)
    mock_429 = MagicMock(spec=httpx.Response)
    mock_429.status_code = 429
    mock_429.text = "Quota exhausted"
    
    with patch("httpx.Client.post", return_value=mock_429) as mock_post:
        with pytest.raises(ExtractionQuotaError):
            extractor._call_gemini("test prompt")
        assert mock_post.call_count == 1


def test_09_503_bounded_exponential_retry():
    """9. HTTP 503 retries up to 3 attempts with exponential backoff and fails cleanly."""
    extractor = FactExtractor(api_key="mock-key", batch_size=5)
    mock_503 = MagicMock(spec=httpx.Response)
    mock_503.status_code = 503
    mock_503.text = "Service Unavailable"
    
    with patch("httpx.Client.post", return_value=mock_503) as mock_post:
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ExtractionError) as exc_info:
                extractor._call_gemini("test prompt")
            assert "HTTP 503" in str(exc_info.value)
            assert mock_post.call_count == 3
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1.0)
            mock_sleep.assert_any_call(2.0)


def test_10_end_to_end_reasoning_pipeline_with_batch_size_5():
    """10. Full document extraction with batch_size=5 flows through normalization, matching, and comparability."""
    pages = [
        PageText(page_number=1, text="India GDP Growth in 2024 was 7.0%.", char_count=34, has_text=True),
        PageText(page_number=2, text="India GDP Growth in 2024 was 6.8%.", char_count=34, has_text=True),
        PageText(page_number=3, text="India GDP Growth in 2024 was 7.2%.", char_count=34, has_text=True),
        PageText(page_number=4, text="China GDP Growth in 2024 was 5.0%.", char_count=34, has_text=True),
        PageText(page_number=5, text="China GDP Growth in 2024 was 5.2%.", char_count=34, has_text=True),
    ]
    doc = ParsedDocument(document_id="doc-flow", document_name="flow.pdf", total_pages=5, pages=pages)
    
    mock_facts = [
        {
            "page_number": 1,
            "entity": "India",
            "metric": "GDP Growth",
            "value_raw": "7.0%",
            "value_numeric": 7.0,
            "unit": "percent",
            "time_period": {"label": "2024"},
            "supporting_text": "India GDP Growth in 2024 was 7.0%.",
        },
        {
            "page_number": 2,
            "entity": "India",
            "metric": "GDP Growth",
            "value_raw": "6.8%",
            "value_numeric": 6.8,
            "unit": "percent",
            "time_period": {"label": "2024"},
            "supporting_text": "India GDP Growth in 2024 was 6.8%.",
        },
    ]
    
    extractor = FactExtractor(api_key="mock-key", batch_size=5, page_filter_enabled=False, context_selector_enabled=False)
    extractor._llm_caller = lambda prompt: {"facts": mock_facts}
    
    result = extractor.extract_document_result(doc)
    assert result.status == ExtractionStatus.COMPLETE
    assert result.requests_used == 1
    assert len(result.facts) == 2
    
    # Downstream normalization
    normalizer = FactNormalizer()
    normalized_facts = normalizer.normalize_batch(result.facts)
    assert len(normalized_facts) == 2
    
    # Downstream candidate matching
    matcher = CandidateMatcher()
    candidate_tuples = matcher.find_candidates(normalized_facts)
    assert len(candidate_tuples) == 1
    
    # Downstream comparability & relationship
    gate = ComparabilityGate()
    rel_engine = RelationshipEngine()
    comp_res = gate.evaluate(candidate_tuples[0][0], candidate_tuples[0][1])
    cand_pair = matcher.match_pair(candidate_tuples[0][0], candidate_tuples[0][1])
    rel_res = rel_engine.determine_relationship(candidate_tuples[0][0], candidate_tuples[0][1], comp_res, cand_pair)
    
    assert rel_res.relationship_type.value in ("CONTRADICTS", "CORROBORATES", "EXTENDS", "SUPERSEDES")
