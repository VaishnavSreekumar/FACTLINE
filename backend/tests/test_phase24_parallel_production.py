"""
Phase 24: Production Integration Regression Tests for Bounded 2-Worker Gemini Extraction.

Covers:
1. workers=2 is bounded.
2. No more than 2 Gemini requests are simultaneously active.
3. Batch order is deterministic after concurrent completion.
4. Page attribution is preserved.
5. EvidenceVerifier validates against original PDF pages.
6. Cross-page contamination is rejected.
7. Worker metadata cannot leak into FactRecords.
8. One worker failure preserves the successful worker's facts.
9. 429 immediately stops further scheduling.
10. 503 retains the existing 3-attempt retry policy.
11. 400/401/403/404 remain non-retryable.
12. 184 eligible pages with batch size 5 produces exactly 37 planned batches.
13. Downstream normalization remains unchanged.
14. Matching remains unchanged.
15. Comparability remains unchanged.
16. Relationship generation remains unchanged.
17. SQLite persistence remains unchanged.
18. COMPLETE is impossible when unprocessed pages > 0.
19. Existing Phase 18A status aggregation remains correct.
"""

import os
import time
import threading
import pytest
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from backend.extraction.fact_extractor import (
    FactExtractor,
    ExtractionError,
    ExtractionQuotaError,
)
from backend.models.document import ParsedDocument, PageText
from backend.models.fact import FactRecord, EpistemicStatus
from backend.quota_experiment.status import ExtractionStatus
from backend.quota_experiment.planner import QuotaPlanner
from backend.normalization.normalizer import FactNormalizer
from backend.reasoning.matcher import CandidateMatcher
from backend.reasoning.comparability import ComparabilityGate
from backend.reasoning.relationships import RelationshipEngine
from backend.db.database import DatabaseRepository


def _create_mock_document(num_pages: int = 10, doc_id: str = "doc-p24") -> ParsedDocument:
    """Helper to create a mock multi-page document with realistic quantitative content."""
    pages = []
    for p_num in range(1, num_pages + 1):
        text = (
            f"Page {p_num} Global GDP growth in 2024 was {2.0 + p_num * 0.1:.1f} percent. "
            f"Inflation reached {3.0 + p_num * 0.1:.1f} percent for Country {p_num}."
        )
        pages.append(
            PageText(
                page_number=p_num,
                text=text,
                char_count=len(text),
                has_text=True,
            )
        )
    return ParsedDocument(
        document_id=doc_id,
        document_name="TestReport.pdf",
        total_pages=num_pages,
        pages=pages,
    )


# 1. workers=2 is bounded (and clamped to max 2)
def test_max_workers_clamped_strictly():
    # Constructor clamped
    ext1 = FactExtractor(api_key="mock", max_workers=5)
    assert ext1.max_workers == 2

    ext0 = FactExtractor(api_key="mock", max_workers=0)
    assert ext0.max_workers == 1

    ext2 = FactExtractor(api_key="mock", max_workers=2)
    assert ext2.max_workers == 2

    # Environment variable clamped
    with patch.dict(os.environ, {"FACTLINE_EXTRACTION_MAX_WORKERS": "10"}):
        ext_env = FactExtractor(api_key="mock")
        assert ext_env.max_workers == 2

    with patch.dict(os.environ, {"FACTLINE_EXTRACTION_MAX_WORKERS": "1"}):
        ext_env1 = FactExtractor(api_key="mock")
        assert ext_env1.max_workers == 1


# 2. No more than 2 Gemini requests are simultaneously active
def test_concurrency_bounded_active_requests():
    doc = _create_mock_document(num_pages=20)  # with batch_size=5 -> 4 batches
    active_threads = 0
    max_active_observed = 0
    lock = threading.Lock()

    def mock_llm(prompt: str) -> Dict[str, Any]:
        nonlocal active_threads, max_active_observed
        with lock:
            active_threads += 1
            if active_threads > max_active_observed:
                max_active_observed = active_threads
        time.sleep(0.05)  # simulate brief network inference
        with lock:
            active_threads -= 1
        return {
            "facts": [
                {
                    "page_number": 1,
                    "entity": "Global",
                    "metric": "GDP Growth",
                    "value_raw": "2.1 percent",
                    "value_numeric": 2.1,
                    "unit": "%",
                    "time_period": {"label": "2024"},
                    "supporting_text": "Global GDP growth in 2024 was 2.1 percent.",
                }
            ]
        }

    extractor = FactExtractor(
        api_key="mock",
        batch_size=5,
        max_workers=2,
        page_filter_enabled=False,
        context_selector_enabled=False,
        llm_caller=mock_llm,
    )
    result = extractor.extract_document_result(doc)
    assert max_active_observed <= 2
    assert result.status == ExtractionStatus.COMPLETE
    assert result.max_workers == 2


# 3. Batch order is deterministic after concurrent completion
def test_deterministic_batch_order_preservation():
    doc = _create_mock_document(num_pages=15)  # 3 batches: [1..5], [6..10], [11..15]

    def mock_llm(prompt: str) -> Dict[str, Any]:
        # Introduce artificial delay on Batch 1 so Batch 2/3 complete first
        if "DOCUMENT PAGE 1 START" in prompt:
            time.sleep(0.08)
            p = 1
        elif "DOCUMENT PAGE 6 START" in prompt:
            time.sleep(0.01)
            p = 6
        else:
            time.sleep(0.01)
            p = 11
        return {
            "facts": [
                {
                    "page_number": p,
                    "entity": f"Entity-{p}",
                    "metric": "GDP Growth",
                    "value_raw": f"{2.0 + p * 0.1:.1f} percent",
                    "value_numeric": 2.0 + p * 0.1,
                    "unit": "%",
                    "time_period": {"label": "2024"},
                    "supporting_text": f"Global GDP growth in 2024 was {2.0 + p * 0.1:.1f} percent.",
                }
            ]
        }

    extractor = FactExtractor(
        api_key="mock",
        batch_size=5,
        max_workers=2,
        page_filter_enabled=False,
        context_selector_enabled=False,
        llm_caller=mock_llm,
    )
    result = extractor.extract_document_result(doc)
    assert result.status == ExtractionStatus.COMPLETE
    # Verify deterministic ordering: page 1 first, page 6 second, page 11 third
    assert [f.provenance.page_number for f in result.facts] == [1, 6, 11]
    assert [f.entity for f in result.facts] == ["Entity-1", "Entity-6", "Entity-11"]


# 4 & 5. Page attribution is preserved & EvidenceVerifier validates against original PDF pages
def test_page_attribution_and_grounding_validation():
    doc = _create_mock_document(num_pages=5)

    def mock_llm(prompt: str) -> Dict[str, Any]:
        return {
            "facts": [
                {
                    "page_number": 3,
                    "entity": "Country 3",
                    "metric": "Inflation",
                    "value_raw": "3.3 percent",
                    "value_numeric": 3.3,
                    "unit": "%",
                    "time_period": {"label": "2024"},
                    "supporting_text": "Inflation reached 3.3 percent for Country 3.",
                }
            ]
        }

    extractor = FactExtractor(
        api_key="mock",
        batch_size=5,
        max_workers=2,
        page_filter_enabled=False,
        context_selector_enabled=False,
        llm_caller=mock_llm,
    )
    result = extractor.extract_document_result(doc)
    assert len(result.facts) == 1
    f = result.facts[0]
    assert f.provenance.page_number == 3
    assert f.provenance.supporting_text == "Inflation reached 3.3 percent for Country 3."


# 6. Cross-page contamination is rejected
def test_cross_page_contamination_rejected():
    doc = _create_mock_document(num_pages=5)

    def mock_llm(prompt: str) -> Dict[str, Any]:
        return {
            "facts": [
                {
                    # Text from page 3 claimed on page 1 -> Contamination!
                    "page_number": 1,
                    "entity": "Country 3",
                    "metric": "Inflation",
                    "value_raw": "3.3 percent",
                    "value_numeric": 3.3,
                    "unit": "%",
                    "time_period": {"label": "2024"},
                    "supporting_text": "Inflation reached 3.3 percent for Country 3.",
                }
            ]
        }

    extractor = FactExtractor(
        api_key="mock",
        batch_size=5,
        max_workers=2,
        page_filter_enabled=False,
        context_selector_enabled=False,
        llm_caller=mock_llm,
    )
    result = extractor.extract_document_result(doc)
    assert len(result.facts) == 0
    assert result.facts_rejected_grounding == 1


# 7. Worker metadata cannot leak into FactRecords
def test_worker_metadata_cannot_leak():
    doc = _create_mock_document(num_pages=5)

    def mock_llm(prompt: str) -> Dict[str, Any]:
        return {
            "facts": [
                {
                    "page_number": 2,
                    "entity": "Global",
                    "metric": "GDP Growth",
                    "value_raw": "2.2 percent",
                    "supporting_text": "Global GDP growth in 2024 was 2.2 percent.",
                    "time_period": {"label": "2024"},
                }
            ]
        }

    extractor = FactExtractor(
        api_key="mock",
        batch_size=5,
        max_workers=2,
        page_filter_enabled=False,
        context_selector_enabled=False,
        llm_caller=mock_llm,
    )
    result = extractor.extract_document_result(doc)
    f_dict = result.facts[0].model_dump()
    for key in ["thread_id", "worker", "worker_id", "execution_mode"]:
        assert key not in f_dict


# 8. One worker failure preserves the successful worker's facts
def test_worker_failure_preserves_successful_facts():
    doc = _create_mock_document(num_pages=10)  # Batch 0: [1..5], Batch 1: [6..10]

    def mock_llm(prompt: str) -> Dict[str, Any]:
        if "DOCUMENT PAGE 1 START" in prompt:
            return {
                "facts": [
                    {
                        "page_number": 1,
                        "entity": "Global",
                        "metric": "GDP Growth",
                        "value_raw": "2.1 percent",
                        "supporting_text": "Global GDP growth in 2024 was 2.1 percent.",
                        "time_period": {"label": "2024"},
                    }
                ]
            }
        else:
            raise RuntimeError("Batch 1 unexpected syntax failure")

    extractor = FactExtractor(
        api_key="mock",
        batch_size=5,
        max_workers=2,
        page_filter_enabled=False,
        context_selector_enabled=False,
        llm_caller=mock_llm,
    )
    result = extractor.extract_document_result(doc)
    assert result.status == ExtractionStatus.FAILED
    assert len(result.facts) == 1
    assert result.facts[0].entity == "Global"
    assert result.processed_pages == [1, 2, 3, 4, 5]
    assert result.unprocessed_pages == [6, 7, 8, 9, 10]


# 9. 429 immediately stops further scheduling
def test_429_immediately_stops_scheduling():
    doc = _create_mock_document(num_pages=20)  # 4 batches: [1..5], [6..10], [11..15], [16..20]
    batches_attempted = 0

    def mock_llm(prompt: str) -> Dict[str, Any]:
        nonlocal batches_attempted
        batches_attempted += 1
        if "DOCUMENT PAGE 1 START" in prompt:
            return {
                "facts": [
                    {
                        "page_number": 1,
                        "entity": "Global",
                        "metric": "GDP Growth",
                        "value_raw": "2.1 percent",
                        "supporting_text": "Global GDP growth in 2024 was 2.1 percent.",
                        "time_period": {"label": "2024"},
                    }
                ]
            }
        elif "DOCUMENT PAGE 6 START" in prompt:
            raise ExtractionQuotaError("HTTP 429 Quota Exhausted")
        else:
            return {"facts": []}

    extractor = FactExtractor(
        api_key="mock",
        batch_size=5,
        max_workers=2,
        page_filter_enabled=False,
        context_selector_enabled=False,
        llm_caller=mock_llm,
    )
    result = extractor.extract_document_result(doc)
    assert result.status == ExtractionStatus.PARTIAL_QUOTA
    assert len(result.facts) >= 1
    # 429 stopped scheduling further batches (batch 4 pages [16..20] were never scheduled)
    assert 16 not in result.processed_pages
    assert batches_attempted < 4


# 10. 503 retains existing 3-attempt retry policy in _call_gemini
def test_503_retry_behavior():
    extractor = FactExtractor(api_key="mock-key", batch_size=5)
    mock_resp_503 = MagicMock()
    mock_resp_503.status_code = 503

    with patch("httpx.Client.post", return_value=mock_resp_503) as mock_post:
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ExtractionError) as exc:
                extractor._call_gemini("test prompt")
            assert "HTTP 503" in str(exc.value)
            assert mock_post.call_count == 3
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1.0)
            mock_sleep.assert_any_call(2.0)


# 11. 400/401/403/404 remain non-retryable
def test_non_retryable_client_errors():
    extractor = FactExtractor(api_key="mock-key", batch_size=5)
    for code in [400, 401, 403, 404]:
        mock_resp = MagicMock()
        mock_resp.status_code = code
        mock_resp.text = f"Error {code}"

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            with pytest.raises(ExtractionError) as exc:
                extractor._call_gemini("test prompt")
            assert f"HTTP {code}" in str(exc.value)
            assert mock_post.call_count == 1  # No retry


# 12. 184 eligible pages with batch size 5 produces exactly 37 planned batches
def test_184_pages_quota_planner_37_batches():
    planner = QuotaPlanner(default_batch_size=5)
    eligible_pages = list(range(1, 185))  # 184 pages
    plan = planner.calculate_plan(eligible_pages=eligible_pages, batch_size=5)
    assert len(plan.batches) == 37
    assert plan.estimated_requests == 37
    assert sum(len(b) for b in plan.batches) == 184


# 13, 14, 15, 16, 17. Downstream Reasoning & Persistence Unchanged
def test_downstream_reasoning_and_persistence_pipeline(tmp_path):
    doc = _create_mock_document(num_pages=10)

    def mock_llm(prompt: str) -> Dict[str, Any]:
        if "DOCUMENT PAGE 1 START" in prompt:
            p = 1
        else:
            p = 6
        return {
            "facts": [
                {
                    "page_number": p,
                    "entity": "Global",
                    "metric": "GDP Growth",
                    "value_raw": f"{2.0 + p * 0.1:.1f} percent",
                    "value_numeric": 2.0 + p * 0.1,
                    "unit": "%",
                    "time_period": {"label": "2024"},
                    "supporting_text": f"Global GDP growth in 2024 was {2.0 + p * 0.1:.1f} percent.",
                }
            ]
        }

    extractor = FactExtractor(
        api_key="mock",
        batch_size=5,
        max_workers=2,
        page_filter_enabled=False,
        context_selector_enabled=False,
        llm_caller=mock_llm,
    )
    result = extractor.extract_document_result(doc)
    assert result.status == ExtractionStatus.COMPLETE
    assert len(result.facts) == 2

    # Normalization
    normalizer = FactNormalizer()
    normalized_facts = normalizer.normalize_batch(result.facts)
    assert len(normalized_facts) == 2
    assert normalized_facts[0].canonical_metric == "gdp growth"

    # Candidate matching
    matcher = CandidateMatcher()
    candidate_tuples = matcher.find_candidates(normalized_facts)
    assert len(candidate_tuples) == 1

    # Comparability & Relationships
    gate = ComparabilityGate()
    rel_engine = RelationshipEngine()
    relationships = []
    for f_a, f_b in candidate_tuples:
        cand_pair = matcher.match_pair(f_a, f_b)
        comp = gate.evaluate(f_a, f_b)
        rel = rel_engine.determine_relationship(
            fact_a=f_a,
            fact_b=f_b,
            comparability=comp,
            candidate_pair=cand_pair,
        )
        relationships.append(rel)

    assert len(relationships) == 1

    # SQLite Persistence
    db_file = str(tmp_path / "test_p24.db")
    repo = DatabaseRepository(db_path=db_file)
    repo.save_analysis(
        analysis_id="analysis-p24",
        documents=[doc],
        normalized_facts=normalized_facts,
        relationships=relationships,
        candidate_pair_count=len(candidate_tuples),
        extraction_status={"status": "COMPLETE", "documents": []},
    )
    loaded = repo.get_analysis("analysis-p24")
    assert loaded is not None
    assert len(loaded["facts"]) == 2
    assert len(loaded["relationships"]) == 1


# 18. COMPLETE is impossible when unprocessed pages > 0
def test_complete_status_impossible_with_unprocessed_pages():
    doc = _create_mock_document(num_pages=10)

    # Simulate extractor where max_pages=1 (budget limited to 1 request = 5 pages)
    extractor = FactExtractor(
        api_key="mock",
        max_pages=1,
        batch_size=5,
        max_workers=2,
        page_filter_enabled=False,
        context_selector_enabled=False,
        llm_caller=lambda p: {"facts": []},
    )
    result = extractor.extract_document_result(doc)
    assert result.status != ExtractionStatus.COMPLETE
    assert result.status == ExtractionStatus.PARTIAL_QUOTA
    assert len(result.unprocessed_pages) == 5


# 19. Existing Phase 18A status aggregation remains correct
def test_phase18a_status_aggregation_with_workers():
    from backend.services.analysis import AnalysisService

    doc = _create_mock_document(num_pages=5)
    service = AnalysisService()
    # Mock extractor
    mock_extractor = MagicMock()
    mock_extractor.extract_from_document.return_value = []
    mock_extractor.last_extraction_result = ExtractionStatus.COMPLETE

    mock_res = MagicMock()
    mock_res.document_id = doc.document_id
    mock_res.status = ExtractionStatus.COMPLETE
    mock_res.total_eligible_pages = 5
    mock_res.processed_pages = [1, 2, 3, 4, 5]
    mock_res.unprocessed_pages = []
    mock_res.requests_used = 1
    mock_res.requests_available = None
    mock_res.batch_size = 5
    mock_res.max_workers = 2
    mock_res.facts = []
    mock_res.facts_rejected_grounding = 0
    mock_res.user_message = "All 5 pages processed"
    mock_res.error_message = None

    mock_extractor.last_extraction_result = mock_res
    service.extractor = mock_extractor
    service.parser.parse_bytes = MagicMock(return_value=doc)

    summary = service.analyze_documents([("TestReport.pdf", b"%PDF-1.4 dummy")])
    assert summary["summary"]["extraction_status"]["status"] == "COMPLETE"
    assert summary["summary"]["documents_processed"] == 1

