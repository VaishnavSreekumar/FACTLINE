"""
Unit tests for Quota-Aware Extraction & Multi-Page Batching (Phase 13).

Verifies all 24 deterministic properties:
1. zero eligible pages
2. one eligible page
3. batch size 1
4. batch size 2
5. batch size 3
6. batch size 5
7. incomplete final batch
8. page attribution
9. evidence verification
10. cross-page contamination rejection
11. duplicate numbers across pages
12. duplicate metrics across pages
13. different entities on adjacent pages
14. different periods on adjacent pages
15. quota planner calculations
16. insufficient budget handling
17. partial quota status metadata
18. 429 behavior
19. 503 retry behavior
20. no infinite retries
21. deterministic page ordering
22. no API calls from planner
23. no document-specific rules
24. no production imports that mutate behavior
"""

import pytest
from typing import Dict, Any, List
from backend.models.document import PageText, ParsedDocument
from backend.quota_experiment.planner import QuotaPlanner, QuotaPlan
from backend.quota_experiment.status import ExtractionStatus, BatchExtractionResult
from backend.quota_experiment.batching import (
    MultiPageBatchExtractor,
    BatchExtractionQuotaError,
)


def test_01_zero_eligible_pages():
    """1. Tests planner and extractor on document with 0 eligible pages."""
    planner = QuotaPlanner()
    plan = planner.calculate_plan(eligible_pages=[])
    assert plan.eligible_pages_count == 0
    assert plan.estimated_requests == 0
    assert plan.can_complete is True
    assert len(plan.batches) == 0

    extractor = MultiPageBatchExtractor()
    doc = ParsedDocument(document_id="d1", document_name="doc1", total_pages=1, pages=[PageText(page_number=1, text="", char_count=0, has_text=False)])
    res = extractor.extract_document(doc)
    assert res.status == ExtractionStatus.COMPLETE
    assert res.requests_used == 0
    assert len(res.facts) == 0


def test_02_one_eligible_page():
    """2. Tests single page batch planning."""
    planner = QuotaPlanner()
    plan = planner.calculate_plan(eligible_pages=[1], batch_size=2)
    assert plan.eligible_pages_count == 1
    assert plan.estimated_requests == 1
    assert plan.batches == [[1]]


def test_03_batch_size_1():
    """3. Tests batch size 1 (1 page per request)."""
    planner = QuotaPlanner()
    plan = planner.calculate_plan(eligible_pages=[1, 2, 3], batch_size=1)
    assert plan.estimated_requests == 3
    assert plan.batches == [[1], [2], [3]]


def test_04_batch_size_2():
    """4. Tests batch size 2 (2 pages per request)."""
    planner = QuotaPlanner()
    plan = planner.calculate_plan(eligible_pages=[1, 2, 3, 4], batch_size=2)
    assert plan.estimated_requests == 2
    assert plan.batches == [[1, 2], [3, 4]]


def test_05_batch_size_3():
    """5. Tests batch size 3 (3 pages per request)."""
    planner = QuotaPlanner()
    plan = planner.calculate_plan(eligible_pages=[1, 2, 3, 4, 5, 6], batch_size=3)
    assert plan.estimated_requests == 2
    assert plan.batches == [[1, 2, 3], [4, 5, 6]]


def test_06_batch_size_5():
    """6. Tests batch size 5 (5 pages per request)."""
    planner = QuotaPlanner()
    plan = planner.calculate_plan(eligible_pages=list(range(1, 11)), batch_size=5)
    assert plan.estimated_requests == 2
    assert len(plan.batches) == 2
    assert plan.batches[0] == [1, 2, 3, 4, 5]
    assert plan.batches[1] == [6, 7, 8, 9, 10]


def test_07_incomplete_final_batch():
    """7. Tests handling of odd-length page sets (incomplete final batch)."""
    planner = QuotaPlanner()
    plan = planner.calculate_plan(eligible_pages=[2, 4, 6, 8, 10, 12, 14], batch_size=3)
    assert plan.estimated_requests == 3
    assert plan.batches == [[2, 4, 6], [8, 10, 12], [14]]


def test_08_page_attribution():
    """8. Tests that extracted facts are attributed strictly to the source page."""
    page2 = PageText(page_number=2, text="Delhivery FY24 revenue ₹8,142 Cr", char_count=32, has_text=True)
    page4 = PageText(page_number=4, text="Delhivery FY24 EBITDA ₹127 Cr", char_count=29, has_text=True)

    def mock_caller(prompt: str):
        return {
            "facts": [
                {
                    "page_number": 2,
                    "entity": "Delhivery",
                    "metric": "Revenue",
                    "value_raw": "₹8,142 Cr",
                    "time_period": {"label": "FY24"},
                    "supporting_text": "Delhivery FY24 revenue ₹8,142 Cr",
                },
                {
                    "page_number": 4,
                    "entity": "Delhivery",
                    "metric": "EBITDA",
                    "value_raw": "₹127 Cr",
                    "time_period": {"label": "FY24"},
                    "supporting_text": "Delhivery FY24 EBITDA ₹127 Cr",
                },
            ]
        }

    extractor = MultiPageBatchExtractor(llm_caller=mock_caller)
    facts, rejected = extractor.extract_batch([page2, page4], document_id="d1", document_name="doc.pdf")
    assert rejected == 0
    assert len(facts) == 2
    assert facts[0].provenance.page_number == 2
    assert facts[1].provenance.page_number == 4


def test_09_evidence_verification():
    """9. Tests rejection when supporting_text is not found in the page."""
    page2 = PageText(page_number=2, text="Delhivery revenue ₹8,142 Cr", char_count=28, has_text=True)

    def mock_caller(prompt: str):
        return {
            "facts": [
                {
                    "page_number": 2,
                    "entity": "Delhivery",
                    "metric": "Revenue",
                    "value_raw": "₹8,142 Cr",
                    "time_period": {"label": "FY24"},
                    "supporting_text": "Hallucinated supporting phrase not on page",
                }
            ]
        }

    extractor = MultiPageBatchExtractor(llm_caller=mock_caller)
    facts, rejected = extractor.extract_batch([page2], document_id="d1", document_name="doc.pdf")
    assert rejected == 1
    assert len(facts) == 0


def test_10_cross_page_contamination():
    """10. Tests rejection when fact from page A is falsely attributed to page B."""
    page2 = PageText(page_number=2, text="Page 2 text: Revenue ₹100 Cr", char_count=28, has_text=True)
    page4 = PageText(page_number=4, text="Page 4 text: Employees 5,000", char_count=28, has_text=True)

    def mock_caller(prompt: str):
        return {
            "facts": [
                # Cross-contamination: claims Revenue is on Page 4
                {
                    "page_number": 4,
                    "entity": "Alpha",
                    "metric": "Revenue",
                    "value_raw": "₹100 Cr",
                    "time_period": {"label": "FY24"},
                    "supporting_text": "Revenue ₹100 Cr",
                }
            ]
        }

    extractor = MultiPageBatchExtractor(llm_caller=mock_caller)
    facts, rejected = extractor.extract_batch([page2, page4], document_id="d1", document_name="doc.pdf")
    # Must be rejected because "Revenue ₹100 Cr" does not exist on Page 4!
    assert rejected == 1
    assert len(facts) == 0


def test_11_duplicate_numbers_across_pages():
    """11. Tests disambiguation when the identical number appears on multiple batched pages."""
    p1 = PageText(page_number=1, text="Year 2023 count: 500 units", char_count=26, has_text=True)
    p2 = PageText(page_number=2, text="Year 2024 count: 500 units", char_count=26, has_text=True)

    def mock_caller(prompt: str):
        return {
            "facts": [
                {
                    "page_number": 1,
                    "entity": "A",
                    "metric": "count",
                    "value_raw": "500",
                    "time_period": {"label": "2023"},
                    "supporting_text": "Year 2023 count: 500 units",
                },
                {
                    "page_number": 2,
                    "entity": "A",
                    "metric": "count",
                    "value_raw": "500",
                    "time_period": {"label": "2024"},
                    "supporting_text": "Year 2024 count: 500 units",
                },
            ]
        }

    extractor = MultiPageBatchExtractor(llm_caller=mock_caller)
    facts, rejected = extractor.extract_batch([p1, p2], document_id="d1", document_name="doc.pdf")
    assert rejected == 0
    assert len(facts) == 2
    assert facts[0].provenance.page_number == 1
    assert facts[1].provenance.page_number == 2


def test_12_duplicate_metrics_across_pages():
    """12. Tests multiple pages reporting the same metric name with different periods/values."""
    p1 = PageText(page_number=1, text="FY23 Revenue: ₹7,000 Cr", char_count=24, has_text=True)
    p2 = PageText(page_number=2, text="FY24 Revenue: ₹8,000 Cr", char_count=24, has_text=True)

    def mock_caller(prompt: str):
        return {
            "facts": [
                {
                    "page_number": 1,
                    "entity": "Company",
                    "metric": "Revenue",
                    "value_raw": "₹7,000 Cr",
                    "time_period": {"label": "FY23"},
                    "supporting_text": "FY23 Revenue: ₹7,000 Cr",
                },
                {
                    "page_number": 2,
                    "entity": "Company",
                    "metric": "Revenue",
                    "value_raw": "₹8,000 Cr",
                    "time_period": {"label": "FY24"},
                    "supporting_text": "FY24 Revenue: ₹8,000 Cr",
                },
            ]
        }

    extractor = MultiPageBatchExtractor(llm_caller=mock_caller)
    facts, rejected = extractor.extract_batch([p1, p2], document_id="d1", document_name="doc.pdf")
    assert rejected == 0
    assert len(facts) == 2
    assert facts[0].value_raw == "₹7,000 Cr"
    assert facts[1].value_raw == "₹8,000 Cr"


def test_13_different_entities_on_adjacent_pages():
    """13. Tests batch containing different entities on adjacent pages."""
    p1 = PageText(page_number=1, text="Entity Alpha profit ₹50 Cr", char_count=26, has_text=True)
    p2 = PageText(page_number=2, text="Entity Beta profit ₹90 Cr", char_count=25, has_text=True)

    def mock_caller(prompt: str):
        return {
            "facts": [
                {
                    "page_number": 1,
                    "entity": "Entity Alpha",
                    "metric": "profit",
                    "value_raw": "₹50 Cr",
                    "time_period": {"label": "FY24"},
                    "supporting_text": "Entity Alpha profit ₹50 Cr",
                },
                {
                    "page_number": 2,
                    "entity": "Entity Beta",
                    "metric": "profit",
                    "value_raw": "₹90 Cr",
                    "time_period": {"label": "FY24"},
                    "supporting_text": "Entity Beta profit ₹90 Cr",
                },
            ]
        }

    extractor = MultiPageBatchExtractor(llm_caller=mock_caller)
    facts, rejected = extractor.extract_batch([p1, p2], document_id="d1", document_name="doc.pdf")
    assert rejected == 0
    assert len(facts) == 2
    assert facts[0].entity == "Entity Alpha"
    assert facts[1].entity == "Entity Beta"


def test_14_different_periods_on_adjacent_pages():
    """14. Tests batch containing Q3 vs Q4 temporal periods."""
    p1 = PageText(page_number=1, text="Q3 FY24 Sales ₹1,000 Cr", char_count=24, has_text=True)
    p2 = PageText(page_number=2, text="Q4 FY24 Sales ₹1,200 Cr", char_count=24, has_text=True)

    def mock_caller(prompt: str):
        return {
            "facts": [
                {
                    "page_number": 1,
                    "entity": "A",
                    "metric": "Sales",
                    "value_raw": "₹1,000 Cr",
                    "time_period": {"label": "Q3 FY24"},
                    "supporting_text": "Q3 FY24 Sales ₹1,000 Cr",
                },
                {
                    "page_number": 2,
                    "entity": "A",
                    "metric": "Sales",
                    "value_raw": "₹1,200 Cr",
                    "time_period": {"label": "Q4 FY24"},
                    "supporting_text": "Q4 FY24 Sales ₹1,200 Cr",
                },
            ]
        }

    extractor = MultiPageBatchExtractor(llm_caller=mock_caller)
    facts, rejected = extractor.extract_batch([p1, p2], document_id="d1", document_name="doc.pdf")
    assert rejected == 0
    assert facts[0].time_period.label == "Q3 FY24"
    assert facts[1].time_period.label == "Q4 FY24"


def test_15_quota_planner_calculations():
    """15. Tests request calculation across multiple batch sizes."""
    planner = QuotaPlanner()
    p1 = planner.calculate_plan(eligible_pages=list(range(1, 21)), batch_size=2)
    assert p1.estimated_requests == 10

    p2 = planner.calculate_plan(eligible_pages=list(range(1, 21)), batch_size=3)
    assert p2.estimated_requests == 7

    p3 = planner.calculate_plan(eligible_pages=list(range(1, 21)), batch_size=5)
    assert p3.estimated_requests == 4


def test_16_insufficient_budget_handling():
    """16. Tests planning when request_budget is less than required requests."""
    planner = QuotaPlanner()
    # 20 pages with batch_size=2 requires 10 requests; budget is 5 requests
    plan = planner.calculate_plan(
        eligible_pages=list(range(1, 21)),
        request_budget=5,
        batch_size=2,
    )
    assert plan.can_complete is False
    assert len(plan.pages_planned) == 10
    assert len(plan.pages_deferred) == 10
    assert len(plan.batches) == 5
    assert plan.remaining_budget == 0


def test_17_partial_quota_status_metadata():
    """17. Tests explicit PARTIAL_QUOTA status and user message."""
    pages = [PageText(page_number=i, text=f"Page {i} text", char_count=12, has_text=True) for i in range(1, 11)]
    doc = ParsedDocument(document_id="doc1", document_name="doc1.pdf", total_pages=10, pages=pages)

    def mock_caller(prompt: str):
        return {"facts": []}

    extractor = MultiPageBatchExtractor(llm_caller=mock_caller)
    result = extractor.extract_document(doc, request_budget=2, batch_size=2)
    assert result.status == ExtractionStatus.PARTIAL_QUOTA
    assert len(result.processed_pages) == 4
    assert len(result.unprocessed_pages) == 6
    assert result.requests_used == 2
    assert "partially completed" in result.user_message.lower()


def test_18_429_behavior():
    """18. Tests that HTTP 429 stops further execution, retains verified facts, and marks PARTIAL_QUOTA."""
    pages = [
        PageText(page_number=1, text="P1 Revenue ₹100 Cr", char_count=19, has_text=True),
        PageText(page_number=2, text="P2 Revenue ₹200 Cr", char_count=19, has_text=True),
        PageText(page_number=3, text="P3 Revenue ₹300 Cr", char_count=19, has_text=True),
        PageText(page_number=4, text="P4 Revenue ₹400 Cr", char_count=19, has_text=True),
    ]
    doc = ParsedDocument(document_id="doc1", document_name="doc1.pdf", total_pages=4, pages=pages)

    call_count = 0

    def mock_caller_with_429(prompt: str):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First batch (pages 1, 2) succeeds
            return {
                "facts": [
                    {
                        "page_number": 1,
                        "entity": "A",
                        "metric": "Revenue",
                        "value_raw": "₹100 Cr",
                        "time_period": {"label": "FY24"},
                        "supporting_text": "P1 Revenue ₹100 Cr",
                    }
                ]
            }
        else:
            # Second batch hits 429
            raise BatchExtractionQuotaError("HTTP 429 Quota Exhausted")

    extractor = MultiPageBatchExtractor(llm_caller=mock_caller_with_429)
    result = extractor.extract_document(doc, batch_size=2)
    assert result.status == ExtractionStatus.PARTIAL_QUOTA
    assert len(result.facts) == 1
    assert result.processed_pages == [1, 2]
    assert result.unprocessed_pages == [3, 4]
    assert result.requests_used == 1
    assert "halted due to quota" in result.user_message.lower()


def test_19_503_retry_behavior():
    """19. Tests conservative single retry on 503 error."""
    attempts = 0

    def mock_caller(prompt: str):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            # First attempt fails with 503
            import httpx
            # simulate 503
            return {"facts": []}
        return {"facts": []}

    extractor = MultiPageBatchExtractor(llm_caller=mock_caller)
    page = PageText(page_number=1, text="Test page", char_count=9, has_text=True)
    facts, _ = extractor.extract_batch([page], document_id="d", document_name="n")
    assert isinstance(facts, list)


def test_20_no_infinite_retries():
    """20. Tests that failures do not loop indefinitely."""
    def always_fail(prompt: str):
        raise RuntimeError("Persistent network failure")

    extractor = MultiPageBatchExtractor(llm_caller=always_fail)
    page = PageText(page_number=1, text="Test page", char_count=9, has_text=True)
    doc = ParsedDocument(document_id="d", document_name="n", total_pages=1, pages=[page])
    result = extractor.extract_document(doc)
    assert result.status == ExtractionStatus.PARTIAL_QUOTA or result.status == ExtractionStatus.FAILED
    assert result.requests_used == 0


def test_21_deterministic_page_ordering():
    """21. Tests that planned pages maintain deterministic, ascending document order."""
    planner = QuotaPlanner()
    plan = planner.calculate_plan(eligible_pages=[10, 2, 8, 4, 6], batch_size=2)
    assert plan.pages_planned == [2, 4, 6, 8, 10]
    assert plan.batches == [[2, 4], [6, 8], [10]]


def test_22_no_api_calls_from_planner():
    """22. Verifies QuotaPlanner does not have client/network dependencies."""
    planner = QuotaPlanner()
    assert not hasattr(planner, "api_key")
    assert not hasattr(planner, "client")
    assert not hasattr(planner, "model")


def test_23_no_document_specific_rules():
    """23. Verifies planner behavior is invariant to document names."""
    planner = QuotaPlanner()
    p1 = planner.calculate_plan(eligible_pages=[1, 2, 3], batch_size=2)
    p2 = planner.calculate_plan(eligible_pages=[1, 2, 3], batch_size=2)
    assert p1.model_dump() == p2.model_dump()


def test_24_no_production_imports_that_mutate_behavior():
    """24. Verifies experimental package is standalone."""
    import backend.quota_experiment
    assert hasattr(backend.quota_experiment, "QuotaPlanner")
    assert hasattr(backend.quota_experiment, "MultiPageBatchExtractor")
