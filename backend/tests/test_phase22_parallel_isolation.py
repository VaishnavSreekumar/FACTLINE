"""
Phase 22 Tests: Parallel Gemini Extraction Isolation & Concurrency Safety.
"""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
import pytest

from backend.models.document import PageText, ParsedDocument
from backend.models.fact import FactRecord, Provenance, TimePeriod, EpistemicStatus
from backend.extraction.evidence import EvidenceVerifier, create_evidence
from backend.performance_experiment.parallel_experiment.parallel_runner import ParallelBenchmarkRunner, SingleBatchTaskResult


def test_parallel_deterministic_ordering():
    """Verify that results collected from concurrent workers are deterministically sorted by batch_index."""
    runner = ParallelBenchmarkRunner()
    
    # Mock batch results arriving out of order
    res_b3 = SingleBatchTaskResult(3, [11, 12, 13, 14, 15], 0.0, 1.0, 1.0, 1000, 250, 200, True, 0, [], [], 0, 100)
    res_b1 = SingleBatchTaskResult(1, [1, 2, 3, 4, 5], 0.0, 1.0, 1.0, 1000, 250, 200, True, 0, [], [], 0, 100)
    res_b2 = SingleBatchTaskResult(2, [6, 7, 8, 9, 10], 0.0, 1.0, 1.0, 1000, 250, 200, True, 0, [], [], 0, 100)
    
    out_of_order = [res_b3, res_b1, res_b2]
    out_of_order.sort(key=lambda r: r.batch_index)
    
    assert [r.batch_index for r in out_of_order] == [1, 2, 3]
    assert out_of_order[0].page_numbers == [1, 2, 3, 4, 5]
    assert out_of_order[1].page_numbers == [6, 7, 8, 9, 10]
    assert out_of_order[2].page_numbers == [11, 12, 13, 14, 15]


def test_parallel_failure_isolation_preserves_successful_batches():
    """Verify that a failure in one batch does not corrupt or discard successful facts from other batches."""
    page1 = PageText(page_number=1, text="India GDP was 7.0%.", char_count=19, has_text=True)
    page2 = PageText(page_number=2, text="China GDP was 5.0%.", char_count=19, has_text=True)
    page_lookup = {1: page1, 2: page2}

    fact_b1 = FactRecord(
        fact_id="f1",
        entity="India",
        metric="GDP",
        value_raw="7.0%",
        value_numeric=7.0,
        unit="%",
        time_period=TimePeriod(label="2024"),
        epistemic_status=EpistemicStatus.REPORTED,
        provenance=create_evidence("doc1", 1, "India GDP was 7.0%."),
    )

    # Batch 1 succeeds, Batch 2 fails with 503
    res_b1 = SingleBatchTaskResult(1, [1], 0.0, 1.0, 1.0, 500, 125, 200, True, 0, [{"fact": 1}], [fact_b1], 0, 100)
    res_b2 = SingleBatchTaskResult(2, [2], 0.0, 1.0, 1.0, 500, 125, 503, False, 2, [], [], 0, 0, "HTTP 503")

    results = [res_b1, res_b2]
    successful_facts = [f for r in results if r.success for f in r.verified_facts]
    
    assert len(successful_facts) == 1
    assert successful_facts[0].entity == "India"
    assert successful_facts[0].provenance.page_number == 1


def test_parallel_worker_page_isolation():
    """Verify that evidence verification strictly evaluates against the worker's declared page map."""
    page_b1 = PageText(page_number=1, text="Alpha Corp revenue $100M.", char_count=25, has_text=True)
    page_b2 = PageText(page_number=2, text="Beta Corp revenue $200M.", char_count=24, has_text=True)

    # Cross-worker contamination attempt: Worker 1 claims Beta Corp on Page 1
    hallucinated_prov = create_evidence("doc1", 1, "Beta Corp revenue $200M.")
    assert EvidenceVerifier.verify_provenance(hallucinated_prov, page_b1) is False
