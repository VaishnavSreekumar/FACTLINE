"""
Tests for Phase 10: Local Extraction Benchmark Module.
"""

import pytest
import os
from backend.local_extraction.docling_parser import DoclingLocalParser, ParsedPage, ParsedBlock
from backend.local_extraction.gliner_extractor import GLiNERLocalExtractor, RawLocalFact
from backend.local_extraction.adapters import (
    adapt_raw_fact_to_fact_record,
    filter_grounded_facts,
    parse_numeric_candidate,
)
from backend.local_extraction.local_extractor import LocalFactExtractor
from backend.models.fact import FactRecord, EpistemicStatus
from backend.extraction.evidence import EvidenceVerifier, create_evidence

SAMPLE_PDF = os.path.join(
    os.path.dirname(__file__),
    "../../sample-data/starter-datasets/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf"
)

def test_docling_local_parser():
    """Verify local parser extracts structured pages and blocks with 1-indexed page numbers."""
    parser = DoclingLocalParser()
    pages = parser.parse_pdf(SAMPLE_PDF, page_numbers=[2])
    
    assert len(pages) == 1
    p = pages[0]
    assert p.page_number == 2
    assert len(p.text) > 0
    assert len(p.blocks) > 0

def test_gliner_extractor_sentence_extraction():
    """Verify GLiNER / local IE extracts entities, metrics, and numerical values."""
    extractor = GLiNERLocalExtractor()
    page = ParsedPage(
        page_number=2,
        text="In FY24, Delhivery Limited reported revenue from operations of ₹81,415.38 million and delivered >4.8Mn tonnes of freight.",
        blocks=[]
    )
    facts = extractor.extract_from_page(page)
    assert len(facts) >= 1
    
    # Check that monetary or numerical values are extracted
    val_texts = [f.value_raw for f in facts]
    assert any("81,415" in v or "4.8" in v for v in val_texts)
    assert all(f.page_number == 2 for f in facts)

def test_adapter_fact_record_mapping():
    """Verify adapter creates compliant FactRecord objects with strict Provenance."""
    raw = RawLocalFact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value_raw="₹81,415.38 million",
        unit="INR",
        time_period="FY24",
        scope="Consolidated",
        geography="India",
        epistemic_status="Reported Fact",
        supporting_text="revenue from operations of ₹81,415.38 million",
        page_number=4,
        confidence=0.92,
    )
    
    fact_record = adapt_raw_fact_to_fact_record(raw, document_id="02-delhivery-annual-report-fy24-excerpt.pdf")
    
    assert isinstance(fact_record, FactRecord)
    assert fact_record.entity == "Delhivery Limited"
    assert fact_record.metric == "Revenue from Operations"
    assert fact_record.value_raw == "₹81,415.38 million"
    assert fact_record.value_numeric == 81415.38
    assert fact_record.time_period.label == "FY24"
    assert fact_record.provenance.page_number == 4
    assert fact_record.provenance.supporting_text == "revenue from operations of ₹81,415.38 million"
    assert fact_record.epistemic_status == EpistemicStatus.REPORTED

def test_strict_evidence_grounding_filter():
    """Verify filter_grounded_facts rejects ungrounded facts."""
    page_text = "Delhivery delivered >4.8Mn tonnes of freight in FY24."
    
    grounded_raw = RawLocalFact(
        entity="Delhivery Limited",
        metric="Part-Truckload Freight Volume",
        value_raw=">4.8Mn tonnes",
        time_period="FY24",
        supporting_text="delivered >4.8Mn tonnes of freight",
        page_number=2,
    )
    
    ungrounded_raw = RawLocalFact(
        entity="Delhivery Limited",
        metric="Hallucinated Metric",
        value_raw="$999 Billion",
        time_period="FY99",
        supporting_text="this hallucinated text does not exist on page 2",
        page_number=2,
    )
    
    f1 = adapt_raw_fact_to_fact_record(grounded_raw, document_id="test.pdf")
    f2 = adapt_raw_fact_to_fact_record(ungrounded_raw, document_id="test.pdf")
    
    filtered = filter_grounded_facts([f1, f2], {2: page_text})
    
    assert len(filtered) == 1
    assert filtered[0].value_raw == ">4.8Mn tonnes"

def test_local_extractor_end_to_end():
    """Verify full end-to-end LocalFactExtractor execution on sample PDF."""
    extractor = LocalFactExtractor()
    res = extractor.extract_from_pdf(SAMPLE_PDF, page_numbers=[2])
    
    assert res.page_count == 1
    assert len(res.facts) > 0
    assert res.total_latency_ms > 0
    # Every extracted fact must pass EvidenceVerifier
    for f in res.facts:
        assert EvidenceVerifier.verify_provenance(f.provenance, f.provenance.supporting_text)
