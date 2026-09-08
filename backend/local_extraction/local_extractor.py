"""
Main Local Fact Extractor Pipeline for FACTLINE (Isolated Experimental).

Combines layout parsing, GLiNER/local IE, FactRecord adaptation, and strict evidence grounding.
"""

from typing import List, Dict, Any, Optional
import time
import os

from backend.models.fact import FactRecord
from backend.local_extraction.docling_parser import DoclingLocalParser, ParsedPage
from backend.local_extraction.gliner_extractor import GLiNERLocalExtractor, RawLocalFact
from backend.local_extraction.adapters import adapt_raw_fact_to_fact_record, filter_grounded_facts

class LocalExtractionResult:
    def __init__(
        self,
        facts: List[FactRecord],
        unsupported_facts_count: int,
        page_count: int,
        parse_latency_ms: float,
        extraction_latency_ms: float,
        total_latency_ms: float,
    ):
        self.facts = facts
        self.unsupported_facts_count = unsupported_facts_count
        self.page_count = page_count
        self.parse_latency_ms = parse_latency_ms
        self.extraction_latency_ms = extraction_latency_ms
        self.total_latency_ms = total_latency_ms

class LocalFactExtractor:
    """
    Isolated local extraction pipeline implementing PDF -> Layout -> Local IE -> Grounded FactRecord.
    """

    def __init__(
        self,
        parser: Optional[DoclingLocalParser] = None,
        extractor: Optional[GLiNERLocalExtractor] = None,
    ):
        self.parser = parser or DoclingLocalParser()
        self.extractor = extractor or GLiNERLocalExtractor()

    def extract_from_pdf(
        self,
        pdf_path: str,
        page_numbers: Optional[List[int]] = None,
        document_id: Optional[str] = None,
        document_date: Optional[str] = None,
    ) -> LocalExtractionResult:
        """
        Execute the full local extraction pipeline on a PDF file.
        """
        t_start = time.time()
        doc_id = document_id or os.path.basename(pdf_path)

        # Step 1: Parse pages
        t0 = time.time()
        parsed_pages = self.parser.parse_pdf(pdf_path, page_numbers=page_numbers)
        parse_latency_ms = (time.time() - t0) * 1000.0

        # Build page text lookup map for grounding
        page_text_map: Dict[int, str] = {p.page_number: p.text for p in parsed_pages}

        # Step 2: Extract candidate facts
        t1 = time.time()
        raw_facts: List[RawLocalFact] = []
        for page in parsed_pages:
            p_facts = self.extractor.extract_from_page(page)
            raw_facts.extend(p_facts)
        extraction_latency_ms = (time.time() - t1) * 1000.0

        # Step 3: Adapt to canonical FactRecord schema
        adapted_facts: List[FactRecord] = [
            adapt_raw_fact_to_fact_record(rf, document_id=doc_id, document_date=document_date)
            for rf in raw_facts
        ]

        # Step 4: Strict Evidence Grounding Verification
        grounded_facts = filter_grounded_facts(adapted_facts, page_text_map)
        unsupported_count = len(adapted_facts) - len(grounded_facts)

        total_latency_ms = (time.time() - t_start) * 1000.0

        return LocalExtractionResult(
            facts=grounded_facts,
            unsupported_facts_count=unsupported_count,
            page_count=len(parsed_pages),
            parse_latency_ms=parse_latency_ms,
            extraction_latency_ms=extraction_latency_ms,
            total_latency_ms=total_latency_ms,
        )
