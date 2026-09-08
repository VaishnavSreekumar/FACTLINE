"""
Local Extraction Module for FACTLINE (Phase 10 Benchmark Architecture).
"""

from backend.local_extraction.docling_parser import DoclingLocalParser, ParsedPage, ParsedBlock
from backend.local_extraction.gliner_extractor import GLiNERLocalExtractor, RawLocalFact
from backend.local_extraction.adapters import adapt_raw_fact_to_fact_record, filter_grounded_facts
from backend.local_extraction.local_extractor import LocalFactExtractor, LocalExtractionResult

__all__ = [
    "DoclingLocalParser",
    "ParsedPage",
    "ParsedBlock",
    "GLiNERLocalExtractor",
    "RawLocalFact",
    "adapt_raw_fact_to_fact_record",
    "filter_grounded_facts",
    "LocalFactExtractor",
    "LocalExtractionResult",
]
