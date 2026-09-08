"""
Local Page Relevance Filter Module for FACTLINE (Phase 11 Experimental).
"""

from backend.page_filter.signals import (
    extract_numeric_signals,
    extract_currency_signals,
    extract_unit_signals,
    extract_metric_signals,
    extract_temporal_signals,
    detect_table_structure,
    detect_boilerplate_signals,
)
from backend.page_filter.relevance import (
    PageRelevanceFilter,
    PageRelevanceScore,
    DEFAULT_RELEVANCE_THRESHOLD,
)

__all__ = [
    "PageRelevanceFilter",
    "PageRelevanceScore",
    "DEFAULT_RELEVANCE_THRESHOLD",
    "extract_numeric_signals",
    "extract_currency_signals",
    "extract_unit_signals",
    "extract_metric_signals",
    "extract_temporal_signals",
    "detect_table_structure",
    "detect_boilerplate_signals",
]
