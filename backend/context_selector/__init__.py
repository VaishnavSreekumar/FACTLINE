"""
Local Fact-Bearing Region & Context Window Selector Package.

Provides deterministic region detection, context expansion, and evidence-grounded
window selection for LLM payload compression experiments.
"""

from backend.context_selector.region_detector import RegionDetector, DetectedRegion
from backend.context_selector.context_expander import ContextExpander, ExpandedContextWindow
from backend.context_selector.selector import ContextSelector, PageContextResult

__all__ = [
    "RegionDetector",
    "DetectedRegion",
    "ContextExpander",
    "ExpandedContextWindow",
    "ContextSelector",
    "PageContextResult",
]
