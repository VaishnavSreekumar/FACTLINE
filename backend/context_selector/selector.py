"""
Local Context Selector for Fact-Bearing Region Benchmarking.

Coordinates deterministic region detection, context expansion, and evidence-grounded
window selection to compress PDF page inputs for downstream extraction.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from backend.models.document import PageText, ParsedDocument
from backend.context_selector.region_detector import RegionDetector, DetectedRegion
from backend.context_selector.context_expander import ContextExpander, ExpandedContextWindow


class PageContextResult(BaseModel):
    """Result of context selection on a single document page."""
    document_id: Optional[str] = Field(None, description="Document identifier")
    page_number: int = Field(..., description="1-indexed page number")
    original_character_count: int = Field(..., description="Original raw character count")
    selected_character_count: int = Field(..., description="Character count across all selected windows")
    compression_ratio: float = Field(..., description="selected_chars / original_chars (lower is more compressed)")
    context_windows: List[ExpandedContextWindow] = Field(default_factory=list, description="Selected context windows")
    anchors: List[str] = Field(default_factory=list, description="All unique anchors detected across windows")
    signals: List[str] = Field(default_factory=list, description="All unique signals triggered on this page")
    combined_source_text: str = Field(..., description="Concatenated verbatim source text from all windows")
    is_empty: bool = Field(..., description="Whether no fact-bearing context was selected")


class ContextSelector:
    """
    Orchestrates deterministic region detection and context expansion.
    """

    def __init__(
        self,
        expansion_radius: int = 2,
        merge_gap_threshold: int = 2,
        min_digit_count: int = 1,
        include_page_header: bool = True,
        header_line_limit: int = 2,
    ):
        self.detector = RegionDetector(
            min_digit_count=min_digit_count,
            detect_tables=True,
        )
        self.expander = ContextExpander(
            expansion_radius=expansion_radius,
            merge_gap_threshold=merge_gap_threshold,
            include_page_header=include_page_header,
            header_line_limit=header_line_limit,
        )

    def select_page_context(
        self,
        page_text: str,
        page_number: int = 1,
        document_id: Optional[str] = None,
    ) -> PageContextResult:
        """
        Executes region detection, context expansion, and statistical compression measurement
        on a single page of text.
        """
        original_char_count = len(page_text) if page_text else 0

        if not page_text or not page_text.strip():
            return PageContextResult(
                document_id=document_id,
                page_number=page_number,
                original_character_count=0,
                selected_character_count=0,
                compression_ratio=0.0,
                context_windows=[],
                anchors=[],
                signals=[],
                combined_source_text="",
                is_empty=True,
            )

        # 1. Detect candidate regions
        regions = self.detector.detect_regions(page_text, page_number=page_number)

        # 2. Expand regions and merge overlapping / adjacent intervals
        windows = self.expander.expand_and_merge(regions, page_text, page_number=page_number)

        # 3. Aggregate unique anchors, signals, and combined verbatim text
        all_anchors: List[str] = []
        all_signals: List[str] = []
        for w in windows:
            all_anchors.extend(w.anchors)
            all_signals.extend(w.signals)

        unique_anchors = list(dict.fromkeys(all_anchors))
        unique_signals = list(dict.fromkeys(all_signals))

        combined_text = "\n\n---\n\n".join(w.source_text for w in windows) if windows else ""
        selected_char_count = sum(w.char_count for w in windows)
        compression_ratio = (
            round(selected_char_count / original_char_count, 4)
            if original_char_count > 0 else 0.0
        )

        return PageContextResult(
            document_id=document_id,
            page_number=page_number,
            original_character_count=original_char_count,
            selected_character_count=selected_char_count,
            compression_ratio=compression_ratio,
            context_windows=windows,
            anchors=unique_anchors,
            signals=unique_signals,
            combined_source_text=combined_text,
            is_empty=len(windows) == 0,
        )

    def select_document_context(
        self,
        document: ParsedDocument,
    ) -> List[PageContextResult]:
        """
        Executes context selection across all pages of a parsed document.
        """
        results: List[PageContextResult] = []
        for page in document.pages:
            res = self.select_page_context(
                page_text=page.text,
                page_number=page.page_number,
                document_id=document.document_id,
            )
            results.append(res)
        return results
