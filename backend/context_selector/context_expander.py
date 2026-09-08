"""
Context Expander and Region Merging Engine.

Expands detected fact-bearing anchors with surrounding lines, headers, and labels,
and merges overlapping/adjacent line intervals deterministically.
"""

from typing import List, Tuple, Optional
import re
from pydantic import BaseModel, Field

from backend.context_selector.region_detector import DetectedRegion


class ExpandedContextWindow(BaseModel):
    """Represents an expanded, merged context window of text on a page."""
    page_number: int = Field(..., description="1-indexed page number")
    start_line: int = Field(..., description="0-indexed start line within original page")
    end_line: int = Field(..., description="0-indexed end line (inclusive) within original page")
    anchors: List[str] = Field(default_factory=list, description="Anchors contained in this window")
    signals: List[str] = Field(default_factory=list, description="Signals triggered in this window")
    source_text: str = Field(..., description="Verbatim extracted source text for this window")
    char_count: int = Field(..., description="Character count of this window's source text")


class ContextExpander:
    """
    Expands detected regions by a configurable line radius and merges nearby windows.
    """

    def __init__(
        self,
        expansion_radius: int = 2,
        merge_gap_threshold: int = 2,
        include_page_header: bool = True,
        header_line_limit: int = 2,
    ):
        """
        Args:
            expansion_radius: Number of lines before and after the detected anchor (±N lines).
            merge_gap_threshold: Maximum line gap between adjacent intervals to merge them.
            include_page_header: Whether to include the top header lines (page title) if fact-bearing regions exist.
            header_line_limit: Maximum number of top lines considered as page header.
        """
        self.expansion_radius = expansion_radius
        self.merge_gap_threshold = merge_gap_threshold
        self.include_page_header = include_page_header
        self.header_line_limit = header_line_limit

    def expand_and_merge(
        self,
        regions: List[DetectedRegion],
        page_text: str,
        page_number: int = 1,
    ) -> List[ExpandedContextWindow]:
        """
        Expands detected regions and merges overlapping or nearby intervals into
        coherent, bounded context windows.
        """
        if not page_text or not page_text.strip() or not regions:
            return []

        lines = page_text.splitlines()
        total_lines = len(lines)

        # Step 1: Compute expanded line intervals for each detected region
        intervals: List[Tuple[int, int, List[str], List[str]]] = []
        for r in regions:
            start = max(0, r.start_line - self.expansion_radius)
            end = min(total_lines - 1, r.end_line + self.expansion_radius)
            anchors = [r.anchor_text] if r.anchor_text else []
            signals = list(r.signals)
            intervals.append((start, end, anchors, signals))

        # Sort intervals by start_line
        intervals.sort(key=lambda x: (x[0], x[1]))

        # Step 2: Merge overlapping or adjacent intervals
        merged_intervals: List[Tuple[int, int, List[str], List[str]]] = []
        for start, end, anchors, signals in intervals:
            if not merged_intervals:
                merged_intervals.append((start, end, list(anchors), list(signals)))
                continue

            last_start, last_end, last_anchors, last_signals = merged_intervals[-1]

            # If current interval overlaps or is within merge_gap_threshold of the previous interval
            if start <= last_end + self.merge_gap_threshold + 1:
                # Merge into previous
                new_start = min(last_start, start)
                new_end = max(last_end, end)
                # Combine distinct anchors and signals
                combined_anchors = list(dict.fromkeys(last_anchors + anchors))
                combined_signals = list(dict.fromkeys(last_signals + signals))
                merged_intervals[-1] = (new_start, new_end, combined_anchors, combined_signals)
            else:
                merged_intervals.append((start, end, list(anchors), list(signals)))

        # Step 3: Check if page header should be prepended to the first window or included
        # If the first window starts within header_line_limit + 2 lines of page top, extend it to line 0
        if self.include_page_header and merged_intervals:
            first_start, first_end, first_anchors, first_signals = merged_intervals[0]
            if first_start <= self.header_line_limit + 1:
                merged_intervals[0] = (0, first_end, first_anchors, first_signals)

        # Step 4: Construct ExpandedContextWindow objects with exact verbatim slice
        results: List[ExpandedContextWindow] = []
        for start, end, anchors, signals in merged_intervals:
            window_lines = lines[start : end + 1]
            verbatim_text = "\n".join(window_lines)
            results.append(
                ExpandedContextWindow(
                    page_number=page_number,
                    start_line=start,
                    end_line=end,
                    anchors=anchors,
                    signals=signals,
                    source_text=verbatim_text,
                    char_count=len(verbatim_text),
                )
            )

        return results
