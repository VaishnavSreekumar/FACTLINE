"""
Deterministic Region Detector for Fact-Bearing Text Segments.

Detects candidate fact-bearing regions (lines/blocks) in document pages using
deterministic structural signals:
- Numeric anchors (quantities, counts, ratios, ranges)
- Currency markers (₹, INR, Rs., $, crores, millions, billions, lakhs)
- Quantitative unit expressions (%, tonnes, customers, shipments, employees, km, units, etc.)
- Metric and business vocabulary (revenue, EBITDA, profit, loss, margin, growth, volume, etc.)
- Table-like rows and tabular structures
"""

from typing import List, Dict, Any, Set, Tuple, Optional
import re
from pydantic import BaseModel, Field

# Currency and monetary expressions
CURRENCY_PATTERN = re.compile(
    r"(?:₹|INR|Rs\.?|USD|US\$|EUR|GBP|\$|\b(?:crores?|millions?|billions?|trillions?|lakhs?)\b)",
    re.IGNORECASE,
)

# Quantitative units and physical / operational dimensions
UNIT_PATTERN = re.compile(
    r"(?:%|\b(?:tonnes?|tons?|kg|km|sq\.?\s*ft\.?|square\s+feet|customers?|employees?|shipments?|parcels?|packages?|units?|pin\s*codes?|centres?|centers?|vehicles?|shares?|bps|basis\s+points|destinations?|gateways?|hubs?)\b)",
    re.IGNORECASE,
)

# Generic business, financial, and macroeconomic metric keywords
METRIC_KEYWORD_PATTERN = re.compile(
    r"\b(?:revenue|ebitda|profit|loss|margin|growth|sales|income|expenditure|expenses?|assets?|liabilities?|capital|turnover|volume|tonnage|gdp|inflation|cpi|wpi|deficit|reserves?|exports?|imports?|pat|cash\s+flow|borrowings?|earnings|operating\s+leverage|market\s+share|capacity|utilization|yield|headcount|orders?)\b",
    re.IGNORECASE,
)

# Temporal markers: fiscal years, quarters, dates
TEMPORAL_PATTERN = re.compile(
    r"\b(?:FY\s*\d{2,4}|Q[1-4]\s*(?:FY\s*\d{2,4})?|20\d{2}-\d{2,4}|20\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{4})\b",
    re.IGNORECASE,
)

# Numeric anchors: standalone numbers, ranges, inequalities, floats, comma-separated figures, percentages
NUMERIC_ANCHOR_PATTERN = re.compile(
    r"(?:[><≥≤~]\s*)?[\d,]+(?:\.\d+)?(?:\+)?(?:\s*%)?",
)

# Table row heuristics: line containing at least 2 distinct numbers separated by whitespace/delimiters
YEAR_COLUMN_PATTERN = re.compile(r"\b(?:FY\s*\d{2,4}|20\d{2}(?:-\d{2,4})?|Q[1-4])\b", re.IGNORECASE)


class DetectedRegion(BaseModel):
    """Represents a detected candidate fact-bearing line interval on a page."""
    page_number: int = Field(..., description="1-indexed page number")
    start_line: int = Field(..., description="0-indexed start line within the page")
    end_line: int = Field(..., description="0-indexed end line (inclusive)")
    anchor_text: str = Field(..., description="Primary anchor text or line snippet")
    signals: List[str] = Field(default_factory=list, description="Triggered detection signals")
    line_text: str = Field(..., description="Full text of the detected line(s)")


class RegionDetector:
    """
    Deterministic detector that identifies lines containing numeric/quantitative anchors
    or table structures without LLMs, embeddings, or document-specific heuristics.
    """

    def __init__(
        self,
        min_digit_count: int = 1,
        detect_tables: bool = True,
    ):
        self.min_digit_count = min_digit_count
        self.detect_tables = detect_tables

    def detect_signals_in_line(self, line: str) -> Tuple[List[str], List[str]]:
        """
        Analyzes a single line of text and extracts detected signal tags and anchor tokens.
        
        Returns:
            (signals, anchors)
        """
        signals: List[str] = []
        anchors: List[str] = []

        if not line or not line.strip():
            return signals, anchors

        # Check numeric anchors
        raw_nums = NUMERIC_ANCHOR_PATTERN.findall(line)
        valid_nums = [
            n.strip() for n in raw_nums
            if any(c.isdigit() for c in n) and len(n.strip()) > 0
        ]
        # Filter out standalone 1-digit numbers unless followed by % or unit
        filtered_nums = [
            n for n in valid_nums
            if len(re.sub(r"[^\d]", "", n)) >= self.min_digit_count
        ]

        if filtered_nums:
            signals.append("numeric")
            anchors.extend(filtered_nums[:3])

        # Check currency
        currencies = CURRENCY_PATTERN.findall(line)
        if currencies:
            signals.append("currency")
            anchors.extend([c.strip() for c in currencies[:2]])

        # Check units
        units = UNIT_PATTERN.findall(line)
        if units:
            signals.append("unit")
            anchors.extend([u.strip() for u in units[:2]])

        # Check percentage
        if "%" in line:
            if "percentage" not in signals:
                signals.append("percentage")

        # Check metric vocabulary
        metrics = METRIC_KEYWORD_PATTERN.findall(line)
        if metrics:
            signals.append("metric_keyword")
            anchors.extend([m.strip() for m in metrics[:2]])

        # Check temporal markers
        temporals = TEMPORAL_PATTERN.findall(line)
        if temporals:
            signals.append("temporal")
            anchors.extend([t.strip() for t in temporals[:2]])

        # Check tabular pattern in line (multiple numbers or year headers)
        if len(valid_nums) >= 2 or len(YEAR_COLUMN_PATTERN.findall(line)) >= 2:
            signals.append("table_row")

        return signals, anchors

    def detect_regions(
        self,
        page_text: str,
        page_number: int = 1,
    ) -> List[DetectedRegion]:
        """
        Scans a page's text line-by-line and extracts all candidate fact-bearing regions.
        
        A line is considered a candidate region if:
        1. It has a numeric anchor AND (currency OR unit OR metric_keyword OR temporal OR percentage), OR
        2. It is a table row (multiple numbers / year columns), OR
        3. It is a metric line immediately adjacent to numbers.
        """
        if not page_text or not page_text.strip():
            return []

        lines = page_text.splitlines()
        detected: List[DetectedRegion] = []

        for line_idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            signals, anchors = self.detect_signals_in_line(stripped)

            # Determine whether this line qualifies as a candidate region
            is_fact_bearing = False

            # Condition 1: Has numeric anchor AND at least one contextual anchor (currency, unit, metric, temporal, percentage)
            has_numeric = "numeric" in signals or "percentage" in signals
            has_context_signal = any(
                s in signals for s in ("currency", "unit", "metric_keyword", "temporal", "percentage")
            )

            if has_numeric and has_context_signal:
                is_fact_bearing = True
            elif "table_row" in signals and has_numeric:
                is_fact_bearing = True
            elif has_numeric and len(anchors) > 0:
                # Strong multi-digit numbers or percentages (e.g. 33,278 or >4.8Mn)
                digits = re.findall(r"\d+", stripped)
                if any(len(d) >= 2 for d in digits) or "%" in stripped or "+" in stripped:
                    is_fact_bearing = True

            if is_fact_bearing:
                anchor_repr = ", ".join(anchors[:3]) if anchors else stripped[:30]
                detected.append(
                    DetectedRegion(
                        page_number=page_number,
                        start_line=line_idx,
                        end_line=line_idx,
                        anchor_text=anchor_repr,
                        signals=signals,
                        line_text=stripped,
                    )
                )

        return detected
