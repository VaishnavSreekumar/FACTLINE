"""
Deterministic Signal Extractors for Page Relevance Filtering.

Extracts interpretable, explainable features from page-level text without any
dependency on external LLMs, embeddings, or document-specific heuristics.
"""

from typing import List, Dict, Any, Tuple
import re

# Currency and monetary terms (Phase 3 safety: bare '$' is detected as a signal, not inferred as USD)
CURRENCY_PATTERN = re.compile(
    r"(?:₹|INR|Rs\.?|USD|US\$|EUR|GBP|\$|\b(?:crores?|millions?|billions?|trillions?|lakhs?)\b)",
    re.IGNORECASE,
)

# Quantitative units and physical dimensions
UNIT_PATTERN = re.compile(
    r"(?:%|\b(?:tonnes?|tons?|kg|km|sq\.?\s*ft\.?|square\s+feet|customers?|employees?|shipments?|units?|pin\s*codes?|centres?|centers?|vehicles?|shares?|bps|basis\s+points)\b)",
    re.IGNORECASE,
)

# Generic business, financial, and macroeconomic metric keywords
METRIC_KEYWORD_PATTERN = re.compile(
    r"\b(?:financial|performance|revenue|profit|loss|ebitda|income|expenditure|expenses?|margin|growth|rate|sales|assets|liabilities|capital|turnover|volume|tonnage|gdp|inflation|cpi|wpi|deficit|reserves|exports?|imports?|advance\s+estimates?|projections?|forecasts?|operating\s+leverage|pat|cash\s+flow)\b",
    re.IGNORECASE,
)

# Numeric patterns: integers, floats, comma-separated figures, inequalities, percentages
NUMERIC_PATTERN = re.compile(
    r"(?:[><≥≤~]\s*)?[\d,]+(?:\.\d+)?(?:\+)?(?:\s*%)?",
)

# Date and temporal markers
TEMPORAL_PATTERN = re.compile(
    r"\b(?:FY\s*\d{2,4}|Q[1-4]\s*(?:FY\s*\d{2,4})?|20\d{2}-\d{2,4}|20\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4})\b",
    re.IGNORECASE,
)

# Low-value / boilerplate markers (used for negative dampening if quantitative density is near-zero)
BOILERPLATE_PATTERN = re.compile(
    r"\b(?:table\s+of\s+contents|contents|all\s+rights\s+reserved|confidential|disclaimer|forward-looking\s+statements|safe\s+harbor|blank\s+page|this\s+page\s+intentionally\s+left\s+blank|corporate\s+information|directory)\b",
    re.IGNORECASE,
)


def extract_numeric_signals(text: str) -> Tuple[int, List[str]]:
    """Counts numerical tokens and returns distinctive quantitative values."""
    matches = NUMERIC_PATTERN.findall(text)
    meaningful = [m.strip() for m in matches if any(c.isdigit() for c in m) and len(m.strip()) > 0]
    filtered = [m for m in meaningful if len(m) > 1 or m.isdigit()]
    return len(filtered), filtered[:10]


def extract_currency_signals(text: str) -> List[str]:
    """Identifies currency and financial denomination indicators."""
    matches = CURRENCY_PATTERN.findall(text)
    return list(set(m.strip() for m in matches))


def extract_unit_signals(text: str) -> List[str]:
    """Identifies quantitative units of measurement."""
    matches = UNIT_PATTERN.findall(text)
    return list(set(m.strip().lower() for m in matches))


def extract_metric_signals(text: str) -> List[str]:
    """Identifies generic business and macroeconomic context terms."""
    matches = METRIC_KEYWORD_PATTERN.findall(text)
    return list(set(m.strip().lower() for m in matches))


def extract_temporal_signals(text: str) -> List[str]:
    """Identifies fiscal years, quarters, and calendar periods."""
    matches = TEMPORAL_PATTERN.findall(text)
    return list(set(m.strip() for m in matches))


def detect_table_structure(text: str) -> Tuple[bool, int]:
    """
    Detects whether text contains tabular structures by scanning for lines
    with multiple whitespace-separated quantitative or numeric columns.
    """
    lines = text.splitlines()
    table_like_rows = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        num_matches = NUMERIC_PATTERN.findall(stripped)
        nums = [n for n in num_matches if any(c.isdigit() for c in n)]
        if len(nums) >= 2 and len(stripped.split()) >= 3:
            table_like_rows += 1

    is_tabular = table_like_rows >= 3
    return is_tabular, table_like_rows


def detect_boilerplate_signals(text: str) -> List[str]:
    """Identifies structural boilerplate or low-value disclaimer phrasing."""
    matches = BOILERPLATE_PATTERN.findall(text)
    return list(set(m.strip().lower() for m in matches))
