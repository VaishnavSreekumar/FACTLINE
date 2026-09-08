"""
Local Extraction Schema and Ontology Definitions for FACTLINE.

Defines the entity labels, zero-shot prompts, and regex patterns used by the local
information extraction pipeline to identify semantic facts matching the FactRecord schema.
"""

from typing import List, Dict, Any, Optional
import re

# GLiNER entity extraction label taxonomy
LOCAL_EXTRACTION_LABELS: List[str] = [
    "entity",
    "financial metric",
    "operational metric",
    "macroeconomic indicator",
    "monetary value",
    "percentage value",
    "numeric quantity",
    "time period",
    "reporting scope",
    "geography",
    "epistemic qualifier",
]

# Canonical mapping from GLiNER label types to FactRecord attributes
LABEL_TO_FIELD_MAP: Dict[str, str] = {
    "entity": "entity",
    "financial metric": "metric",
    "operational metric": "metric",
    "macroeconomic indicator": "metric",
    "monetary value": "value_raw",
    "percentage value": "value_raw",
    "numeric quantity": "value_raw",
    "time period": "time_period",
    "reporting scope": "scope",
    "geography": "geography",
    "epistemic qualifier": "epistemic_status",
}

# Regex patterns for local rule-based feature verification
PATTERNS = {
    "monetary": re.compile(
        r"(?:₹|Rs\.?|INR|\$|USD)\s*[\d,]+(?:\.\d+)?(?:\s*(?:Cr(?:ore)?|Mn|Million|Bn|Billion|Lakh))?",
        re.IGNORECASE,
    ),
    "percentage": re.compile(r"[-+]?[\d,]+(?:\.\d+)?\s*%", re.IGNORECASE),
    "inequality": re.compile(
        r"([><≥≤~]|at least|over|more than|approximately)\s*([\d,]+(?:\.\d+)?(?:\s*(?:Cr(?:ore)?|Mn|Million|Bn|Billion|Lakh))?(?:\s*(?:tonnes?|shipments?|customers?|pin\s*codes?|\%))?)",
        re.IGNORECASE,
    ),
    "exact_count": re.compile(
        r"\b([\d,]{4,}(?:\.\d+)?(?:\+)?)(?:\s*(?:customers?|pin\s*codes?|employees?))?\b",
        re.IGNORECASE,
    ),
    "fiscal_period": re.compile(
        r"\b(FY\s*\d{2,4}|Q[1-4]\s*FY\s*\d{2,4}|FY\s*\d{2,4}-\d{2,4})\b", re.IGNORECASE
    ),
    "calendar_period": re.compile(r"\b(20\d{2}-20\d{2}|20\d{2}-\d{2}|20\d{2})\b"),
    "scale_unit": re.compile(
        r"\b(Cr(?:ore)?|Mn|Million|Bn|Billion|Lakh|tonnes?|shipments?|customers?|pin\s*codes?|centers?|sq\.?\s*ft\.?)\b",
        re.IGNORECASE,
    ),
}

def detect_epistemic_status(text: str) -> str:
    """Classify epistemic status based on grounded contextual cues."""
    t_lower = text.lower()
    if any(k in t_lower for k in ["projected", "baseline projection", "forecast", "expected to grow", "outlook"]):
        return "Baseline Projection"
    elif any(k in t_lower for k in ["advance estimate", "first advance estimate", "second advance estimate", "provisional"]):
        return "Official Estimate"
    elif any(k in t_lower for k in ["target", "ambition", "aim", "goal"]):
        return "Target / Ambition"
    return "Reported Fact"
