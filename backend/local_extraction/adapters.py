"""
Adapter Layer mapping Local Raw Facts to Canonical FactRecords.

Ensures complete schema compliance with FACTLINE core and enforces strict evidence grounding.
"""

import uuid
import re
from typing import Optional, List, Dict, Any

from backend.models.fact import FactRecord, EpistemicStatus, TimePeriod, Provenance
from backend.extraction.evidence import EvidenceVerifier, create_evidence
from backend.local_extraction.gliner_extractor import RawLocalFact

def parse_numeric_candidate(value_raw: str) -> Optional[float]:
    """Parse raw value string into floating point number if deterministically possible."""
    clean = value_raw.replace("₹", "").replace("$", "").replace("Rs.", "").replace("INR", "").replace("USD", "")
    clean = clean.replace(">", "").replace("<", "").replace("~", "").replace("%", "").strip()
    # Remove standard unit words
    clean = re.sub(r"\b(Cr(?:ore)?|Mn|Million|Bn|Billion|Lakh|tonnes?|shipments?|customers?)\b", "", clean, flags=re.IGNORECASE).strip()
    clean = clean.replace(",", "")
    try:
        return float(clean)
    except (ValueError, TypeError):
        return None

def map_epistemic_status(status_str: str) -> EpistemicStatus:
    """Map string status to EpistemicStatus enum."""
    s = (status_str or "").lower()
    if "project" in s or "forecast" in s:
        return EpistemicStatus.PROJECTED
    elif "estimate" in s:
        return EpistemicStatus.ESTIMATED
    elif "target" in s or "ambition" in s:
        return EpistemicStatus.TARGET
    elif "audit" in s:
        return EpistemicStatus.AUDITED
    return EpistemicStatus.REPORTED

def adapt_raw_fact_to_fact_record(
    raw_fact: RawLocalFact,
    document_id: str,
    document_date: Optional[str] = None,
) -> FactRecord:
    """
    Transforms a RawLocalFact into a canonical FACTLINE FactRecord.
    """
    fact_id = f"fact-loc-{uuid.uuid4().hex[:8]}"
    
    tp_label = raw_fact.time_period if raw_fact.time_period else "Unspecified Period"
    time_period = TimePeriod(label=tp_label)

    provenance = create_evidence(
        document_id=document_id,
        page_number=raw_fact.page_number,
        supporting_text=raw_fact.supporting_text,
        document_date=document_date,
    )

    numeric_val = parse_numeric_candidate(raw_fact.value_raw)

    return FactRecord(
        fact_id=fact_id,
        entity=raw_fact.entity if raw_fact.entity else "Unspecified Entity",
        metric=raw_fact.metric if raw_fact.metric else "Unspecified Metric",
        value_raw=raw_fact.value_raw,
        value_numeric=numeric_val,
        unit=raw_fact.unit,
        time_period=time_period,
        scope=raw_fact.scope,
        geography=raw_fact.geography,
        epistemic_status=map_epistemic_status(raw_fact.epistemic_status),
        data_vintage=document_date,
        provenance=provenance,
        extraction_confidence=raw_fact.confidence,
    )

def filter_grounded_facts(
    facts: List[FactRecord],
    page_text_map: Dict[int, str],
) -> List[FactRecord]:
    """
    Strictly filters out any fact whose supporting text cannot be verified on its cited page.
    """
    grounded_facts: List[FactRecord] = []
    for f in facts:
        page_num = f.provenance.page_number
        page_text = page_text_map.get(page_num, "")
        if EvidenceVerifier.verify_provenance(f.provenance, page_text):
            grounded_facts.append(f)
        else:
            # Fact rejected due to ungrounded evidence invariant
            pass
    return grounded_facts
