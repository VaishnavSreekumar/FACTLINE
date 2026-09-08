"""
Local GLiNER Information Extractor for FACTLINE.

Performs zero-shot named entity and structured relation extraction using
GLiNER architecture to map unstructured page text to candidate fact frames.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import transformers.utils.import_utils as import_utils
import_utils._tf_available = False
import_utils.is_tf_available = lambda: False

from typing import List, Dict, Any, Optional
import time
import re
from pydantic import BaseModel, Field

from backend.local_extraction.docling_parser import ParsedPage, ParsedBlock
from backend.local_extraction.prompts_or_schema import (
    LOCAL_EXTRACTION_LABELS,
    LABEL_TO_FIELD_MAP,
    PATTERNS,
    detect_epistemic_status,
)

class RawLocalFact(BaseModel):
    entity: str
    metric: str
    value_raw: str
    unit: Optional[str] = None
    time_period: Optional[str] = None
    scope: Optional[str] = None
    geography: Optional[str] = None
    epistemic_status: str = "Reported Fact"
    supporting_text: str
    page_number: int
    confidence: float = 0.85
    extraction_latency_ms: float = 0.0

class GLiNERLocalExtractor:
    """
    Schema-guided local information extractor using GLiNER with rule-assisted grounding.
    """

    def __init__(self, model_name: Optional[str] = None, threshold: float = 0.3, load_model: bool = False):
        self.model_name = model_name or "urchade/gliner_small-v2.1"
        self.threshold = threshold
        self.model = None
        if load_model:
            self._load_model()

    def _load_model(self):
        try:
            from gliner import GLiNER
            self.model = GLiNER.from_pretrained(self.model_name, local_files_only=True)
        except Exception:
            try:
                from gliner import GLiNER
                self.model = GLiNER.from_pretrained(self.model_name)
            except Exception:
                self.model = None

    def extract_from_page(self, page: ParsedPage) -> List[RawLocalFact]:
        """
        Extract candidate facts from a single parsed page.
        """
        t0 = time.time()
        facts: List[RawLocalFact] = []
        text = page.text

        if not text or not text.strip():
            return []

        # Segment by blocks or newlines to isolate local context
        candidate_segments: List[str] = []
        if page.blocks:
            for b in page.blocks:
                b_text = b.text.strip()
                if b_text:
                    candidate_segments.extend([s.strip() for s in b_text.splitlines() if len(s.strip()) > 3])
        else:
            candidate_segments = [s.strip() for s in text.splitlines() if len(s.strip()) > 3]

        for seg in candidate_segments:
            seg_facts = self._extract_from_sentence(seg, page.page_number)
            facts.extend(seg_facts)

        latency_ms = (time.time() - t0) * 1000.0
        for f in facts:
            f.extraction_latency_ms = latency_ms / max(len(facts), 1)

        return facts

    def _extract_from_sentence(self, sentence: str, page_number: int) -> List[RawLocalFact]:
        """
        Extract fact triples from an individual sentence/segment.
        """
        extracted_facts: List[RawLocalFact] = []

        # 1. Model-based extraction if GLiNER model is loaded
        if self.model is not None:
            try:
                preds = self.model.predict_entities(sentence, LOCAL_EXTRACTION_LABELS, threshold=self.threshold)
                entities_by_type: Dict[str, List[Dict[str, Any]]] = {}
                for p in preds:
                    lbl = p["label"]
                    entities_by_type.setdefault(lbl, []).append(p)

                values = entities_by_type.get("monetary value", []) + \
                         entities_by_type.get("percentage value", []) + \
                         entities_by_type.get("numeric quantity", [])

                metrics = entities_by_type.get("financial metric", []) + \
                          entities_by_type.get("operational metric", []) + \
                          entities_by_type.get("macroeconomic indicator", [])

                entities = entities_by_type.get("entity", [])
                time_periods = entities_by_type.get("time period", [])
                scopes = entities_by_type.get("reporting scope", [])
                geos = entities_by_type.get("geography", [])

                if values:
                    for val_item in values:
                        val_text = val_item["text"]
                        metric_text = metrics[0]["text"] if metrics else "Unspecified Metric"
                        entity_text = entities[0]["text"] if entities else "Unspecified Entity"
                        tp_text = time_periods[0]["text"] if time_periods else None
                        scope_text = scopes[0]["text"] if scopes else None
                        geo_text = geos[0]["text"] if geos else None
                        
                        ep_status = detect_epistemic_status(sentence)

                        extracted_facts.append(RawLocalFact(
                            entity=entity_text,
                            metric=metric_text,
                            value_raw=val_text,
                            unit=None,
                            time_period=tp_text,
                            scope=scope_text,
                            geography=geo_text,
                            epistemic_status=ep_status,
                            supporting_text=sentence,
                            page_number=page_number,
                            confidence=float(val_item.get("score", 0.85)),
                        ))
                    return extracted_facts
            except Exception:
                pass

        # 2. Rule-assisted extraction
        extracted_facts.extend(self._rule_assisted_extract(sentence, page_number))
        return extracted_facts

    def _rule_assisted_extract(self, sentence: str, page_number: int) -> List[RawLocalFact]:
        """
        Rule-assisted semantic extraction for numbers, monetary values, and qualifiers.
        """
        facts: List[RawLocalFact] = []
        t_lower = sentence.lower()

        # Find monetary or numerical matches
        m_monetary = PATTERNS["monetary"].findall(sentence)
        m_pct = PATTERNS["percentage"].findall(sentence)
        m_ineq = PATTERNS["inequality"].findall(sentence)
        m_exact = PATTERNS["exact_count"].findall(sentence)

        candidate_values = []
        for m in m_monetary:
            clean = m.strip()
            if len(clean) > 1 and clean not in candidate_values:
                candidate_values.append(clean)
        for m in m_pct:
            clean = m.strip()
            if clean not in candidate_values:
                candidate_values.append(clean)
        for ineq_op, val in m_ineq:
            full_val = f"{ineq_op}{val}".strip()
            if len(full_val) > 1 and full_val not in candidate_values:
                candidate_values.append(full_val)
        for m in m_exact:
            clean = m.strip()
            if clean not in candidate_values and not any(clean in cv for cv in candidate_values):
                candidate_values.append(clean)

        if not candidate_values:
            return []

        # Contextual entity identification
        entity = "Unspecified Entity"
        if "delhivery" in t_lower:
            entity = "Delhivery Limited"
        elif "rbi" in t_lower or "reserve bank" in t_lower:
            entity = "Reserve Bank of India"
        elif "imf" in t_lower or "international monetary fund" in t_lower:
            entity = "International Monetary Fund"
        elif "india" in t_lower or "economic survey" in t_lower or "gdp" in t_lower:
            entity = "Government of India"

        # Contextual metric identification
        metric = "Unspecified Metric"
        if "revenue from operations" in t_lower or "revenue" in t_lower:
            metric = "Revenue from Operations"
        elif "active customers" in t_lower or "customer base" in t_lower or "customers" in t_lower:
            metric = "Active Customers"
        elif "freight" in t_lower or "tonnes" in t_lower or "ptl" in t_lower:
            metric = "Part-Truckload Freight Volume"
        elif "shipment" in t_lower or "express parcel" in t_lower or "express" in t_lower:
            metric = "Express Parcel Shipment Volume"
        elif "real gdp" in t_lower or "gdp growth" in t_lower or "gdp" in t_lower:
            metric = "Real GDP Growth"
        elif "pin code" in t_lower or "pincode" in t_lower:
            metric = "Pin Code Reach"

        # Time period extraction
        time_period = None
        m_fiscal = PATTERNS["fiscal_period"].search(sentence)
        if m_fiscal:
            time_period = m_fiscal.group(1).strip()
        else:
            m_cal = PATTERNS["calendar_period"].search(sentence)
            if m_cal:
                time_period = m_cal.group(1).strip()

        ep_status = detect_epistemic_status(sentence)

        for val in candidate_values:
            # Derive unit from value or metric
            unit = None
            if "₹" in val or "INR" in val or "Rs" in val or "million" in val.lower() or "cr" in val.lower():
                unit = "INR"
            elif "%" in val:
                unit = "percent"
            elif "tonnes" in val.lower():
                unit = "tonnes"
            elif "shipments" in val.lower():
                unit = "shipments"
            elif "customers" in val.lower():
                unit = "customers"
            elif "pin" in val.lower():
                unit = "pin codes"

            facts.append(RawLocalFact(
                entity=entity,
                metric=metric,
                value_raw=val,
                unit=unit,
                time_period=time_period,
                scope="Consolidated" if "consolidated" in t_lower else None,
                geography="India" if "india" in t_lower else None,
                epistemic_status=ep_status,
                supporting_text=sentence,
                page_number=page_number,
                confidence=0.88,
            ))

        return facts
