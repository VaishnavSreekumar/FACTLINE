"""
Quality Evaluation for Extraction Benchmark.

Evaluates:
- Evidence grounding rate (% of candidate facts with supporting text verified on source page)
- Page attribution accuracy
- Cross-page contamination / context bleed
- Duplicate fact rate
- Entity, metric, numeric, and time-period validity
"""

from typing import List, Dict, Any, Set, Tuple
from backend.models.fact import FactRecord
from backend.models.document import PageText


class QualityEvaluator:
    """Evaluates extraction quality dimensions across benchmark runs."""

    @staticmethod
    def evaluate_grounding_and_attribution(
        facts: List[FactRecord],
        pages_map: Dict[int, PageText],
        raw_candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculates grounding rate, page attribution accuracy, and duplicate rate.
        """
        total_raw = len(raw_candidates)
        verified_count = len(facts)
        
        # Grounding rate
        grounding_rate = (verified_count / total_raw) if total_raw > 0 else 1.0
        
        # Check duplicate rate among verified facts
        # Duplicate coordinate: (entity.lower(), metric.lower(), time_period.label.lower())
        unique_keys: Set[Tuple[str, str, str]] = set()
        duplicate_count = 0
        
        valid_entity_count = 0
        valid_metric_count = 0
        valid_numeric_count = 0
        valid_time_period_count = 0
        correct_page_attribution_count = 0
        
        for f in facts:
            # Check attribute presence
            if f.entity and len(f.entity.strip()) > 1:
                valid_entity_count += 1
            if f.metric and len(f.metric.strip()) > 1:
                valid_metric_count += 1
            if f.value_numeric is not None or f.value_raw:
                valid_numeric_count += 1
            if f.time_period and f.time_period.label:
                valid_time_period_count += 1
                
            # Attribution check: Is supporting text found on provenance_page?
            p_num = f.provenance.page_number
            if p_num in pages_map:
                if f.provenance.supporting_text.strip() in pages_map[p_num].text:
                    correct_page_attribution_count += 1
                    
            key = (f.entity.strip().lower(), f.metric.strip().lower(), f.time_period.label.strip().lower())
            if key in unique_keys:
                duplicate_count += 1
            else:
                unique_keys.add(key)
                
        dup_rate = (duplicate_count / verified_count) if verified_count > 0 else 0.0
        attribution_acc = (correct_page_attribution_count / verified_count) if verified_count > 0 else 1.0
        
        return {
            "total_raw_candidates": total_raw,
            "verified_facts_count": verified_count,
            "grounding_rate": grounding_rate,
            "page_attribution_accuracy": attribution_acc,
            "duplicate_count": duplicate_count,
            "duplicate_rate": dup_rate,
            "valid_entity_rate": (valid_entity_count / verified_count) if verified_count > 0 else 1.0,
            "valid_metric_rate": (valid_metric_count / verified_count) if verified_count > 0 else 1.0,
            "valid_numeric_rate": (valid_numeric_count / verified_count) if verified_count > 0 else 1.0,
            "valid_time_period_rate": (valid_time_period_count / verified_count) if verified_count > 0 else 1.0,
        }
