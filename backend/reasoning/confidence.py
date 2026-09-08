"""Confidence scoring calculation interface (Placeholder)."""

from typing import List
from backend.models.fact import FactRecord


class ConfidenceCalculator:
    """Calculates extraction and reasoning confidence scores."""

    def compute_pair_confidence(self, fact_a: FactRecord, fact_b: FactRecord, contextual_factors: List[str]) -> float:
        """Computes confidence score based on extraction confidence and dimension alignment.
        
        To be implemented in reasoning milestone.
        """
        raise NotImplementedError("Confidence calculator is not implemented yet.")
