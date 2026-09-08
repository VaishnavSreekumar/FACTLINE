"""
Deterministic Page Relevance Filter for FACTLINE.

Evaluates parsed PDF pages to determine whether they are likely to contain
extractable quantitative facts, reducing LLM calls while maintaining high recall.
"""

from typing import List, Dict, Any, Optional
import os
from pydantic import BaseModel, Field

from backend.models.document import PageText
from backend.page_filter.signals import (
    extract_numeric_signals,
    extract_currency_signals,
    extract_unit_signals,
    extract_metric_signals,
    extract_temporal_signals,
    detect_table_structure,
    detect_boilerplate_signals,
)

# Default threshold favoring recall over precision
DEFAULT_RELEVANCE_THRESHOLD = 0.35


class PageRelevanceScore(BaseModel):
    """Explainable relevance evaluation for a single document page."""

    page_number: int = Field(..., description="1-indexed page number")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Deterministic score [0.0, 1.0]")
    selected: bool = Field(..., description="Whether page meets selection threshold for extraction")
    signals_triggered: List[str] = Field(default_factory=list, description="List of positive signal names triggered")
    signal_breakdown: Dict[str, float] = Field(default_factory=dict, description="Numerical contribution per feature")
    explanation: str = Field(..., description="Human-interpretable explanation of score derivation")


class PageRelevanceFilter:
    """
    Deterministic page relevance scoring filter.
    """

    def __init__(self, threshold: Optional[float] = None):
        if threshold is not None:
            self.threshold = threshold
        else:
            env_thresh = os.getenv("FACTLINE_PAGE_FILTER_THRESHOLD")
            if env_thresh is not None and env_thresh.strip():
                try:
                    self.threshold = float(env_thresh.strip())
                except ValueError:
                    self.threshold = DEFAULT_RELEVANCE_THRESHOLD
            else:
                self.threshold = DEFAULT_RELEVANCE_THRESHOLD

    def score_page(
        self,
        page: PageText,
        prev_page: Optional[PageText] = None,
        next_page: Optional[PageText] = None,
        custom_threshold: Optional[float] = None,
    ) -> PageRelevanceScore:
        """
        Evaluates a single page and computes its explainable relevance score.
        """
        threshold = custom_threshold if custom_threshold is not None else self.threshold

        # 1. Immediate rejection for empty or whitespace-only pages
        if not page.has_text or not page.text or not page.text.strip():
            return PageRelevanceScore(
                page_number=page.page_number,
                relevance_score=0.0,
                selected=False,
                signals_triggered=[],
                signal_breakdown={"empty_page": 0.0},
                explanation="Page contains no extractable text content.",
            )

        text = page.text
        signals_triggered: List[str] = []
        breakdown: Dict[str, float] = {}

        # 2. Extract distinct signals
        num_count, sample_nums = extract_numeric_signals(text)
        currencies = extract_currency_signals(text)
        units = extract_unit_signals(text)
        metrics = extract_metric_signals(text)
        temporals = extract_temporal_signals(text)
        is_table, table_rows = detect_table_structure(text)
        boilerplates = detect_boilerplate_signals(text)

        # 3. Calculate positive signal contributions
        # Numeric density (up to 0.30)
        num_score = min(num_count * 0.05, 0.30)
        if num_count > 0:
            signals_triggered.append(f"numeric_density({num_count})")
            breakdown["numeric_density"] = round(num_score, 3)

        # Currency & monetary terms (up to 0.25)
        curr_score = min(len(currencies) * 0.15, 0.25)
        if currencies:
            signals_triggered.append(f"currency_indicator({','.join(currencies[:3])})")
            breakdown["currency"] = round(curr_score, 3)

        # Quantitative units (up to 0.20)
        unit_score = min(len(units) * 0.10, 0.20)
        if units:
            signals_triggered.append(f"quantitative_unit({','.join(units[:3])})")
            breakdown["unit"] = round(unit_score, 3)

        # Business / macroeconomic metric keywords (up to 0.25)
        metric_score = min(len(metrics) * 0.10, 0.25)
        if metrics:
            signals_triggered.append(f"metric_keyword({','.join(metrics[:3])})")
            breakdown["metric"] = round(metric_score, 3)

        # Table-like structured rows (up to 0.15)
        table_score = 0.15 if is_table else min(table_rows * 0.05, 0.10)
        if table_rows > 0:
            signals_triggered.append(f"tabular_structure({table_rows}_rows)")
            breakdown["table_structure"] = round(table_score, 3)

        # Temporal indicators (up to 0.10)
        temp_score = min(len(temporals) * 0.05, 0.10)
        if temporals:
            signals_triggered.append(f"temporal_marker({','.join(temporals[:2])})")
            breakdown["temporal"] = round(temp_score, 3)

        # 4. Optional neighboring page context boost (up to 0.05)
        neighbor_boost = 0.0
        if num_count >= 2:
            neighbor_has_metric = False
            if prev_page and prev_page.has_text and extract_metric_signals(prev_page.text):
                neighbor_has_metric = True
            elif next_page and next_page.has_text and extract_metric_signals(next_page.text):
                neighbor_has_metric = True

            if neighbor_has_metric and not metrics:
                neighbor_boost = 0.05
                signals_triggered.append("neighbor_metric_context")
                breakdown["neighbor_context"] = 0.05

        # 5. Negative boilerplate dampening (e.g. table of contents or disclaimer with no metrics)
        penalty = 0.0
        if boilerplates and not currencies and not metrics and num_count <= 4:
            penalty = -0.30
            signals_triggered.append(f"boilerplate_penalty({','.join(boilerplates[:2])})")
            breakdown["boilerplate_penalty"] = -0.30

        # 6. Composite Score Derivation
        total_raw = num_score + curr_score + unit_score + metric_score + table_score + temp_score + neighbor_boost + penalty
        final_score = max(0.0, min(1.0, round(total_raw, 3)))
        is_selected = final_score >= threshold

        explanation_parts = [f"Score: {final_score:.3f} (Threshold: {threshold})"]
        if signals_triggered:
            explanation_parts.append(f"Signals: {', '.join(signals_triggered)}")
        else:
            explanation_parts.append("No quantitative signals identified.")

        return PageRelevanceScore(
            page_number=page.page_number,
            relevance_score=final_score,
            selected=is_selected,
            signals_triggered=signals_triggered,
            signal_breakdown=breakdown,
            explanation="; ".join(explanation_parts),
        )

    def filter_pages(
        self,
        pages: List[PageText],
        threshold: Optional[float] = None,
    ) -> List[PageRelevanceScore]:
        """
        Scores all pages in a document and returns detailed score reports.
        """
        results: List[PageRelevanceScore] = []
        n = len(pages)
        for i, page in enumerate(pages):
            prev_p = pages[i - 1] if i > 0 else None
            next_p = pages[i + 1] if i < n - 1 else None
            score = self.score_page(page, prev_page=prev_p, next_page=next_p, custom_threshold=threshold)
            results.append(score)
        return results

    def get_selected_pages(
        self,
        pages: List[PageText],
        threshold: Optional[float] = None,
    ) -> List[PageText]:
        """
        Returns the subset of PageText objects that pass the relevance filter threshold.
        """
        scored = self.filter_pages(pages, threshold=threshold)
        selected_numbers = {s.page_number for s in scored if s.selected}
        return [p for p in pages if p.page_number in selected_numbers]
