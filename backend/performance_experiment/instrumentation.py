"""
Performance Instrumentation for FACTLINE Extraction & Reasoning Pipeline.

Measures granular execution latencies across:
1. PDF parsing
2. Page relevance filtering
3. Context selection / compression
4. Gemini HTTP requests (per-request latency, retries, medians, max)
5. Evidence verification
6. Normalization
7. Candidate matching
8. Relationship evaluation
9. Persistence
10. Total wall-clock analysis time
"""

import time
import statistics
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GeminiRequestMetric(BaseModel):
    """Metrics for a single Gemini HTTP request."""
    batch_index: int
    page_numbers: List[int]
    character_count: int
    latency_seconds: float
    status_code: int
    success: bool
    retries_used: int = 0
    facts_extracted_raw: int = 0


class StageTimingBreakdown(BaseModel):
    """Time spent in each pipeline stage in seconds."""
    pdf_parsing_seconds: float = 0.0
    page_filtering_seconds: float = 0.0
    context_selection_seconds: float = 0.0
    gemini_request_seconds: float = 0.0
    evidence_verification_seconds: float = 0.0
    normalization_seconds: float = 0.0
    candidate_matching_seconds: float = 0.0
    relationship_generation_seconds: float = 0.0
    persistence_seconds: float = 0.0
    total_wall_clock_seconds: float = 0.0


class GeminiAggregateMetrics(BaseModel):
    """Aggregate statistics for Gemini API calls."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_retries: int = 0
    average_request_latency_seconds: float = 0.0
    median_request_latency_seconds: float = 0.0
    max_request_latency_seconds: float = 0.0
    min_request_latency_seconds: float = 0.0
    total_gemini_wall_clock_seconds: float = 0.0


class BenchmarkRunResult(BaseModel):
    """Complete results from a single benchmark configuration run."""
    batch_size: int
    total_pdf_pages: int
    eligible_pages_count: int
    eligible_pages: List[int]
    processed_pages_count: int
    processed_pages: List[int]
    unprocessed_pages_count: int
    unprocessed_pages: List[int]
    status: str
    
    # Counts
    raw_facts_extracted: int = 0
    verified_facts_count: int = 0
    rejected_grounding_count: int = 0
    candidate_pairs_count: int = 0
    relationships_count: int = 0
    
    # Performance & Timing
    timings: StageTimingBreakdown
    gemini_metrics: GeminiAggregateMetrics
    individual_gemini_requests: List[GeminiRequestMetric] = Field(default_factory=list)
    
    # Quality metrics
    evidence_grounding_rate: float = 0.0
    page_attribution_accuracy: float = 0.0
    duplicate_fact_rate: float = 0.0
    error_message: Optional[str] = None
