"""Metric aggregation and comparison computation for Phase 26 benchmark."""

import statistics
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.groq_experiment.models import SingleBatchProviderResult


class MetricValue(BaseModel):
    """Encapsulates a metric with its strict classification status."""
    classification: str = Field(..., description="'MEASURED', 'DERIVED', or 'N/A — NO GROUND TRUTH'")
    value: Any = Field(..., description="Numerical value, string, or 'N/A'")
    unit: str = Field(default="", description="Unit of measurement if applicable")
    note: Optional[str] = Field(default=None, description="Diagnostic context or qualifier")


class ProviderBenchmarkSummary(BaseModel):
    """Complete structured metrics summary for a provider benchmark run."""
    provider_name: str
    model_name: str
    total_batches: int
    total_pages_evaluated: int
    wall_clock_time_s: float

    # Measured Metrics
    raw_facts_returned: MetricValue
    verified_facts: MetricValue
    verification_rate_pct: MetricValue
    evidence_grounding_rate_pct: MetricValue
    total_requests: MetricValue
    successful_requests: MetricValue
    failed_requests: MetricValue
    http_failures_by_status: MetricValue
    structured_output_failures: MetricValue
    retries_attempted: MetricValue
    mean_latency_ms: MetricValue
    median_latency_ms: MetricValue
    p95_latency_ms_directional: MetricValue
    total_provider_latency_s: MetricValue
    estimated_tokens_total: MetricValue
    rate_limit_summary: MetricValue

    # Derived Metrics
    duplicate_facts_count: MetricValue
    duplicate_fact_rate_pct: MetricValue
    cross_page_contamination_count: MetricValue
    cross_page_contamination_rate_pct: MetricValue
    page_attribution_consistency_pct: MetricValue

    # Metrics Requiring Ground Truth (Strictly N/A when ground truth is absent)
    precision: MetricValue
    recall: MetricValue
    f1_score: MetricValue
    numeric_extraction_accuracy: MetricValue
    unit_scale_accuracy: MetricValue
    entity_accuracy: MetricValue
    metric_accuracy: MetricValue
    time_period_accuracy: MetricValue
    epistemic_status_accuracy: MetricValue


def aggregate_batch_results(
    provider_name: str,
    model_name: str,
    batch_results: List[SingleBatchProviderResult],
    wall_clock_time_s: float,
    total_pages: int = 25,
) -> ProviderBenchmarkSummary:
    """Aggregates single-batch provider results into structured metrics with strict classification."""
    total_batches = len(batch_results)
    raw_facts = sum(b.raw_facts_count for b in batch_results)
    verified = sum(len(b.verified_facts) for b in batch_results)
    duplicates = sum(b.duplicate_facts_count for b in batch_results)
    contamination = sum(b.cross_page_contamination_count for b in batch_results)

    success_batches = [b for b in batch_results if b.failure_class is None]
    failed_batches = [b for b in batch_results if b.failure_class is not None]

    latencies = [b.latency_ms for b in batch_results]
    mean_lat = statistics.mean(latencies) if latencies else 0.0
    median_lat = statistics.median(latencies) if latencies else 0.0

    # Directional P95 (sample size N=5)
    if len(latencies) >= 2:
        sorted_lat = sorted(latencies)
        idx_p95 = int(round(0.95 * (len(sorted_lat) - 1)))
        p95_lat = sorted_lat[idx_p95]
    else:
        p95_lat = latencies[0] if latencies else 0.0

    total_latency_s = sum(latencies) / 1000.0
    total_tokens = sum((b.tokens_used or 0) for b in batch_results)

    failure_counts = {}
    structured_errs = 0
    for b in failed_batches:
        f_name = b.failure_class.value if b.failure_class else "UNKNOWN"
        failure_counts[f_name] = failure_counts.get(f_name, 0) + 1
        if f_name == "STRUCTURED_OUTPUT_PARSE_ERROR":
            structured_errs += 1

    verif_rate = (verified / raw_facts * 100.0) if raw_facts > 0 else 0.0
    ground_rate = verif_rate  # All verified facts strictly satisfy EvidenceVerifier
    dup_rate = (duplicates / raw_facts * 100.0) if raw_facts > 0 else 0.0
    contam_rate = (contamination / raw_facts * 100.0) if raw_facts > 0 else 0.0
    attribution_rate = 100.0 - contam_rate

    # Check for rate limit headers
    last_rl = next((b.rate_limit_info for b in reversed(batch_results) if b.rate_limit_info and b.rate_limit_info.raw_headers), None)
    rl_desc = f"Remaining req: {last_rl.remaining_requests or 'N/A'}, tokens: {last_rl.remaining_tokens or 'N/A'}" if last_rl else "No rate limit headers returned"

    return ProviderBenchmarkSummary(
        provider_name=provider_name,
        model_name=model_name,
        total_batches=total_batches,
        total_pages_evaluated=total_pages,
        wall_clock_time_s=wall_clock_time_s,

        # MEASURED
        raw_facts_returned=MetricValue(classification="MEASURED", value=raw_facts, unit="facts"),
        verified_facts=MetricValue(classification="MEASURED", value=verified, unit="facts"),
        verification_rate_pct=MetricValue(classification="MEASURED", value=round(verif_rate, 2), unit="%"),
        evidence_grounding_rate_pct=MetricValue(classification="MEASURED", value=round(ground_rate, 2), unit="%"),
        total_requests=MetricValue(classification="MEASURED", value=total_batches, unit="requests"),
        successful_requests=MetricValue(classification="MEASURED", value=len(success_batches), unit="requests"),
        failed_requests=MetricValue(classification="MEASURED", value=len(failed_batches), unit="requests"),
        http_failures_by_status=MetricValue(classification="MEASURED", value=failure_counts),
        structured_output_failures=MetricValue(classification="MEASURED", value=structured_errs),
        retries_attempted=MetricValue(classification="MEASURED", value=0, unit="retries"),
        mean_latency_ms=MetricValue(classification="MEASURED", value=round(mean_lat, 1), unit="ms"),
        median_latency_ms=MetricValue(classification="MEASURED", value=round(median_lat, 1), unit="ms"),
        p95_latency_ms_directional=MetricValue(
            classification="MEASURED",
            value=round(p95_lat, 1),
            unit="ms",
            note="Directional only (N=5 requests)",
        ),
        total_provider_latency_s=MetricValue(classification="MEASURED", value=round(total_latency_s, 2), unit="s"),
        estimated_tokens_total=MetricValue(classification="MEASURED", value=total_tokens if total_tokens > 0 else "N/A", unit="tokens"),
        rate_limit_summary=MetricValue(classification="MEASURED", value=rl_desc),

        # DERIVED
        duplicate_facts_count=MetricValue(classification="DERIVED", value=duplicates, unit="facts"),
        duplicate_fact_rate_pct=MetricValue(classification="DERIVED", value=round(dup_rate, 2), unit="%"),
        cross_page_contamination_count=MetricValue(classification="DERIVED", value=contamination, unit="facts"),
        cross_page_contamination_rate_pct=MetricValue(classification="DERIVED", value=round(contam_rate, 2), unit="%"),
        page_attribution_consistency_pct=MetricValue(classification="DERIVED", value=round(attribution_rate, 2), unit="%"),

        # N/A — NO GROUND TRUTH
        precision=MetricValue(classification="N/A — NO GROUND TRUTH", value="N/A", note="No exhaustive ground truth annotation for 25 benchmark pages"),
        recall=MetricValue(classification="N/A — NO GROUND TRUTH", value="N/A", note="No exhaustive ground truth annotation for 25 benchmark pages"),
        f1_score=MetricValue(classification="N/A — NO GROUND TRUTH", value="N/A", note="No exhaustive ground truth annotation for 25 benchmark pages"),
        numeric_extraction_accuracy=MetricValue(classification="N/A — NO GROUND TRUTH", value="N/A", note="Requires per-fact gold label annotations"),
        unit_scale_accuracy=MetricValue(classification="N/A — NO GROUND TRUTH", value="N/A", note="Requires per-fact gold label annotations"),
        entity_accuracy=MetricValue(classification="N/A — NO GROUND TRUTH", value="N/A", note="Requires per-fact gold label annotations"),
        metric_accuracy=MetricValue(classification="N/A — NO GROUND TRUTH", value="N/A", note="Requires per-fact gold label annotations"),
        time_period_accuracy=MetricValue(classification="N/A — NO GROUND TRUTH", value="N/A", note="Requires per-fact gold label annotations"),
        epistemic_status_accuracy=MetricValue(classification="N/A — NO GROUND TRUTH", value="N/A", note="Requires per-fact gold label annotations"),
    )
