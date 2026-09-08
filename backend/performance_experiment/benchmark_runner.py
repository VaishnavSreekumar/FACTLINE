"""
Isolated Speed & Quality Benchmark Runner for FACTLINE (Phase 19).

Measures granular pipeline latencies and extraction quality across
batch sizes 3, 5, 8, and 10 on the April IMF PDF.
"""

import os
import sys
import time
import math
import json
import sqlite3
import statistics
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath("."))

import httpx
from dotenv import load_dotenv

load_dotenv()

from backend.extraction.pdf_parser import PDFParser
from backend.page_filter.relevance import PageRelevanceFilter
from backend.context_selector.selector import ContextSelector
from backend.quota_experiment.planner import QuotaPlanner
from backend.extraction.fact_extractor import (
    FactExtractor,
    BATCH_EXTRACTION_SYSTEM_PROMPT,
    BATCH_EXTRACTION_JSON_SCHEMA,
    ExtractionError,
    ExtractionQuotaError,
)
from backend.models.document import ParsedDocument, PageText
from backend.models.fact import FactRecord, Provenance, TimePeriod, EpistemicStatus
from backend.extraction.evidence import EvidenceVerifier, create_evidence
from backend.normalization.normalizer import FactNormalizer
from backend.reasoning.matcher import CandidateMatcher
from backend.reasoning.comparability import ComparabilityGate
from backend.reasoning.relationships import RelationshipEngine
from backend.db.database import DatabaseRepository
from backend.performance_experiment.instrumentation import (
    BenchmarkRunResult,
    StageTimingBreakdown,
    GeminiAggregateMetrics,
    GeminiRequestMetric,
)
from backend.performance_experiment.quality_evaluator import QualityEvaluator


class PerformanceBenchmarkRunner:
    """Orchestrates isolated speed and quality benchmarks across different Gemini batch sizes."""

    def __init__(
        self,
        pdf_path: str = r"C:\Users\vaish\Downloads\APRIL IMF.pdf",
        eligible_page_limit: int = 30,
        gemini_model: Optional[str] = None,
    ):
        self.pdf_path = pdf_path
        self.eligible_page_limit = eligible_page_limit
        self.model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.api_key = os.getenv("GEMINI_API_KEY", "")

    def run_benchmark(self, batch_sizes: List[int] = [3, 5, 8, 10]) -> Dict[str, Any]:
        """Runs the benchmark across all target batch sizes on the exact same eligible pages."""
        print("=" * 80)
        print("PHASE 19 — EXTRACTION SPEED & QUALITY BENCHMARK")
        print("=" * 80)
        print(f"Document Path: {self.pdf_path}")
        print(f"Gemini Model: {self.model}")
        print(f"Eligible Pages to Process per Run: {self.eligible_page_limit}")
        print(f"Batch Sizes to Benchmark: {batch_sizes}")

        # Stage 1: Measure PDF Parsing Time
        t0 = time.perf_counter()
        parser = PDFParser()
        with open(self.pdf_path, "rb") as f:
            pdf_bytes = f.read()
        parsed_doc = parser.parse_bytes(content=pdf_bytes, document_name=os.path.basename(self.pdf_path))
        pdf_parsing_time = time.perf_counter() - t0
        print(f"\n[Stage 1] PDF Parsing Time: {pdf_parsing_time:.3f}s ({parsed_doc.total_pages} total pages)")

        # Stage 2: Measure Page Relevance Filtering Time
        t0 = time.perf_counter()
        page_filter = PageRelevanceFilter(threshold=0.30)
        text_pages = [p for p in parsed_doc.pages if p.has_text and p.text.strip()]
        filter_results = page_filter.filter_pages(text_pages)
        page_filtering_time = time.perf_counter() - t0

        scores_map = {fr.page_number: fr.relevance_score for fr in filter_results}
        all_eligible_pages = [fr.page_number for fr in filter_results if fr.selected]
        print(f"[Stage 2] Page Filtering Time: {page_filtering_time:.3f}s ({len(all_eligible_pages)} eligible pages found)")

        # Select fixed, identical subset of eligible pages for FAIR comparison
        benchmark_eligible_pages = all_eligible_pages[: self.eligible_page_limit]
        print(f"Benchmark Test Set: {len(benchmark_eligible_pages)} pages: {benchmark_eligible_pages}")

        results: Dict[int, BenchmarkRunResult] = {}

        for bs in batch_sizes:
            print(f"\n" + "-" * 70)
            print(f"RUNNING BENCHMARK: BATCH SIZE = {bs}")
            print("-" * 70)
            run_res = self._execute_single_batch_size_run(
                parsed_doc=parsed_doc,
                eligible_pages=benchmark_eligible_pages,
                scores_map=scores_map,
                batch_size=bs,
                pdf_parsing_time=pdf_parsing_time,
                page_filtering_time=page_filtering_time,
            )
            results[bs] = run_res
            print(f"Batch Size {bs} Summary:")
            print(f"  Total Wall-Clock: {run_res.timings.total_wall_clock_seconds:.2f}s")
            print(f"  Gemini Requests: {run_res.gemini_metrics.total_requests}")
            print(f"  Gemini Total Time: {run_res.gemini_metrics.total_gemini_wall_clock_seconds:.2f}s")
            print(f"  Average Request Latency: {run_res.gemini_metrics.average_request_latency_seconds:.2f}s")
            print(f"  Verified Facts: {run_res.verified_facts_count} (Raw: {run_res.raw_facts_extracted})")
            print(f"  Grounding Rate: {run_res.evidence_grounding_rate * 100:.1f}%")
            print(f"  Relationships Created: {run_res.relationships_count}")

            # Small pause between runs to avoid burst rate limits
            time.sleep(1.0)

        # Stage Projections
        projections = self._calculate_projections(results)

        output_data = {
            "document": parsed_doc.document_name,
            "total_pages": parsed_doc.total_pages,
            "benchmark_pages_count": len(benchmark_eligible_pages),
            "benchmark_pages": benchmark_eligible_pages,
            "results": {str(k): v.model_dump() for k, v in results.items()},
            "projections": projections,
        }

        # Save to benchmark_results.json
        out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n[Complete] Benchmark results saved to {out_path}")

        return output_data

    def _execute_single_batch_size_run(
        self,
        parsed_doc: ParsedDocument,
        eligible_pages: List[int],
        scores_map: Dict[int, float],
        batch_size: int,
        pdf_parsing_time: float,
        page_filtering_time: float,
    ) -> BenchmarkRunResult:
        """Executes full pipeline for a specific batch size with isolated instrumentation."""
        total_wall_start = time.perf_counter()

        page_lookup: Dict[int, PageText] = {p.page_number: p for p in parsed_doc.pages}
        context_selector = ContextSelector(expansion_radius=2)
        planner = QuotaPlanner(default_batch_size=batch_size)

        # Plan batches
        plan = planner.calculate_plan(
            eligible_pages=eligible_pages,
            request_budget=None,
            batch_size=batch_size,
            page_relevance_scores=scores_map,
        )

        # Stage 3: Context Selection & Compression Timing
        t0 = time.perf_counter()
        batches_prompt_data: List[List[Tuple[int, str]]] = []
        for batch_p_nums in plan.batches:
            batch_items = []
            for p_num in batch_p_nums:
                p = page_lookup[p_num]
                c_res = context_selector.select_page_context(
                    page_text=p.text,
                    page_number=p.page_number,
                    document_id=parsed_doc.document_id,
                )
                p_text = c_res.combined_source_text if not c_res.is_empty else p.text
                batch_items.append((p_num, p_text))
            batches_prompt_data.append(batch_items)
        context_selection_time = time.perf_counter() - t0

        # Stage 4 & 5: Gemini API Calls and Evidence Verification
        gemini_request_metrics: List[GeminiRequestMetric] = []
        all_raw_candidates: List[Dict[str, Any]] = []
        all_verified_facts: List[FactRecord] = []
        total_rejected_grounding = 0
        total_gemini_time = 0.0
        total_evidence_time = 0.0
        processed_pages: List[int] = []
        run_status = "COMPLETE"
        error_msg = None

        extractor = FactExtractor(
            api_key=self.api_key,
            model=self.model,
            batch_size=batch_size,
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        for b_idx, (batch_p_nums, batch_prompt_items) in enumerate(zip(plan.batches, batches_prompt_data)):
            prompt = extractor.format_batch_prompt(batch_prompt_items, parsed_doc.document_name)
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "systemInstruction": {"parts": [{"text": BATCH_EXTRACTION_SYSTEM_PROMPT}]},
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": BATCH_EXTRACTION_JSON_SCHEMA,
                    "temperature": 0.0,
                },
            }

            req_start = time.perf_counter()
            retries = 0
            call_success = False
            raw_facts_in_batch: List[Dict[str, Any]] = []
            status_code = 200

            with httpx.Client(timeout=60.0) as client:
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        resp = client.post(url, json=payload)
                        status_code = resp.status_code
                        if resp.status_code == 200:
                            data = resp.json()
                            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                            if parts:
                                parsed_json = json.loads(parts[0].get("text", "{}"))
                                raw_facts_in_batch = parsed_json.get("facts", [])
                            call_success = True
                            break
                        elif resp.status_code == 429:
                            run_status = "QUOTA_EXHAUSTED"
                            error_msg = "429 Quota Exhausted"
                            break
                        elif resp.status_code == 503:
                            if attempt < max_attempts - 1:
                                retries += 1
                                backoff = 1.0 * (2 ** attempt)
                                time.sleep(backoff)
                                continue
                            run_status = "FAILED"
                            error_msg = "503 Service Unavailable (3 attempts exhausted)"
                            break
                        else:
                            run_status = "FAILED"
                            error_msg = f"HTTP {resp.status_code}"
                            break
                    except Exception as e:
                        run_status = "FAILED"
                        error_msg = f"Network exception: {e}"
                        break

            req_duration = time.perf_counter() - req_start
            total_gemini_time += req_duration

            gemini_request_metrics.append(
                GeminiRequestMetric(
                    batch_index=b_idx + 1,
                    page_numbers=batch_p_nums,
                    character_count=len(prompt),
                    latency_seconds=req_duration,
                    status_code=status_code,
                    success=call_success,
                    retries_used=retries,
                    facts_extracted_raw=len(raw_facts_in_batch),
                )
            )

            if not call_success:
                break

            processed_pages.extend(batch_p_nums)
            all_raw_candidates.extend(raw_facts_in_batch)

            # Stage 5: Evidence Verification on this batch
            ev_start = time.perf_counter()
            orig_page_map = {p_num: page_lookup[p_num] for p_num in batch_p_nums if p_num in page_lookup}
            for candidate in raw_facts_in_batch:
                try:
                    rep_p_num = candidate.get("page_number")
                    if rep_p_num is None and len(batch_p_nums) == 1:
                        rep_p_num = batch_p_nums[0]
                    if not isinstance(rep_p_num, int) or rep_p_num not in orig_page_map:
                        total_rejected_grounding += 1
                        continue

                    target_orig_page = orig_page_map[rep_p_num]
                    entity = str(candidate.get("entity", "")).strip()
                    metric = str(candidate.get("metric", "")).strip()
                    value_raw = str(candidate.get("value_raw", "")).strip()
                    supporting_text = str(candidate.get("supporting_text", "")).strip()

                    if not entity or not metric or not value_raw or not supporting_text:
                        total_rejected_grounding += 1
                        continue

                    provenance = create_evidence(
                        document_id=parsed_doc.document_id,
                        page_number=rep_p_num,
                        supporting_text=supporting_text,
                    )

                    if not EvidenceVerifier.verify_provenance(provenance, target_orig_page):
                        total_rejected_grounding += 1
                        continue

                    # Construct FactRecord
                    raw_status = candidate.get("epistemic_status", "reported")
                    try:
                        ep_status = EpistemicStatus(str(raw_status).lower())
                    except ValueError:
                        ep_status = EpistemicStatus.REPORTED

                    val_num = candidate.get("value_numeric")
                    if val_num is not None:
                        try:
                            val_num = float(val_num)
                        except (ValueError, TypeError):
                            val_num = None

                    unit = candidate.get("unit")
                    unit = str(unit).strip() if unit else None

                    tp_dict = candidate.get("time_period", {})
                    tp_label = str(tp_dict.get("label", "Unspecified")).strip() if isinstance(tp_dict, dict) else "Unspecified"
                    time_period = TimePeriod(label=tp_label)

                    fact_id = FactExtractor.generate_fact_id(
                        document_id=parsed_doc.document_id,
                        page_number=rep_p_num,
                        entity=entity,
                        metric=metric,
                        value_raw=value_raw,
                        time_period_label=time_period.label,
                    )

                    fact_rec = FactRecord(
                        fact_id=fact_id,
                        entity=entity,
                        metric=metric,
                        value_raw=value_raw,
                        value_numeric=val_num,
                        unit=unit,
                        time_period=time_period,
                        epistemic_status=ep_status,
                        provenance=provenance,
                        extraction_confidence=float(candidate.get("extraction_confidence", 0.9)),
                    )
                    all_verified_facts.append(fact_rec)
                except Exception:
                    total_rejected_grounding += 1
            total_evidence_time += (time.perf_counter() - ev_start)

        unprocessed_pages = [p for p in eligible_pages if p not in set(processed_pages)]

        # Stage 6: Normalization
        t0 = time.perf_counter()
        normalizer = FactNormalizer()
        normalized_facts = normalizer.normalize_batch(all_verified_facts)
        normalization_time = time.perf_counter() - t0

        # Stage 7: Candidate Matching
        t0 = time.perf_counter()
        matcher = CandidateMatcher()
        candidate_tuples = matcher.find_candidates(normalized_facts)
        matching_time = time.perf_counter() - t0

        # Stage 8: Relationship Evaluation
        t0 = time.perf_counter()
        gate = ComparabilityGate()
        rel_engine = RelationshipEngine()
        relationships = []
        for fact_a, fact_b in candidate_tuples:
            cand_pair = matcher.match_pair(fact_a, fact_b)
            if cand_pair is None:
                continue
            comp_res = gate.evaluate(fact_a, fact_b)
            rel_res = rel_engine.determine_relationship(
                fact_a=fact_a,
                fact_b=fact_b,
                comparability=comp_res,
                candidate_pair=cand_pair,
            )
            relationships.append(rel_res)
        relationship_time = time.perf_counter() - t0

        # Stage 9: Persistence (using isolated temporary SQLite memory/scratch DB)
        t0 = time.perf_counter()
        temp_db_path = f"scratch/bench_temp_{batch_size}.db"
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except OSError:
                pass
        temp_repo = DatabaseRepository(db_path=temp_db_path)
        temp_repo.save_analysis(
            analysis_id=f"bench-run-bs{batch_size}",
            documents=[parsed_doc],
            normalized_facts=normalized_facts,
            relationships=relationships,
            candidate_pair_count=len(candidate_tuples),
            extraction_status={
                "status": run_status,
                "total_eligible_pages": len(eligible_pages),
                "total_processed_pages": len(processed_pages),
                "total_unprocessed_pages": len(unprocessed_pages),
                "requests_used": len(gemini_request_metrics),
            },
        )
        persistence_time = time.perf_counter() - t0
        # Clean up temp db
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except OSError:
                pass

        total_wall_time = time.perf_counter() - total_wall_start

        # Quality evaluation
        quality = QualityEvaluator.evaluate_grounding_and_attribution(
            facts=all_verified_facts,
            pages_map=page_lookup,
            raw_candidates=all_raw_candidates,
        )

        # Aggregate Gemini metrics
        req_latencies = [m.latency_seconds for m in gemini_request_metrics]
        avg_lat = statistics.mean(req_latencies) if req_latencies else 0.0
        med_lat = statistics.median(req_latencies) if req_latencies else 0.0
        max_lat = max(req_latencies) if req_latencies else 0.0
        min_lat = min(req_latencies) if req_latencies else 0.0
        succ_reqs = sum(1 for m in gemini_request_metrics if m.success)
        fail_reqs = sum(1 for m in gemini_request_metrics if not m.success)
        total_retries = sum(m.retries_used for m in gemini_request_metrics)

        gemini_agg = GeminiAggregateMetrics(
            total_requests=len(gemini_request_metrics),
            successful_requests=succ_reqs,
            failed_requests=fail_reqs,
            total_retries=total_retries,
            average_request_latency_seconds=avg_lat,
            median_request_latency_seconds=med_lat,
            max_request_latency_seconds=max_lat,
            min_request_latency_seconds=min_lat,
            total_gemini_wall_clock_seconds=total_gemini_time,
        )

        timings = StageTimingBreakdown(
            pdf_parsing_seconds=pdf_parsing_time,
            page_filtering_seconds=page_filtering_time,
            context_selection_seconds=context_selection_time,
            gemini_request_seconds=total_gemini_time,
            evidence_verification_seconds=total_evidence_time,
            normalization_seconds=normalization_time,
            candidate_matching_seconds=matching_time,
            relationship_generation_seconds=relationship_time,
            persistence_seconds=persistence_time,
            total_wall_clock_seconds=total_wall_time,
        )

        return BenchmarkRunResult(
            batch_size=batch_size,
            total_pdf_pages=parsed_doc.total_pages,
            eligible_pages_count=len(eligible_pages),
            eligible_pages=eligible_pages,
            processed_pages_count=len(processed_pages),
            processed_pages=processed_pages,
            unprocessed_pages_count=len(unprocessed_pages),
            unprocessed_pages=unprocessed_pages,
            status=run_status,
            raw_facts_extracted=len(all_raw_candidates),
            verified_facts_count=len(all_verified_facts),
            rejected_grounding_count=total_rejected_grounding,
            candidate_pairs_count=len(candidate_tuples),
            relationships_count=len(relationships),
            timings=timings,
            gemini_metrics=gemini_agg,
            individual_gemini_requests=gemini_request_metrics,
            evidence_grounding_rate=quality["grounding_rate"],
            page_attribution_accuracy=quality["page_attribution_accuracy"],
            duplicate_fact_rate=quality["duplicate_rate"],
            error_message=error_msg,
        )

    def _calculate_projections(self, results: Dict[int, BenchmarkRunResult]) -> Dict[str, Any]:
        """Calculates projected request counts and latency for 50, 100, 184, 300, 500 eligible pages."""
        page_milestones = [50, 100, 184, 300, 500]
        projections: Dict[str, Any] = {}

        for bs, res in results.items():
            avg_lat = res.gemini_metrics.average_request_latency_seconds
            bs_projs = {}
            for pages in page_milestones:
                reqs = math.ceil(pages / bs)
                proj_gemini_sec = reqs * avg_lat
                proj_gemini_min = proj_gemini_sec / 60.0
                bs_projs[f"{pages}_pages"] = {
                    "eligible_pages": pages,
                    "batch_size": bs,
                    "projected_requests": reqs,
                    "measured_avg_latency_sec": round(avg_lat, 2),
                    "projected_gemini_seconds": round(proj_gemini_sec, 1),
                    "projected_gemini_minutes": round(proj_gemini_min, 2),
                }
            projections[f"batch_size_{bs}"] = bs_projs

        return projections


if __name__ == "__main__":
    runner = PerformanceBenchmarkRunner()
    runner.run_benchmark(batch_sizes=[3, 5, 8, 10])
