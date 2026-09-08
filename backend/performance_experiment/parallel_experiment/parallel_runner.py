"""
Parallel vs Sequential Gemini Extraction Benchmark Runner (Phase 22).

Evaluates bounded concurrency (workers=1 vs workers=2) across:
- Wall-clock analysis latency
- Individual Gemini request latency
- Evidence grounding rate & page attribution accuracy
- Cross-batch isolation & deterministic result assembly
- Downstream normalization and relationship evaluation
"""

import os
import sys
import time
import math
import json
import statistics
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
load_dotenv(override=True)

import httpx

from backend.extraction.pdf_parser import PDFParser
from backend.page_filter.relevance import PageRelevanceFilter
from backend.context_selector.selector import ContextSelector
from backend.quota_experiment.planner import QuotaPlanner
from backend.extraction.fact_extractor import (
    FactExtractor,
    BATCH_EXTRACTION_SYSTEM_PROMPT,
    BATCH_EXTRACTION_JSON_SCHEMA,
)
from backend.models.document import ParsedDocument, PageText
from backend.models.fact import FactRecord, Provenance, TimePeriod, EpistemicStatus
from backend.extraction.evidence import EvidenceVerifier, create_evidence
from backend.normalization.normalizer import FactNormalizer
from backend.reasoning.matcher import CandidateMatcher
from backend.reasoning.comparability import ComparabilityGate
from backend.reasoning.relationships import RelationshipEngine
from backend.performance_experiment.quality_evaluator import QualityEvaluator


class SingleBatchTaskResult:
    """Isolated extraction result for a single batch."""
    def __init__(
        self,
        batch_index: int,
        page_numbers: List[int],
        start_time: float,
        end_time: float,
        latency_seconds: float,
        prompt_characters: int,
        estimated_input_tokens: int,
        status_code: int,
        success: bool,
        retries_used: int,
        raw_facts: List[Dict[str, Any]],
        verified_facts: List[FactRecord],
        rejected_count: int,
        output_size_bytes: int,
        error_message: Optional[str] = None,
    ):
        self.batch_index = batch_index
        self.page_numbers = page_numbers
        self.start_time = start_time
        self.end_time = end_time
        self.latency_seconds = latency_seconds
        self.prompt_characters = prompt_characters
        self.estimated_input_tokens = estimated_input_tokens
        self.status_code = status_code
        self.success = success
        self.retries_used = retries_used
        self.raw_facts = raw_facts
        self.verified_facts = verified_facts
        self.rejected_count = rejected_count
        self.output_size_bytes = output_size_bytes
        self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "page_numbers": self.page_numbers,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "latency_seconds": round(self.latency_seconds, 3),
            "prompt_characters": self.prompt_characters,
            "estimated_input_tokens": self.estimated_input_tokens,
            "status_code": self.status_code,
            "success": self.success,
            "retries_used": self.retries_used,
            "raw_facts_count": len(self.raw_facts),
            "raw_facts": self.raw_facts,
            "verified_facts_count": len(self.verified_facts),
            "verified_facts": [f.model_dump() for f in self.verified_facts],
            "rejected_count": self.rejected_count,
            "output_size_bytes": self.output_size_bytes,
            "error_message": self.error_message,
        }


class ParallelBenchmarkRunner:
    """Orchestrates isolated sequential (workers=1) and parallel (workers=2) runs."""

    def __init__(
        self,
        pdf_path: str = r"C:\Users\vaish\Downloads\APRIL IMF.pdf",
        batch_size: int = 5,
        target_batches_count: int = 4,
    ):
        self.pdf_path = pdf_path
        self.batch_size = batch_size
        self.target_batches_count = target_batches_count
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.parser = PDFParser()
        self.page_filter = PageRelevanceFilter(threshold=0.30)
        self.context_selector = ContextSelector(expansion_radius=2)
        self.planner = QuotaPlanner(default_batch_size=batch_size)

    def run_experiment(self) -> Dict[str, Any]:
        """Runs sequential vs parallel benchmark across identical batches."""
        print("=" * 80)
        print("PHASE 22 — PARALLEL EXTRACTION BENCHMARK (WORKERS 1 vs 2)")
        print("=" * 80)

        # 1. Parse and filter
        with open(self.pdf_path, "rb") as f:
            pdf_bytes = f.read()

        doc = self.parser.parse_bytes(pdf_bytes, os.path.basename(self.pdf_path))
        text_pages = [p for p in doc.pages if p.has_text and p.text.strip()]
        filter_results = self.page_filter.filter_pages(text_pages)

        scores_map = {fr.page_number: fr.relevance_score for fr in filter_results}
        all_eligible = [fr.page_number for fr in filter_results if fr.selected]
        page_lookup = {p.page_number: p for p in doc.pages}

        plan = self.planner.calculate_plan(
            eligible_pages=all_eligible,
            request_budget=None,
            batch_size=self.batch_size,
            page_relevance_scores=scores_map,
        )

        target_batches = plan.batches[: self.target_batches_count]
        target_pages = [p for b in target_batches for p in b]
        print(f"Target Batches ({len(target_batches)}): {target_batches}")
        print(f"Target Eligible Pages ({len(target_pages)}): {target_pages}")

        # Precompute context-selected prompt items for exact fairness
        extractor = FactExtractor(batch_size=self.batch_size)
        precomputed_prompts: List[Tuple[int, List[int], str]] = []
        for b_idx, batch_p_nums in enumerate(target_batches):
            batch_items = []
            for p_num in batch_p_nums:
                p = page_lookup[p_num]
                c_res = self.context_selector.select_page_context(p.text, p.page_number, doc.document_id)
                p_text = c_res.combined_source_text if not c_res.is_empty else p.text
                batch_items.append((p_num, p_text))
            prompt_text = extractor.format_batch_prompt(batch_items, doc.document_name)
            precomputed_prompts.append((b_idx + 1, batch_p_nums, prompt_text))

        # 2. Sequential Run (workers=1)
        print("\n" + "-" * 70)
        print("EXECUTING CONFIGURATION A: SEQUENTIAL (WORKERS = 1)")
        print("-" * 70)
        seq_results = self._run_configuration(
            doc=doc,
            page_lookup=page_lookup,
            prompts_data=precomputed_prompts,
            workers=1,
        )
        print(f"Sequential Wall-Clock Time: {seq_results['total_wall_clock']:.2f}s")
        print(f"Sequential Facts Extracted: {seq_results['verified_facts_count']}")
        print(f"Sequential Relationships: {seq_results['relationships_count']}")

        # Short cool-down pause
        time.sleep(2.0)

        # 3. Parallel Run (workers=2)
        print("\n" + "-" * 70)
        print("EXECUTING CONFIGURATION B: PARALLEL (WORKERS = 2)")
        print("-" * 70)
        par_results = self._run_configuration(
            doc=doc,
            page_lookup=page_lookup,
            prompts_data=precomputed_prompts,
            workers=2,
        )
        print(f"Parallel Wall-Clock Time: {par_results['total_wall_clock']:.2f}s")
        print(f"Parallel Facts Extracted: {par_results['verified_facts_count']}")
        print(f"Parallel Relationships: {par_results['relationships_count']}")

        # Speed metrics
        seq_wall = seq_results["total_wall_clock"]
        par_wall = par_results["total_wall_clock"]
        speedup = seq_wall / par_wall if par_wall > 0 else 1.0
        reduction = (1.0 - (par_wall / seq_wall)) if seq_wall > 0 else 0.0

        print("\n" + "=" * 80)
        print("SPEEDUP COMPARISON")
        print("=" * 80)
        print(f"Sequential Wall Time : {seq_wall:.2f}s")
        print(f"Parallel Wall Time   : {par_wall:.2f}s")
        print(f"Measured Speedup     : {speedup:.2f}x")
        print(f"Wall-Clock Reduction : {reduction * 100:.1f}%")

        out_data = {
            "document": doc.document_name,
            "target_batches": target_batches,
            "target_pages": target_pages,
            "sequential_workers_1": seq_results,
            "parallel_workers_2": par_results,
            "comparison": {
                "sequential_wall_time_sec": round(seq_wall, 2),
                "parallel_wall_time_sec": round(par_wall, 2),
                "speedup_ratio": round(speedup, 2),
                "wall_clock_reduction_pct": round(reduction * 100, 1),
            },
        }

        out_path = os.path.join(os.path.dirname(__file__), "parallel_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nResults saved to {out_path}")

        return out_data

    def _execute_single_batch(
        self,
        batch_index: int,
        page_numbers: List[int],
        prompt_text: str,
        doc_id: str,
        doc_name: str,
        page_lookup: Dict[int, PageText],
    ) -> SingleBatchTaskResult:
        """Executes a single isolated batch request with full provenance and timing."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "systemInstruction": {"parts": [{"text": BATCH_EXTRACTION_SYSTEM_PROMPT}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": BATCH_EXTRACTION_JSON_SCHEMA,
                "temperature": 0.0,
            },
        }

        t_start = time.perf_counter()
        retries = 0
        call_success = False
        raw_facts: List[Dict[str, Any]] = []
        status_code = 200
        output_bytes = 0
        error_msg = None

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
                            raw_text = parts[0].get("text", "{}")
                            output_bytes = len(raw_text.encode("utf-8"))
                            parsed_json = json.loads(raw_text)
                            raw_facts = parsed_json.get("facts", [])
                        call_success = True
                        break
                    elif resp.status_code == 429:
                        error_msg = "HTTP 429 Quota Exhausted"
                        break
                    elif resp.status_code == 503:
                        if attempt < max_attempts - 1:
                            retries += 1
                            time.sleep(1.0 * (2 ** attempt))
                            continue
                        error_msg = "HTTP 503 Service Unavailable (3 attempts exhausted)"
                        break
                    else:
                        error_msg = f"HTTP {resp.status_code}"
                        break
                except Exception as e:
                    error_msg = f"Exception: {type(e).__name__}: {e}"
                    break

        t_end = time.perf_counter()
        req_latency = t_end - t_start

        # Evidence Verification on this isolated batch
        verified_facts: List[FactRecord] = []
        rejected_count = 0
        orig_page_map = {p_num: page_lookup[p_num] for p_num in page_numbers if p_num in page_lookup}

        for candidate in raw_facts:
            try:
                rep_p_num = candidate.get("page_number")
                if rep_p_num is None and len(page_numbers) == 1:
                    rep_p_num = page_numbers[0]
                if not isinstance(rep_p_num, int) or rep_p_num not in orig_page_map:
                    rejected_count += 1
                    continue

                target_orig_page = orig_page_map[rep_p_num]
                entity = str(candidate.get("entity", "")).strip()
                metric = str(candidate.get("metric", "")).strip()
                value_raw = str(candidate.get("value_raw", "")).strip()
                supporting_text = str(candidate.get("supporting_text", "")).strip()

                if not entity or not metric or not value_raw or not supporting_text:
                    rejected_count += 1
                    continue

                provenance = create_evidence(
                    document_id=doc_id,
                    page_number=rep_p_num,
                    supporting_text=supporting_text,
                )

                if not EvidenceVerifier.verify_provenance(provenance, target_orig_page):
                    rejected_count += 1
                    continue

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
                    document_id=doc_id,
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
                verified_facts.append(fact_rec)
            except Exception:
                rejected_count += 1

        return SingleBatchTaskResult(
            batch_index=batch_index,
            page_numbers=page_numbers,
            start_time=t_start,
            end_time=t_end,
            latency_seconds=req_latency,
            prompt_characters=len(prompt_text),
            estimated_input_tokens=len(prompt_text) // 4,
            status_code=status_code,
            success=call_success,
            retries_used=retries,
            raw_facts=raw_facts,
            verified_facts=verified_facts,
            rejected_count=rejected_count,
            output_size_bytes=output_bytes,
            error_message=error_msg,
        )

    def _run_configuration(
        self,
        doc: ParsedDocument,
        page_lookup: Dict[int, PageText],
        prompts_data: List[Tuple[int, List[int], str]],
        workers: int,
    ) -> Dict[str, Any]:
        """Runs the benchmark under the given worker concurrency."""
        wall_start = time.perf_counter()
        batch_results: List[SingleBatchTaskResult] = []

        if workers == 1:
            for b_idx, p_nums, p_text in prompts_data:
                res = self._execute_single_batch(
                    batch_index=b_idx,
                    page_numbers=p_nums,
                    prompt_text=p_text,
                    doc_id=doc.document_id,
                    doc_name=doc.document_name,
                    page_lookup=page_lookup,
                )
                batch_results.append(res)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_batch = {
                    executor.submit(
                        self._execute_single_batch,
                        b_idx,
                        p_nums,
                        p_text,
                        doc.document_id,
                        doc.document_name,
                        page_lookup,
                    ): b_idx
                    for b_idx, p_nums, p_text in prompts_data
                }
                for future in as_completed(future_to_batch):
                    batch_results.append(future.result())

        # Sort deterministically by batch_index to guarantee order preservation
        batch_results.sort(key=lambda r: r.batch_index)
        wall_time = time.perf_counter() - wall_start

        # Assemble verified facts deterministically
        all_verified_facts: List[FactRecord] = []
        all_raw_facts: List[Dict[str, Any]] = []
        total_rejected = 0
        successful_batches = 0
        failed_batches = 0
        retries_count = 0
        c_429 = 0
        c_503 = 0

        for r in batch_results:
            if r.success:
                successful_batches += 1
                all_verified_facts.extend(r.verified_facts)
                all_raw_facts.extend(r.raw_facts)
            else:
                failed_batches += 1
                if r.status_code == 429:
                    c_429 += 1
                elif r.status_code == 503:
                    c_503 += 1
            total_rejected += r.rejected_count
            retries_count += r.retries_used

        # Downstream reasoning pipeline
        t0 = time.perf_counter()
        normalizer = FactNormalizer()
        normalized_facts = normalizer.normalize_batch(all_verified_facts)

        matcher = CandidateMatcher()
        candidate_tuples = matcher.find_candidates(normalized_facts)

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

        reasoning_time = time.perf_counter() - t0

        # Concurrency analysis for workers > 1
        intervals = [(r.start_time, r.end_time) for r in batch_results]
        max_concurrent = 1
        if workers > 1 and len(intervals) >= 2:
            # check overlaps
            for i in range(len(intervals)):
                active = 1
                for j in range(len(intervals)):
                    if i != j and not (intervals[i][1] < intervals[j][0] or intervals[i][0] > intervals[j][1]):
                        active += 1
                if active > max_concurrent:
                    max_concurrent = active

        # Quality evaluation
        quality = QualityEvaluator.evaluate_grounding_and_attribution(
            facts=all_verified_facts,
            pages_map=page_lookup,
            raw_candidates=all_raw_facts,
        )

        latencies = [r.latency_seconds for r in batch_results]
        avg_lat = statistics.mean(latencies) if latencies else 0.0

        return {
            "workers": workers,
            "total_wall_clock": round(wall_time, 3),
            "gemini_cumulative_active_time": round(sum(latencies), 3),
            "average_request_latency": round(avg_lat, 3),
            "max_concurrent_observed": max_concurrent,
            "requests_count": len(batch_results),
            "successful_requests": successful_batches,
            "failed_requests": failed_batches,
            "count_429": c_429,
            "count_503": c_503,
            "retries_count": retries_count,
            "processed_pages_count": len([p for r in batch_results if r.success for p in r.page_numbers]),
            "raw_facts_count": len(all_raw_facts),
            "verified_facts_count": len(all_verified_facts),
            "rejected_grounding_count": total_rejected,
            "candidate_pairs_count": len(candidate_tuples),
            "relationships_count": len(relationships),
            "grounding_rate": quality["grounding_rate"],
            "page_attribution_accuracy": quality["page_attribution_accuracy"],
            "duplicate_rate": quality["duplicate_rate"],
            "reasoning_time_seconds": round(reasoning_time, 4),
            "requests_detail": [r.to_dict() for r in batch_results],
        }


if __name__ == "__main__":
    runner = ParallelBenchmarkRunner()
    runner.run_experiment()
