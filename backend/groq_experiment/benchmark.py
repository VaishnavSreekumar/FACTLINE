"""Phase 26 Benchmark Runner: Gemini vs Groq on Fixed Diverse 25-Page Corpus."""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.context_selector.selector import ContextSelector
from backend.extraction.evidence import EvidenceVerifier, create_evidence
from backend.extraction.fact_extractor import FactExtractor
from backend.extraction.pdf_parser import PDFParser
from backend.groq_experiment.extractor import GroqFactExtractor, compute_semantic_input_hash
from backend.groq_experiment.metrics import (
    MetricValue,
    ProviderBenchmarkSummary,
    aggregate_batch_results,
)
from backend.groq_experiment.models import SingleBatchProviderResult
from backend.groq_experiment.provider import GroqProviderClient
from backend.models.document import PageText, ParsedDocument
from backend.models.fact import EpistemicStatus, FactRecord, TimePeriod


# Fixed 25-Page Deterministic Benchmark Dataset across 5 diverse categories
BENCHMARK_BATCH_DEFINITIONS = [
    {
        "batch_id": "batch_1_prose_macro",
        "description": "Category 1: Prose-Heavy Macroeconomic Narrative & Policy",
        "document_path": "sample-data/starter-datasets/india-macroeconomy/01-india-economic-survey-2024-25-excerpt.pdf",
        "document_name": "01-india-economic-survey-2024-25-excerpt.pdf",
        "page_numbers": [4, 5, 9, 12, 13],
    },
    {
        "batch_id": "batch_2_dense_numerical",
        "description": "Category 2: Dense Numerical & Operational Scale Highlights",
        "document_path": "sample-data/starter-datasets/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf",
        "document_name": "02-delhivery-annual-report-fy24-excerpt.pdf",
        "page_numbers": [2, 4, 6, 8, 13],
    },
    {
        "batch_id": "batch_3_tables_negative",
        "description": "Category 3: Tables, Statistical Splits, and Negative Values",
        "document_path": "sample-data/starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf",
        "document_name": "03-delhivery-q4-fy24-earnings-presentation.pdf",
        "page_numbers": [8, 9, 11, 13, 14],
    },
    {
        "batch_id": "batch_4_central_bank_dates",
        "description": "Category 4: Central Banking Data, Currency, Foreign Exchange & Dates",
        "document_path": "sample-data/starter-datasets/india-macroeconomy/02-rbi-annual-report-2024-25-excerpt.pdf",
        "document_name": "02-rbi-annual-report-2024-25-excerpt.pdf",
        "page_numbers": [6, 7, 8, 10, 12],
    },
    {
        "batch_id": "batch_5_structural_low_fact",
        "description": "Category 5: Low-Fact / Title Slides, Disclaimers & Multi-Document Structure",
        "is_multi_document": True,
        "items": [
            ("sample-data/starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf", "03-delhivery-q4-fy24-earnings-presentation.pdf", 2),
            ("sample-data/starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf", "03-delhivery-q4-fy24-earnings-presentation.pdf", 3),
            ("sample-data/starter-datasets/delhivery/01-delhivery-prospectus-2022-excerpt.pdf", "01-delhivery-prospectus-2022-excerpt.pdf", 2),
            ("sample-data/starter-datasets/india-macroeconomy/03-imf-india-2025-article-iv-excerpt.pdf", "03-imf-india-2025-article-iv-excerpt.pdf", 5),
            ("sample-data/starter-datasets/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf", "02-delhivery-annual-report-fy24-excerpt.pdf", 1),
        ],
    },
]


def load_benchmark_batches(
    project_root: str,
) -> List[Tuple[str, str, List[PageText]]]:
    """Parses and extracts the exact 25 benchmark PageText instances."""
    parser = PDFParser()
    batches = []

    for defn in BENCHMARK_BATCH_DEFINITIONS:
        if defn.get("is_multi_document"):
            # Multi-document batch 5
            pages = []
            for rel_path, doc_name, page_num in defn["items"]:
                full_path = os.path.join(project_root, rel_path)
                parsed_doc = parser.parse(full_path, document_name=doc_name)
                matching = [p for p in parsed_doc.pages if p.page_number == page_num]
                if matching:
                    pages.append(matching[0])
            batches.append((defn["batch_id"], "multi-doc-structural", pages))
        else:
            rel_path = defn["document_path"]
            doc_name = defn["document_name"]
            full_path = os.path.join(project_root, rel_path)
            parsed_doc = parser.parse(full_path, document_name=doc_name)
            req_pages = defn["page_numbers"]
            pages = [p for p in parsed_doc.pages if p.page_number in req_pages]
            batches.append((defn["batch_id"], doc_name, pages))

    return batches


def run_gemini_benchmark_batch(
    gemini_extractor: FactExtractor,
    batch_pages: List[PageText],
    document_id: str,
    document_name: str,
    batch_index: int,
) -> SingleBatchProviderResult:
    """Executes a single benchmark batch using existing production FactExtractor."""
    context_selector = ContextSelector(
        expansion_radius=2,
        merge_gap_threshold=2,
        include_page_header=True,
    )
    batch_prompt_texts = []
    original_page_map = {}

    for page in batch_pages:
        original_page_map[page.page_number] = page
        context_res = context_selector.select_page_context(
            page_text=page.text,
            page_number=page.page_number,
            document_id=document_id,
        )
        prompt_text = context_res.combined_source_text if not context_res.is_empty else page.text
        batch_prompt_texts.append((page.page_number, prompt_text))

    input_hash = compute_semantic_input_hash(
        batch_pages=batch_prompt_texts,
        document_name=document_name,
    )

    t0 = time.time()
    try:
        verified_facts, rejected_count = gemini_extractor.extract_batch(
            batch_pages=batch_pages,
            document_id=document_id,
            document_name=document_name,
        )
        latency_ms = (time.time() - t0) * 1000.0

        # Duplicate checking
        seen_ids = set()
        dup_count = 0
        deduped = []
        for f in verified_facts:
            if f.fact_id in seen_ids:
                dup_count += 1
            else:
                seen_ids.add(f.fact_id)
                deduped.append(f)

        return SingleBatchProviderResult(
            batch_index=batch_index,
            document_id=document_id,
            page_numbers=[p.page_number for p in batch_pages],
            semantic_input_hash=input_hash,
            latency_ms=latency_ms,
            raw_facts_count=len(verified_facts) + rejected_count,
            verified_facts=deduped,
            rejected_facts_count=rejected_count,
            duplicate_facts_count=dup_count,
            cross_page_contamination_count=0,
            failure_class=None,
        )
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000.0
        return SingleBatchProviderResult(
            batch_index=batch_index,
            document_id=document_id,
            page_numbers=[p.page_number for p in batch_pages],
            semantic_input_hash=input_hash,
            latency_ms=latency_ms,
            raw_facts_count=0,
            verified_facts=[],
            rejected_facts_count=0,
            error_message=str(e),
        )


def run_full_benchmark(
    project_root: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    groq_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs the controlled, provider-blind Gemini vs Groq benchmark."""
    root = project_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    batches = load_benchmark_batches(root)

    print("=" * 70)
    print("   FACTLINE PHASE 26: GROQ EXTRACTION PROVIDER BENCHMARK")
    print("=" * 70)
    print(f"Total Batches: {len(batches)} | Total Pages: 25 | Concurrency: 1 worker\n")

    # Initialize extractors
    gemini_extractor = FactExtractor(batch_size=5, max_workers=1)
    groq_client = GroqProviderClient(api_key=groq_api_key, model=groq_model)
    groq_extractor = GroqFactExtractor(provider_client=groq_client)

    gemini_batch_results: List[SingleBatchProviderResult] = []
    groq_batch_results: List[SingleBatchProviderResult] = []
    fingerprint_comparisons: List[Dict[str, Any]] = []

    # 1. Execute Gemini Batches
    print("--> Executing Gemini Benchmark Batches (5 batches, 25 pages)...")
    t_gem_start = time.time()
    for idx, (b_id, doc_name, pages) in enumerate(batches, 1):
        print(f"  [Gemini Batch {idx}/5] {b_id} ({len(pages)} pages: {[p.page_number for p in pages]})...", end="", flush=True)
        res = run_gemini_benchmark_batch(
            gemini_extractor=gemini_extractor,
            batch_pages=pages,
            document_id=doc_name,
            document_name=doc_name,
            batch_index=idx,
        )
        gemini_batch_results.append(res)
        print(f" Done ({res.latency_ms:.0f}ms, {len(res.verified_facts)} facts)")
    gemini_wall_time = time.time() - t_gem_start

    # 2. Execute Groq Batches
    print("\n--> Executing Groq Benchmark Batches (5 batches, 25 pages)...")
    t_groq_start = time.time()
    for idx, (b_id, doc_name, pages) in enumerate(batches, 1):
        print(f"  [Groq Batch {idx}/5] {b_id} ({len(pages)} pages: {[p.page_number for p in pages]})...", end="", flush=True)
        res = groq_extractor.extract_batch(
            batch_pages=pages,
            document_id=doc_name,
            document_name=doc_name,
            batch_index=idx,
        )
        groq_batch_results.append(res)
        if res.failure_class:
            print(f" FAILED ({res.failure_class.value}: {res.error_message})")
        else:
            print(f" Done ({res.latency_ms:.0f}ms, {len(res.verified_facts)} facts)")
    groq_wall_time = time.time() - t_groq_start

    # 3. Verify Input Fingerprints
    all_fingerprints_match = True
    for idx in range(len(batches)):
        gem_hash = gemini_batch_results[idx].semantic_input_hash
        groq_hash = groq_batch_results[idx].semantic_input_hash
        match = (gem_hash == groq_hash)
        if not match:
            all_fingerprints_match = False
        fingerprint_comparisons.append({
            "batch_index": idx + 1,
            "batch_id": batches[idx][0],
            "gemini_input_hash": gem_hash,
            "groq_input_hash": groq_hash,
            "hashes_match": match,
        })

    # 4. Aggregate Metrics
    gemini_summary = aggregate_batch_results(
        provider_name="Google Gemini",
        model_name=gemini_extractor.model,
        batch_results=gemini_batch_results,
        wall_clock_time_s=gemini_wall_time,
    )

    groq_summary = aggregate_batch_results(
        provider_name="Groq",
        model_name=groq_client.model,
        batch_results=groq_batch_results,
        wall_clock_time_s=groq_wall_time,
    )

    result_payload = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_pages": 25,
            "total_batches": 5,
            "concurrency": 1,
            "batch_size": 5,
            "fingerprints_match": all_fingerprints_match,
        },
        "fingerprints": fingerprint_comparisons,
        "gemini_summary": gemini_summary.model_dump(),
        "groq_summary": groq_summary.model_dump(),
    }

    # Write results to JSON artifact
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    return result_payload


if __name__ == "__main__":
    run_full_benchmark()
