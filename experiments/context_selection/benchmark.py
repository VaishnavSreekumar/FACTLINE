"""
Phase 12 Local Fact-Bearing Region & Context Window Benchmark Runner.

Evaluates deterministic structural context selection against gold-annotated benchmark pages.
Measures:
1. Region Recall
2. Context Completeness (Entity, Metric, Value, Unit, Time, Scope)
3. Evidence Preservation (100% target)
4. Character and Content Reduction
5. Context Window Count per page
6. Context-Size Sweep (±1, ±2, ±3, ±5 lines)
7. Limited Quota-Safe Gemini Validation (3 representative cases)
8. Unseen PDF Generalization Test
"""

import json
import os
import sys
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

from backend.models.document import PageText, ParsedDocument
from backend.extraction.pdf_parser import PDFParser
from backend.context_selector.selector import ContextSelector, PageContextResult


def evaluate_fact_coverage(
    gold_region: Dict[str, Any],
    selected_text: str,
) -> Dict[str, bool]:
    """
    Evaluates whether each required context element of a gold fact is preserved
    in the selected context text.
    """
    selected_lower = selected_text.lower()
    required = gold_region.get("required_context_elements", [])
    key_tokens = gold_region.get("key_tokens", {})

    element_coverage: Dict[str, bool] = {}

    for elem in ["entity", "metric", "value", "unit", "time_period", "scope"]:
        if elem not in required:
            continue
        
        tokens = key_tokens.get(elem, [])
        if not tokens:
            element_coverage[elem] = True
            continue

        # Check if any of the valid token representations are found
        matched = any(t.lower() in selected_lower for t in tokens)
        element_coverage[elem] = matched

    return element_coverage


def run_benchmark_for_radius(
    benchmark_cases: List[Dict[str, Any]],
    parsed_pages_cache: Dict[str, Dict[int, str]],
    radius: int,
) -> Dict[str, Any]:
    """
    Runs context selection benchmark for a specific expansion radius.
    """
    selector = ContextSelector(
        expansion_radius=radius,
        merge_gap_threshold=2,
        include_page_header=True,
    )

    total_gold_facts = 0
    covered_gold_facts = 0

    element_totals = {"entity": 0, "metric": 0, "value": 0, "unit": 0, "time_period": 0, "scope": 0}
    element_covered = {"entity": 0, "metric": 0, "value": 0, "unit": 0, "time_period": 0, "scope": 0}

    total_orig_chars = 0
    total_selected_chars = 0
    char_reductions: List[float] = []
    window_counts: List[int] = []
    evidence_valid_count = 0
    total_evaluated_pages = len(benchmark_cases)

    case_details: List[Dict[str, Any]] = []

    for case in benchmark_cases:
        doc_name = case["document"]
        page_num = case["page"]
        page_text = parsed_pages_cache[doc_name].get(page_num, "")

        result: PageContextResult = selector.select_page_context(
            page_text=page_text,
            page_number=page_num,
            document_id=doc_name,
        )

        orig_len = len(page_text)
        sel_len = result.selected_character_count
        total_orig_chars += orig_len
        total_selected_chars += sel_len

        red_ratio = (1.0 - (sel_len / orig_len)) if orig_len > 0 else 0.0
        char_reductions.append(red_ratio)
        window_counts.append(len(result.context_windows))

        # Evidence preservation check: every line in each window must exist verbatim in source
        is_evidence_valid = True
        for w in result.context_windows:
            for line in w.source_text.splitlines():
                if line.strip() and line not in page_text:
                    is_evidence_valid = False
                    break
        if is_evidence_valid:
            evidence_valid_count += 1

        # Evaluate gold facts for fact-bearing pages
        gold_regions = case.get("gold_regions", [])
        case_fact_evals = []

        for g in gold_regions:
            total_gold_facts += 1
            cov = evaluate_fact_coverage(g, result.combined_source_text)
            
            # Value + metric is minimum for region recall
            val_ok = cov.get("value", True)
            met_ok = cov.get("metric", True)
            is_region_recalled = val_ok and met_ok
            if is_region_recalled:
                covered_gold_facts += 1

            for elem, status in cov.items():
                element_totals[elem] += 1
                if status:
                    element_covered[elem] += 1

            case_fact_evals.append({
                "fact_id": g["fact_id"],
                "region_recalled": is_region_recalled,
                "element_coverage": cov,
            })

        case_details.append({
            "case_id": case["id"],
            "case_type": case["case_type"],
            "original_chars": orig_len,
            "selected_chars": sel_len,
            "reduction_ratio": round(red_ratio, 4),
            "window_count": len(result.context_windows),
            "fact_evaluations": case_fact_evals,
        })

    region_recall = (covered_gold_facts / total_gold_facts) if total_gold_facts > 0 else 1.0
    evidence_preservation_pct = (evidence_valid_count / total_evaluated_pages) * 100.0 if total_evaluated_pages > 0 else 100.0
    avg_reduction_pct = (1.0 - (total_selected_chars / total_orig_chars)) * 100.0 if total_orig_chars > 0 else 0.0
    median_reduction_pct = statistics.median(char_reductions) * 100.0 if char_reductions else 0.0
    avg_windows_per_page = statistics.mean(window_counts) if window_counts else 0.0
    max_windows_per_page = max(window_counts) if window_counts else 0

    completeness = {}
    for elem in element_totals:
        tot = element_totals[elem]
        cov = element_covered[elem]
        pct = (cov / tot * 100.0) if tot > 0 else 100.0
        completeness[elem] = {
            "covered": cov,
            "total": tot,
            "percentage": round(pct, 2),
        }

    return {
        "expansion_radius": radius,
        "region_recall_pct": round(region_recall * 100.0, 2),
        "gold_facts_covered": f"{covered_gold_facts}/{total_gold_facts}",
        "evidence_preservation_pct": round(evidence_preservation_pct, 2),
        "total_original_chars": total_orig_chars,
        "total_selected_chars": total_selected_chars,
        "average_character_reduction_pct": round(avg_reduction_pct, 2),
        "median_character_reduction_pct": round(median_reduction_pct, 2),
        "average_windows_per_page": round(avg_windows_per_page, 2),
        "max_windows_per_page": max_windows_per_page,
        "context_completeness": completeness,
        "case_details": case_details,
    }


def run_gemini_feasibility_check(
    benchmark_cases: List[Dict[str, Any]],
    parsed_pages_cache: Dict[str, Dict[int, str]],
    selector: ContextSelector,
) -> Dict[str, Any]:
    """
    Runs a limited, quota-safe feasibility comparison with Gemini (3 cases max)
    if API key is present and active.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return {
            "executed": False,
            "reason": "GEMINI_API_KEY not configured in environment."
        }

    target_case_ids = [
        "case-01-delhivery-ar24-p2",
        "case-02-delhivery-ar24-p4",
        "case-06-delhivery-q4-p5",
    ]

    selected_cases = [c for c in benchmark_cases if c["id"] in target_case_ids]

    import httpx
    from backend.extraction.prompts import EXTRACTION_JSON_SCHEMA, EXTRACTION_SYSTEM_PROMPT

    def _fast_gemini_call(prompt_text: str) -> Optional[Dict[str, Any]]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "systemInstruction": {"parts": [{"text": EXTRACTION_SYSTEM_PROMPT}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": EXTRACTION_JSON_SCHEMA,
                "temperature": 0.0,
            },
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    if parts:
                        return json.loads(parts[0].get("text", "{}"))
                    return {"facts": []}
                else:
                    return None
        except Exception:
            return None

    comparison_results = []

    for c in selected_cases:
        doc_name = c["document"]
        page_num = c["page"]
        page_text = parsed_pages_cache[doc_name].get(page_num, "")

        context_res = selector.select_page_context(page_text, page_number=page_num, document_id=doc_name)
        compressed_text = context_res.combined_source_text

        prompt_base = f"Document: {doc_name}\nPage Number: {page_num}\n\n--- PAGE TEXT ---\n{page_text}"
        prompt_exp = f"Document: {doc_name}\nPage Number: {page_num}\n\n--- PAGE TEXT ---\n{compressed_text}"

        base_data = _fast_gemini_call(prompt_base)
        if base_data is None:
            return {
                "executed": False,
                "reason": "Gemini validation not executed because quota was unavailable or API call was rejected."
            }

        exp_data = _fast_gemini_call(prompt_exp)
        if exp_data is None:
            return {
                "executed": False,
                "reason": "Gemini validation not executed because quota was unavailable or API call was rejected."
            }

        comparison_results.append({
            "case_id": c["id"],
            "page": page_num,
            "original_chars": len(page_text),
            "compressed_chars": len(compressed_text),
            "char_reduction_pct": round((1.0 - len(compressed_text)/len(page_text)) * 100.0, 1),
            "baseline_facts_count": len(base_data.get("facts", [])),
            "experimental_facts_count": len(exp_data.get("facts", [])),
            "baseline_sample_facts": [
                {
                    "metric": f.get("metric"),
                    "value_raw": f.get("value_raw"),
                    "time_period": f.get("time_period", {}).get("label") if isinstance(f.get("time_period"), dict) else None,
                    "entity": f.get("entity")
                } for f in base_data.get("facts", [])[:3]
            ],
            "experimental_sample_facts": [
                {
                    "metric": f.get("metric"),
                    "value_raw": f.get("value_raw"),
                    "time_period": f.get("time_period", {}).get("label") if isinstance(f.get("time_period"), dict) else None,
                    "entity": f.get("entity")
                } for f in exp_data.get("facts", [])[:3]
            ],
            "status": "SUCCESS"
        })

    return {
        "executed": len(comparison_results) > 0,
        "comparisons": comparison_results,
    }


def run_unseen_pdf_test(
    selector: ContextSelector,
    parser: PDFParser,
) -> Dict[str, Any]:
    """
    Evaluates context selection on an unseen document (Delhivery Prospectus pages 11-20).
    """
    pdf_path = Path("sample-data/starter-datasets/delhivery/01-delhivery-prospectus-2022-excerpt.pdf")
    if not pdf_path.exists():
        return {"status": "SKIPPED", "reason": "Prospectus PDF not found"}

    parsed = parser.parse(pdf_path)
    # Test on unseen pages 11 to 20
    test_pages = [p for p in parsed.pages if 11 <= p.page_number <= 20]

    orig_chars = 0
    sel_chars = 0
    windows_count = 0

    page_stats = []
    for p in test_pages:
        res = selector.select_page_context(p.text, page_number=p.page_number, document_id="prospectus-unseen")
        orig_chars += len(p.text)
        sel_chars += res.selected_character_count
        windows_count += len(res.context_windows)
        page_stats.append({
            "page_number": p.page_number,
            "original_chars": len(p.text),
            "selected_chars": res.selected_character_count,
            "windows": len(res.context_windows),
            "anchors": res.anchors[:3],
        })

    reduction_pct = (1.0 - (sel_chars / orig_chars)) * 100.0 if orig_chars > 0 else 0.0
    avg_windows = windows_count / len(test_pages) if test_pages else 0.0

    return {
        "document": "01-delhivery-prospectus-2022-excerpt.pdf (Pages 11-20)",
        "total_unseen_pages": len(test_pages),
        "total_original_chars": orig_chars,
        "total_selected_chars": sel_chars,
        "character_reduction_pct": round(reduction_pct, 2),
        "average_windows_per_page": round(avg_windows, 2),
        "page_breakdown": page_stats,
    }


def main():
    cases_path = Path("experiments/context_selection/benchmark_cases.json")
    if not cases_path.exists():
        print(f"Error: {cases_path} not found.")
        sys.exit(1)

    with open(cases_path, "r", encoding="utf-8") as f:
        cases_data = json.load(f)

    benchmark_cases = cases_data["cases"]
    parser = PDFParser()

    # Cache parsed text for all unique documents in benchmark
    parsed_pages_cache: Dict[str, Dict[int, str]] = {}
    for case in benchmark_cases:
        doc_name = case["document"]
        pdf_path = case["pdf_path"]
        if doc_name not in parsed_pages_cache:
            parsed = parser.parse(pdf_path)
            parsed_pages_cache[doc_name] = {p.page_number: p.text for p in parsed.pages}

    print(f"Loaded {len(benchmark_cases)} benchmark cases across {len(parsed_pages_cache)} documents.")

    # Sweep context expansion sizes: radius 1 (±1), 2 (±2), 3 (±3), 5 (±5)
    sweep_radii = [1, 2, 3, 5]
    sweep_results = []

    print("\n--- Running Context Size Sweep ---")
    for r in sweep_radii:
        res = run_benchmark_for_radius(benchmark_cases, parsed_pages_cache, radius=r)
        sweep_results.append(res)
        print(
            f"Radius ±{r} lines: Region Recall={res['region_recall_pct']}%, "
            f"Val Cov={res['context_completeness']['value']['percentage']}%, "
            f"Metric Cov={res['context_completeness']['metric']['percentage']}%, "
            f"Time Cov={res['context_completeness']['time_period']['percentage']}%, "
            f"Char Red={res['average_character_reduction_pct']}%, "
            f"Avg Windows/pg={res['average_windows_per_page']}"
        )

    # Primary recommended selector (radius = 2)
    optimal_selector = ContextSelector(expansion_radius=2, merge_gap_threshold=2)

    print("\n--- Running Gemini Feasibility Check (3 cases max) ---")
    gemini_result = run_gemini_feasibility_check(benchmark_cases, parsed_pages_cache, optimal_selector)
    print(f"Gemini Feasibility Executed: {gemini_result.get('executed', False)}")
    if not gemini_result.get("executed", False):
        print(f"Reason: {gemini_result.get('reason')}")
    else:
        print("Gemini Feasibility Comparisons:")
        for cmp in gemini_result.get("comparisons", []):
            print(f"  {cmp['case_id']}: baseline={cmp['baseline_facts_count']} facts, exp={cmp['experimental_facts_count']} facts (char red {cmp['char_reduction_pct']}%)")

    print("\n--- Running Unseen PDF Generalization Test ---")
    unseen_result = run_unseen_pdf_test(optimal_selector, parser)
    print(f"Unseen PDF Reduction: {unseen_result.get('character_reduction_pct')}% across {unseen_result.get('total_unseen_pages')} pages")

    final_results = {
        "benchmark_metadata": {
            "total_benchmark_cases": len(benchmark_cases),
            "fact_bearing_cases": sum(1 for c in benchmark_cases if len(c.get("gold_regions", [])) > 0),
            "narrative_noise_cases": sum(1 for c in benchmark_cases if len(c.get("gold_regions", [])) == 0),
            "sweep_radii_tested": sweep_radii,
        },
        "sweep_results": sweep_results,
        "gemini_feasibility_validation": gemini_result,
        "unseen_pdf_generalization": unseen_result,
    }

    out_file = Path("experiments/context_selection/benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nSaved benchmark results to {out_file}")


if __name__ == "__main__":
    main()
