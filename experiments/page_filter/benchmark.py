"""
FACTLINE Phase 11: Local Page Relevance Filter Benchmark Runner.

Evaluates deterministic page selection across:
1. Multi-threshold performance curve ([0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70])
2. Page Recall, Page Precision, and Gemini API call reduction
3. Missed fact-bearing pages (False Negatives) and False Positives audit
4. Full starter-corpus scan across all 6 starter PDFs
5. Unseen PDF generalization test
"""

import os
import sys
import json
import time
import fitz

# Ensure UTF-8 console output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.models.document import PageText
from backend.page_filter.relevance import PageRelevanceFilter, PageRelevanceScore


def load_page_text(pdf_path: str, page_number: int) -> PageText:
    """Helper to extract PageText from PDF via fitz."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        text = page.get_text() or ""
        return PageText(
            page_number=page_number,
            text=text,
            char_count=len(text),
            has_text=bool(text.strip()),
        )
    finally:
        doc.close()


def run_benchmark():
    benchmark_file = os.path.join(os.path.dirname(__file__), "benchmark_cases.json")
    with open(benchmark_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    cases = bench_data.get("cases", [])
    print("=" * 70)
    print("   FACTLINE PHASE 11: LOCAL PAGE RELEVANCE FILTER BENCHMARK")
    print("=" * 70)
    print(f"Loaded {len(cases)} gold-standard labeled benchmark pages.\n")

    # Pre-load all benchmark page texts
    page_records = []
    for c in cases:
        full_pdf = os.path.join(PROJECT_ROOT, c["pdf_path"])
        p_text = load_page_text(full_pdf, c["page"])
        page_records.append({
            "case_id": c["id"],
            "document": c["document"],
            "page_number": c["page"],
            "ground_truth": c["ground_truth"],
            "description": c["description"],
            "page_text": p_text,
        })

    filter_engine = PageRelevanceFilter()
    thresholds = [0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    threshold_results = []

    print("--- THRESHOLD TRADEOFF ANALYSIS ---")
    print(f"{'Threshold':>9} | {'Recall':>8} | {'Precision':>9} | {'Selected':>8} | {'Call Reduction':>14} | {'Missed Pages':>12}")
    print("-" * 72)

    total_fact_bearing = sum(1 for c in page_records if c["ground_truth"] == "FACT_BEARING")
    total_non_fact = sum(1 for c in page_records if c["ground_truth"] == "NOT_FACT_BEARING")
    total_pages = len(page_records)

    for thresh in thresholds:
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        missed_cases = []
        fp_cases = []

        for item in page_records:
            score = filter_engine.score_page(item["page_text"], custom_threshold=thresh)
            is_selected = score.selected
            is_fact_bearing = item["ground_truth"] == "FACT_BEARING"

            if is_fact_bearing and is_selected:
                tp += 1
            elif not is_fact_bearing and is_selected:
                fp += 1
                fp_cases.append(f"{item['document']} p.{item['page_number']}")
            elif is_fact_bearing and not is_selected:
                fn += 1
                missed_cases.append(f"{item['document']} p.{item['page_number']}")
            else:
                tn += 1

        recall = (tp / total_fact_bearing) * 100.0 if total_fact_bearing else 0.0
        precision = (tp / (tp + fp)) * 100.0 if (tp + fp) else 0.0
        selected_count = tp + fp
        reduction = ((total_pages - selected_count) / total_pages) * 100.0

        threshold_results.append({
            "threshold": thresh,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "recall_pct": round(recall, 1),
            "precision_pct": round(precision, 1),
            "selected_pages": selected_count,
            "total_pages": total_pages,
            "gemini_reduction_pct": round(reduction, 1),
            "missed_pages": missed_cases,
            "false_positive_pages": fp_cases,
        })

        missed_str = f"{fn} missed" if fn > 0 else "0 missed"
        print(f"{thresh:9.2f} | {recall:7.1f}% | {precision:8.1f}% | {selected_count:4d}/{total_pages:2d} | {reduction:13.1f}% | {missed_str:>12}")

    # Detailed inspection at recommended default threshold (0.35)
    default_res = next(r for r in threshold_results if r["threshold"] == 0.35)
    print("\n" + "=" * 70)
    print("      BENCHMARK PERFORMANCE SUMMARY (Default Threshold = 0.35)")
    print("=" * 70)
    print(f"Total Benchmark Pages:            {total_pages} (14 Fact-Bearing, 10 Non-Fact-Bearing)")
    print(f"Page Recall:                      {default_res['recall_pct']:.1f}% ({default_res['tp']}/{total_fact_bearing} fact-bearing pages retained)")
    print(f"Page Precision:                   {default_res['precision_pct']:.1f}% ({default_res['tp']}/{default_res['selected_pages']} selected pages are valid)")
    print(f"Pages Sent to Gemini:             {default_res['selected_pages']} / {total_pages}")
    print(f"Gemini API Call Reduction:        {default_res['gemini_reduction_pct']:.1f}%")
    print(f"Missed Fact-Bearing Pages (FN):   {default_res['fn']} -> {default_res['missed_pages']}")
    print(f"False Positives Retained (FP):    {default_res['fp']} -> {default_res['false_positive_pages']}")
    print("=" * 70)

    # Full Starter Corpus Scan
    print("\n--- FULL STARTER CORPUS EVALUATION ---")
    starter_pdfs = [
        "sample-data/starter-datasets/delhivery/01-delhivery-prospectus-2022-excerpt.pdf",
        "sample-data/starter-datasets/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf",
        "sample-data/starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf",
        "sample-data/starter-datasets/india-macroeconomy/01-india-economic-survey-2024-25-excerpt.pdf",
        "sample-data/starter-datasets/india-macroeconomy/02-rbi-annual-report-2024-25-excerpt.pdf",
        "sample-data/starter-datasets/india-macroeconomy/03-imf-india-2025-article-iv-excerpt.pdf",
    ]

    corpus_summary = []
    total_corpus_pages = 0
    total_corpus_text_pages = 0
    total_corpus_selected = 0

    print(f"{'Document':<48} | {'Total':>5} | {'Text':>5} | {'Selected':>8} | {'Reduction':>9}")
    print("-" * 84)

    for pdf_rel in starter_pdfs:
        pdf_full = os.path.join(PROJECT_ROOT, pdf_rel)
        doc = fitz.open(pdf_full)
        doc_name = os.path.basename(pdf_rel)
        tot_p = len(doc)
        
        pages_list = []
        for pno in range(1, tot_p + 1):
            t = doc[pno - 1].get_text() or ""
            pages_list.append(PageText(
                page_number=pno,
                text=t,
                char_count=len(t),
                has_text=bool(t.strip())
            ))
        doc.close()

        text_pages = [p for p in pages_list if p.has_text]
        scores = filter_engine.filter_pages(pages_list, threshold=0.35)
        selected_pages = [s for s in scores if s.selected]

        tot_sel = len(selected_pages)
        tot_txt = len(text_pages)
        reduct = ((tot_txt - tot_sel) / tot_txt) * 100.0 if tot_txt else 0.0

        total_corpus_pages += tot_p
        total_corpus_text_pages += tot_txt
        total_corpus_selected += tot_sel

        corpus_summary.append({
            "document": doc_name,
            "total_pages": tot_p,
            "text_pages": tot_txt,
            "selected_pages": tot_sel,
            "reduction_pct": round(reduct, 1),
        })

        print(f"{doc_name:<48} | {tot_p:5d} | {tot_txt:5d} | {tot_sel:8d} | {reduct:8.1f}%")

    overall_reduction = ((total_corpus_text_pages - total_corpus_selected) / total_corpus_text_pages) * 100.0
    print("-" * 84)
    print(f"{'AGGREGATE STARTER CORPUS':<48} | {total_corpus_pages:5d} | {total_corpus_text_pages:5d} | {total_corpus_selected:8d} | {overall_reduction:8.1f}%\n")

    # Unseen PDF Evaluation (Prospectus Pages 5–15)
    print("--- UNSEEN PDF GENERALIZATION EVALUATION ---")
    unseen_doc_rel = "sample-data/starter-datasets/delhivery/01-delhivery-prospectus-2022-excerpt.pdf"
    doc_unseen = fitz.open(os.path.join(PROJECT_ROOT, unseen_doc_rel))
    unseen_pages = []
    for pno in range(5, 16):
        t = doc_unseen[pno - 1].get_text() or ""
        unseen_pages.append(PageText(page_number=pno, text=t, char_count=len(t), has_text=bool(t.strip())))
    doc_unseen.close()

    unseen_scores = filter_engine.filter_pages(unseen_pages, threshold=0.35)
    unseen_selected = [s for s in unseen_scores if s.selected]
    print(f"Evaluated 11 unseen prospectus pages (pages 5–15):")
    print(f"  Selected for Gemini: {len(unseen_selected)} / 11 pages ({(len(unseen_selected)/11)*100.0:.1f}%)")
    for s in unseen_scores:
        status = "[SELECTED]" if s.selected else "[SKIPPED] "
        print(f"   * Page {s.page_number:02d} {status} (Score: {s.relevance_score:.3f}) | {s.signals_triggered[:3]}")

    # Output artifact
    benchmark_artifact = {
        "benchmark_summary": {
            "default_threshold": 0.35,
            "page_recall_pct": default_res["recall_pct"],
            "page_precision_pct": default_res["precision_pct"],
            "gemini_reduction_pct": default_res["gemini_reduction_pct"],
            "total_benchmark_pages": total_pages,
            "missed_fact_bearing_pages": default_res["missed_pages"],
            "false_positive_pages": default_res["false_positive_pages"],
        },
        "threshold_tradeoffs": threshold_results,
        "starter_corpus_evaluation": {
            "total_documents": len(starter_pdfs),
            "total_pages": total_corpus_pages,
            "total_text_pages": total_corpus_text_pages,
            "total_selected_pages": total_corpus_selected,
            "aggregate_reduction_pct": round(overall_reduction, 1),
            "documents": corpus_summary,
        },
        "unseen_pdf_evaluation": {
            "document": os.path.basename(unseen_doc_rel),
            "evaluated_pages": 11,
            "selected_pages": len(unseen_selected),
            "selection_rate_pct": round((len(unseen_selected) / 11) * 100.0, 1),
        }
    }

    out_json = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_artifact, f, indent=2)
    print(f"\nBenchmark results artifact saved to: {out_json}")


if __name__ == "__main__":
    run_benchmark()
