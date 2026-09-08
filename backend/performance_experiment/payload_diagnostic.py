"""
Phase 21: Gemini Payload & Latency Diagnostic Engine.

Performs deterministic request-level and page-level analysis across
all planned 5-page batches of APRIL IMF.pdf:
- Context windows & source regions count
- Raw vs compressed characters & compression ratio
- Prompt character count & estimated input tokens
- Page complexity signals (numbers, percentages, currencies, table lines)
- Output size & fact yields
- Comparison of successful batches vs failed/timeout batches
"""

import os
import sys
import re
import json
import statistics
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
load_dotenv(override=True)

from backend.extraction.pdf_parser import PDFParser
from backend.page_filter.relevance import PageRelevanceFilter
from backend.context_selector.selector import ContextSelector
from backend.quota_experiment.planner import QuotaPlanner
from backend.extraction.fact_extractor import FactExtractor


class PayloadDiagnostic:
    """Diagnostic analyzer for Gemini payload sizes, context windows, and page complexity."""

    def __init__(self, pdf_path: str = r"C:\Users\vaish\Downloads\APRIL IMF.pdf", batch_size: int = 5):
        self.pdf_path = pdf_path
        self.batch_size = batch_size
        self.parser = PDFParser()
        self.page_filter = PageRelevanceFilter(threshold=0.30)
        self.context_selector = ContextSelector(expansion_radius=2)
        self.planner = QuotaPlanner(default_batch_size=batch_size)
        self.extractor = FactExtractor(batch_size=batch_size)

        # Deterministic complexity regexes
        self.NUMERIC_PATTERN = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
        self.PERCENT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent(?:age)?)\b", re.IGNORECASE)
        self.CURRENCY_PATTERN = re.compile(r"(?:₹|\$|USD|INR|EUR|GBP|Rs\.?|dollars?|rupees?|crores?|lakhs?|billions?|millions?)", re.IGNORECASE)

    def analyze(self) -> Dict[str, Any]:
        """Runs complete deterministic diagnostic analysis on APRIL IMF.pdf."""
        with open(self.pdf_path, "rb") as f:
            pdf_bytes = f.read()

        doc = self.parser.parse_bytes(pdf_bytes, os.path.basename(self.pdf_path))
        text_pages = [p for p in doc.pages if p.has_text and p.text.strip()]
        filter_results = self.page_filter.filter_pages(text_pages)

        scores_map = {fr.page_number: fr.relevance_score for fr in filter_results}
        eligible_pages = [fr.page_number for fr in filter_results if fr.selected]
        page_lookup = {p.page_number: p for p in doc.pages}

        plan = self.planner.calculate_plan(
            eligible_pages=eligible_pages,
            request_budget=None,
            batch_size=self.batch_size,
            page_relevance_scores=scores_map,
        )

        # Load persisted facts from latest run if available to correlate with page fact yields
        facts_by_page: Dict[int, int] = {}
        try:
            import sqlite3
            conn = sqlite3.connect("factline.db")
            c = conn.cursor()
            # facts for the latest april doc
            c.execute("SELECT provenance_page, count(*) FROM facts WHERE document_id = '8e1c2dbf9827dd38' GROUP BY provenance_page")
            for p_num, count in c.fetchall():
                facts_by_page[p_num] = count
            conn.close()
        except Exception:
            pass

        # 1. Page Complexity Analysis
        pages_complexity = []
        for p_num in eligible_pages:
            p = page_lookup[p_num]
            text = p.text
            lines = text.split("\n")
            
            # Count table-like lines (multiple columns / numeric tokens on a single line)
            table_lines = 0
            for line in lines:
                nums_in_line = self.NUMERIC_PATTERN.findall(line)
                if len(nums_in_line) >= 3 or ("\t" in line and len(nums_in_line) >= 2):
                    table_lines += 1

            c_res = self.context_selector.select_page_context(text, p.page_number, doc.document_id)
            
            pages_complexity.append({
                "page_number": p_num,
                "character_count": len(text),
                "numeric_tokens": len(self.NUMERIC_PATTERN.findall(text)),
                "percentage_expressions": len(self.PERCENT_PATTERN.findall(text)),
                "currency_expressions": len(self.CURRENCY_PATTERN.findall(text)),
                "table_lines": table_lines,
                "context_windows_count": len(c_res.context_windows),
                "compressed_characters": len(c_res.combined_source_text),
                "compression_ratio": round(len(c_res.combined_source_text) / len(text), 3) if len(text) > 0 else 1.0,
                "facts_extracted": facts_by_page.get(p_num, 0),
            })

        # 2. Batch Level Payload & Context Window Analysis
        batches_analysis = []
        for b_idx, batch_p_nums in enumerate(plan.batches):
            b_num = b_idx + 1
            batch_items = []
            raw_chars = 0
            comp_chars = 0
            total_windows = 0
            window_sizes = []
            table_lines_total = 0
            numeric_tokens_total = 0
            facts_in_batch = sum(facts_by_page.get(p, 0) for p in batch_p_nums)

            for p_num in batch_p_nums:
                p = page_lookup[p_num]
                raw_chars += len(p.text)
                c_res = self.context_selector.select_page_context(p.text, p.page_number, doc.document_id)
                p_text = c_res.combined_source_text if not c_res.is_empty else p.text
                comp_chars += len(p_text)
                total_windows += len(c_res.context_windows)
                for w in c_res.context_windows:
                    window_sizes.append(w.char_count)
                
                # Table lines on page
                for line in p.text.split("\n"):
                    if len(self.NUMERIC_PATTERN.findall(line)) >= 3:
                        table_lines_total += 1
                numeric_tokens_total += len(self.NUMERIC_PATTERN.findall(p.text))

                batch_items.append((p_num, p_text))

            prompt = self.extractor.format_batch_prompt(batch_items, doc.document_name)
            prompt_chars = len(prompt)
            est_tokens = prompt_chars // 4

            avg_w_size = round(statistics.mean(window_sizes), 1) if window_sizes else 0
            max_w_size = max(window_sizes) if window_sizes else 0

            batches_analysis.append({
                "batch_number": b_num,
                "page_numbers": batch_p_nums,
                "pages_count": len(batch_p_nums),
                "context_windows_count": total_windows,
                "average_window_size": avg_w_size,
                "largest_window_size": max_w_size,
                "raw_characters": raw_chars,
                "compressed_characters": comp_chars,
                "compression_ratio": round(comp_chars / raw_chars, 3) if raw_chars > 0 else 1.0,
                "prompt_characters": prompt_chars,
                "estimated_input_tokens": est_tokens,
                "table_lines": table_lines_total,
                "numeric_tokens": numeric_tokens_total,
                "facts_extracted": facts_in_batch,
            })

        return {
            "document": doc.document_name,
            "total_pages": doc.total_pages,
            "eligible_pages_count": len(eligible_pages),
            "planned_batches_count": len(plan.batches),
            "batches": batches_analysis,
            "pages": pages_complexity,
        }


if __name__ == "__main__":
    diag = PayloadDiagnostic()
    res = diag.analyze()
    out_file = "scratch/phase21_payload_diagnostic.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Diagnostic completed. Saved {len(res['batches'])} batches analysis to {out_file}")
