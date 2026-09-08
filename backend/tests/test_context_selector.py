"""
Unit tests for Local Fact-Bearing Region & Context Window Selection (Phase 12).

Verifies all 22 required deterministic properties:
1. numeric anchor detection
2. currency anchor
3. percentage anchor
4. quantitative unit
5. metric vocabulary
6. table-like region
7. region merging
8. context expansion
9. context boundaries
10. evidence preservation
11. overlapping regions
12. empty page
13. no-number page
14. numeric-noise page
15. deterministic repeated output
16. bounded output
17. no paraphrasing
18. no LLM calls
19. no filename-specific rules
20. no page-number-specific rules
21. arbitrary entities
22. arbitrary metrics
"""

import pytest
from backend.context_selector.region_detector import RegionDetector, DetectedRegion
from backend.context_selector.context_expander import ContextExpander, ExpandedContextWindow
from backend.context_selector.selector import ContextSelector, PageContextResult


def test_01_numeric_anchor_detection():
    """1. Tests detection of numeric quantities and counts."""
    detector = RegionDetector()
    text = "We delivered 33,278 parcels across 17,000+ pin codes."
    regions = detector.detect_regions(text, page_number=1)
    assert len(regions) >= 1
    assert "numeric" in regions[0].signals
    assert any("33,278" in r.anchor_text for r in regions)


def test_02_currency_anchor():
    """2. Tests detection of monetary symbols and financial denominations."""
    detector = RegionDetector()
    text = "Revenue from operations reached ₹8,142 Cr in FY24."
    regions = detector.detect_regions(text, page_number=1)
    assert len(regions) >= 1
    assert "currency" in regions[0].signals
    assert "numeric" in regions[0].signals


def test_03_percentage_anchor():
    """3. Tests detection of percentage metrics."""
    detector = RegionDetector()
    text = "The adjusted EBITDA margin expanded to 12.4% during Q4."
    regions = detector.detect_regions(text, page_number=1)
    assert len(regions) >= 1
    assert "percentage" in regions[0].signals


def test_04_quantitative_unit():
    """4. Tests detection of operational and physical units (tonnes, customers, shipments)."""
    detector = RegionDetector()
    text = "Handled freight volume exceeding >4.8Mn tonnes with 33,200 active customers."
    regions = detector.detect_regions(text, page_number=1)
    assert len(regions) >= 1
    assert "unit" in regions[0].signals


def test_05_metric_vocabulary():
    """5. Tests detection of generic financial/macroeconomic metric keywords."""
    detector = RegionDetector()
    text = "Annual net profit and operational turnover improved significantly."
    signals, _ = detector.detect_signals_in_line(text)
    assert "metric_keyword" in signals
    assert "profit" in text.lower()


def test_06_table_like_region():
    """6. Tests detection of multi-column tabular rows."""
    detector = RegionDetector()
    text = "Metric       FY23      FY24\nRevenue     ₹7,225 Cr  ₹8,142 Cr\nEBITDA      ₹450 Cr    ₹680 Cr"
    regions = detector.detect_regions(text, page_number=1)
    assert len(regions) >= 2
    assert any("table_row" in r.signals for r in regions)


def test_07_region_merging():
    """7. Tests merging of adjacent detected lines into a unified window."""
    expander = ContextExpander(expansion_radius=1, merge_gap_threshold=2)
    regions = [
        DetectedRegion(page_number=1, start_line=4, end_line=4, anchor_text="₹8,142 Cr", signals=["currency"], line_text="Revenue ₹8,142 Cr"),
        DetectedRegion(page_number=1, start_line=6, end_line=6, anchor_text="12.4%", signals=["percentage"], line_text="Growth 12.4%"),
    ]
    page_text = "\n".join([f"Line {i}" for i in range(10)])
    windows = expander.expand_and_merge(regions, page_text, page_number=1)
    assert len(windows) == 1
    assert windows[0].start_line <= 3
    assert windows[0].end_line >= 7


def test_08_context_expansion():
    """8. Tests expansion of ±N lines around an anchor."""
    expander = ContextExpander(expansion_radius=2, include_page_header=False)
    regions = [
        DetectedRegion(page_number=1, start_line=5, end_line=5, anchor_text="100", signals=["numeric"], line_text="Target Line 5: 100"),
    ]
    page_text = "\n".join([f"Line {i}" for i in range(12)])
    windows = expander.expand_and_merge(regions, page_text, page_number=1)
    assert len(windows) == 1
    assert windows[0].start_line == 3
    assert windows[0].end_line == 7


def test_09_context_boundaries():
    """9. Tests boundary clamping at line 0 and line total_lines-1."""
    expander = ContextExpander(expansion_radius=5, include_page_header=False)
    regions = [
        DetectedRegion(page_number=1, start_line=0, end_line=0, anchor_text="50", signals=["numeric"], line_text="Top 50"),
    ]
    page_text = "Line 0: 50\nLine 1\nLine 2"
    windows = expander.expand_and_merge(regions, page_text, page_number=1)
    assert len(windows) == 1
    assert windows[0].start_line == 0
    assert windows[0].end_line == 2


def test_10_evidence_preservation():
    """10. Tests that selected source text is 100% verbatim extract from the source page."""
    selector = ContextSelector(expansion_radius=2)
    page_text = (
        "Delhivery Limited\n"
        "Financial Highlights FY24\n"
        "Operating metrics:\n"
        "Revenue from operations was ₹8,142 Cr.\n"
        "Total express shipments crossed 740 million packages.\n"
        "End of summary."
    )
    result = selector.select_page_context(page_text, page_number=1)
    assert not result.is_empty
    for w in result.context_windows:
        assert w.source_text in page_text


def test_11_overlapping_regions():
    """11. Tests that overlapping regions merge gracefully into single window."""
    expander = ContextExpander(expansion_radius=2)
    regions = [
        DetectedRegion(page_number=1, start_line=3, end_line=3, anchor_text="100", signals=["numeric"], line_text="A: 100"),
        DetectedRegion(page_number=1, start_line=4, end_line=4, anchor_text="200", signals=["numeric"], line_text="B: 200"),
    ]
    page_text = "\n".join([f"Line {i}" for i in range(10)])
    windows = expander.expand_and_merge(regions, page_text, page_number=1)
    assert len(windows) == 1
    assert windows[0].start_line <= 1
    assert windows[0].end_line >= 6


def test_12_empty_page():
    """12. Tests handling of an empty or whitespace-only page."""
    selector = ContextSelector()
    result = selector.select_page_context("", page_number=1)
    assert result.is_empty
    assert result.selected_character_count == 0
    assert result.compression_ratio == 0.0
    assert len(result.context_windows) == 0


def test_13_no_number_page():
    """13. Tests handling of narrative text with zero numbers or quantifiers."""
    selector = ContextSelector()
    text = (
        "This chapter introduces the conceptual philosophy of the organisation.\n"
        "We strive to provide excellent customer service through dedicated values\n"
        "and collaborative teamwork across all branches."
    )
    result = selector.select_page_context(text, page_number=1)
    assert result.is_empty
    assert result.selected_character_count == 0


def test_14_numeric_noise_page():
    """14. Tests page with unanchored standalone single digits or disclaimers."""
    detector = RegionDetector(min_digit_count=2)
    text = "Section a . 1 . Refer to note 2 regarding paragraph 3 ."
    regions = detector.detect_regions(text, page_number=1)
    # Standalone single digits with no currency, metric, or unit context should not explode
    assert len(regions) == 0


def test_15_deterministic_repeated_output():
    """15. Tests that identical input produces exact identical output across multiple runs."""
    selector = ContextSelector(expansion_radius=2)
    text = "FY24 Revenue from Operations: ₹8,142 Cr (Growth: 12.4%). Active clients: 33,278."
    res1 = selector.select_page_context(text, page_number=1)
    res2 = selector.select_page_context(text, page_number=1)
    assert res1.model_dump() == res2.model_dump()


def test_16_bounded_output():
    """16. Tests that selected character count is strictly <= original character count."""
    selector = ContextSelector(expansion_radius=3)
    text = "\n".join([f"Metric row {i}: ₹{i * 100} Cr in FY2{i%5}" for i in range(20)])
    res = selector.select_page_context(text, page_number=1)
    assert res.selected_character_count <= res.original_character_count
    assert 0.0 <= res.compression_ratio <= 1.0


def test_17_no_paraphrasing():
    """17. Tests that the selector does not modify words, alter casing, or generate new text."""
    selector = ContextSelector(expansion_radius=1)
    text = "Line A: Company Alpha reported EBITDA of ₹550.75 Cr.\nLine B: Operating leverage improved."
    res = selector.select_page_context(text, page_number=1)
    for window in res.context_windows:
        # Every line in window must be an exact line from original text
        for line in window.source_text.splitlines():
            assert line in text


def test_18_no_llm_calls():
    """18. Confirms that ContextSelector is 100% deterministic and has no network/LLM dependencies."""
    selector = ContextSelector()
    assert not hasattr(selector, "client")
    assert not hasattr(selector, "gemini")
    assert not hasattr(selector, "model")


def test_19_no_filename_specific_rules():
    """19. Verifies that the selector does not inspect document_name for hardcoded filenames."""
    selector = ContextSelector()
    res1 = selector.select_page_context("Revenue ₹100 Cr", page_number=1, document_id="doc1")
    res2 = selector.select_page_context("Revenue ₹100 Cr", page_number=1, document_id="random_unseen_file.pdf")
    assert res1.selected_character_count == res2.selected_character_count
    assert res1.anchors == res2.anchors


def test_20_no_page_number_specific_rules():
    """20. Verifies that the selector behaves identically regardless of page_number."""
    selector = ContextSelector()
    res1 = selector.select_page_context("EBITDA ₹500 Cr", page_number=1)
    res2 = selector.select_page_context("EBITDA ₹500 Cr", page_number=999)
    assert res1.selected_character_count == res2.selected_character_count
    assert res1.signals == res2.signals


def test_21_arbitrary_entities():
    """21. Tests that arbitrary unseen entities (e.g. Acme Space Exploration) are captured correctly."""
    selector = ContextSelector(expansion_radius=2)
    text = (
        "Acme Space Exploration Private Limited\n"
        "Q3 FY26 Interim Performance\n"
        "Satellite launches reached 42 units generating ₹1,850 Cr in commercial revenue."
    )
    res = selector.select_page_context(text, page_number=1)
    assert not res.is_empty
    assert "Acme Space Exploration" in res.combined_source_text
    assert "₹1,850 Cr" in res.combined_source_text


def test_22_arbitrary_metrics():
    """22. Tests that arbitrary numeric metrics (e.g. quantum coherence duration 45.2 microseconds) are captured."""
    selector = ContextSelector(expansion_radius=2)
    text = (
        "Research Laboratory Findings\n"
        "Quantum coherence duration achieved 45.2% improvement over baseline with 8,500 test cycles."
    )
    res = selector.select_page_context(text, page_number=1)
    assert not res.is_empty
    assert "45.2%" in res.combined_source_text
    assert "8,500" in res.combined_source_text
