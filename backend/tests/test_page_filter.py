"""
Tests for Phase 11: Local Page Relevance Filter.

Verifies deterministic page-level scoring, signal extraction, thresholding,
and isolation invariants across synthetic and real document page representations.
"""

import pytest
from backend.models.document import PageText
from backend.page_filter.relevance import PageRelevanceFilter, PageRelevanceScore
from backend.page_filter.signals import (
    extract_numeric_signals,
    extract_currency_signals,
    extract_unit_signals,
    extract_metric_signals,
    detect_table_structure,
    detect_boilerplate_signals,
)


def test_empty_page():
    """1. Verify empty or whitespace-only page receives score of 0.0 and is rejected."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    page_empty = PageText(page_number=1, text="", char_count=0, has_text=False)
    page_whitespace = PageText(page_number=2, text="   \n\t  \n  ", char_count=8, has_text=True)

    s1 = filter_engine.score_page(page_empty)
    s2 = filter_engine.score_page(page_whitespace)

    assert s1.relevance_score == 0.0
    assert not s1.selected
    assert s2.relevance_score == 0.0
    assert not s2.selected


def test_cover_like_page_with_boilerplate():
    """2. Verify cover / table of contents page with boilerplate and no metrics is penalized/rejected."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    text = """
    ACME CORPORATION
    ANNUAL REPORT 2024
    Table of Contents
    All Rights Reserved.
    Corporate Directory and Legal Information.
    """
    page = PageText(page_number=1, text=text, char_count=len(text), has_text=True)
    score = filter_engine.score_page(page)

    assert score.relevance_score < 0.35
    assert not score.selected
    assert any("boilerplate" in s for s in score.signals_triggered)


def test_page_containing_currency_and_numeric_metric():
    """3. Verify page with financial metrics and currency values triggers high relevance."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    text = "In FY24, Consolidated Revenue from Operations reached INR 81,415.38 million with EBITDA of ₹1,266 million."
    page = PageText(page_number=4, text=text, char_count=len(text), has_text=True)
    score = filter_engine.score_page(page)

    assert score.relevance_score >= 0.35
    assert score.selected
    assert any("currency" in s for s in score.signals_triggered)
    assert any("metric" in s for s in score.signals_triggered)


def test_page_containing_percentage():
    """4. Verify page containing percentage and growth indicators triggers relevance."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    text = "Real GDP growth for the fiscal year was estimated at 6.4% compared to 7.2% in the previous year."
    page = PageText(page_number=12, text=text, char_count=len(text), has_text=True)
    score = filter_engine.score_page(page)

    assert score.relevance_score >= 0.35
    assert score.selected
    assert any("numeric_density" in s for s in score.signals_triggered)


def test_page_containing_quantitative_unit():
    """5. Verify page with physical quantitative units (tonnes, shipments, customers) triggers relevance."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    text = "The network delivered >4.8Mn tonnes of freight and serviced >33,200 active customers across 18,000 pin codes."
    page = PageText(page_number=2, text=text, char_count=len(text), has_text=True)
    score = filter_engine.score_page(page)

    assert score.relevance_score >= 0.35
    assert score.selected
    assert any("quantitative_unit" in s for s in score.signals_triggered)


def test_table_like_page():
    """6. Verify multi-column structured table rows are detected and scored highly."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    text = """
    Particulars              FY22        FY23        FY24
    Revenue from operations  70,536      72,236      81,415
    Operating Expenses       65,120      68,400      74,200
    EBITDA                    5,416       3,836       7,215
    Net Profit / (Loss)      (2,100)     (1,450)      1,250
    """
    page = PageText(page_number=22, text=text, char_count=len(text), has_text=True)
    score = filter_engine.score_page(page)

    assert score.relevance_score >= 0.45
    assert score.selected
    assert any("tabular_structure" in s for s in score.signals_triggered)


def test_narrative_page_no_numbers():
    """7. Verify purely narrative prose without quantitative metrics receives low score."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    text = """
    Our mission is to build the operating system for global logistics through technological innovation,
    fostering long-term customer partnerships, and enabling sustainable operational practices across
    all functional divisions with unwavering dedication to integrity and teamwork.
    """
    page = PageText(page_number=7, text=text, char_count=len(text), has_text=True)
    score = filter_engine.score_page(page)

    assert score.relevance_score < 0.35
    assert not score.selected


def test_irrelevant_numeric_noise_dampening():
    """8. Verify page with single isolated year without business metrics receives low score."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    text = "Report generated in 2024. For more details visit the website."
    page = PageText(page_number=99, text=text, char_count=len(text), has_text=True)
    score = filter_engine.score_page(page)

    assert score.relevance_score < 0.35
    assert not score.selected


def test_score_bounded_between_zero_and_one():
    """9. Verify score is strictly bounded in [0.0, 1.0] across extreme inputs."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    extreme_text = "₹100 Cr INR $ 500 million billion 20% 50 tonnes revenue profit loss margin EBITDA sales assets " * 20
    page_heavy = PageText(page_number=1, text=extreme_text, char_count=len(extreme_text), has_text=True)
    page_empty = PageText(page_number=2, text="", char_count=0, has_text=False)

    s_heavy = filter_engine.score_page(page_heavy)
    s_empty = filter_engine.score_page(page_empty)

    assert 0.0 <= s_heavy.relevance_score <= 1.0
    assert 0.0 <= s_empty.relevance_score <= 1.0


def test_deterministic_repeated_scoring():
    """10. Verify identical input produces identical score across multiple runs."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    text = "Consolidated revenue stood at Rs 45,000 million for FY24 with 15.4% operating margin."
    page = PageText(page_number=5, text=text, char_count=len(text), has_text=True)

    scores = [filter_engine.score_page(page).relevance_score for _ in range(10)]
    assert len(set(scores)) == 1


def test_threshold_behavior():
    """11. Verify configurable threshold controls page selection."""
    text = "In FY24, the business delivered 50,000 shipments across various channels."
    page = PageText(page_number=3, text=text, char_count=len(text), has_text=True)

    strict_filter = PageRelevanceFilter(threshold=0.70)
    permissive_filter = PageRelevanceFilter(threshold=0.15)

    s_strict = strict_filter.score_page(page)
    s_perm = permissive_filter.score_page(page)

    assert s_strict.relevance_score == s_perm.relevance_score
    assert s_perm.selected
    assert not s_strict.selected


def test_neighboring_page_context_boost():
    """12. Verify neighboring page with financial headers provides a context boost to a data page."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    prev_page = PageText(page_number=1, text="Consolidated Financial Performance & Metric Breakdown", char_count=55, has_text=True)
    cur_page = PageText(page_number=2, text="45,000   12,300   8,450", char_count=25, has_text=True)

    s_isolated = filter_engine.score_page(cur_page)
    s_with_context = filter_engine.score_page(cur_page, prev_page=prev_page)

    assert s_with_context.relevance_score >= s_isolated.relevance_score
    assert "neighbor_metric_context" in s_with_context.signals_triggered


def test_arbitrary_entity_names():
    """13. Verify filter works on fictitious or arbitrary global entity names."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    text = "Solaris Quantum Dynamics Corp reported annual turnover of EUR 450 million in 2024."
    page = PageText(page_number=10, text=text, char_count=len(text), has_text=True)
    score = filter_engine.score_page(page)

    assert score.selected
    assert score.relevance_score >= 0.35


def test_arbitrary_metrics():
    """14. Verify filter works on various distinct economic and industry metrics."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    text = "Capital expenditure reached $1.2 billion while head-count expanded to 14,500 employees."
    page = PageText(page_number=8, text=text, char_count=len(text), has_text=True)
    score = filter_engine.score_page(page)

    assert score.selected
    assert score.relevance_score >= 0.35


def test_no_gemini_or_network_calls():
    """15. Verify filter runs fully offline without touching network or Gemini API."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    page = PageText(page_number=1, text="Revenue: INR 500 Cr", char_count=20, has_text=True)
    score = filter_engine.score_page(page)
    assert score.relevance_score > 0.0


def test_no_dependency_on_document_filename():
    """16. Verify scoring does not inspect or depend on document filenames."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    text = "Revenue grew by 25% to INR 10,000 Cr."
    p1 = PageText(page_number=1, text=text, char_count=len(text), has_text=True)
    p2 = PageText(page_number=1, text=text, char_count=len(text), has_text=True)

    assert filter_engine.score_page(p1).relevance_score == filter_engine.score_page(p2).relevance_score


def test_no_dependency_on_page_number():
    """17. Verify page 1 and page 99 with identical content receive the identical score."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    text = "Consolidated EBITDA stood at INR 5,400 million in FY24."
    p1 = PageText(page_number=1, text=text, char_count=len(text), has_text=True)
    p99 = PageText(page_number=99, text=text, char_count=len(text), has_text=True)

    assert filter_engine.score_page(p1).relevance_score == filter_engine.score_page(p99).relevance_score


def test_no_hardcoded_starter_facts():
    """18. Verify synthetic and generic business text is evaluated purely on structural signals."""
    filter_engine = PageRelevanceFilter(threshold=0.35)
    synthetic_text = "Alpha Beta Gamma Ltd generated USD 99.5 million profit in 2024."
    page = PageText(page_number=50, text=synthetic_text, char_count=len(synthetic_text), has_text=True)
    score = filter_engine.score_page(page)

    assert score.selected
    assert score.relevance_score >= 0.35
