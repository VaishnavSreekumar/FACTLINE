"""Date and temporal period normalization tests."""

import pytest
from backend.normalization.dates import DateNormalizer


@pytest.fixture
def normalizer():
    return DateNormalizer()


def test_bare_fy_requires_established_convention(normalizer):
    """Test that bare FY24 does NOT blindly assume April-March unless explicitly established."""
    # Bare FY24 without convention -> returns None with warning
    start, end, warning = normalizer.normalize_period("FY24")
    assert start is None
    assert end is None
    assert "Fiscal-year convention not explicitly established" in (warning or "")

    # FY24 with explicit april-march context -> safely normalized
    start_c, end_c, warn_c = normalizer.normalize_period("FY24", fiscal_calendar="april-march")
    assert start_c == "2023-04-01"
    assert end_c == "2024-03-31"
    assert warn_c is None


def test_explicit_cross_year_fiscal_span(normalizer):
    """Test explicit cross-year span (e.g. FY 2023-24) is safely normalized."""
    start1, end1, warn1 = normalizer.normalize_period("FY 2023-24")
    assert start1 == "2023-04-01"
    assert end1 == "2024-03-31"
    assert warn1 is None

    start2, end2, warn2 = normalizer.normalize_period("2023-24")
    assert start2 == "2023-04-01"
    assert end2 == "2024-03-31"
    assert warn2 is None


def test_quarter_normalization(normalizer):
    """Test fiscal quarter parsing with explicit span and convention requirements."""
    # Explicit span quarter: Q1 FY 2023-24 -> 2023-04-01 to 2023-06-30
    s_q1, e_q1, _ = normalizer.normalize_period("Q1 FY 2023-24")
    assert s_q1 == "2023-04-01"
    assert e_q1 == "2023-06-30"

    # Bare Q4 FY24 without convention -> None
    s_bare, e_bare, warn_bare = normalizer.normalize_period("Q4 FY24")
    assert s_bare is None
    assert e_bare is None
    assert "convention not explicitly established" in (warn_bare or "")

    # Bare Q4 FY24 with april-march convention -> 2024-01-01 to 2024-03-31
    s_q4, e_q4, _ = normalizer.normalize_period("Q4 FY24", fiscal_calendar="april-march")
    assert s_q4 == "2024-01-01"
    assert e_q4 == "2024-03-31"


def test_exact_date_normalization(normalizer):
    """Test exact text and ISO dates."""
    s1, e1, _ = normalizer.normalize_period("March 31, 2024")
    assert s1 == "2024-03-31"
    assert e1 == "2024-03-31"

    s2, e2, _ = normalizer.normalize_period("31 March 2024")
    assert s2 == "2024-03-31"
    assert e2 == "2024-03-31"

    s3, e3, _ = normalizer.normalize_period("2024-03-31")
    assert s3 == "2024-03-31"
    assert e3 == "2024-03-31"


def test_calendar_year_normalization(normalizer):
    """Test explicit calendar year notation."""
    s_cy, e_cy, _ = normalizer.normalize_period("CY2024")
    assert s_cy == "2024-01-01"
    assert e_cy == "2024-12-31"


def test_unresolvable_date_label(normalizer):
    """Test unresolvable or empty date string returns None and warning."""
    s_unres, e_unres, warn_unres = normalizer.normalize_period("Period of Growth")
    assert s_unres is None
    assert e_unres is None
    assert "Unresolvable temporal label" in (warn_unres or "")
