"""Unit, monetary, and scale normalization tests."""

import pytest
from backend.models.normalization import NormalizationStatus
from backend.normalization.units import UnitNormalizer


@pytest.fixture
def normalizer():
    return UnitNormalizer()


def test_monetary_crore_and_million_scaling(normalizer):
    """Test exact Decimal scaling for Indian and Western monetary representations."""
    # ₹8,142 Cr -> exactly 81,420,000,000 INR
    res1 = normalizer.normalize("₹8,142 Cr")
    assert res1.numeric_value == 81420000000.0
    assert res1.canonical_unit == "INR"
    assert res1.currency == "INR"
    assert res1.scale == "crore"
    assert res1.normalization_status == NormalizationStatus.NORMALIZED

    # ₹8,142 crore
    res2 = normalizer.normalize("₹8,142 crore")
    assert res2.numeric_value == 81420000000.0
    assert res2.canonical_unit == "INR"

    # INR 8,142 crore
    res3 = normalizer.normalize("INR 8,142 crore")
    assert res3.numeric_value == 81420000000.0
    assert res3.canonical_unit == "INR"

    # ₹81,415.38 million -> exactly 81,415,380,000 INR
    res4 = normalizer.normalize("₹81,415.38 million")
    assert res4.numeric_value == 81415380000.0
    assert res4.canonical_unit == "INR"
    assert res4.currency == "INR"
    assert res4.scale == "million"
    assert res4.normalization_status == NormalizationStatus.NORMALIZED


def test_indian_units_scale(normalizer):
    """Test 1 lakh, 1 crore, 10 crore multipliers."""
    res_lakh = normalizer.normalize("1 lakh", unit="INR")
    assert res_lakh.numeric_value == 100000.0
    assert res_lakh.scale == "lakh"

    res_cr = normalizer.normalize("1 crore")
    assert res_cr.numeric_value == 10000000.0
    assert res_cr.scale == "crore"

    res_10cr = normalizer.normalize("10 crore")
    assert res_10cr.numeric_value == 100000000.0
    assert res_10cr.scale == "crore"


def test_western_units_scale(normalizer):
    """Test 1 million, 1 billion, 1 trillion multipliers."""
    res_mn = normalizer.normalize("1 million")
    assert res_mn.numeric_value == 1000000.0
    assert res_mn.scale == "million"

    res_bn = normalizer.normalize("1 billion")
    assert res_bn.numeric_value == 1000000000.0
    assert res_bn.scale == "billion"

    res_tn = normalizer.normalize("1 trillion")
    assert res_tn.numeric_value == 1000000000000.0
    assert res_tn.scale == "trillion"


def test_percentage_preservation(normalizer):
    """Test percentage is preserved as percentage points and never converted to 0.064."""
    res_pct = normalizer.normalize("6.4%")
    assert res_pct.numeric_value == 6.4
    assert res_pct.canonical_unit == "percent"
    assert res_pct.numeric_value != 0.064
    assert res_pct.normalization_status == NormalizationStatus.NORMALIZED

    res_pct_text = normalizer.normalize("6.4 percent")
    assert res_pct_text.numeric_value == 6.4
    assert res_pct_text.canonical_unit == "percent"


def test_quantity_count_normalization(normalizer):
    """Test count parsing with comma delimiters."""
    res_count = normalizer.normalize("33,278", unit="customers")
    assert res_count.numeric_value == 33278.0
    assert res_count.canonical_unit == "customers"
    assert res_count.normalization_status == NormalizationStatus.NORMALIZED

    res_count_prefix = normalizer.normalize(">33,200")
    assert res_count_prefix.numeric_value == 33200.0
    assert res_count_prefix.canonical_unit == "count"


def test_ambiguous_dollar_safety(normalizer):
    """Test that bare '$' is NOT automatically assumed to be USD."""
    res_bare_dollar = normalizer.normalize("$500 million")
    # Invariant: bare '$' must remain unresolved/partial currency
    assert res_bare_dollar.currency is None
    assert res_bare_dollar.normalization_status == NormalizationStatus.PARTIAL
    assert "Ambiguous currency symbol '$'" in (res_bare_dollar.normalization_notes or "")

    # Explicit US$ or USD is safely recognized
    res_explicit = normalizer.normalize("US$ 500 million")
    assert res_explicit.currency == "USD"
    assert res_explicit.canonical_unit == "USD"
    assert res_explicit.numeric_value == 500000000.0
    assert res_explicit.normalization_status == NormalizationStatus.NORMALIZED


def test_invalid_unresolvable_value(normalizer):
    """Test non-numeric unresolvable text produces UNRESOLVED status."""
    res_invalid = normalizer.normalize("N/A or unavailable")
    assert res_invalid.numeric_value is None
    assert res_invalid.normalization_status == NormalizationStatus.UNRESOLVED
    assert res_invalid.normalization_notes is not None


def test_tonnes_and_shipments_scale_and_qualifier_separation(normalizer):
    """Phase 9 Test: Scale and semantic measurement unit are strictly separated."""
    # >4.8Mn tonnes -> numeric=4,800,000, unit='tonnes', scale='million', qualifier='>'
    res1 = normalizer.normalize(">4.8Mn tonnes", unit="million tonnes")
    assert res1.numeric_value == 4800000.0
    assert res1.canonical_unit == "tonnes"
    assert res1.scale == "million"
    assert res1.value_qualifier == ">"
    assert res1.normalization_status == NormalizationStatus.NORMALIZED

    # >4.8Mn tonnes without unit param
    res1_b = normalizer.normalize(">4.8Mn tonnes")
    assert res1_b.numeric_value == 4800000.0
    assert res1_b.canonical_unit == "tonnes"
    assert res1_b.scale == "million"
    assert res1_b.value_qualifier == ">"

    # >2.8Bn shipments -> numeric=2,800,000,000, unit='shipments', scale='billion', qualifier='>'
    res2 = normalizer.normalize(">2.8Bn", unit="shipments")
    assert res2.numeric_value == 2800000000.0
    assert res2.canonical_unit == "shipments"
    assert res2.scale == "billion"
    assert res2.value_qualifier == ">"
    assert res2.normalization_status == NormalizationStatus.NORMALIZED


def test_qualifiers_preservation(normalizer):
    """Phase 9 Test: Inequalities and lower bounds are preserved."""
    assert normalizer.normalize(">=500").value_qualifier == ">="
    assert normalizer.normalize("<=200").value_qualifier == "<="
    assert normalizer.normalize(">1000").value_qualifier == ">"
    assert normalizer.normalize("<50").value_qualifier == "<"
    assert normalizer.normalize("=100").value_qualifier == "="
    assert normalizer.normalize("17,000+").value_qualifier == ">="
    assert normalizer.normalize("100").value_qualifier is None
