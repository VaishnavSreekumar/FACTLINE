"""End-to-end Fact Normalization integration and preservation tests."""

import pytest
from backend.models.fact import EpistemicStatus, FactRecord, Provenance, TimePeriod
from backend.models.normalization import NormalizationStatus, NormalizedFact
from backend.normalization.entities import EntityNormalizer, MetricNormalizer
from backend.normalization.normalizer import FactNormalizer


@pytest.fixture
def normalizer():
    return FactNormalizer()


def test_entity_canonicalization_presentation_friendly():
    """Verify entity canonicalization strips legal suffixes while preserving presentation casing."""
    assert EntityNormalizer.normalize("Delhivery Limited") == "Delhivery"
    assert EntityNormalizer.normalize("Delhivery Ltd.") == "Delhivery"
    assert EntityNormalizer.normalize("Delhivery Ltd") == "Delhivery"
    assert EntityNormalizer.normalize("Tata Motors Pvt. Ltd.") == "Tata Motors"
    assert EntityNormalizer.normalize("Tata Motors Private Limited") == "Tata Motors"
    assert EntityNormalizer.normalize("Apple Inc.") == "Apple"
    assert EntityNormalizer.normalize("Microsoft Corporation") == "Microsoft"
    assert EntityNormalizer.normalize("Infosys Limited") == "Infosys"


def test_metric_canonicalization_no_semantic_merging():
    """Verify metric canonicalization cleans casing/whitespace without conflating distinct metrics."""
    assert MetricNormalizer.normalize("Revenue from Operations") == "revenue from operations"
    assert MetricNormalizer.normalize("  revenue   from  operations  ") == "revenue from operations"
    assert MetricNormalizer.normalize("Operating Revenue") == "operating revenue"
    assert MetricNormalizer.normalize("Net Revenue") == "net revenue"

    # Distinct metrics remain distinct strings
    assert MetricNormalizer.normalize("Revenue from Operations") != MetricNormalizer.normalize("Revenue")


def test_fact_record_preservation_invariant(normalizer):
    """Confirm normalization NEVER mutates or destroys the source FactRecord."""
    prov = Provenance(
        document_id="delhivery-fy24",
        document_date="2024-05-17",
        page_number=12,
        supporting_text="Revenue from operations was ₹81,415.38 million in FY24.",
    )
    original_fact = FactRecord(
        fact_id="fact-delhivery-rev",
        entity="Delhivery Limited",
        metric="Revenue from operations",
        value_raw="₹81,415.38 million",
        value_numeric=81415.38,
        unit="million INR",
        time_period=TimePeriod(label="FY 2023-24"),
        scope="Consolidated",
        geography="India",
        epistemic_status=EpistemicStatus.REPORTED,
        provenance=prov,
        extraction_confidence=0.98,
    )

    normalized = normalizer.normalize(original_fact)

    # Source fact is intact and unaltered
    assert normalized.fact.value_raw == "₹81,415.38 million"
    assert normalized.fact.entity == "Delhivery Limited"
    assert normalized.fact.metric == "Revenue from operations"
    assert normalized.fact.time_period.label == "FY 2023-24"
    assert normalized.fact.provenance.supporting_text == "Revenue from operations was ₹81,415.38 million in FY24."

    # Derived normalization fields
    assert normalized.canonical_entity == "Delhivery"
    assert normalized.canonical_metric == "revenue from operations"
    assert normalized.normalized_value.numeric_value == 81415380000.0
    assert normalized.normalized_value.canonical_unit == "INR"
    assert normalized.normalized_value.currency == "INR"
    assert normalized.normalized_value.scale == "million"
    assert normalized.normalized_time_period is not None
    assert normalized.normalized_time_period.start_date == "2023-04-01"
    assert normalized.normalized_time_period.end_date == "2024-03-31"


def test_starter_dataset_crore_and_million_canonical_representations(normalizer):
    """Verify ₹8,142 Cr and ₹81,415.38 million normalize to canonical numeric representations."""
    prov = Provenance(document_id="d1", page_number=1, supporting_text="Text")

    # Claim A: ₹8,142 Cr
    fact_a = FactRecord(
        fact_id="fact-a",
        entity="Delhivery Limited",
        metric="Revenue from operations",
        value_raw="₹8,142 Cr",
        time_period=TimePeriod(label="FY 2023-24"),
        provenance=prov,
    )

    # Claim B: ₹81,415.38 million
    fact_b = FactRecord(
        fact_id="fact-b",
        entity="Delhivery Limited",
        metric="Revenue from operations",
        value_raw="₹81,415.38 million",
        time_period=TimePeriod(label="FY 2023-24"),
        provenance=prov,
    )

    norm_a = normalizer.normalize(fact_a)
    norm_b = normalizer.normalize(fact_b)

    assert norm_a.normalized_value.numeric_value == 81420000000.0
    assert norm_a.normalized_value.canonical_unit == "INR"
    assert norm_a.normalized_value.currency == "INR"

    assert norm_b.normalized_value.numeric_value == 81415380000.0
    assert norm_b.normalized_value.canonical_unit == "INR"
    assert norm_b.normalized_value.currency == "INR"


def test_determinism_invariant(normalizer):
    """Verify that identical inputs produce identical NormalizedFact objects."""
    prov = Provenance(document_id="d1", page_number=5, supporting_text="GDP growth 6.4%")
    fact = FactRecord(
        fact_id="fact-gdp",
        entity="India",
        metric="Real GDP Growth",
        value_raw="6.4%",
        time_period=TimePeriod(label="2024-25"),
        provenance=prov,
    )

    norm1 = normalizer.normalize(fact)
    norm2 = normalizer.normalize(fact)

    assert norm1.model_dump() == norm2.model_dump()


def test_batch_normalization(normalizer):
    """Verify batch normalization across multiple facts."""
    prov = Provenance(document_id="d1", page_number=1, supporting_text="test")
    facts = [
        FactRecord(
            fact_id=f"f-{i}",
            entity="Company Ltd.",
            metric="Metric",
            value_raw=f"{i * 10} million",
            time_period=TimePeriod(label="2024"),
            provenance=prov,
        )
        for i in range(1, 4)
    ]

    batch_norm = normalizer.normalize_batch(facts)
    assert len(batch_norm) == 3
    assert batch_norm[0].canonical_entity == "Company"
    assert batch_norm[0].normalized_value.numeric_value == 10000000.0
    assert batch_norm[1].normalized_value.numeric_value == 20000000.0
    assert batch_norm[2].normalized_value.numeric_value == 30000000.0
