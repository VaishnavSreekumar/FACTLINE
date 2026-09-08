"""Orchestration service for deterministic fact normalization."""

from typing import List, Optional
from backend.models.fact import FactRecord, TimePeriod
from backend.models.normalization import NormalizedFact, NormalizedValue
from backend.normalization.dates import DateNormalizer
from backend.normalization.entities import EntityNormalizer, MetricNormalizer
from backend.normalization.units import UnitNormalizer


class FactNormalizer:
    """Deterministically normalizes FactRecords into canonical NormalizedFact representations."""

    def __init__(self):
        self.unit_normalizer = UnitNormalizer()
        self.date_normalizer = DateNormalizer()
        self.entity_normalizer = EntityNormalizer()
        self.metric_normalizer = MetricNormalizer()

    def normalize(
        self,
        fact: FactRecord,
        fiscal_calendar: Optional[str] = None,
    ) -> NormalizedFact:
        """Normalizes an individual FactRecord deterministically without mutating source fact.

        Args:
            fact: Immutable source FactRecord.
            fiscal_calendar: Optional established fiscal calendar convention ('april-march', 'calendar').

        Returns:
            Derived NormalizedFact preserving original FactRecord and computed canonical dimensions.
        """
        warnings: List[str] = []

        # 1. Numeric and Unit / Monetary Normalization
        normalized_value = self.unit_normalizer.normalize(
            value_raw=fact.value_raw,
            unit=fact.unit,
            value_numeric=fact.value_numeric,
        )
        if normalized_value.normalization_notes:
            warnings.append(normalized_value.normalization_notes)

        # 2. Date / Time Period Normalization
        start_date, end_date, date_warn = self.date_normalizer.normalize_period(
            label=fact.time_period.label,
            fiscal_calendar=fiscal_calendar,
            existing_start=fact.time_period.start_date,
            existing_end=fact.time_period.end_date,
        )

        normalized_time_period: Optional[TimePeriod] = None
        if start_date and end_date:
            normalized_time_period = TimePeriod(
                label=fact.time_period.label,
                start_date=start_date,
                end_date=end_date,
            )
        elif date_warn:
            warnings.append(date_warn)

        # 3. Entity Canonicalization (Presentation-friendly casing + legal suffix removal)
        canonical_entity = self.entity_normalizer.normalize(fact.entity)

        # 4. Metric Canonicalization (Deterministic whitespace and case standardization)
        canonical_metric = self.metric_normalizer.normalize(fact.metric)

        return NormalizedFact(
            fact=fact,
            normalized_value=normalized_value,
            canonical_entity=canonical_entity,
            canonical_metric=canonical_metric,
            normalized_time_period=normalized_time_period,
            normalization_warnings=warnings,
        )

    def normalize_batch(
        self,
        facts: List[FactRecord],
        fiscal_calendar: Optional[str] = None,
    ) -> List[NormalizedFact]:
        """Normalizes a list of FactRecords deterministically."""
        return [self.normalize(fact, fiscal_calendar=fiscal_calendar) for fact in facts]
