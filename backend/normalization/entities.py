"""Deterministic Entity and Metric Canonicalization Module."""

import re


class EntityNormalizer:
    """Normalizes entity names into presentation-friendly canonical forms deterministically."""

    # Generic legal corporate suffixes (case-insensitive, matched at end of string)
    LEGAL_SUFFIX_PATTERN = re.compile(
        r"[\s,]+(?:private\s+limited|pvt\.?\s*ltd\.?|limited|ltd\.?|incorporated|inc\.?|corporation|corp\.?|llc|plc|gmbh|s\.a\.?|n\.v\.?)\.?$",
        re.IGNORECASE,
    )

    @classmethod
    def normalize(cls, raw_entity: str) -> str:
        """Resolves raw entity name into presentation-friendly canonical base name.

        Example:
            'Delhivery Limited' -> 'Delhivery'
            'Tata Motors Ltd.' -> 'Tata Motors'

        Args:
            raw_entity: Raw entity string from extraction.

        Returns:
            Cleaned, canonical entity name preserving presentation casing.
        """
        if not raw_entity or not raw_entity.strip():
            return ""

        # Normalize inner whitespace
        cleaned = " ".join(raw_entity.split())

        # Strip trailing legal suffix generically
        canonical = cls.LEGAL_SUFFIX_PATTERN.sub("", cleaned).strip()

        # Remove any lingering trailing comma or period
        canonical = canonical.rstrip(",. ")

        return canonical if canonical else cleaned


class MetricNormalizer:
    """Standardizes metric text representations without performing semantic conflation."""

    @classmethod
    def normalize(cls, raw_metric: str) -> str:
        """Standardizes whitespace and case formatting of a metric.

        Preserves distinct metric phrasing without semantic collapsing.

        Args:
            raw_metric: Raw metric text from extraction.

        Returns:
            Whitespace-normalized lowercase metric string.
        """
        if not raw_metric or not raw_metric.strip():
            return ""

        # Collapse whitespace and convert to lowercase for deterministic comparison
        return " ".join(raw_metric.split()).lower()
