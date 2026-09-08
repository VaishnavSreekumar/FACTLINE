"""Deterministic Unit, Scale, and Monetary Normalization."""

from decimal import Decimal, InvalidOperation
import re
from typing import Optional, Tuple
from backend.models.normalization import NormalizationStatus, NormalizedValue


class UnitNormalizer:
    """Normalizes Indian (crore, lakh) and Western (million, billion) scales deterministically."""

    # Exact scale multipliers as Decimals to avoid IEEE-754 precision loss
    SCALES = {
        "crore": (Decimal("10000000"), "crore"),
        "crores": (Decimal("10000000"), "crore"),
        "cr": (Decimal("10000000"), "crore"),
        "crs": (Decimal("10000000"), "crore"),
        "lakh": (Decimal("100000"), "lakh"),
        "lakhs": (Decimal("100000"), "lakh"),
        "lac": (Decimal("100000"), "lakh"),
        "lacs": (Decimal("100000"), "lakh"),
        "thousand": (Decimal("1000"), "thousand"),
        "thousands": (Decimal("1000"), "thousand"),
        "k": (Decimal("1000"), "thousand"),
        "million": (Decimal("1000000"), "million"),
        "millions": (Decimal("1000000"), "million"),
        "mn": (Decimal("1000000"), "million"),
        "mns": (Decimal("1000000"), "million"),
        "billion": (Decimal("1000000000"), "billion"),
        "billions": (Decimal("1000000000"), "billion"),
        "bn": (Decimal("1000000000"), "billion"),
        "bns": (Decimal("1000000000"), "billion"),
        "trillion": (Decimal("1000000000000"), "trillion"),
        "trillions": (Decimal("1000000000000"), "trillion"),
        "tn": (Decimal("1000000000000"), "trillion"),
        "tns": (Decimal("1000000000000"), "trillion"),
    }

    SCALE_PATTERN = re.compile(
        r"\b(crores?|crs?|lakhs?|lacs?|thousands?|k|millions?|mns?|billions?|bns?|trillions?|tns?)\b",
        re.IGNORECASE,
    )

    # Currencies with explicit symbol / naming (using word boundaries for text tokens)
    EXPLICIT_CURRENCIES = [
        (re.compile(r"(?:₹|\binr\b|\brs\.?\b|\brupees?\b)", re.IGNORECASE), "INR"),
        (re.compile(r"(?:us\$|\busd\b|\bus\s+dollars?\b|\bu\.s\.\s*dollars?\b)", re.IGNORECASE), "USD"),
        (re.compile(r"(?:€|\beur\b|\beuros?\b)", re.IGNORECASE), "EUR"),
        (re.compile(r"(?:£|\bgbp\b|\bpounds?\b)", re.IGNORECASE), "GBP"),
    ]

    PERCENT_PATTERN = re.compile(r"(?:%|percent(?:age)?)", re.IGNORECASE)

    # Unicode dash normalization mapping: U+2014, U+2013, U+2212, U+2010, U+2011, U+2012, U+2015
    UNICODE_DASHES = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")

    # Explicit deterministic list of negative directional terms
    DIRECTIONAL_NEGATIVE_PATTERN = re.compile(
        r"\b(decline|declined|declines|declining|decrease|decreased|decreases|decreasing|drop|dropped|drops|dropping|reduction|reductions|reduced|reducing|fall|fell|falls|falling|contraction|contracted|contractions|contracting|loss|losses)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _extract_qualifier(cls, raw: str) -> Optional[str]:
        """Extracts comparison/inequality qualifiers such as '>', '<', '>=', '<=', '=', 'approx', 'around', etc."""
        raw_str = raw.strip()
        if raw_str.startswith(">="):
            return ">="
        elif raw_str.startswith("<="):
            return "<="
        elif raw_str.startswith(">"):
            return ">"
        elif raw_str.startswith("<"):
            return "<"
        elif raw_str.startswith("="):
            return "="
        elif raw_str.startswith("+") or raw_str.endswith("+"):
            return ">="
        elif raw_str.startswith("~"):
            return "~"

        raw_lower = raw_str.lower()
        if raw_lower.startswith("more than"):
            return ">"
        elif raw_lower.startswith("less than"):
            return "<"
        elif raw_lower.startswith("at least"):
            return ">="
        elif raw_lower.startswith("at most"):
            return "<="
        elif raw_lower.startswith("approximately"):
            return "approximately"
        elif raw_lower.startswith("approx.") or raw_lower.startswith("approx"):
            return "approx"
        elif raw_lower.startswith("around"):
            return "around"
        elif raw_lower.startswith("about"):
            return "about"
        elif raw_lower.startswith("nearly"):
            return "nearly"
        return None

    @classmethod
    def _extract_number(cls, raw: str) -> Optional[Decimal]:
        """Extracts the primary numeric component from a string using exact Decimal."""
        # 1. Normalize Unicode dashes to ASCII '-' first
        raw_norm = cls.UNICODE_DASHES.sub("-", raw)
        # Match standard integer or decimal with commas, optional leading + or - or >
        cleaned = re.sub(r"[^\d.,\-+]", "", raw_norm.replace(",", ""))
        if not cleaned:
            return None
        # Handle cases like '+17000' or '>33200' or '17000+'
        cleaned = cleaned.lstrip(">+~<=").rstrip("+")
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None

    @classmethod
    def _clean_semantic_unit(cls, unit_str: Optional[str]) -> Optional[str]:
        """Removes scale descriptors, numbers, and currency symbols from unit string to yield pure semantic measurement unit."""
        if not unit_str or not unit_str.strip():
            return None
        # Extract alphabetic words that are not scales, currencies, or generic numeric artifacts
        words = re.findall(r"[a-zA-Z]+", unit_str)
        meaningful_words = []
        for w in words:
            w_lower = w.lower()
            if w_lower in cls.SCALES:
                continue
            if w_lower in {"inr", "usd", "eur", "gbp", "rs", "rupee", "rupees", "dollar", "dollars", "percent", "percentage"}:
                continue
            if cls.DIRECTIONAL_NEGATIVE_PATTERN.match(w_lower):
                continue
            meaningful_words.append(w_lower)
        if meaningful_words:
            return " ".join(meaningful_words)
        return None

    @classmethod
    def normalize(
        cls,
        value_raw: str,
        unit: Optional[str] = None,
        value_numeric: Optional[float] = None,
    ) -> NormalizedValue:
        """Deterministically normalizes a raw value and unit.

        Args:
            value_raw: Exact string from source document (e.g., '₹81,415.38 million', '>4.8Mn tonnes').
            unit: Optional unit string extracted from source.
            value_numeric: Optional numeric component from extraction.

        Returns:
            NormalizedValue containing canonical magnitude, currency, scale, qualifier, and status.
        """
        combined = f"{value_raw} {unit or ''}".strip().lower()
        original_raw = value_raw
        qualifier = cls._extract_qualifier(value_raw)

        # Detect negative directional terms or negative extracted numeric value
        is_directional_neg = bool(cls.DIRECTIONAL_NEGATIVE_PATTERN.search(combined))
        is_extracted_neg = (value_numeric is not None and value_numeric < 0)

        # 1. Percentage Normalization (Invariant: 6.4% -> 6.4 percent, never 0.064)
        if cls.PERCENT_PATTERN.search(combined):
            num = cls._extract_number(value_raw)
            if num is None and value_numeric is not None:
                num = Decimal(str(value_numeric))

            if num is not None:
                # Never invert negative sign; apply negative direction if indicated
                if (is_directional_neg or is_extracted_neg) and num > 0:
                    num = -num

                return NormalizedValue(
                    numeric_value=float(num),
                    canonical_unit="percent",
                    scale=None,
                    value_qualifier=qualifier,
                    currency=None,
                    original_value_raw=original_raw,
                    normalization_status=NormalizationStatus.NORMALIZED,
                    normalization_notes="Percentage value retained as percentage points.",
                )
            return NormalizedValue(
                original_value_raw=original_raw,
                value_qualifier=qualifier,
                normalization_status=NormalizationStatus.UNRESOLVED,
                normalization_notes="Failed to parse numeric value for percentage.",
            )

        # 2. Currency Identification
        detected_currency: Optional[str] = None
        for pattern, curr_code in cls.EXPLICIT_CURRENCIES:
            if pattern.search(combined):
                detected_currency = curr_code
                break

        # Check for ambiguous bare '$' symbol without explicit US identifier
        ambiguous_dollar = False
        if not detected_currency and "$" in combined:
            ambiguous_dollar = True

        # 3. Scale Identification
        detected_multiplier = Decimal("1")
        detected_scale_label: Optional[str] = None

        # Look for scale tokens across words in combined string
        tokens = re.findall(r"[a-z]+", combined)
        for token in tokens:
            if token in cls.SCALES:
                mult, scale_lbl = cls.SCALES[token]
                detected_multiplier = mult
                detected_scale_label = scale_lbl
                break

        # 4. Numeric Value Extraction
        num_decimal = cls._extract_number(value_raw)
        if num_decimal is None and value_numeric is not None:
            num_decimal = Decimal(str(value_numeric))

        if num_decimal is None:
            return NormalizedValue(
                original_value_raw=original_raw,
                value_qualifier=qualifier,
                normalization_status=NormalizationStatus.UNRESOLVED,
                normalization_notes="Unable to extract numeric component from raw value.",
            )

        # Never invert negative sign; apply negative direction if indicated
        if (is_directional_neg or is_extracted_neg) and num_decimal > 0:
            num_decimal = -num_decimal

        # Exact Decimal calculation
        canonical_magnitude = num_decimal * detected_multiplier

        # 5. Determine Canonical Semantic Unit and Status
        canonical_unit: Optional[str] = None
        status = NormalizationStatus.NORMALIZED
        notes: Optional[str] = None

        if detected_currency:
            canonical_unit = detected_currency
        elif ambiguous_dollar:
            # Invariant: bare '$' must NOT be automatically converted to USD without context
            canonical_unit = None
            status = NormalizationStatus.PARTIAL
            notes = "Ambiguous currency symbol '$' without explicit country code."
        else:
            # Clean semantic unit by stripping scale descriptors from extracted unit
            semantic_from_unit = cls._clean_semantic_unit(unit)
            if semantic_from_unit:
                canonical_unit = semantic_from_unit
            else:
                # Check for residual semantic measurement words in value_raw (e.g. '>4.8Mn tonnes' -> 'tonnes')
                semantic_from_raw = cls._clean_semantic_unit(value_raw)
                if semantic_from_raw:
                    canonical_unit = semantic_from_raw
                else:
                    canonical_unit = "count"

        return NormalizedValue(
            numeric_value=float(canonical_magnitude),
            canonical_unit=canonical_unit,
            scale=detected_scale_label,
            value_qualifier=qualifier,
            currency=detected_currency,
            original_value_raw=original_raw,
            normalization_status=status,
            normalization_notes=notes,
        )
