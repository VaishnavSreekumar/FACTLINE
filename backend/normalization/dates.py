"""Date and temporal period normalization module."""

import re
from typing import Optional, Tuple
from backend.models.fact import TimePeriod


class DateNormalizer:
    """Parses fiscal years, quarters, and exact dates into bounded ISO ranges deterministically."""

    # Month name to 2-digit mapping
    MONTH_MAP = {
        "january": "01", "jan": "01",
        "february": "02", "feb": "02",
        "march": "03", "mar": "03",
        "april": "04", "apr": "04",
        "may": "05",
        "june": "06", "jun": "06",
        "july": "07", "jul": "07",
        "august": "08", "aug": "08",
        "september": "09", "sep": "09", "sept": "09",
        "october": "10", "oct": "10",
        "november": "11", "nov": "11",
        "december": "12", "dec": "12",
    }

    # Regex patterns
    # Exact ISO date: 2024-03-31
    ISO_DATE_PATTERN = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

    # Text date: March 31, 2024 or 31 March 2024
    TEXT_DATE_1 = re.compile(r"(?:as\s+of\s+)?([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", re.IGNORECASE)
    TEXT_DATE_2 = re.compile(r"(?:as\s+of\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+),?\s+(\d{4})", re.IGNORECASE)

    # Explicit cross-year fiscal span: FY 2023-24 or FY 2023-2024 or 2023-24
    SPAN_FY_PATTERN = re.compile(r"(?:fy|fiscal\s+year)?\s*(\d{4})\s*[-/]\s*(\d{2,4})", re.IGNORECASE)

    # Bare FY: FY24, FY 2024, FY2024
    BARE_FY_PATTERN = re.compile(r"^fy\s*(\d{2,4})$", re.IGNORECASE)

    # Quarter: Q1 FY24, Q4 FY 2023-24, Q4 FY24
    QUARTER_SPAN_PATTERN = re.compile(r"q([1-4])\s*(?:fy|fiscal\s+year)?\s*(\d{4})\s*[-/]\s*(\d{2,4})", re.IGNORECASE)
    BARE_QUARTER_PATTERN = re.compile(r"q([1-4])\s*(?:fy\s*)?(\d{2,4})", re.IGNORECASE)

    # Calendar Year: CY2024 or Calendar Year 2024
    CY_PATTERN = re.compile(r"^(?:cy|calendar\s+year)\s*(\d{4})$", re.IGNORECASE)

    @classmethod
    def _parse_short_year(cls, yy_str: str) -> int:
        """Expands 2-digit year to 4-digit century year."""
        val = int(yy_str)
        if val < 100:
            return 2000 + val
        return val

    @classmethod
    def normalize_period(
        cls,
        label: str,
        fiscal_calendar: Optional[str] = None,
        existing_start: Optional[str] = None,
        existing_end: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Normalizes a temporal label into bounded ISO dates (start_date, end_date, warning).

        Args:
            label: Raw time period label (e.g. 'March 31, 2024', 'FY 2023-24', 'FY24').
            fiscal_calendar: Optional established convention ('april-march', 'calendar').
                             If None, bare FY labels will NOT be assumed as April-March.
            existing_start: Existing ISO start date if already validated.
            existing_end: Existing ISO end date if already validated.

        Returns:
            Tuple of (start_date, end_date, warning_message).
        """
        if existing_start and existing_end:
            return existing_start, existing_end, None

        if not label or not label.strip():
            return None, None, "Empty temporal label."

        raw = label.strip()

        # 1. Exact ISO Date check
        iso_match = cls.ISO_DATE_PATTERN.search(raw)
        if iso_match:
            d = f"{iso_match.group(1)}-{iso_match.group(2)}-{iso_match.group(3)}"
            return d, d, None

        # 2. Text Date check: March 31, 2024
        m1 = cls.TEXT_DATE_1.search(raw)
        if m1:
            month_name = m1.group(1).lower()
            day = int(m1.group(2))
            year = int(m1.group(3))
            month_num = cls.MONTH_MAP.get(month_name)
            if month_num:
                d = f"{year:04d}-{month_num}-{day:02d}"
                return d, d, None

        # 2b. Text Date check: 31 March 2024
        m2 = cls.TEXT_DATE_2.search(raw)
        if m2:
            day = int(m2.group(1))
            month_name = m2.group(2).lower()
            year = int(m2.group(3))
            month_num = cls.MONTH_MAP.get(month_name)
            if month_num:
                d = f"{year:04d}-{month_num}-{day:02d}"
                return d, d, None

        # 3. Explicit Calendar Year: CY2024
        m_cy = cls.CY_PATTERN.match(raw)
        if m_cy:
            year = int(m_cy.group(1))
            return f"{year:04d}-01-01", f"{year:04d}-12-31", None

        # 4. Explicit Cross-Year Fiscal Span: FY 2023-24 or 2023-24
        m_span = cls.SPAN_FY_PATTERN.match(raw)
        if m_span:
            start_yr = int(m_span.group(1))
            end_short = m_span.group(2)
            end_yr = cls._parse_short_year(end_short)
            # Cross-year span (e.g. 2023-24) explicitly establishes the April-March fiscal interval
            return f"{start_yr:04d}-04-01", f"{end_yr:04d}-03-31", None

        # 5. Quarter with Cross-Year Span: Q4 FY 2023-24
        m_q_span = cls.QUARTER_SPAN_PATTERN.match(raw)
        if m_q_span:
            q_num = int(m_q_span.group(1))
            start_yr = int(m_q_span.group(2))
            end_yr = cls._parse_short_year(m_q_span.group(3))
            return cls._quarter_dates(q_num, start_yr, end_yr)

        # 6. Bare Quarter: Q4 FY24
        m_bare_q = cls.BARE_QUARTER_PATTERN.match(raw)
        if m_bare_q:
            q_num = int(m_bare_q.group(1))
            yr_val = cls._parse_short_year(m_bare_q.group(2))
            # Invariant: Bare FY/Quarter without established convention must NOT blindly assume April-March
            if fiscal_calendar == "april-march":
                start_yr = yr_val - 1
                end_yr = yr_val
                return cls._quarter_dates(q_num, start_yr, end_yr)
            return (
                None,
                None,
                f"Fiscal quarter '{raw}' convention not explicitly established (requires April-March context).",
            )

        # 7. Bare FY: FY24 or FY 2024
        m_bare_fy = cls.BARE_FY_PATTERN.match(raw)
        if m_bare_fy:
            yr_val = cls._parse_short_year(m_bare_fy.group(1))
            # Invariant: Bare FY must NOT blindly assume April-March unless explicitly established
            if fiscal_calendar == "april-march":
                return f"{yr_val - 1:04d}-04-01", f"{yr_val:04d}-03-31", None
            return (
                None,
                None,
                f"Fiscal-year convention not explicitly established for bare label '{raw}'.",
            )

        # 8. Single 4-digit year: '2024' (unqualified)
        if re.match(r"^\d{4}$", raw):
            year = int(raw)
            return f"{year:04d}-01-01", f"{year:04d}-12-31", None

        return None, None, f"Unresolvable temporal label: '{raw}'."

    @staticmethod
    def _quarter_dates(quarter: int, start_year: int, end_year: int) -> Tuple[str, str, None]:
        """Calculates exact date bounds for an April-March fiscal quarter."""
        if quarter == 1:
            return f"{start_year:04d}-04-01", f"{start_year:04d}-06-30", None
        elif quarter == 2:
            return f"{start_year:04d}-07-01", f"{start_year:04d}-09-30", None
        elif quarter == 3:
            return f"{start_year:04d}-10-01", f"{start_year:04d}-12-31", None
        elif quarter == 4:
            return f"{end_year:04d}-01-01", f"{end_year:04d}-03-31", None
        return "", "", None
