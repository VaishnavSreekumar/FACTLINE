"""Evidence and provenance creation and verification module."""

import re
from typing import Optional, Union
from backend.models.fact import Provenance
from backend.models.document import PageText


def create_evidence(
    document_id: str,
    page_number: int,
    supporting_text: str,
    document_date: Optional[str] = None,
) -> Provenance:
    """Creates a grounded Provenance evidence reference.

    Args:
        document_id: Unique identifier for the source document.
        page_number: 1-indexed page number where the text appears.
        supporting_text: Exact supporting text snippet extracted from the PDF page.
        document_date: Optional document publication/release date.

    Returns:
        Provenance object representing verifiable evidence.
    """
    if page_number < 1:
        raise ValueError(f"page_number must be >= 1 (1-indexed), got {page_number}")

    return Provenance(
        document_id=document_id,
        document_date=document_date,
        page_number=page_number,
        supporting_text=supporting_text,
    )


class EvidenceVerifier:
    """Verifies that an extracted fact's supporting text exists on the specified page."""

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapses consecutive whitespace characters without altering characters/numbers."""
        return " ".join(text.split())

    @classmethod
    def verify_provenance(
        cls,
        provenance: Provenance,
        page_source: Union[PageText, str],
    ) -> bool:
        """Verifies supporting text presence in page content.

        Args:
            provenance: The Provenance object containing supporting text and page info.
            page_source: Either a PageText model or raw string from that page.

        Returns:
            True if supporting text is present in the page text, False otherwise.
        """
        if isinstance(page_source, PageText):
            page_text = page_source.text
        else:
            page_text = str(page_source)

        if not provenance.supporting_text:
            return False

        target = provenance.supporting_text.strip()
        if not target:
            return False

        # 1. Exact substring match
        if target in page_text:
            return True

        # 2. Whitespace-normalized match (handles PDF multi-line breaks)
        norm_target = cls._normalize_whitespace(target)
        norm_page = cls._normalize_whitespace(page_text)
        return norm_target in norm_page
