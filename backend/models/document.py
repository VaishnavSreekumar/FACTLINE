"""Document and PageText models for page-aware extraction."""

from typing import List, Optional
from pydantic import BaseModel, Field


class PageText(BaseModel):
    """Represents extracted text from a single document page."""

    page_number: int = Field(..., description="1-indexed human-readable page number")
    text: str = Field(..., description="Raw text extracted from this page")
    char_count: int = Field(..., description="Number of characters on this page")
    has_text: bool = Field(..., description="Whether this page contains extractable text")


class ParsedDocument(BaseModel):
    """Represents a fully parsed document containing page-level extracted text."""

    document_id: str = Field(..., description="Unique deterministic identifier for the document")
    document_name: str = Field(..., description="File name or user-assigned label of the document")
    total_pages: int = Field(..., description="Total number of pages in the document")
    pages: List[PageText] = Field(default_factory=list, description="List of 1-indexed page records")
