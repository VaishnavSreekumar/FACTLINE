"""
Execution and Quota Status Models for Experimental Multi-Page Batching (Phase 13).
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from backend.models.fact import FactRecord


class ExtractionStatus(str, Enum):
    """Explicit status indicating extraction outcome under quota constraints."""
    COMPLETE = "COMPLETE"
    PARTIAL_QUOTA = "PARTIAL_QUOTA"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    FAILED = "FAILED"


class BatchExtractionResult(BaseModel):
    """Represents the complete result of a quota-planned multi-page batch extraction."""
    status: ExtractionStatus = Field(..., description="High-level completion status")
    document_id: str = Field(..., description="Target document identifier")
    total_eligible_pages: int = Field(..., description="Number of fact-bearing / eligible pages")
    processed_pages: List[int] = Field(default_factory=list, description="List of page numbers successfully processed")
    unprocessed_pages: List[int] = Field(default_factory=list, description="List of page numbers deferred / unattempted")
    requests_used: int = Field(..., description="Number of Gemini API requests executed")
    requests_available: Optional[int] = Field(None, description="Configured request budget ceiling")
    batch_size: int = Field(..., description="Pages per Gemini request batch")
    max_workers: int = Field(default=2, description="Maximum concurrent worker threads used for Gemini extraction")
    facts: List[FactRecord] = Field(default_factory=list, description="Validated, evidence-grounded FactRecords")
    facts_rejected_grounding: int = Field(default=0, description="Facts rejected due to ungrounded or wrong-page attribution")
    error_message: Optional[str] = Field(None, description="Error explanation if quota or network failed")
    user_message: str = Field(..., description="Explicit, human-readable summary of work completed and data coverage")
