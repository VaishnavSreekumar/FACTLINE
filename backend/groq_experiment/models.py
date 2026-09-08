"""Models and schemas for Phase 26 Groq Extraction Provider Benchmark."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.models.fact import FactRecord


class ProviderFailureClass(str, Enum):
    """Categorized HTTP and operational failure modes."""
    QUOTA_RATE_LIMIT = "429_QUOTA_RATE_LIMIT"
    REQUEST_SCHEMA = "400_REQUEST_SCHEMA"
    AUTH_PERMISSION = "401_403_AUTH_PERMISSION"
    MODEL_CONFIG = "404_MODEL_NOT_FOUND"
    TIMEOUT = "408_TIMEOUT"
    SERVER_ERROR = "5XX_SERVER_ERROR"
    STRUCTURED_OUTPUT_ERROR = "STRUCTURED_OUTPUT_PARSE_ERROR"
    UNKNOWN = "UNKNOWN_ERROR"


class RateLimitHeaders(BaseModel):
    """Captured rate-limit headers and metadata from provider response."""
    limit_requests: Optional[str] = None
    limit_tokens: Optional[str] = None
    remaining_requests: Optional[str] = None
    remaining_tokens: Optional[str] = None
    reset_requests: Optional[str] = None
    reset_tokens: Optional[str] = None
    raw_headers: Dict[str, str] = Field(default_factory=dict)


class SingleBatchProviderResult(BaseModel):
    """Execution result for a single provider extraction batch."""
    batch_index: int
    document_id: str
    page_numbers: List[int]
    semantic_input_hash: str
    latency_ms: float
    raw_facts_count: int
    verified_facts: List[FactRecord] = Field(default_factory=list)
    rejected_facts_count: int = 0
    duplicate_facts_count: int = 0
    cross_page_contamination_count: int = 0
    raw_facts: List[Dict[str, Any]] = Field(default_factory=list)
    failure_class: Optional[ProviderFailureClass] = None
    error_message: Optional[str] = None
    rate_limit_info: Optional[RateLimitHeaders] = None
    tokens_used: Optional[int] = None
