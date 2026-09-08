"""Groq OpenAI-compatible provider client with Strict Structured Outputs."""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx

from backend.extraction.fact_extractor import BATCH_EXTRACTION_SYSTEM_PROMPT
from backend.groq_experiment.models import ProviderFailureClass, RateLimitHeaders


# Strict JSON Schema for Groq structured outputs matching FACTLINE's exact extraction contract
GROQ_EXTRACTION_JSON_SCHEMA = {
    "name": "structured_fact_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page_number": {
                            "type": "integer",
                            "description": "1-indexed page number containing the fact",
                        },
                        "entity": {
                            "type": "string",
                            "description": "Canonical entity name",
                        },
                        "metric": {
                            "type": "string",
                            "description": "Standardized metric name",
                        },
                        "value_raw": {
                            "type": "string",
                            "description": "Verbatim raw value from text",
                        },
                        "value_numeric": {
                            "type": ["number", "null"],
                            "description": "Parsed numeric value if extractable",
                        },
                        "unit": {
                            "type": ["string", "null"],
                            "description": "Normalized unit if present",
                        },
                        "time_period": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "start_date": {"type": ["string", "null"]},
                                "end_date": {"type": ["string", "null"]},
                            },
                            "required": ["label", "start_date", "end_date"],
                            "additionalProperties": False,
                        },
                        "scope": {
                            "type": ["string", "null"],
                            "description": "Reporting scope (e.g. consolidated, standalone)",
                        },
                        "geography": {
                            "type": ["string", "null"],
                            "description": "Geographic scope (e.g. India, Global)",
                        },
                        "epistemic_status": {
                            "type": "string",
                            "enum": ["reported", "estimated", "projected", "target", "audited"],
                        },
                        "data_vintage": {
                            "type": ["string", "null"],
                            "description": "Data vintage or revision identifier",
                        },
                        "supporting_text": {
                            "type": "string",
                            "description": "Exact verbatim excerpt from the page",
                        },
                        "extraction_confidence": {
                            "type": "number",
                            "description": "Model extraction confidence between 0.0 and 1.0",
                        },
                    },
                    "required": [
                        "page_number",
                        "entity",
                        "metric",
                        "value_raw",
                        "value_numeric",
                        "unit",
                        "time_period",
                        "scope",
                        "geography",
                        "epistemic_status",
                        "data_vintage",
                        "supporting_text",
                        "extraction_confidence",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["facts"],
        "additionalProperties": False,
    },
}


class GroqProviderClient:
    """Isolated Groq API client with strict structured JSON schema output."""

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 60.0,
        http_client: Optional[httpx.Client] = None,
    ):
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            scratch_key_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scratch/.groq_key"))
            if os.path.exists(scratch_key_path):
                try:
                    with open(scratch_key_path, "r", encoding="utf-8") as f:
                        api_key = f.read().strip()
                except Exception:
                    pass
        self.api_key = api_key
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client

    def build_payload(self, user_prompt: str) -> Dict[str, Any]:
        """Constructs OpenAI-compatible chat completion payload with strict json_schema."""
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": BATCH_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": GROQ_EXTRACTION_JSON_SCHEMA,
            },
            "temperature": 0.0,
        }

    def _extract_rate_limit_headers(self, headers: httpx.Headers) -> RateLimitHeaders:
        """Extracts standard rate limit headers from HTTP response."""
        h_dict = {k.lower(): v for k, v in headers.items()}
        return RateLimitHeaders(
            limit_requests=h_dict.get("x-ratelimit-limit-requests"),
            limit_tokens=h_dict.get("x-ratelimit-limit-tokens"),
            remaining_requests=h_dict.get("x-ratelimit-remaining-requests"),
            remaining_tokens=h_dict.get("x-ratelimit-remaining-tokens"),
            reset_requests=h_dict.get("x-ratelimit-reset-requests"),
            reset_tokens=h_dict.get("x-ratelimit-reset-tokens"),
            raw_headers={k: v for k, v in h_dict.items() if k.startswith("x-ratelimit")},
        )

    def call_structured_extraction(
        self,
        user_prompt: str,
    ) -> Tuple[Optional[Dict[str, Any]], float, Optional[ProviderFailureClass], Optional[str], Optional[RateLimitHeaders], Optional[int]]:
        """Invokes Groq API and returns (parsed_json, latency_ms, failure_class, error_msg, rate_limits, tokens_used)."""
        if not self.api_key:
            return (
                None,
                0.0,
                ProviderFailureClass.AUTH_PERMISSION,
                "GROQ_API_KEY is not configured.",
                None,
                None,
            )

        payload = self.build_payload(user_prompt)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        t_start = time.time()
        client = self._http_client or httpx.Client(timeout=self.timeout_seconds)

        try:
            resp = client.post(self.GROQ_API_URL, json=payload, headers=headers)
            latency_ms = (time.time() - t_start) * 1000.0
            rate_limits = self._extract_rate_limit_headers(resp.headers)

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        return (
                            {"facts": []},
                            latency_ms,
                            None,
                            None,
                            rate_limits,
                            data.get("usage", {}).get("total_tokens"),
                        )
                    message_content = choices[0].get("message", {}).get("content", "{}")
                    parsed_facts = json.loads(message_content)
                    tokens = data.get("usage", {}).get("total_tokens")
                    return parsed_facts, latency_ms, None, None, rate_limits, tokens
                except Exception as parse_err:
                    return (
                        None,
                        latency_ms,
                        ProviderFailureClass.STRUCTURED_OUTPUT_ERROR,
                        f"Failed to parse structured JSON: {parse_err}",
                        rate_limits,
                        None,
                    )

            elif resp.status_code == 429:
                return (
                    None,
                    latency_ms,
                    ProviderFailureClass.QUOTA_RATE_LIMIT,
                    f"Groq Rate Limit/Quota Exceeded (HTTP 429): {resp.text}",
                    rate_limits,
                    None,
                )
            elif resp.status_code == 400:
                return (
                    None,
                    latency_ms,
                    ProviderFailureClass.REQUEST_SCHEMA,
                    f"Groq Bad Request / Schema Error (HTTP 400): {resp.text}",
                    rate_limits,
                    None,
                )
            elif resp.status_code in (401, 403):
                return (
                    None,
                    latency_ms,
                    ProviderFailureClass.AUTH_PERMISSION,
                    f"Groq Auth / Permission Failure (HTTP {resp.status_code}): {resp.text}",
                    rate_limits,
                    None,
                )
            elif resp.status_code == 404:
                return (
                    None,
                    latency_ms,
                    ProviderFailureClass.MODEL_CONFIG,
                    f"Groq Model/Endpoint Not Found (HTTP 404): {resp.text}",
                    rate_limits,
                    None,
                )
            elif resp.status_code == 408:
                return (
                    None,
                    latency_ms,
                    ProviderFailureClass.TIMEOUT,
                    f"Groq Request Timeout (HTTP 408): {resp.text}",
                    rate_limits,
                    None,
                )
            elif resp.status_code >= 500:
                return (
                    None,
                    latency_ms,
                    ProviderFailureClass.SERVER_ERROR,
                    f"Groq Server Error (HTTP {resp.status_code}): {resp.text}",
                    rate_limits,
                    None,
                )
            else:
                return (
                    None,
                    latency_ms,
                    ProviderFailureClass.UNKNOWN,
                    f"Unexpected Groq HTTP {resp.status_code}: {resp.text}",
                    rate_limits,
                    None,
                )

        except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            latency_ms = (time.time() - t_start) * 1000.0
            return (
                None,
                latency_ms,
                ProviderFailureClass.TIMEOUT,
                f"Groq HTTP Timeout: {type(e).__name__}: {e}",
                None,
                None,
            )
        except Exception as e:
            latency_ms = (time.time() - t_start) * 1000.0
            return (
                None,
                latency_ms,
                ProviderFailureClass.UNKNOWN,
                f"Groq Client Exception: {type(e).__name__}: {e}",
                None,
                None,
            )
        finally:
            if self._http_client is None:
                client.close()
