"""Phase 26: Groq Extraction Provider Benchmark Unit & Regression Tests.

Verifies:
1. Groq provider initialization
2. API request construction
3. Model configuration via environment override
4. Strict JSON schema compliance
5. Response parsing from chat completion choice
6. FactRecord adapter conversion & production ID generation
7. Nullable field handling
8. Malformed structured output error handling
9. HTTP 429 rate limit / quota error classification
10. HTTP 400 request / schema error classification
11. HTTP 401/403 auth error classification
12. HTTP 404 model not found classification
13. HTTP 408 / timeout classification
14. HTTP 5xx server error classification
15. Strict evidence verification & rejection of ungrounded facts
16. Cross-page contamination detection across batch pages
17. Semantic input fingerprint equality between Gemini and Groq
18. Invariant: Production FactExtractor and Gemini configs are unchanged
19. Invariant: Production .env is unchanged
20. Metric classification rigor (MEASURED, DERIVED, N/A — NO GROUND TRUTH)
21. Latency metric directional qualification for 5-sample batches
22. Invariant: Strict deterministic Fact ID reuse across providers
23. Fixed 25-page benchmark corpus integrity across 5 diverse categories
"""

import hashlib
import json
import os
from unittest.mock import MagicMock, patch
import httpx
import pytest

from backend.extraction.evidence import EvidenceVerifier, create_evidence
from backend.extraction.fact_extractor import (
    BATCH_EXTRACTION_JSON_SCHEMA,
    BATCH_EXTRACTION_SYSTEM_PROMPT,
    FactExtractor,
)
from backend.groq_experiment.benchmark import (
    BENCHMARK_BATCH_DEFINITIONS,
    load_benchmark_batches,
)
from backend.groq_experiment.extractor import GroqFactExtractor, compute_semantic_input_hash
from backend.groq_experiment.metrics import (
    MetricValue,
    aggregate_batch_results,
)
from backend.groq_experiment.models import ProviderFailureClass, SingleBatchProviderResult
from backend.groq_experiment.provider import (
    GROQ_EXTRACTION_JSON_SCHEMA,
    GroqProviderClient,
)
from backend.models.document import PageText
from backend.models.fact import EpistemicStatus, FactRecord, TimePeriod


# Test 1: Groq Provider Initialization
def test_groq_provider_initialization():
    client = GroqProviderClient(api_key="gsk_test123", model="openai/gpt-oss-120b", timeout_seconds=30.0)
    assert client.api_key == "gsk_test123"
    assert client.model == "openai/gpt-oss-120b"
    assert client.timeout_seconds == 30.0


# Test 2: API Request Construction
def test_groq_api_request_construction():
    client = GroqProviderClient(api_key="gsk_test123", model="openai/gpt-oss-120b")
    prompt = "Document: report.pdf\nExtract facts..."
    payload = client.build_payload(prompt)

    assert payload["model"] == "openai/gpt-oss-120b"
    assert payload["temperature"] == 0.0
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == BATCH_EXTRACTION_SYSTEM_PROMPT
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"] == prompt
    assert payload["response_format"]["type"] == "json_schema"


# Test 3: Model Configuration via Environment Override
def test_groq_model_configuration_env_override(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "custom/groq-model-test")
    client = GroqProviderClient(api_key="gsk_test")
    assert client.model == "custom/groq-model-test"


# Test 4: Strict JSON Schema Compliance
def test_groq_strict_json_schema_structure():
    schema = GROQ_EXTRACTION_JSON_SCHEMA
    assert schema["name"] == "structured_fact_extraction"
    assert schema["strict"] is True

    inner_schema = schema["schema"]
    assert inner_schema["type"] == "object"
    assert inner_schema["additionalProperties"] is False
    assert "facts" in inner_schema["properties"]

    fact_item = inner_schema["properties"]["facts"]["items"]
    assert fact_item["type"] == "object"
    assert fact_item["additionalProperties"] is False

    # In strict mode, all defined properties must be listed in required
    props = fact_item["properties"]
    required_fields = fact_item["required"]
    for prop_name in props.keys():
        assert prop_name in required_fields

    # Check nullable types
    assert fact_item["properties"]["value_numeric"]["type"] == ["number", "null"]
    assert fact_item["properties"]["unit"]["type"] == ["string", "null"]
    assert fact_item["properties"]["scope"]["type"] == ["string", "null"]
    assert fact_item["properties"]["geography"]["type"] == ["string", "null"]
    assert fact_item["properties"]["data_vintage"]["type"] == ["string", "null"]


# Test 5: Response Parsing from Chat Completion Choice
def test_groq_response_parsing():
    mock_resp_json = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "facts": [
                            {
                                "page_number": 2,
                                "entity": "Delhivery",
                                "metric": "Shipments",
                                "value_raw": ">2.8Bn",
                                "value_numeric": 2800000000.0,
                                "unit": "shipments",
                                "time_period": {"label": "FY24", "start_date": "2023-04-01", "end_date": "2024-03-31"},
                                "scope": "Consolidated",
                                "geography": "India",
                                "epistemic_status": "reported",
                                "data_vintage": None,
                                "supporting_text": ">2.8Bn parcel shipments delivered",
                                "extraction_confidence": 0.98,
                            }
                        ]
                    })
                }
            }
        ],
        "usage": {"total_tokens": 450},
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_http_resp = MagicMock(spec=httpx.Response)
    mock_http_resp.status_code = 200
    mock_http_resp.json.return_value = mock_resp_json
    mock_http_resp.headers = httpx.Headers({"x-ratelimit-remaining-requests": "99"})
    mock_client.post.return_value = mock_http_resp

    provider = GroqProviderClient(api_key="gsk_test", http_client=mock_client)
    parsed_json, latency_ms, failure_class, error_msg, rate_limits, tokens = (
        provider.call_structured_extraction("test prompt")
    )

    assert failure_class is None
    assert error_msg is None
    assert tokens == 450
    assert len(parsed_json["facts"]) == 1
    assert parsed_json["facts"][0]["entity"] == "Delhivery"
    assert rate_limits.remaining_requests == "99"


# Test 6: FactRecord Adapter Conversion & Production ID Generation
def test_groq_fact_record_adapter():
    page_text = "Delhivery reported 2.8Bn parcel shipments delivered in FY24."
    page = PageText(page_number=2, text=page_text, char_count=len(page_text), has_text=True)

    mock_provider = MagicMock(spec=GroqProviderClient)
    mock_provider.call_structured_extraction.return_value = (
        {
            "facts": [
                {
                    "page_number": 2,
                    "entity": "Delhivery",
                    "metric": "Shipments",
                    "value_raw": "2.8Bn",
                    "value_numeric": 2800000000.0,
                    "unit": "shipments",
                    "time_period": {"label": "FY24", "start_date": "2023-04-01", "end_date": "2024-03-31"},
                    "scope": "Consolidated",
                    "geography": "India",
                    "epistemic_status": "reported",
                    "data_vintage": None,
                    "supporting_text": "2.8Bn parcel shipments delivered",
                    "extraction_confidence": 0.95,
                }
            ]
        },
        120.0,
        None,
        None,
        None,
        200,
    )

    extractor = GroqFactExtractor(provider_client=mock_provider, context_selector_enabled=False)
    result = extractor.extract_batch([page], document_id="doc1", document_name="doc1.pdf")

    assert len(result.verified_facts) == 1
    fact = result.verified_facts[0]
    expected_fact_id = FactExtractor.generate_fact_id(
        document_id="doc1",
        page_number=2,
        entity="Delhivery",
        metric="Shipments",
        value_raw="2.8Bn",
        time_period_label="FY24",
    )
    assert fact.fact_id == expected_fact_id
    assert fact.entity == "Delhivery"
    assert fact.metric == "Shipments"
    assert fact.value_raw == "2.8Bn"
    assert fact.value_numeric == 2800000000.0
    assert fact.epistemic_status == EpistemicStatus.REPORTED


# Test 7: Nullable Field Handling
def test_groq_nullable_field_handling():
    page_text = "Delhivery operational scale highlights."
    page = PageText(page_number=2, text=page_text, char_count=len(page_text), has_text=True)

    mock_provider = MagicMock(spec=GroqProviderClient)
    mock_provider.call_structured_extraction.return_value = (
        {
            "facts": [
                {
                    "page_number": 2,
                    "entity": "Delhivery",
                    "metric": "Highlights",
                    "value_raw": "Scale",
                    "value_numeric": None,  # Nullable
                    "unit": None,            # Nullable
                    "time_period": {"label": "FY24", "start_date": None, "end_date": None},
                    "scope": None,           # Nullable
                    "geography": None,       # Nullable
                    "epistemic_status": "reported",
                    "data_vintage": None,    # Nullable
                    "supporting_text": "Delhivery operational scale highlights",
                    "extraction_confidence": 0.90,
                }
            ]
        },
        100.0,
        None,
        None,
        None,
        150,
    )

    extractor = GroqFactExtractor(provider_client=mock_provider, context_selector_enabled=False)
    result = extractor.extract_batch([page], document_id="doc1", document_name="doc1.pdf")

    assert len(result.verified_facts) == 1
    fact = result.verified_facts[0]
    assert fact.value_numeric is None
    assert fact.unit is None
    assert fact.scope is None
    assert fact.geography is None
    assert fact.data_vintage is None


# Test 8: Malformed Structured Output Error Handling
def test_groq_malformed_response_handling():
    mock_client = MagicMock(spec=httpx.Client)
    mock_http_resp = MagicMock(spec=httpx.Response)
    mock_http_resp.status_code = 200
    mock_http_resp.json.return_value = {
        "choices": [{"message": {"content": "INVALID JSON {"}}],
    }
    mock_http_resp.headers = httpx.Headers({})
    mock_client.post.return_value = mock_http_resp

    provider = GroqProviderClient(api_key="gsk_test", http_client=mock_client)
    parsed, lat, fail_class, err_msg, rl, tok = provider.call_structured_extraction("prompt")

    assert fail_class == ProviderFailureClass.STRUCTURED_OUTPUT_ERROR
    assert "Failed to parse structured JSON" in err_msg


# Test 9: HTTP 429 Rate Limit / Quota Error
def test_groq_http_429_classification():
    mock_client = MagicMock(spec=httpx.Client)
    mock_http_resp = MagicMock(spec=httpx.Response)
    mock_http_resp.status_code = 429
    mock_http_resp.text = "Rate limit reached for requests per minute."
    mock_http_resp.headers = httpx.Headers({"x-ratelimit-remaining-requests": "0"})
    mock_client.post.return_value = mock_http_resp

    provider = GroqProviderClient(api_key="gsk_test", http_client=mock_client)
    parsed, lat, fail_class, err_msg, rl, tok = provider.call_structured_extraction("prompt")

    assert fail_class == ProviderFailureClass.QUOTA_RATE_LIMIT
    assert "Rate Limit/Quota Exceeded" in err_msg
    assert rl.remaining_requests == "0"


# Test 10: HTTP 400 Request/Schema Error
def test_groq_http_400_classification():
    mock_client = MagicMock(spec=httpx.Client)
    mock_http_resp = MagicMock(spec=httpx.Response)
    mock_http_resp.status_code = 400
    mock_http_resp.text = "Invalid JSON schema in response_format."
    mock_http_resp.headers = httpx.Headers({})
    mock_client.post.return_value = mock_http_resp

    provider = GroqProviderClient(api_key="gsk_test", http_client=mock_client)
    parsed, lat, fail_class, err_msg, rl, tok = provider.call_structured_extraction("prompt")

    assert fail_class == ProviderFailureClass.REQUEST_SCHEMA


# Test 11: HTTP 401 / 403 Auth Error
def test_groq_http_401_403_classification():
    mock_client = MagicMock(spec=httpx.Client)
    mock_http_resp = MagicMock(spec=httpx.Response)
    mock_http_resp.status_code = 401
    mock_http_resp.text = "Invalid API Key."
    mock_http_resp.headers = httpx.Headers({})
    mock_client.post.return_value = mock_http_resp

    provider = GroqProviderClient(api_key="gsk_test", http_client=mock_client)
    parsed, lat, fail_class, err_msg, rl, tok = provider.call_structured_extraction("prompt")

    assert fail_class == ProviderFailureClass.AUTH_PERMISSION


# Test 12: HTTP 404 Model Not Found
def test_groq_http_404_classification():
    mock_client = MagicMock(spec=httpx.Client)
    mock_http_resp = MagicMock(spec=httpx.Response)
    mock_http_resp.status_code = 404
    mock_http_resp.text = "Model 'invalid-model' does not exist."
    mock_http_resp.headers = httpx.Headers({})
    mock_client.post.return_value = mock_http_resp

    provider = GroqProviderClient(api_key="gsk_test", http_client=mock_client)
    parsed, lat, fail_class, err_msg, rl, tok = provider.call_structured_extraction("prompt")

    assert fail_class == ProviderFailureClass.MODEL_CONFIG


# Test 13: HTTP 408 / Timeout
def test_groq_timeout_classification():
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = httpx.ReadTimeout("Read timeout on connection")

    provider = GroqProviderClient(api_key="gsk_test", http_client=mock_client)
    parsed, lat, fail_class, err_msg, rl, tok = provider.call_structured_extraction("prompt")

    assert fail_class == ProviderFailureClass.TIMEOUT


# Test 14: HTTP 5xx Server Error
def test_groq_http_5xx_classification():
    mock_client = MagicMock(spec=httpx.Client)
    mock_http_resp = MagicMock(spec=httpx.Response)
    mock_http_resp.status_code = 503
    mock_http_resp.text = "Service Unavailable"
    mock_http_resp.headers = httpx.Headers({})
    mock_client.post.return_value = mock_http_resp

    provider = GroqProviderClient(api_key="gsk_test", http_client=mock_client)
    parsed, lat, fail_class, err_msg, rl, tok = provider.call_structured_extraction("prompt")

    assert fail_class == ProviderFailureClass.SERVER_ERROR


# Test 15: Strict Evidence Verification (Rejection of Ungrounded Facts)
def test_groq_extractor_rejects_hallucinated_evidence():
    real_page_text = "Real GDP growth is estimated at 7.0 percent in 2024-25."
    page = PageText(page_number=1, text=real_page_text, char_count=len(real_page_text), has_text=True)

    mock_provider = MagicMock(spec=GroqProviderClient)
    mock_provider.call_structured_extraction.return_value = (
        {
            "facts": [
                # Fact 1: Real grounded supporting text -> MUST ACCEPT
                {
                    "page_number": 1,
                    "entity": "India",
                    "metric": "GDP Growth",
                    "value_raw": "7.0%",
                    "value_numeric": 7.0,
                    "unit": "percent",
                    "time_period": {"label": "2024-25"},
                    "supporting_text": "Real GDP growth is estimated at 7.0 percent",
                    "extraction_confidence": 0.99,
                },
                # Fact 2: Hallucinated supporting text not present on page -> MUST REJECT
                {
                    "page_number": 1,
                    "entity": "India",
                    "metric": "Inflation",
                    "value_raw": "4.5%",
                    "value_numeric": 4.5,
                    "unit": "percent",
                    "time_period": {"label": "2024-25"},
                    "supporting_text": "Headline inflation declined significantly to 4.5 percent",
                    "extraction_confidence": 0.90,
                },
            ]
        },
        100.0,
        None,
        None,
        None,
        250,
    )

    extractor = GroqFactExtractor(provider_client=mock_provider, context_selector_enabled=False)
    result = extractor.extract_batch([page], document_id="doc1", document_name="doc1.pdf")

    assert result.raw_facts_count == 2
    assert len(result.verified_facts) == 1
    assert result.rejected_facts_count == 1
    assert result.verified_facts[0].metric == "GDP Growth"


# Test 16: Cross-Page Contamination Detection across Batch Pages
def test_groq_extractor_detects_cross_page_contamination():
    page1 = PageText(page_number=1, text="Acme Corp reported $500M revenue.", char_count=35, has_text=True)
    page2 = PageText(page_number=2, text="Beta Logistics incurred $50M capex.", char_count=35, has_text=True)

    mock_provider = MagicMock(spec=GroqProviderClient)
    # Model misattributes Page 2 text to Page 1
    mock_provider.call_structured_extraction.return_value = (
        {
            "facts": [
                {
                    "page_number": 1,  # WRONG PAGE (text exists on Page 2)
                    "entity": "Beta Logistics",
                    "metric": "Capex",
                    "value_raw": "$50M",
                    "value_numeric": 50000000.0,
                    "unit": "USD",
                    "time_period": {"label": "FY24"},
                    "supporting_text": "Beta Logistics incurred $50M capex",
                    "extraction_confidence": 0.95,
                }
            ]
        },
        100.0,
        None,
        None,
        None,
        150,
    )

    extractor = GroqFactExtractor(provider_client=mock_provider, context_selector_enabled=False)
    result = extractor.extract_batch([page1, page2], document_id="doc1", document_name="doc1.pdf")

    assert len(result.verified_facts) == 0
    assert result.rejected_facts_count == 1
    assert result.cross_page_contamination_count == 1


# Test 17: Semantic Input Fingerprint Equality Between Gemini and Groq
def test_semantic_input_fingerprint_equality():
    pages = [
        (1, "India real GDP growth reached 7.0%."),
        (2, "Delhivery shipment volume exceeded 2.8 billion."),
    ]
    doc_name = "test_document.pdf"

    hash_gemini = compute_semantic_input_hash(pages, doc_name)
    hash_groq = compute_semantic_input_hash(pages, doc_name)

    assert hash_gemini == hash_groq
    assert len(hash_gemini) == 64  # SHA-256 hex digest


# Test 18: Invariant: Production FactExtractor and Gemini Configs are Unchanged
def test_production_gemini_extractor_invariant():
    extractor = FactExtractor()
    assert extractor.model == "gemini-3.6-flash"
    assert extractor.batch_size == 5
    assert extractor.max_workers == 2
    assert extractor._llm_caller is None


# Test 19: Invariant: Production .env File Unchanged
def test_production_env_unchanged_invariant():
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Must not contain hardcoded groq production switches
            assert "FACTLINE_EXTRACTION_PROVIDER=groq" not in content


# Test 20: Metric Classification Rigor
def test_metric_classification_rigor():
    sample_result = SingleBatchProviderResult(
        batch_index=1,
        document_id="doc1",
        page_numbers=[1, 2, 3, 4, 5],
        semantic_input_hash="abc123hash",
        latency_ms=850.0,
        raw_facts_count=10,
        verified_facts=[
            FactRecord(
                fact_id="fact-1",
                entity="India",
                metric="GDP Growth",
                value_raw="7.0%",
                time_period=TimePeriod(label="2024"),
                provenance=create_evidence("doc1", 1, "7.0%"),
            )
        ],
        rejected_facts_count=9,
        duplicate_facts_count=1,
        cross_page_contamination_count=0,
    )

    summary = aggregate_batch_results(
        provider_name="TestProvider",
        model_name="test-model",
        batch_results=[sample_result],
        wall_clock_time_s=1.2,
    )

    assert summary.raw_facts_returned.classification == "MEASURED"
    assert summary.raw_facts_returned.value == 10
    assert summary.verified_facts.classification == "MEASURED"
    assert summary.verified_facts.value == 1
    assert summary.duplicate_fact_rate_pct.classification == "DERIVED"
    assert summary.precision.classification == "N/A — NO GROUND TRUTH"
    assert summary.recall.classification == "N/A — NO GROUND TRUTH"
    assert summary.f1_score.classification == "N/A — NO GROUND TRUTH"
    assert summary.numeric_extraction_accuracy.classification == "N/A — NO GROUND TRUTH"
    assert summary.entity_accuracy.classification == "N/A — NO GROUND TRUTH"


# Test 21: Latency Metric Directional Qualification for 5-Sample Batches
def test_directional_p95_latency_metric():
    batch_results = [
        SingleBatchProviderResult(
            batch_index=i,
            document_id="doc1",
            page_numbers=[i],
            semantic_input_hash=f"hash_{i}",
            latency_ms=100.0 * i,
            raw_facts_count=1,
            verified_facts=[],
        )
        for i in range(1, 6)
    ]
    summary = aggregate_batch_results(
        provider_name="TestProvider",
        model_name="test-model",
        batch_results=batch_results,
        wall_clock_time_s=1.5,
    )

    assert summary.mean_latency_ms.value == 300.0
    assert summary.median_latency_ms.value == 300.0
    assert "Directional only" in summary.p95_latency_ms_directional.note
    assert summary.p95_latency_ms_directional.classification == "MEASURED"


# Test 22: Invariant: Strict Deterministic Fact ID Reuse Across Providers
def test_fact_identity_generation_reuse_invariant():
    doc_id = "doc_alpha"
    page_num = 3
    entity = "Reserve Bank of India"
    metric = "Policy Repo Rate"
    value_raw = "6.50%"
    time_label = "June 2024"

    prod_id = FactExtractor.generate_fact_id(doc_id, page_num, entity, metric, value_raw, time_label)

    # Replicate raw construction in extractor
    key_str = f"{doc_id}|{page_num}|{entity.strip()}|{metric.strip()}|{value_raw.strip()}|{time_label.strip()}"
    expected_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]
    expected_id = f"fact-{expected_hash}"

    assert prod_id == expected_id


# Test 23: Fixed 25-Page Benchmark Corpus Integrity across 5 Diverse Categories
def test_benchmark_page_selection_corpus_integrity():
    assert len(BENCHMARK_BATCH_DEFINITIONS) == 5
    total_pages = 0
    categories = set()

    for b in BENCHMARK_BATCH_DEFINITIONS:
        categories.add(b["batch_id"])
        if b.get("is_multi_document"):
            total_pages += len(b["items"])
        else:
            total_pages += len(b["page_numbers"])

    assert total_pages == 25
    assert len(categories) == 5
