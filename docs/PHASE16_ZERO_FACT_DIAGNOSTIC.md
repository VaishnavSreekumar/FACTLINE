# PHASE 16 — ZERO-FACT EXTRACTION REGRESSION DIAGNOSTIC REPORT

---

## 1. Executive Diagnosis

When any PDF document is analyzed in FACTLINE, the pipeline currently extracts **0 facts**, forms **0 relationships**, processes **0 pages**, and marks the analysis as **`PARTIAL_QUOTA` (0 / $N$ pages processed)**.

This regression is caused by two compounding root causes at the Gemini API invocation boundary:
1. **JSON Schema Proto Violation (HTTP 400)**: The `BATCH_EXTRACTION_JSON_SCHEMA` defined in `backend/extraction/fact_extractor.py` uses JSON Schema type unions (e.g. `"type": ["string", "null"]` and `"type": ["number", "null"]`). Google Gemini's REST API (`responseSchema` OpenAPI protobuf parser) strictly rejects list-valued `type` fields with HTTP 400 (`"Proto field is not repeating, cannot start list"`).
2. **Model Deprecation / Discontinuation (HTTP 404)**: The environment configuration `.env` specifies `GEMINI_MODEL=gemini-2.5-flash`. Google Gemini's endpoint returns HTTP 404 (`"This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash..."`).

When request #1 fails with HTTP 400 or 404, `FactExtractor.extract_document_result` catches the `ExtractionError`, halts further batch processing, records 0 processed pages and 0 facts, and falls into the `PARTIAL_QUOTA` status branch (`len(unprocessed_pages) > 0`).

---

## 2. Exact Failing Stage

- **Failing Component**: `backend/extraction/fact_extractor.py` $\rightarrow$ `FactExtractor._call_gemini` during `extract_batch`.
- **Failing Line**: `resp = client.post(url, json=payload)`
- **Exception Raised**: `ExtractionError: Gemini API error (HTTP 400 / HTTP 404)`
- **Exception Caught**: In `extract_document_result` (`except Exception as e:` block).

---

## 3. Complete Execution Trace

```text
1. POST /analysis
   │
   ▼
2. PDFParser.parse_bytes
   │  ✓ Extracted page text successfully for all pages.
   ▼
3. FactExtractor.extract_document_result
   │  ✓ PageRelevanceFilter: filtered non-relevant pages (e.g., 224 eligible out of 284).
   │  ✓ QuotaPlanner: partitioned 224 pages into batches of size 3 (e.g., [6, 7, 9], ...).
   ▼
4. FactExtractor.extract_batch (Batch 1: pages [6, 7, 9])
   │  ✓ ContextSelector: extracted fact-bearing context window.
   │  ✓ Formatted prompt with delimiters: "===== DOCUMENT PAGE 6 START ====="
   ▼
5. FactExtractor._call_gemini
   │  ✗ HTTP POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=...
   │  ✗ Gemini API rejects payload with HTTP 400 (Proto list error on "type": ["string", "null"])
   │    AND HTTP 404 (gemini-2.5-flash unavailable).
   │  ✗ Raises ExtractionError("Gemini API error (HTTP 400)...")
   ▼
6. Loop Termination & Quota Fallback
   │  ✗ Loop catches ExtractionError and breaks on Batch 1.
   │  ✗ requests_executed = 0, processed_pages = [], all_facts = [].
   │  ✗ Evaluates: len(unprocessed_pages) > 0 (224 remaining) -> status = PARTIAL_QUOTA.
   │  ✗ user_message = "Analysis partially completed ... 0 of 224 eligible pages were processed."
   ▼
7. Downstream Stages
   │  ✓ FactNormalizer.normalize_batch([]) -> []
   │  ✓ CandidateMatcher.find_candidates([]) -> []
   │  ✓ RelationshipEngine -> []
   │  ✓ DatabaseRepository.save_analysis persists summary with extraction_status error.
   ▼
8. UI Output
   │  3 documents · 0 facts · 0 relationships
   │  Partial Analysis · 0 / 279 pages
```

---

## 4. Gemini Configuration State

- **GEMINI_API_KEY**: PRESENT (Length: 53 characters, loaded from `.env`).
- **GEMINI_MODEL**: `gemini-2.5-flash` (loaded from `.env`).
- **Endpoint URL**: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=...`
- **Canonical Model Resolution**: A single canonical resolution path exists in `FactExtractor.__init__`.
- **Cached/Stale Model Configuration**: No stale fallback model is being used.

---

## 5. HTTP / Provider Error Details

### Error A: Schema Format Violation (HTTP 400)
```json
{
  "error": {
    "code": 400,
    "message": "Invalid JSON payload received. Unknown name \"type\" at 'generation_config.response_schema.properties[0].value.items.properties[4].value': Proto field is not repeating, cannot start list.\nInvalid JSON payload received. Unknown name \"type\" at 'generation_config.response_schema.properties[0].value.items.properties[5].value': Proto field is not repeating, cannot start list.\nInvalid JSON payload received. Unknown name \"type\" at 'generation_config.response_schema.properties[0].value.items.properties[6].value.properties[1].value': Proto field is not repeating, cannot start list.",
    "status": "INVALID_ARGUMENT"
  }
}
```

### Error B: Model Not Found / Discontinued (HTTP 404)
```json
{
  "error": {
    "code": 404,
    "message": "This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements.",
    "status": "NOT_FOUND"
  }
}
```

---

## 6. Quota Handling Behavior

- **HTTP 429 vs Non-429 Error Classification**:
  - If Gemini returns HTTP 429 $\rightarrow$ `ExtractionQuotaError` is raised, loop halts, and `status` is set to `PARTIAL_QUOTA` with `user_message: "Analysis halted due to quota limit."`
  - If Gemini returns HTTP 400 / 404 $\rightarrow$ `ExtractionError` is raised, loop halts, but because `len(unprocessed_pages) > 0`, the status determination block sets `status = PARTIAL_QUOTA` and masks the underlying HTTP error behind `"Analysis partially completed under configured request limit."`
- **Result**: The UI displayed a generic `Partial Analysis · 0 / 279 pages` banner instead of an explicit configuration/provider error notice.

---

## 7. Page Selection & Processing Counts (for recent failing run)

| Metric | Value |
| :--- | :--- |
| Total PDF Pages | 284 |
| Eligible Pages (Selected by PageFilter) | 224 |
| Planned Gemini Batches (size 3) | 75 batches |
| Gemini Requests Attempted | 1 |
| Successful Requests | 0 |
| Failed Requests | 1 (HTTP 400 / 404) |
| Pages Marked Processed | 0 |
| Pages Remaining / Deferred | 224 |

---

## 8. Extraction Response & Evidence Verification Behavior

- The pipeline never reached JSON response parsing or `EvidenceVerifier` because request #1 was rejected upstream by the Google API gateway before execution.

---

## 9. Existing Test Results

- **Backend Pytest Suite**: `177 passed, 1 skipped in 28.52s` (100% pass rate).
  - *Why unit tests passed*: In unit tests, `FactExtractor._call_gemini` is mocked with valid dictionaries or mock responses; the mock tests did not validate Google's remote OpenAPI proto schema compiler constraints against live Gemini endpoints.
- **Frontend Production Build**: `npm run build` succeeded in `141ms` with 0 errors.

---

## 10. Root Cause Summary

1. **Schema Syntax Incompatibility**: In `BATCH_EXTRACTION_JSON_SCHEMA`, nullable properties used JSON Schema draft 7 type lists `["string", "null"]` and `["number", "null"]`. Google Gemini's REST API `responseSchema` requires OpenAPI 3.0 scalar types with `"nullable": True` (e.g., `{"type": "STRING", "nullable": True}`).
2. **Model Availability**: The environment variable `GEMINI_MODEL=gemini-2.5-flash` targets a discontinued model name on the v1beta API endpoint, producing HTTP 404.
3. **Error Status Masking**: When an `ExtractionError` occurs during batch iteration, `extract_document_result` breaks out of the loop with 0 facts and marks the result as `PARTIAL_QUOTA`, obscuring the actual API failure in the top-level status.

---

## 11. Confidence in Root Cause

**100% High Confidence**. Verified with live API curl/HTTP requests reproducing the exact HTTP 400 and HTTP 404 responses.

---

## 12. Recommended Next Fix (Proposal Only — Not Implemented)

1. **Schema Fix**: In `backend/extraction/fact_extractor.py`, update `BATCH_EXTRACTION_JSON_SCHEMA` to use valid OpenAPI 3.0 types:
   - Replace `{"type": ["string", "null"]}` with `{"type": "STRING", "nullable": True}` (or `"type": "string", "nullable": true`).
   - Replace `{"type": ["number", "null"]}` with `{"type": "NUMBER", "nullable": True}`.
2. **Model Fix**: Update `GEMINI_MODEL` in `.env` and `.env.example` to an active model (e.g. `gemini-3.6-flash` or `gemini-2.0-flash` / `gemini-1.5-flash` supported by the provider key).
3. **Status Reporting Clarity**: In `extract_document_result`, if an unexpected exception occurs during extraction and 0 batches succeeded, set `status = ExtractionStatus.FAILED` (or preserve explicit `error_message` in UI) so configuration/schema errors are not reported as normal partial quota deferrals.
