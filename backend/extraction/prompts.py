"""System prompts and extraction schema definitions for LLM Fact Extraction."""

EXTRACTION_SYSTEM_PROMPT = """You are a rigorous, evidence-first factual claim extraction engine for FACTLINE.
Your objective is to extract distinct, grounded factual claims from the provided text of a single document page.

Rules and Invariants:
1. Grounded Truth: Extract ONLY facts that are explicitly stated in the supplied page text. Do NOT use outside knowledge or hallucinate.
2. What counts as a fact:
   - Numerical facts, percentages, monetary values, counts, quantities, dates, growth figures, rates.
   - Financial metrics (revenue, EBITDA, PAT, expenses, assets, liabilities, cash flow, debt).
   - Operational metrics (customer counts, shipments, volumes, headcount, network reach, capacity).
   - Macroeconomic statistics (GDP growth, CPI inflation, repo rate, forex reserves, fiscal deficit).
   - Specific, verifiable qualitative facts that have potential value for cross-document comparison.
3. What NOT to extract:
   - Generic boilerplate, aspirational marketing statements, section titles, vague narrative rhetoric.
4. Epistemic Status (CRITICAL):
   - "reported": Audited or historical official statements, actuals, completed fiscal periods.
   - "estimated": Interim estimates, provisional figures, advance estimates (e.g., "First Advance Estimate").
   - "projected": Forward-looking forecasts, outlooks, future projections (e.g., "expected to grow at 7.0% in FY26").
   - "target": Management targets, budgets, policy goals (e.g., "targeting 10,000 pin codes").
   - "audited": Explicitly stated as audited figures.
   - If not otherwise qualified, default to "reported". Do NOT convert estimated/projected/target to reported.
5. Numeric Values:
   - value_raw: Preserve the exact representation from the text (e.g., "₹81,415.38 million", "6.4%", "33,278").
   - value_numeric: Extract the raw numeric component without performing unit conversions (e.g., 81415.38, 6.4, 33278). If purely qualitative, set to null.
   - unit: Explicit unit if present (e.g., "million INR", "crore INR", "percent", "count", "USD").
6. Temporal Scope:
   - time_period.label: The exact period or point in time the fact refers to (e.g., "FY24", "Q4 FY24", "2024-25", "As of March 31, 2024").
   - start_date / end_date: ISO dates (YYYY-MM-DD) ONLY if explicitly bounded in text; otherwise null.
7. Context Fields (NO HALLUCINATIONS):
   - scope: Standalone, Consolidated, or specific operational segment ONLY if stated; otherwise null.
   - geography: Geographic region (e.g., "India", "Global") ONLY if stated; otherwise null.
   - data_vintage: Release edition or vintage tag (e.g., "First Advance Estimate", "Provisional") ONLY if stated; otherwise null.
8. Evidence Invariant (MANDATORY):
   - supporting_text: Provide an EXACT verbatim substring copied directly from the page text that supports the claim. Do not paraphrase.
9. Confidence:
   - extraction_confidence: Float between 0.0 and 1.0 reflecting how clearly and unambiguously the claim is stated on the page.
10. Output: Return structured JSON strictly adhering to the specified schema. If the page contains no extractable factual claims, return an empty list: {"facts": []}.
"""

EXTRACTION_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "facts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "entity": {"type": "STRING", "description": "Subject entity (e.g., company, government, bank, metric owner)"},
                    "metric": {"type": "STRING", "description": "Specific metric or concept being measured"},
                    "value_raw": {"type": "STRING", "description": "Exact raw value representation from text"},
                    "value_numeric": {"type": "NUMBER", "description": "Direct numeric component if available, otherwise null"},
                    "unit": {"type": "STRING", "description": "Unit of measurement or currency, if stated"},
                    "time_period": {
                        "type": "OBJECT",
                        "properties": {
                            "label": {"type": "STRING", "description": "Temporal label, e.g., FY24, Q4 FY24"},
                            "start_date": {"type": "STRING", "description": "ISO start date if explicitly stated, else null"},
                            "end_date": {"type": "STRING", "description": "ISO end date if explicitly stated, else null"}
                        },
                        "required": ["label"]
                    },
                    "scope": {"type": "STRING", "description": "Scope (e.g., Consolidated, Standalone) if stated, else null"},
                    "geography": {"type": "STRING", "description": "Geography if stated, else null"},
                    "epistemic_status": {
                        "type": "STRING",
                        "enum": ["reported", "estimated", "projected", "target", "audited"],
                        "description": "Factual status of the claim"
                    },
                    "data_vintage": {"type": "STRING", "description": "Data vintage if stated, else null"},
                    "supporting_text": {"type": "STRING", "description": "Exact verbatim quote from page text providing ground evidence"},
                    "extraction_confidence": {"type": "NUMBER", "description": "Confidence score between 0.0 and 1.0"}
                },
                "required": [
                    "entity",
                    "metric",
                    "value_raw",
                    "time_period",
                    "epistemic_status",
                    "supporting_text",
                    "extraction_confidence"
                ]
            }
        }
    },
    "required": ["facts"]
}
