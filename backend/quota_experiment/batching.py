"""
Multi-Page Batch Extractor (Phase 13 Experiment).

Constructs structured multi-page prompts with explicit delimiters, parses structured
JSON responses with page attribution, and applies EvidenceVerifier across all pages.
"""

import hashlib
import json
import os
import time
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import httpx
from dotenv import load_dotenv

load_dotenv()

from backend.models.document import PageText, ParsedDocument
from backend.models.fact import FactRecord, EpistemicStatus, TimePeriod, Provenance
from backend.extraction.evidence import EvidenceVerifier, create_evidence
from backend.quota_experiment.planner import QuotaPlanner, QuotaPlan
from backend.quota_experiment.status import ExtractionStatus, BatchExtractionResult


BATCH_EXTRACTION_SYSTEM_PROMPT = """You are FACTLINE's Multi-Page Batch Fact Extractor.
Extract atomic, evidence-grounded quantitative and business facts from the provided multi-page document excerpt.

CRITICAL INVARIANTS:
1. Every extracted fact MUST be explicitly attributed to the exact 'page_number' where it appears.
2. 'supporting_text' MUST be an exact, unparaphrased substring copied directly from that specific page.
3. Never attribute a fact from Page A to Page B.
4. Output valid JSON matching the specified schema.
"""

BATCH_EXTRACTION_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "facts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "page_number": {"type": "INTEGER", "description": "1-indexed page number containing the fact"},
                    "entity": {"type": "STRING", "description": "Canonical entity name"},
                    "metric": {"type": "STRING", "description": "Standardized metric name"},
                    "value_raw": {"type": "STRING", "description": "Verbatim raw value from text"},
                    "value_numeric": {"type": "NUMBER", "nullable": True, "description": "Parsed numeric value"},
                    "unit": {"type": "STRING", "nullable": True, "description": "Normalized unit if present"},
                    "time_period": {
                        "type": "OBJECT",
                        "properties": {
                            "label": {"type": "STRING"},
                            "start_date": {"type": "STRING", "nullable": True},
                            "end_date": {"type": "STRING", "nullable": True},
                        },
                        "required": ["label"],
                    },
                    "scope": {"type": "STRING", "nullable": True},
                    "geography": {"type": "STRING", "nullable": True},
                    "epistemic_status": {
                        "type": "STRING",
                        "enum": ["reported", "estimated", "projected", "target", "audited"],
                    },
                    "supporting_text": {"type": "STRING", "description": "Verbatim excerpt from the page"},
                    "extraction_confidence": {"type": "NUMBER"},
                },
                "required": [
                    "page_number",
                    "entity",
                    "metric",
                    "value_raw",
                    "time_period",
                    "supporting_text",
                ],
            },
        },
    },
    "required": ["facts"],
}


class BatchExtractionQuotaError(Exception):
    """Raised when Gemini API quota (HTTP 429) is exhausted during batch extraction."""
    pass


class MultiPageBatchExtractor:
    """
    Experimental extractor grouping multiple page texts into a single Gemini prompt
    while enforcing strict per-page provenance and evidence verification.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        llm_caller: Optional[Callable[[str], Dict[str, Any]]] = None,
        default_batch_size: int = 2,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self._llm_caller = llm_caller
        self.default_batch_size = max(1, default_batch_size)
        self.planner = QuotaPlanner(default_batch_size=self.default_batch_size)

    @staticmethod
    def generate_fact_id(
        document_id: str,
        page_number: int,
        entity: str,
        metric: str,
        value_raw: str,
        time_period_label: str,
    ) -> str:
        """Generates a deterministic 16-character Fact ID from coordinates."""
        key_str = f"{document_id}|{page_number}|{entity.strip()}|{metric.strip()}|{value_raw.strip()}|{time_period_label.strip()}"
        digest = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]
        return f"fact-{digest}"

    def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        """Calls Gemini API or custom llm_caller with retry and quota handling."""
        if self._llm_caller is not None:
            return self._llm_caller(prompt)

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": BATCH_EXTRACTION_SYSTEM_PROMPT}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": BATCH_EXTRACTION_JSON_SCHEMA,
                "temperature": 0.0,
            },
        }

        with httpx.Client(timeout=60.0) as client:
            for attempt in range(2):
                try:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        if parts:
                            return json.loads(parts[0].get("text", "{}"))
                        return {"facts": []}
                    elif resp.status_code == 429:
                        raise BatchExtractionQuotaError(
                            "Gemini API quota (HTTP 429) exhausted during batch extraction."
                        )
                    elif resp.status_code == 503:
                        if attempt == 0:
                            time.sleep(1.0)
                            continue
                        raise RuntimeError("Gemini API 503 Service Unavailable on retry.")
                    else:
                        raise RuntimeError(f"Gemini API returned HTTP {resp.status_code}: {resp.text}")
                except (httpx.RequestError, json.JSONDecodeError) as e:
                    if attempt == 0:
                        time.sleep(0.5)
                        continue
                    raise RuntimeError(f"Batch extraction request failed: {e}") from e

        return {"facts": []}

    def format_batch_prompt(
        self,
        batch_pages: List[PageText],
        document_name: str,
        document_date: Optional[str] = None,
    ) -> str:
        """Formats multiple pages with unambiguous delimiters and page attributions."""
        sections = [
            f"Document: {document_name}",
            f"Document Date: {document_date or 'Unknown'}",
            f"Pages in this batch: {', '.join(str(p.page_number) for p in batch_pages)}",
            "",
            "Extract all facts and assign each fact strictly to its source page_number.",
            "",
        ]

        for p in batch_pages:
            sections.append(f"===== DOCUMENT PAGE {p.page_number} START =====")
            sections.append(p.text.strip())
            sections.append(f"===== DOCUMENT PAGE {p.page_number} END =====")
            sections.append("")

        return "\n".join(sections)

    def extract_batch(
        self,
        batch_pages: List[PageText],
        document_id: str,
        document_name: str,
        document_date: Optional[str] = None,
    ) -> Tuple[List[FactRecord], int]:
        """
        Extracts facts from a batch of pages, applying strict per-page evidence verification.
        
        Returns:
            (verified_facts, rejected_count)
        """
        if not batch_pages:
            return [], 0

        # Map page_number -> PageText for fast evidence verification
        page_map: Dict[int, PageText] = {p.page_number: p for p in batch_pages}
        prompt = self.format_batch_prompt(batch_pages, document_name, document_date)

        raw_response = self._call_gemini(prompt)
        candidates = raw_response.get("facts", [])

        verified_facts: List[FactRecord] = []
        rejected_count = 0

        for item in candidates:
            # 1. Verify page_number is valid for this batch
            reported_page_num = item.get("page_number")
            if not isinstance(reported_page_num, int) or reported_page_num not in page_map:
                rejected_count += 1
                continue

            target_page = page_map[reported_page_num]
            supporting_text = str(item.get("supporting_text", "")).strip()
            if not supporting_text:
                rejected_count += 1
                continue

            # 2. Verify evidence against the specified page
            provenance = create_evidence(
                document_id=document_id,
                page_number=reported_page_num,
                supporting_text=supporting_text,
                document_date=document_date,
            )

            if not EvidenceVerifier.verify_provenance(provenance, target_page):
                rejected_count += 1
                continue

            # 3. Construct validated FactRecord
            entity = str(item.get("entity", "")).strip() or "Unknown"
            metric = str(item.get("metric", "")).strip() or "Unknown"
            value_raw = str(item.get("value_raw", "")).strip()
            if not value_raw:
                rejected_count += 1
                continue

            tp_dict = item.get("time_period", {})
            tp_label = str(tp_dict.get("label", "")).strip() if isinstance(tp_dict, dict) else ""
            if not tp_label:
                tp_label = "Unspecified"
            time_period = TimePeriod(
                label=tp_label,
                start_date=tp_dict.get("start_date") if isinstance(tp_dict, dict) else None,
                end_date=tp_dict.get("end_date") if isinstance(tp_dict, dict) else None,
            )

            raw_status = str(item.get("epistemic_status", "reported")).lower()
            try:
                epistemic_status = EpistemicStatus(raw_status)
            except ValueError:
                epistemic_status = EpistemicStatus.REPORTED

            value_numeric = item.get("value_numeric")
            if value_numeric is not None:
                try:
                    value_numeric = float(value_numeric)
                except (ValueError, TypeError):
                    value_numeric = None

            unit = item.get("unit")
            unit = str(unit).strip() if unit else None

            raw_conf = item.get("extraction_confidence", 0.0)
            try:
                confidence = max(0.0, min(1.0, float(raw_conf)))
            except (ValueError, TypeError):
                confidence = 0.0

            fact_id = self.generate_fact_id(
                document_id=document_id,
                page_number=reported_page_num,
                entity=entity,
                metric=metric,
                value_raw=value_raw,
                time_period_label=time_period.label,
            )

            fact_record = FactRecord(
                fact_id=fact_id,
                entity=entity,
                metric=metric,
                value_raw=value_raw,
                value_numeric=value_numeric,
                unit=unit,
                time_period=time_period,
                scope=item.get("scope"),
                geography=item.get("geography"),
                epistemic_status=epistemic_status,
                provenance=provenance,
                extraction_confidence=confidence,
            )
            verified_facts.append(fact_record)

        return verified_facts, rejected_count

    def extract_document(
        self,
        document: ParsedDocument,
        request_budget: Optional[int] = None,
        batch_size: Optional[int] = None,
        page_relevance_scores: Optional[Dict[int, float]] = None,
        document_date: Optional[str] = None,
    ) -> BatchExtractionResult:
        """
        Executes quota-planned multi-page batch extraction over an entire ParsedDocument.
        """
        b_size = batch_size if (batch_size is not None and batch_size > 0) else self.default_batch_size
        eligible_pages = [p.page_number for p in document.pages if p.has_text and p.text.strip()]

        plan = self.planner.calculate_plan(
            eligible_pages=eligible_pages,
            request_budget=request_budget,
            batch_size=b_size,
            page_relevance_scores=page_relevance_scores,
        )

        all_facts: List[FactRecord] = []
        total_rejected = 0
        processed_pages: List[int] = []
        requests_executed = 0
        error_msg: Optional[str] = None
        quota_hit = False

        page_lookup: Dict[int, PageText] = {p.page_number: p for p in document.pages}

        for batch_p_nums in plan.batches:
            batch_objs = [page_lookup[p] for p in batch_p_nums if p in page_lookup]
            if not batch_objs:
                continue

            try:
                facts, rejected = self.extract_batch(
                    batch_pages=batch_objs,
                    document_id=document.document_id,
                    document_name=document.document_name,
                    document_date=document_date,
                )
                requests_executed += 1
                all_facts.extend(facts)
                total_rejected += rejected
                processed_pages.extend(batch_p_nums)
            except BatchExtractionQuotaError as e:
                quota_hit = True
                error_msg = str(e)
                break
            except Exception as e:
                error_msg = f"Batch extraction failed on pages {batch_p_nums}: {e}"
                break

        unprocessed_pages = [p for p in eligible_pages if p not in set(processed_pages)]

        # Determine explicit status and informative user message
        if quota_hit:
            status = ExtractionStatus.QUOTA_EXHAUSTED if not processed_pages else ExtractionStatus.PARTIAL_QUOTA
            user_msg = (
                f"Analysis halted due to quota limit. "
                f"{len(processed_pages)} of {len(eligible_pages)} eligible pages were processed "
                f"across {requests_executed} Gemini requests ({len(unprocessed_pages)} pages left unprocessed)."
            )
        elif error_msg is not None:
            status = ExtractionStatus.FAILED
            user_msg = (
                f"Batch extraction failed: {error_msg}. "
                f"{len(processed_pages)} of {len(eligible_pages)} eligible pages were processed."
            )
        elif not plan.can_complete or len(unprocessed_pages) > 0:
            status = ExtractionStatus.PARTIAL_QUOTA
            user_msg = (
                f"Analysis partially completed under configured request budget ({request_budget} requests). "
                f"{len(processed_pages)} of {len(eligible_pages)} eligible pages were processed. "
                f"{len(unprocessed_pages)} pages were deferred."
            )
        else:
            status = ExtractionStatus.COMPLETE
            user_msg = (
                f"Analysis completed successfully. "
                f"All {len(processed_pages)} eligible pages processed across {requests_executed} Gemini requests."
            )

        return BatchExtractionResult(
            status=status,
            document_id=document.document_id,
            total_eligible_pages=len(eligible_pages),
            processed_pages=processed_pages,
            unprocessed_pages=unprocessed_pages,
            requests_used=requests_executed,
            requests_available=request_budget,
            batch_size=b_size,
            facts=all_facts,
            facts_rejected_grounding=total_rejected,
            error_message=error_msg,
            user_message=user_msg,
        )
