"""Structured Fact Extractor integrating Page Filtering, Context Selection, and Multi-Page Batching."""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
import httpx

load_dotenv()

from backend.extraction.evidence import EvidenceVerifier, create_evidence
from backend.models.document import PageText, ParsedDocument
from backend.models.fact import EpistemicStatus, FactRecord, Provenance, TimePeriod
from backend.page_filter.relevance import PageRelevanceFilter
from backend.context_selector.selector import ContextSelector
from backend.quota_experiment.planner import QuotaPlanner, QuotaPlan
from backend.quota_experiment.status import ExtractionStatus, BatchExtractionResult


class ExtractionError(Exception):
    """Raised when an error occurs during LLM invocation or candidate extraction."""
    pass


class ExtractionQuotaError(ExtractionError):
    """Raised when LLM API quota or rate limit is exhausted."""
    pass


class SingleBatchExecutionResult:
    """Isolated extraction result for a single batch under bounded concurrency."""

    def __init__(
        self,
        batch_index: int,
        page_numbers: List[int],
        facts: List[FactRecord],
        rejected_count: int,
        quota_error: Optional[ExtractionQuotaError] = None,
        error: Optional[Exception] = None,
    ):
        self.batch_index = batch_index
        self.page_numbers = page_numbers
        self.facts = facts
        self.rejected_count = rejected_count
        self.quota_error = quota_error
        self.error = error
        self.success = (quota_error is None and error is None)


# Canonical multi-page structured batch extraction prompt & schema
BATCH_EXTRACTION_SYSTEM_PROMPT = """You are FACTLINE's Multi-Page Structured Fact Extractor.
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
                    "data_vintage": {"type": "STRING", "nullable": True},
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


class FactExtractor:
    """Production Fact Extractor orchestrating Page Filtering, Context Selection,

    Multi-Page Batching, and Strict Evidence Verification.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        llm_caller: Optional[Callable[[str], Dict[str, Any]]] = None,
        max_pages: Optional[int] = None,
        page_selection: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        page_filter_enabled: Optional[bool] = None,
        page_relevance_threshold: Optional[float] = None,
        context_selector_enabled: Optional[bool] = None,
        context_radius: Optional[int] = None,
    ):
        """Initializes the FactExtractor with unified configuration."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        # Single canonical model configuration
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self._llm_caller = llm_caller

        # 1. Page control configurations
        if max_pages is not None:
            if not isinstance(max_pages, int) or max_pages <= 0:
                raise ValueError("max_pages must be a positive integer (> 0).")
            self.max_pages = max_pages
        else:
            env_max = os.getenv("FACTLINE_EXTRACTION_MAX_PAGES")
            if env_max is not None and env_max.strip():
                try:
                    parsed_max = int(env_max.strip())
                    if parsed_max <= 0:
                        raise ValueError(f"FACTLINE_EXTRACTION_MAX_PAGES must be a positive integer, got '{env_max}'.")
                    self.max_pages = parsed_max
                except ValueError as e:
                    if "invalid literal for int()" in str(e):
                        raise ValueError(f"FACTLINE_EXTRACTION_MAX_PAGES must be a valid integer, got '{env_max}'.") from e
                    raise
            else:
                self.max_pages = None

        if page_selection is not None:
            self.page_selection = [p for p in page_selection if isinstance(p, int) and p > 0]
        elif max_pages is None:
            env_selection = os.getenv("FACTLINE_EXTRACTION_PAGE_SELECTION")
            if env_selection is not None and env_selection.strip():
                parsed_pages = []
                for item in env_selection.split(","):
                    item_str = item.strip()
                    if item_str:
                        try:
                            p_num = int(item_str)
                            if p_num > 0:
                                parsed_pages.append(p_num)
                        except ValueError:
                            pass
                self.page_selection = parsed_pages if parsed_pages else None
            else:
                self.page_selection = None
        else:
            self.page_selection = None

        # 2. Phase 11 Page Relevance Filter settings
        if page_filter_enabled is not None:
            self.page_filter_enabled = page_filter_enabled
        else:
            env_pf = os.getenv("FACTLINE_PAGE_FILTER_ENABLED", "true").lower()
            self.page_filter_enabled = env_pf not in ("false", "0", "no")

        if page_relevance_threshold is not None:
            self.page_relevance_threshold = page_relevance_threshold
        else:
            env_thresh = os.getenv("FACTLINE_PAGE_RELEVANCE_THRESHOLD", "0.30")
            try:
                self.page_relevance_threshold = float(env_thresh)
            except ValueError:
                self.page_relevance_threshold = 0.30

        self.page_filter = PageRelevanceFilter(threshold=self.page_relevance_threshold)

        # 3. Phase 12 Context Selector settings
        if context_selector_enabled is not None:
            self.context_selector_enabled = context_selector_enabled
        else:
            env_cs = os.getenv("FACTLINE_CONTEXT_SELECTOR_ENABLED", "true").lower()
            self.context_selector_enabled = env_cs not in ("false", "0", "no")

        if context_radius is not None:
            self.context_radius = context_radius
        else:
            env_rad = os.getenv("FACTLINE_CONTEXT_RADIUS", "2")
            try:
                self.context_radius = int(env_rad)
            except ValueError:
                self.context_radius = 2

        self.context_selector = ContextSelector(
            expansion_radius=self.context_radius,
            merge_gap_threshold=2,
            include_page_header=True,
        )

        # 4. Phase 13/20 Batching and Quota Planner settings
        if batch_size is not None:
            self.batch_size = max(1, batch_size)
        else:
            env_bs = os.getenv("FACTLINE_EXTRACTION_BATCH_SIZE", "5")
            try:
                self.batch_size = max(1, int(env_bs))
            except ValueError:
                self.batch_size = 5

        # 5. Phase 24 Bounded Concurrency (clamped strictly to 1 or 2 workers)
        if max_workers is not None:
            parsed_workers = max_workers
        else:
            env_mw = os.getenv("FACTLINE_EXTRACTION_MAX_WORKERS", "2")
            try:
                parsed_workers = int(env_mw)
            except ValueError:
                parsed_workers = 2
        self.max_workers = max(1, min(2, parsed_workers))

        self.planner = QuotaPlanner(default_batch_size=self.batch_size)
        self.last_extraction_result: Optional[BatchExtractionResult] = None

    @staticmethod
    def generate_fact_id(
        document_id: str,
        page_number: int,
        entity: str,
        metric: str,
        value_raw: str,
        time_period_label: str,
    ) -> str:
        """Derives a stable, deterministic Fact ID from extraction coordinates."""
        key_str = f"{document_id}|{page_number}|{entity.strip()}|{metric.strip()}|{value_raw.strip()}|{time_period_label.strip()}"
        digest = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]
        return f"fact-{digest}"

    def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        """Calls the Gemini REST API with structured JSON output constraints."""
        if self._llm_caller is not None:
            return self._llm_caller(prompt)

        if not self.api_key:
            raise ExtractionError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in environment."
            )

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
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        if parts:
                            return json.loads(parts[0].get("text", "{}"))
                        return {"facts": []}
                    elif resp.status_code == 429:
                        # CRITICAL: Treat 429 as quota exhaustion. Halt immediately without retry.
                        raise ExtractionQuotaError(
                            "Gemini API quota is currently exhausted for this project/model. "
                            "No further extraction requests were attempted."
                        )
                    elif resp.status_code == 503:
                        if attempt < max_attempts - 1:
                            backoff = 1.0 * (2 ** attempt)  # attempt 0: 1.0s, attempt 1: 2.0s
                            time.sleep(backoff)
                            continue
                        raise ExtractionError(
                            f"Gemini API service temporarily unavailable (HTTP 503 after {max_attempts} attempts)"
                        )
                    else:
                        raise ExtractionError(f"Gemini API error (HTTP {resp.status_code}): {resp.text}")
                except ExtractionQuotaError:
                    raise
                except ExtractionError:
                    raise
                except (httpx.RequestError, json.JSONDecodeError) as e:
                    raise ExtractionError(f"Network error while calling Gemini API: {type(e).__name__}: {e}") from e

    def format_batch_prompt(
        self,
        batch_prompt_texts: List[Tuple[int, str]],
        document_name: str,
        document_date: Optional[str] = None,
    ) -> str:
        """Formats multiple pages with unambiguous delimiters and page attributions."""
        sections = [
            f"Document: {document_name}",
            f"Document Date: {document_date or 'Unknown'}",
            f"Pages in this batch: {', '.join(str(p_num) for p_num, _ in batch_prompt_texts)}",
            "",
            "Extract all facts and assign each fact strictly to its source page_number.",
            "",
        ]

        for p_num, prompt_text in batch_prompt_texts:
            sections.append(f"===== DOCUMENT PAGE {p_num} START =====")
            sections.append(prompt_text.strip())
            sections.append(f"===== DOCUMENT PAGE {p_num} END =====")
            sections.append("")

        return "\n".join(sections)

    def filter_pages(self, pages: List[PageText]) -> List[PageText]:
        """Filters pages based on page selection or max pages limits."""
        if not pages:
            return []

        if self.page_selection is not None:
            allowed_set = set(self.page_selection)
            eligible = [p for p in pages if p.page_number in allowed_set]
        elif self.max_pages is not None:
            eligible = pages[: self.max_pages]
        else:
            eligible = list(pages)

        return eligible

    def extract_batch(
        self,
        batch_pages: List[PageText],
        document_id: str,
        document_name: str,
        document_date: Optional[str] = None,
    ) -> Tuple[List[FactRecord], int]:
        """Extracts facts from a batch of pages, verifying evidence strictly against original PageText."""
        if not batch_pages:
            return [], 0

        valid_text_pages = [p for p in batch_pages if p.has_text and p.text.strip()]
        if not valid_text_pages:
            return [], 0

        # Authoritative original page lookup
        original_page_map: Dict[int, PageText] = {p.page_number: p for p in batch_pages}

        # Prepare prompt texts (using ContextSelector if enabled, else original text)
        batch_prompt_items: List[Tuple[int, str]] = []
        for p in valid_text_pages:
            if self.context_selector_enabled:
                context_res = self.context_selector.select_page_context(
                    page_text=p.text,
                    page_number=p.page_number,
                    document_id=document_id,
                )
                prompt_text = context_res.combined_source_text if not context_res.is_empty else p.text
            else:
                prompt_text = p.text

            batch_prompt_items.append((p.page_number, prompt_text))

        prompt = self.format_batch_prompt(batch_prompt_items, document_name, document_date)
        raw_response = self._call_gemini(prompt)
        candidates = raw_response.get("facts", [])

        verified_facts: List[FactRecord] = []
        rejected_count = 0

        for candidate in candidates:
            try:
                reported_page_num = candidate.get("page_number")
                if reported_page_num is None and len(batch_pages) == 1:
                    reported_page_num = batch_pages[0].page_number
                if not isinstance(reported_page_num, int) or reported_page_num not in original_page_map:
                    rejected_count += 1
                    continue

                # CRITICAL: Verify evidence strictly against AUTHORITATIVE original page text
                target_original_page = original_page_map[reported_page_num]

                entity = str(candidate.get("entity", "")).strip()
                metric = str(candidate.get("metric", "")).strip()
                value_raw = str(candidate.get("value_raw", "")).strip()
                supporting_text = str(candidate.get("supporting_text", "")).strip()

                if not entity or not metric or not value_raw or not supporting_text:
                    rejected_count += 1
                    continue

                provenance = create_evidence(
                    document_id=document_id,
                    page_number=reported_page_num,
                    supporting_text=supporting_text,
                    document_date=document_date,
                )

                # Reject if supporting text is not found on the declared original page
                if not EvidenceVerifier.verify_provenance(provenance, target_original_page):
                    rejected_count += 1
                    continue

                # Parse optional fields
                raw_status = candidate.get("epistemic_status", "reported")
                try:
                    epistemic_status = EpistemicStatus(str(raw_status).lower())
                except ValueError:
                    epistemic_status = EpistemicStatus.REPORTED

                value_numeric = candidate.get("value_numeric")
                if value_numeric is not None:
                    try:
                        value_numeric = float(value_numeric)
                    except (ValueError, TypeError):
                        value_numeric = None

                unit = candidate.get("unit")
                unit = str(unit).strip() if unit else None

                tp_dict = candidate.get("time_period", {})
                tp_label = str(tp_dict.get("label", "")).strip() if isinstance(tp_dict, dict) else ""
                if not tp_label:
                    tp_label = "Unspecified"
                time_period = TimePeriod(
                    label=tp_label,
                    start_date=tp_dict.get("start_date") if isinstance(tp_dict, dict) else None,
                    end_date=tp_dict.get("end_date") if isinstance(tp_dict, dict) else None,
                )

                scope = candidate.get("scope")
                scope = str(scope).strip() if scope else None

                geography = candidate.get("geography")
                geography = str(geography).strip() if geography else None

                data_vintage = candidate.get("data_vintage")
                data_vintage = str(data_vintage).strip() if data_vintage else None

                raw_conf = candidate.get("extraction_confidence", 0.0)
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
                    scope=scope,
                    geography=geography,
                    epistemic_status=epistemic_status,
                    data_vintage=data_vintage,
                    provenance=provenance,
                    extraction_confidence=confidence,
                )
                verified_facts.append(fact_record)

            except Exception:
                rejected_count += 1
                continue

        return verified_facts, rejected_count

    def _execute_batch_isolated(
        self,
        batch_index: int,
        page_numbers: List[int],
        document_id: str,
        document_name: str,
        page_lookup: Dict[int, PageText],
        document_date: Optional[str] = None,
    ) -> SingleBatchExecutionResult:
        """Executes a single batch with full isolation and strict error capture."""
        batch_objs = [page_lookup[p] for p in page_numbers if p in page_lookup]
        if not batch_objs:
            return SingleBatchExecutionResult(
                batch_index=batch_index,
                page_numbers=page_numbers,
                facts=[],
                rejected_count=0,
            )

        try:
            facts, rejected = self.extract_batch(
                batch_pages=batch_objs,
                document_id=document_id,
                document_name=document_name,
                document_date=document_date,
            )
            return SingleBatchExecutionResult(
                batch_index=batch_index,
                page_numbers=page_numbers,
                facts=facts,
                rejected_count=rejected,
            )
        except ExtractionQuotaError as q_err:
            return SingleBatchExecutionResult(
                batch_index=batch_index,
                page_numbers=page_numbers,
                facts=[],
                rejected_count=0,
                quota_error=q_err,
            )
        except Exception as err:
            return SingleBatchExecutionResult(
                batch_index=batch_index,
                page_numbers=page_numbers,
                facts=[],
                rejected_count=0,
                error=err,
            )

    def extract_from_page(
        self,
        page: PageText,
        document_id: str,
        document_name: str,
        document_date: Optional[str] = None,
    ) -> List[FactRecord]:
        """Single-page extraction interface (delegates to batch of size 1)."""
        facts, _ = self.extract_batch(
            batch_pages=[page],
            document_id=document_id,
            document_name=document_name,
            document_date=document_date,
        )
        return facts

    def extract_from_document(
        self,
        document: ParsedDocument,
        document_date: Optional[str] = None,
    ) -> List[FactRecord]:
        """Orchestrates end-to-end extraction across a parsed document and returns verified facts."""
        result = self.extract_document_result(document, document_date=document_date)
        return result.facts

    def extract_document_result(
        self,
        document: ParsedDocument,
        document_date: Optional[str] = None,
    ) -> BatchExtractionResult:
        """Full extraction orchestration returning BatchExtractionResult with status and metadata."""
        if not document.pages:
            res = BatchExtractionResult(
                status=ExtractionStatus.COMPLETE,
                document_id=document.document_id,
                total_eligible_pages=0,
                processed_pages=[],
                unprocessed_pages=[],
                requests_used=0,
                requests_available=self.max_pages,
                batch_size=self.batch_size,
                facts=[],
                facts_rejected_grounding=0,
                user_message="Document contains 0 pages.",
            )
            self.last_extraction_result = res
            return res

        # Step 1: Determine candidate pages
        text_pages = [p for p in document.pages if p.has_text and p.text.strip()]
        page_relevance_scores: Dict[int, float] = {}

        if self.page_selection is not None:
            allowed_set = set(self.page_selection)
            eligible_pages = [p.page_number for p in text_pages if p.page_number in allowed_set]
        elif self.page_filter_enabled:
            # Score pages with Phase 11 Page Relevance Filter
            filter_results = self.page_filter.filter_pages(text_pages)
            eligible_pages = []
            for fr in filter_results:
                page_relevance_scores[fr.page_number] = fr.relevance_score
                if fr.selected:
                    eligible_pages.append(fr.page_number)
        else:
            eligible_pages = [p.page_number for p in text_pages]

        # Step 2: Quota Planning
        plan: QuotaPlan = self.planner.calculate_plan(
            eligible_pages=eligible_pages,
            request_budget=self.max_pages,
            batch_size=self.batch_size,
            page_relevance_scores=page_relevance_scores if page_relevance_scores else None,
        )

        page_lookup: Dict[int, PageText] = {p.page_number: p for p in document.pages}
        all_facts: List[FactRecord] = []
        total_rejected = 0
        processed_pages: List[int] = []
        requests_executed = 0
        quota_hit = False
        error_msg: Optional[str] = None

        # Step 3: Bounded Concurrency Batch Extraction
        batch_results: List[SingleBatchExecutionResult] = []

        if self.max_workers == 1 or len(plan.batches) <= 1:
            for b_idx, batch_p_nums in enumerate(plan.batches):
                res = self._execute_batch_isolated(
                    batch_index=b_idx,
                    page_numbers=batch_p_nums,
                    document_id=document.document_id,
                    document_name=document.document_name,
                    page_lookup=page_lookup,
                    document_date=document_date,
                )
                batch_results.append(res)
                if not res.success:
                    break
        else:
            stop_scheduling = False
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                batch_iter = enumerate(plan.batches)
                in_flight: Dict[Any, int] = {}

                # Seed initial worker pool up to max_workers (max 2)
                while len(in_flight) < self.max_workers:
                    try:
                        b_idx, b_p_nums = next(batch_iter)
                        fut = executor.submit(
                            self._execute_batch_isolated,
                            b_idx,
                            b_p_nums,
                            document.document_id,
                            document.document_name,
                            page_lookup,
                            document_date,
                        )
                        in_flight[fut] = b_idx
                    except StopIteration:
                        break

                while in_flight:
                    done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                    for fut in done:
                        in_flight.pop(fut)
                        res = fut.result()
                        batch_results.append(res)
                        if not res.success:
                            stop_scheduling = True

                    if not stop_scheduling:
                        while len(in_flight) < self.max_workers:
                            try:
                                b_idx, b_p_nums = next(batch_iter)
                                fut = executor.submit(
                                    self._execute_batch_isolated,
                                    b_idx,
                                    b_p_nums,
                                    document.document_id,
                                    document.document_name,
                                    page_lookup,
                                    document_date,
                                )
                                in_flight[fut] = b_idx
                            except StopIteration:
                                break

        # Deterministic sorting by batch_index to guarantee order preservation
        batch_results.sort(key=lambda r: r.batch_index)

        for res in batch_results:
            if res.success:
                requests_executed += 1
                all_facts.extend(res.facts)
                total_rejected += res.rejected_count
                processed_pages.extend(res.page_numbers)
            elif res.quota_error is not None:
                quota_hit = True
                if error_msg is None:
                    error_msg = str(res.quota_error)
            elif res.error is not None:
                if error_msg is None:
                    error_msg = f"Extraction failed on pages {res.page_numbers}: {res.error}"

        unprocessed_pages = [p for p in eligible_pages if p not in set(processed_pages)]

        # Step 4: Status Determination
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
                f"Extraction failed: {error_msg}. "
                f"{len(processed_pages)} of {len(eligible_pages)} eligible pages were processed."
            )
        elif not plan.can_complete or len(unprocessed_pages) > 0:
            status = ExtractionStatus.PARTIAL_QUOTA
            user_msg = (
                f"Analysis partially completed under configured request limit. "
                f"{len(processed_pages)} of {len(eligible_pages)} eligible pages were processed. "
                f"{len(unprocessed_pages)} pages deferred."
            )
        else:
            status = ExtractionStatus.COMPLETE
            user_msg = (
                f"Analysis completed successfully. "
                f"All {len(processed_pages)} eligible pages processed across {requests_executed} Gemini requests."
            )

        res = BatchExtractionResult(
            status=status,
            document_id=document.document_id,
            total_eligible_pages=len(eligible_pages),
            processed_pages=processed_pages,
            unprocessed_pages=unprocessed_pages,
            requests_used=requests_executed,
            requests_available=self.max_pages,
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            facts=all_facts,
            facts_rejected_grounding=total_rejected,
            error_message=error_msg,
            user_message=user_msg,
        )
        self.last_extraction_result = res
        return res
