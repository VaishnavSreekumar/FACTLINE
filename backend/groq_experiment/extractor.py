"""Groq Fact Extractor with Evidence Verification and Semantic Input Fingerprinting."""

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.context_selector.selector import ContextSelector
from backend.extraction.evidence import EvidenceVerifier, create_evidence
from backend.extraction.fact_extractor import (
    BATCH_EXTRACTION_JSON_SCHEMA,
    BATCH_EXTRACTION_SYSTEM_PROMPT,
    FactExtractor,
)
from backend.groq_experiment.models import SingleBatchProviderResult
from backend.groq_experiment.provider import GroqProviderClient
from backend.models.document import PageText
from backend.models.fact import EpistemicStatus, FactRecord, TimePeriod


def compute_semantic_input_hash(
    batch_pages: List[Tuple[int, str]],
    document_name: str,
    document_date: Optional[str] = None,
) -> str:
    """Computes a provider-independent SHA-256 hash of the exact shared semantic inputs.

    Covers:
    - ordered page numbers
    - selected context text per page
    - shared system extraction instructions
    - shared user delimiter format
    - extraction schema semantics
    - batch ordering
    """
    semantic_dict = {
        "system_prompt": BATCH_EXTRACTION_SYSTEM_PROMPT.strip(),
        "schema_keys": sorted(BATCH_EXTRACTION_JSON_SCHEMA["properties"]["facts"]["items"]["properties"].keys()),
        "document_name": document_name,
        "document_date": document_date or "Unknown",
        "batch_pages": [
            {"page_number": p_num, "text": text.strip()}
            for p_num, text in batch_pages
        ],
    }
    canonical_json = json.dumps(semantic_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class GroqFactExtractor:
    """Isolated Groq Fact Extractor for benchmark evaluation."""

    def __init__(
        self,
        provider_client: Optional[GroqProviderClient] = None,
        context_selector_enabled: bool = True,
        context_radius: int = 2,
    ):
        self.provider = provider_client or GroqProviderClient()
        self.context_selector_enabled = context_selector_enabled
        self.context_radius = context_radius
        self.context_selector = ContextSelector(
            expansion_radius=context_radius,
            merge_gap_threshold=2,
            include_page_header=True,
        )

    def format_batch_prompt(
        self,
        batch_prompt_texts: List[Tuple[int, str]],
        document_name: str,
        document_date: Optional[str] = None,
    ) -> str:
        """Formats multi-page batch prompt identically to production Gemini extractor."""
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

    def extract_batch(
        self,
        batch_pages: List[PageText],
        document_id: str,
        document_name: str,
        document_date: Optional[str] = None,
        batch_index: int = 0,
    ) -> SingleBatchProviderResult:
        """Executes structured extraction for a single batch of pages via Groq."""
        # 1. Prepare context-selected text
        batch_prompt_texts: List[Tuple[int, str]] = []
        original_page_map: Dict[int, PageText] = {}

        for page in batch_pages:
            original_page_map[page.page_number] = page
            if self.context_selector_enabled:
                context_res = self.context_selector.select_page_context(
                    page_text=page.text,
                    page_number=page.page_number,
                    document_id=document_id,
                )
                prompt_text = context_res.combined_source_text if not context_res.is_empty else page.text
                batch_prompt_texts.append((page.page_number, prompt_text))
            else:
                batch_prompt_texts.append((page.page_number, page.text))

        # 2. Compute semantic input fingerprint
        input_hash = compute_semantic_input_hash(
            batch_pages=batch_prompt_texts,
            document_name=document_name,
            document_date=document_date,
        )

        user_prompt = self.format_batch_prompt(
            batch_prompt_texts=batch_prompt_texts,
            document_name=document_name,
            document_date=document_date,
        )

        # 3. Call Groq provider
        raw_response, latency_ms, failure_class, error_msg, rate_limits, tokens = (
            self.provider.call_structured_extraction(user_prompt)
        )

        if failure_class is not None or raw_response is None:
            return SingleBatchProviderResult(
                batch_index=batch_index,
                document_id=document_id,
                page_numbers=[p.page_number for p in batch_pages],
                semantic_input_hash=input_hash,
                latency_ms=latency_ms,
                raw_facts_count=0,
                verified_facts=[],
                rejected_facts_count=0,
                failure_class=failure_class,
                error_message=error_msg,
                rate_limit_info=rate_limits,
                tokens_used=tokens,
            )

        # 4. Strict Evidence Verification & FactRecord Construction
        candidate_facts = raw_response.get("facts", [])
        if not isinstance(candidate_facts, list):
            candidate_facts = []

        verified_facts: List[FactRecord] = []
        rejected_count = 0
        duplicate_count = 0
        contamination_count = 0
        seen_fact_ids = set()

        batch_page_numbers = set(original_page_map.keys())

        for cand in candidate_facts:
            if not isinstance(cand, dict):
                rejected_count += 1
                continue

            try:
                reported_page_num = cand.get("page_number")
                if reported_page_num not in batch_page_numbers:
                    # Page attribution outside the batch is an immediate rejection
                    rejected_count += 1
                    contamination_count += 1
                    continue

                target_original_page = original_page_map[reported_page_num]

                entity = str(cand.get("entity", "")).strip()
                metric = str(cand.get("metric", "")).strip()
                value_raw = str(cand.get("value_raw", "")).strip()
                supporting_text = str(cand.get("supporting_text", "")).strip()

                if not entity or not metric or not value_raw or not supporting_text:
                    rejected_count += 1
                    continue

                provenance = create_evidence(
                    document_id=document_id,
                    page_number=reported_page_num,
                    supporting_text=supporting_text,
                    document_date=document_date,
                )

                # Authoritative Evidence Verification against original page
                if not EvidenceVerifier.verify_provenance(provenance, target_original_page):
                    # Check if text was accidentally pulled from another page in the batch (cross-page contamination)
                    is_contam = False
                    for other_p_num, other_page in original_page_map.items():
                        if other_p_num != reported_page_num and EvidenceVerifier.verify_provenance(
                            create_evidence(document_id, other_p_num, supporting_text), other_page
                        ):
                            is_contam = True
                            break
                    if is_contam:
                        contamination_count += 1
                    rejected_count += 1
                    continue

                # Parse optional fields
                raw_status = cand.get("epistemic_status", "reported")
                try:
                    epistemic_status = EpistemicStatus(str(raw_status).lower())
                except ValueError:
                    epistemic_status = EpistemicStatus.REPORTED

                val_num = cand.get("value_numeric")
                value_numeric = float(val_num) if val_num is not None else None

                unit = cand.get("unit")
                unit = str(unit).strip() if unit else None

                tp_dict = cand.get("time_period", {})
                tp_label = str(tp_dict.get("label", "")).strip() if isinstance(tp_dict, dict) else ""
                if not tp_label:
                    tp_label = "Unspecified"
                time_period = TimePeriod(
                    label=tp_label,
                    start_date=tp_dict.get("start_date") if isinstance(tp_dict, dict) else None,
                    end_date=tp_dict.get("end_date") if isinstance(tp_dict, dict) else None,
                )

                scope = cand.get("scope")
                scope = str(scope).strip() if scope else None

                geography = cand.get("geography")
                geography = str(geography).strip() if geography else None

                data_vintage = cand.get("data_vintage")
                data_vintage = str(data_vintage).strip() if data_vintage else None

                raw_conf = cand.get("extraction_confidence", 1.0)
                try:
                    confidence = max(0.0, min(1.0, float(raw_conf)))
                except (ValueError, TypeError):
                    confidence = 1.0

                # Deterministic fact identity using existing production coordinate logic
                fact_id = FactExtractor.generate_fact_id(
                    document_id=document_id,
                    page_number=reported_page_num,
                    entity=entity,
                    metric=metric,
                    value_raw=value_raw,
                    time_period_label=time_period.label,
                )

                if fact_id in seen_fact_ids:
                    duplicate_count += 1
                    rejected_count += 1
                    continue
                seen_fact_ids.add(fact_id)

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

        return SingleBatchProviderResult(
            batch_index=batch_index,
            document_id=document_id,
            page_numbers=[p.page_number for p in batch_pages],
            semantic_input_hash=input_hash,
            latency_ms=latency_ms,
            raw_facts_count=len(candidate_facts),
            verified_facts=verified_facts,
            rejected_facts_count=rejected_count,
            duplicate_facts_count=duplicate_count,
            cross_page_contamination_count=contamination_count,
            raw_facts=candidate_facts,
            failure_class=None,
            error_message=None,
            rate_limit_info=rate_limits,
            tokens_used=tokens,
        )
