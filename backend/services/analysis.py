"""Analysis Orchestration Service for FACTLINE."""

import uuid
from typing import Any, Dict, List, Optional, Tuple

from backend.db.database import DatabaseRepository
from backend.extraction.fact_extractor import ExtractionError, FactExtractor
from backend.extraction.pdf_parser import PDFParser, PDFParsingError
from backend.models.document import ParsedDocument
from backend.models.fact import FactRecord
from backend.models.normalization import NormalizedFact
from backend.models.relationship import CandidatePair, RelationshipResult
from backend.normalization.normalizer import FactNormalizer
from backend.reasoning.comparability import ComparabilityGate
from backend.reasoning.matcher import CandidateMatcher
from backend.reasoning.relationships import RelationshipEngine
from backend.reasoning.surfacing import RelationshipSurfacingFilter


class AnalysisService:
    """Orchestrates end-to-end PDF parsing, fact extraction, normalization,

    candidate matching, comparability gating, relationship evaluation, and persistence.
    """

    def __init__(
        self,
        parser: Optional[PDFParser] = None,
        extractor: Optional[FactExtractor] = None,
        normalizer: Optional[FactNormalizer] = None,
        matcher: Optional[CandidateMatcher] = None,
        gate: Optional[ComparabilityGate] = None,
        surfacing_filter: Optional[RelationshipSurfacingFilter] = None,
        rel_engine: Optional[RelationshipEngine] = None,
        repository: Optional[DatabaseRepository] = None,
    ):
        self.parser = parser or PDFParser()
        self.extractor = extractor or FactExtractor()
        self.normalizer = normalizer or FactNormalizer()
        self.matcher = matcher or CandidateMatcher()
        self.gate = gate or ComparabilityGate()
        self.surfacing_filter = surfacing_filter or RelationshipSurfacingFilter()
        self.rel_engine = rel_engine or RelationshipEngine()
        self.repository = repository or DatabaseRepository()

    def analyze_documents(
        self,
        files: List[Tuple[str, bytes]],
    ) -> Dict[str, Any]:
        """Runs the complete end-to-end analysis pipeline across N uploaded PDF files.

        Args:
            files: List of (filename, bytes) tuples.

        Returns:
            Structured summary dictionary containing analysis_id, metrics, and persistence confirmation.
        """
        if not files:
            raise ValueError("At least one document file must be provided for analysis.")

        # Step 1: Parse all documents
        parsed_documents: List[ParsedDocument] = []
        for filename, content in files:
            if not filename.lower().endswith(".pdf"):
                raise ValueError(f"File '{filename}' is not a valid PDF document.")
            parsed_doc = self.parser.parse_bytes(content=content, document_name=filename)
            parsed_documents.append(parsed_doc)

        # Step 2: Extract facts page-by-page across all documents using quota-aware extraction
        all_facts: List[FactRecord] = []
        doc_extraction_statuses: List[Dict[str, Any]] = []
        overall_status = "COMPLETE"
        total_requests_used = 0
        total_eligible_pages = 0
        total_processed_pages = 0
        total_unprocessed_pages = 0

        for doc in parsed_documents:
            doc_facts = self.extractor.extract_from_document(doc)
            all_facts.extend(doc_facts)

            res = getattr(self.extractor, "last_extraction_result", None)
            if res is not None and getattr(res, "document_id", None) == doc.document_id:
                status_str = res.status.value if hasattr(res.status, "value") else str(res.status)
                doc_extraction_statuses.append({
                    "document_id": doc.document_id,
                    "document_name": doc.document_name,
                    "status": status_str,
                    "total_eligible_pages": res.total_eligible_pages,
                    "processed_pages": res.processed_pages,
                    "unprocessed_pages": res.unprocessed_pages,
                    "requests_used": res.requests_used,
                    "requests_available": res.requests_available,
                    "batch_size": res.batch_size,
                    "max_workers": getattr(res, "max_workers", 2),
                    "facts_extracted": len(res.facts),
                    "facts_rejected_grounding": res.facts_rejected_grounding,
                    "user_message": res.user_message,
                    "error_message": res.error_message,
                })
                total_requests_used += res.requests_used
                total_eligible_pages += res.total_eligible_pages
                total_processed_pages += len(res.processed_pages)
                total_unprocessed_pages += len(res.unprocessed_pages)
                if status_str == "FAILED":
                    overall_status = "FAILED"
                elif status_str in ("QUOTA_EXHAUSTED", "PARTIAL_QUOTA") and overall_status != "FAILED":
                    overall_status = status_str
            else:
                total_eligible_pages += doc.total_pages
                total_processed_pages += doc.total_pages
                total_requests_used += 1

        extraction_status_summary = {
            "status": overall_status,
            "total_eligible_pages": total_eligible_pages,
            "total_processed_pages": total_processed_pages,
            "total_unprocessed_pages": total_unprocessed_pages,
            "requests_used": total_requests_used,
            "documents": doc_extraction_statuses if doc_extraction_statuses else None,
        }

        # Step 3: Deterministically normalize all facts
        normalized_facts: List[NormalizedFact] = self.normalizer.normalize_batch(all_facts)

        # Step 4: Deterministic candidate fact pair matching
        # Uses existing Phase 4 CandidateMatcher without modifying candidate logic
        candidate_tuples = self.matcher.find_candidates(normalized_facts)
        candidate_pair_count = len(candidate_tuples)

        # Steps 5 & 6: Comparability Gate, Surfacing Filter, and Relationship Engine
        relationships: List[RelationshipResult] = []
        for fact_a, fact_b in candidate_tuples:
            candidate_pair = self.matcher.match_pair(fact_a, fact_b)
            if candidate_pair is None:
                continue

            comp_result = self.gate.evaluate(fact_a, fact_b)
            surfacing_decision = self.surfacing_filter.evaluate(
                fact_a=fact_a,
                fact_b=fact_b,
                comparability=comp_result,
                candidate_pair=candidate_pair,
            )
            if not surfacing_decision.should_surface:
                continue

            rel_result = self.rel_engine.determine_relationship(
                fact_a=fact_a,
                fact_b=fact_b,
                comparability=comp_result,
                candidate_pair=candidate_pair,
            )
            relationships.append(rel_result)

        # Step 7: Atomic SQLite Persistence with generated execution UUID
        analysis_id = str(uuid.uuid4())
        persisted_result = self.repository.save_analysis(
            analysis_id=analysis_id,
            documents=parsed_documents,
            normalized_facts=normalized_facts,
            relationships=relationships,
            candidate_pair_count=candidate_pair_count,
            extraction_status=extraction_status_summary,
        )

        return persisted_result

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves structured analysis records (documents, facts with provenance, relationships)."""
        return self.repository.get_analysis(analysis_id)
