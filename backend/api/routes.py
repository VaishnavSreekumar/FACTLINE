"""FastAPI Router definitions."""

from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from backend.extraction.fact_extractor import (
    ExtractionError,
    ExtractionQuotaError,
    FactExtractor,
)
from backend.extraction.pdf_parser import PDFParser, PDFParsingError
from backend.models.document import ParsedDocument
from backend.models.fact import FactRecord
from backend.models.normalization import NormalizedFact
from backend.models.relationship import RelationshipResult
from backend.reasoning.comparability import ComparabilityGate
from backend.reasoning.matcher import CandidateMatcher
from backend.reasoning.relationships import RelationshipEngine

router = APIRouter()
parser = PDFParser()
extractor = FactExtractor()
gate = ComparabilityGate()
matcher = CandidateMatcher()
rel_engine = RelationshipEngine()


class ExtractionResponse(BaseModel):
    """Response payload for document fact extraction."""

    document_id: str = Field(..., description="Deterministic document ID")
    document_name: str = Field(..., description="Document file name")
    total_pages: int = Field(..., description="Total number of pages parsed")
    total_facts: int = Field(..., description="Total verified candidate facts extracted")
    facts: List[FactRecord] = Field(default_factory=list, description="List of verified FactRecords")


class RelationshipRequest(BaseModel):
    """Request payload for determining relationship between two normalized facts."""

    fact_a: NormalizedFact
    fact_b: NormalizedFact


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@router.post(
    "/documents/parse",
    response_model=ParsedDocument,
    summary="Parse PDF into 1-indexed page text records",
)
async def parse_document(
    file: UploadFile = File(..., description="PDF document to parse"),
):
    """Parses an uploaded PDF file into page-level text representations."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a PDF document.",
        )

    try:
        content = await file.read()
        parsed_doc = parser.parse_bytes(
            content=content,
            document_name=file.filename,
        )
        return parsed_doc
    except PDFParsingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse PDF document: {str(e)}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while parsing document: {str(e)}",
        ) from e


@router.post(
    "/documents/extract-facts",
    response_model=ExtractionResponse,
    summary="Parse PDF and extract verified structured candidate facts",
)
async def extract_facts(
    file: UploadFile = File(..., description="PDF document to extract facts from"),
):
    """Parses an uploaded PDF file and extracts verified candidate FactRecords page-by-page."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a PDF document.",
        )

    try:
        content = await file.read()
        parsed_doc = parser.parse_bytes(
            content=content,
            document_name=file.filename,
        )
        facts = extractor.extract_from_document(parsed_doc)

        return ExtractionResponse(
            document_id=parsed_doc.document_id,
            document_name=parsed_doc.document_name,
            total_pages=parsed_doc.total_pages,
            total_facts=len(facts),
            facts=facts,
        )
    except PDFParsingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse PDF document: {str(e)}",
        ) from e
    except ExtractionQuotaError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        ) from e
    except ExtractionError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM fact extraction error: {str(e)}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while extracting facts: {str(e)}",
        ) from e


@router.post(
    "/reason/relationship",
    response_model=RelationshipResult,
    summary="Evaluate deterministic relationship between two normalized facts",
)
def evaluate_relationship(
    request: RelationshipRequest,
):
    """Evaluates candidate matching, comparability, and relationship between two normalized facts."""
    cand = matcher.match_pair(request.fact_a, request.fact_b)
    comp = gate.evaluate(request.fact_a, request.fact_b)
    rel = rel_engine.determine_relationship(
        fact_a=request.fact_a,
        fact_b=request.fact_b,
        comparability=comp,
        candidate_pair=cand,
    )
    return rel


class AnalysisSummaryResponse(BaseModel):
    """Response payload for end-to-end analysis orchestration."""

    analysis_id: str = Field(..., description="Unique generated analysis execution ID")
    created_at: str = Field(..., description="Timestamp of analysis creation")
    summary: dict = Field(..., description="Summary counts of documents, facts, candidate pairs, relationships")


@router.post(
    "/analysis",
    response_model=AnalysisSummaryResponse,
    summary="Run end-to-end analysis across uploaded PDF documents and persist results",
)
async def create_analysis(
    files: List[UploadFile] = File(..., description="One or more PDF documents to analyze"),
):
    """Parses PDFs, extracts facts, normalizes, discovers candidates, gates comparability,

    evaluates relationships, and persists the complete graph in SQLite.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one PDF document must be provided.",
        )

    file_payloads: List[tuple] = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{file.filename}' is not a valid PDF document.",
            )
        content = await file.read()
        file_payloads.append((file.filename, content))

    try:
        from backend.services.analysis import AnalysisService

        service = AnalysisService()
        result = service.analyze_documents(file_payloads)
        return AnalysisSummaryResponse(**result)
    except PDFParsingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse PDF document: {str(e)}",
        ) from e
    except ExtractionQuotaError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        ) from e
    except ExtractionError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM fact extraction error: {str(e)}",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis orchestration failed: {str(e)}",
        ) from e


@router.get(
    "/analysis/{analysis_id}",
    summary="Retrieve structured analysis results including documents, facts, and relationships",
)
def get_analysis(
    analysis_id: str,
):
    """Retrieves full analysis record by analysis_id from SQLite."""
    from backend.services.analysis import AnalysisService

    service = AnalysisService()
    analysis = service.get_analysis(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID '{analysis_id}' not found.",
        )
    return analysis

