# FACTLINE
### Evidence-First Cross-Document Fact Intelligence

FACTLINE is an evidence-first fact intelligence and cross-document reconciliation system designed for complex, heterogeneous document corpora (such as financial annual reports, investor presentations, and macroeconomic surveys).

---

## 1. What FACTLINE Is

FACTLINE extracts structured quantitative and business facts from multi-page PDFs, verifies them strictly against verbatim source evidence, normalizes their dimensions deterministically, enforces a non-negotiable **Comparability Gate**, and explains cross-document relationships (`CORROBORATES`, `CONTRADICTS`, `CONTEXT_RESOLVES`, `EVOLVES_FROM`, `SUPERSEDES`, or `UNRESOLVED`) with full provenance.

---

## 2. The Core Problem

In cross-document research, apparent disagreements frequently arise between documents published at different times, using different units, accounting conventions, reporting scopes, or revision cycles. 

Existing approaches suffer from critical flaws:
- **Blind Metric Comparison**: Treating `₹81,415.38 million` and `₹8,142 crore` as a numerical conflict simply because the raw strings differ, when they are mathematically identical under unit normalization and display rounding.
- **Premature Similarity**: Comparing quarterly figures against annual figures (e.g. `Q4 FY24` active customers vs `FY24` full-year active customers) or First Advance Estimates against Second Advance Estimates without vintage awareness.
- **Hallucinated Syntheses**: Black-box LLMs asserting cross-document agreement or contradiction without grounding claims in verifiable page excerpts.

FACTLINE's foundational principle:
> **"Never compare two values before establishing that the claims are comparable."**

---

## 3. End-to-End Pipeline Architecture

```text
                       Uploaded PDF Documents
                                 │
                                 ▼
                     Page-Aware PDF Parser (PyMuPDF)
                     (1-indexed PageText representation)
                                 │
                                 ▼
                     Page Relevance Filter
                     (Eliminates boilerplate, covers, TOCs)
                                 │
                                 ▼
                     Context Selector & Compression
                     (Extracts fact-dense windows + headers)
                                 │
                                 ▼
                     Quota Planner & Multi-Page Batching
                     (Batch size = 5, Bounded 2-worker concurrency)
                                 │
                                 ▼
                     Gemini Fact Extractor (gemini-3.6-flash)
                     (Extracts candidate facts + page attribution)
                                 │
                                 ▼
                     Strict EvidenceVerifier Gating
                     (Verifies verbatim substring on declared page;
                      rejects hallucinations and cross-page contamination)
                                 │
                                 ▼
                     Deterministic Normalization Engine
                     (Decimal scaling, units, dates, entity canonicalization)
                                 │
                                 ▼
                     Candidate Matcher (High Precision)
                     (Entity alignment + Metric noun token overlap;
                      "Numbers never create candidates")
                                 │
                                 ▼
                     COMPARABILITY GATE
                     (Evaluates 8 dimensions: entity, metric, time,
                      unit family, scope, geography, epistemic status, vintage)
                                 │
                   ┌─────────────┼─────────────┐
                   ▼             ▼             ▼
              COMPARABLE   INSUFFICIENT   NON_COMPARABLE
                   │          CONTEXT          │
                   │             │             │
                   ▼             └─────────┐   │
           Relationship Engine             │   │
                   │                       │   │
     ┌─────────────┼─────────────┬─────────┴───┴───────┐
     ▼             ▼             ▼                     ▼
CORROBORATES  CONTRADICTS  CONTEXT_RESOLVES       UNRESOLVED
     │                           │
     ▼                           ▼
EVOLVES_FROM / SUPERSEDES   Display Rounding
                            Reconciliation
                                 │
                                 ▼
                     Relationship Surfacing Filter
                     (Hides uninteresting non-comparable clutter;
                      surfaces high-signal insights)
                                 │
                                 ▼
                     SQLite Persistence & Evidence-First UI
```

---

## 4. Why This Is Not Simply RAG

Standard Retrieval-Augmented Generation (RAG) retrieves chunks by vector embedding similarity and prompts an LLM to generate an unconstrained prose summary. This introduces hallucinations, cannot audit mathematical rounding, and fails to distinguish between vintage revisions and factual contradictions.

In contrast, FACTLINE:
- Extracts **discrete, typed Fact Records** with explicit coordinates (`entity`, `metric`, `value_raw`, `value_numeric`, `unit`, `time_period`, `scope`, `geography`, `epistemic_status`, `data_vintage`).
- Normalizes all mathematical and temporal coordinates in **pure deterministic code**.
- Evaluates comparability through an **explicit 8-dimensional gate** before allowing numerical comparison.
- Generates **structured, auditable relationship graphs** backed by exact page-level provenance.

---

## 5. Why Provenance Matters

Every accepted fact in FACTLINE requires a verified `Provenance` object specifying:
- `document_id` and `document_name`
- 1-indexed `page_number`
- `supporting_text`: verbatim substring extracted directly from that specific page.

`EvidenceVerifier` checks every candidate fact against the authoritative source page text. If a fact's supporting text is missing, hallucinated, or attributed to the wrong page in a batch, it is **immediately rejected**.

---

## 6. Deterministic Normalization

Normalization is executed strictly in deterministic Python using `Decimal` fixed-point arithmetic:
- **Monetary Scaling**: `₹8,142 Cr` $\rightarrow$ `81,420,000,000 INR`; `₹81,415.38 million` $\rightarrow$ `81,415,380,000 INR`.
- **Percentage Preservation**: `6.4%` is normalized to `6.4 percent` (never inverted or multiplied to `0.064`).
- **Dates & Fiscal Years**: `FY24` under Indian convention parses to `2023-04-01` to `2024-03-31`; `Q4 FY24` parses to `2024-01-01` to `2024-03-31`.
- **Ambiguous Units**: Bare `$` without country context is flagged as ambiguous to prevent cross-currency distortion.

---

## 7. The Comparability Gate

The `ComparabilityGate` inspects 8 core dimensions before allowing relationship classification:
1. **Entity Alignment**: Target entities must match or resolve canonically.
2. **Metric Semantics**: Metrics must represent compatible underlying indicators.
3. **Unit Family**: Currency cannot be compared with counts or percentages.
4. **Temporal Overlap**: Annual periods cannot be directly compared with single quarters.
5. **Reporting Scope**: Consolidated corporate scope cannot be compared with standalone branch scope.
6. **Geographic Scope**: National figures cannot be compared with regional splits.
7. **Epistemic Status**: Reported actuals cannot contradict forward-looking projections or targets.
8. **Data Vintage**: Revisions across advance estimates require vintage-aware progression.

If any dimension is incompatible or missing context, the gate emits `NON_COMPARABLE` or `INSUFFICIENT_CONTEXT`, routing the pair strictly to `UNRESOLVED`.

---

## 8. Relationship Engine Vocabulary

When claims pass the Comparability Gate as `COMPARABLE`, the `RelationshipEngine` evaluates them against deterministic rules:

| Relationship | Condition |
| :--- | :--- |
| **`CORROBORATES`** | Exact normalized value agreement across distinct documents for the same metric and period. |
| **`CONTEXT_RESOLVES`** | Surface differences resolve through unit-aware display resolution and mathematical rounding reconciliation. |
| **`EVOLVES_FROM`** | Material differences explained by sequential data vintage revisions (e.g. First Advance Estimate $\rightarrow$ Second Advance Estimate). |
| **`SUPERSEDES`** | Explicitly stated accounting restatements or retroactive revisions. |
| **`CONTRADICTS`** | Material disagreements beyond rounding precision without vintage, scope, or accounting explanation. |
| **`UNRESOLVED`** | Pairs blocked by the Comparability Gate due to mismatched periods, differing scopes, or insufficient context. |

---

## 9. Handling Ambiguity & Missing Context

When crucial metadata (such as time period, scope, or unit) is omitted in the source document, FACTLINE refuses to guess. The Comparability Gate flags the missing dimension with structured reason codes (`MISSING_TIME_PERIOD`, `AMBIGUOUS_UNIT`, `SCOPE_MISMATCH`) and classifies the relationship as `UNRESOLVED` with an explanatory diagnostic.

---

## 10. Extraction Failure & Quota Handling

Extraction operates under bounded budgets:
- **HTTP 429 Quota Exhaustion**: Execution halts immediately without infinite retries; all facts verified prior to the 429 error are preserved and persisted.
- **HTTP 503 Transient Errors**: Bounded exponential backoff retry (up to 3 attempts) before safe termination.
- **Analysis Statuses**: `COMPLETE`, `PARTIAL_QUOTA`, `PARTIAL_ANALYSIS`, or `EXTRACTION_FAILED` are recorded explicitly in database metadata and surfaced in the UI.

---

## 11. Running the Application

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- Google Gemini API Key

### Backend Setup
```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_gemini_api_key

# 4. Start FastAPI server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Backend Health Check: `GET http://127.0.0.1:8000/health`

### Frontend Setup
```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start Vite dev server
npm run dev
```
Frontend UI available at: `http://localhost:5173`

### Running the Test Suite
```bash
pytest backend/tests/ -v
```
*(Runs 278 automated unit, regression, and invariant tests across 26 test suites).*

---

## 12. Golden Demo Cases

FACTLINE is validated against four established benchmark scenarios:

### Case 1 — Corroboration / Rounding Reconciliation (`CONTEXT_RESOLVES`)
- **Document A** (*Delhivery FY24 Annual Report*): Revenue reported as `₹81,415.38 million` (1-indexed PDF page 22, corresponding to printed report page 43; also summarized on PDF page 4 as `₹81,415Mn`).
- **Document B** (*Delhivery Q4 FY24 Earnings Presentation*): Revenue reported as `₹8,142 Cr` (1-indexed PDF page 6, corresponding to slide 5; also disclosed in summary tables on PDF pages 9 and 10).
- **Page Numbering Note**: FACTLINE consistently references and indexes documents by **1-indexed PDF page numbers** (`page_number`), which are distinct from internal printed report pagination or presentation slide numbers.
- **Outcome**: Normalized values resolve to `81,415,380,000 INR` and `81,420,000,000 INR`. The difference (4.62M INR) is within the display resolution of the crore-scale figure ($\pm 5.0\text{M INR}$). Deterministically identified as **`CONTEXT_RESOLVES`**.

### Case 2 — Vintage Evolution (`EVOLVES_FROM`)
- **Document A** (*Economic Survey 2024-25*, PDF p. 4): FY25 Real GDP Growth estimated at `6.4%` (*First Advance Estimate*).
- **Document B** (*RBI Annual Report 2024-25*, PDF p. 6): FY25 Real GDP Growth estimated at `6.5%` (*Second Advance Estimate*).
- **Outcome**: Comparability Gate identifies sequential vintage progression. Identified as **`EVOLVES_FROM`** rather than a contradiction.

### Case 3 — Contextual Boundary Distinction (`UNRESOLVED` / Non-Comparable)
- **Document A** (*Delhivery Annual Report*, PDF p. 2): `>33,200 active customers` (*FY24 Full Year*).
- **Document B** (*Delhivery Earnings Presentation*, PDF p. 4): `33,278 active customers` (*Q4 FY24 Quarter*).
- **Outcome**: Comparability Gate catches `TIME_MISMATCH` and `SCOPE_MISMATCH`. Identified as **`UNRESOLVED`** without generating a false contradiction.

### Case 4 — Controlled Quota & Partial Analysis
- Multi-page document extracted against bounded budget limits.
- **Outcome**: Status persists as **`PARTIAL_QUOTA`**, retaining 100% of verified facts extracted before the budget ceiling with zero data corruption.

### Case 5 — Held-out PDF Generalization Test
- **Document**: `03-imf-india-2025-article-iv-excerpt.pdf` (95-page official report).
- **Evaluation Note**: This document was previously used for evaluation, but the final run was performed through the frozen production pipeline without document-specific hardcoding, rules, or facts.
- **Outcome**: Bounded budget processed 30 eligible pages, yielding **77 verified facts** with **100% evidence grounding** (0 grounding rejections) and 55 candidate relationship pairs evaluated. Status recorded as `PARTIAL_QUOTA`.

---

## 13. Alternative Provider Evaluation (Groq Experiment Note)

As part of Phase 26, Groq (`openai/gpt-oss-120b`) was evaluated in an isolated benchmark ([backend/groq_experiment/](backend/groq_experiment/)) against Gemini (`gemini-3.6-flash`) across a fixed, content-diverse 25-page corpus receiving semantically identical inputs:
- **Gemini**: 5/5 batch requests successful (100%), extracting 48 verified facts with 100% evidence grounding.
- **Groq**: 1/5 batch requests successful (20%). Groq encountered server-side schema validation failures (`json_validate_failed`) on multi-page prompts and breached the 8,000 TPM limit on larger batches.
- **Decision**: Google Gemini (`gemini-3.6-flash`) remains the sole locked production extraction provider. No Groq dependencies or fallbacks exist in production runtime.

---

## 14. Known Limitations & Future Improvements

1. **Complex Embedded PDF Tables**: Highly irregular, borderless multi-column layouts in scanned PDFs can present extraction challenges; OCR preprocessing (e.g. Docling/Tesseract) is a planned enhancement.
2. **Multi-Step Deductive Arithmetic**: FACTLINE validates explicit display rounding and direct unit scaling; multi-step financial formulas (e.g. deriving Free Cash Flow from Operating Cash Flow minus Capex) require explicit user calculation directives.
3. **Cross-Document Entity Aliasing**: Currently utilizes curated corporate/macroeconomic alias dictionaries and fuzzy matching; advanced knowledge-graph linking will be explored in future phases.
