# FACTLINE — System Architecture Specification

## 1. Problem Statement

Modern financial, regulatory, and macroeconomic analysis requires synthesizing facts distributed across heterogeneous documents—such as draft prospectuses, annual reports, earnings calls, and official statistical releases.

In practice:
- Metrics change over time (e.g. FY23 vs FY24 vs Q4 FY24).
- Reporting standards and units differ (e.g. ₹ million vs ₹ crore vs USD).
- Revisions and data vintages alter figures (e.g. First Advance Estimate vs Revised Estimate).
- Scopes vary (e.g. Standalone vs Consolidated, Express Parcel vs Total Logistics).
- Geographies and currency bases introduce nuance.

Naïve RAG and LLM chatbots often report false contradictions or falsely corroborate mismatched metrics because they perform raw string or vector comparisons without validating contextual equivalence.

---

## 2. Core Design Principle

> **"Never compare two values before establishing that the claims are comparable."**

Comparability is not raw string equality (`fact_a.metric == fact_b.metric`). Comparability is a semantic and contextual gate that evaluates whether two claims refer to the same underlying reality across all essential dimensions before comparing their numerical or qualitative values.

---

## 3. High-Level Architecture & Pipeline

```text
                    PDF Documents
                         │
                         ▼
                Page-Aware PDF Parser (PyMuPDF)
                         │
                         ▼
                 Fact Extraction (LLM + Structured Schema)
                         │
                         ▼
                Grounded Fact Records
                         │ (fact_id, entity, metric, value, time_period,
                         │  provenance, page, supporting_text)
                         │
                         ▼
                  Normalization Layer
                         │ (Units, Scales, Dates, Entity aliases)
                         │
                         ▼
                  Candidate Matcher
                         │
                         ▼
                COMPARABILITY GATE
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
         Comparable   Insufficient   Non-comparable
             │          Context          │
             ▼             │              ▼
       Relationship        └──────►   UNRESOLVED
          Engine
             │
      ┌──────┼─────────┬───────────────┐
      ▼      ▼         ▼               ▼
 CORROBORATES  CONTRADICTS  CONTEXT_RESOLVES
      │
      ▼
 EVOLVES_FROM / SUPERSEDES
             │
             ▼
       Explanation Layer (Grounded provenance & contextual resolution)
             │
             ▼
       Evidence-First UI (Side-by-side claim & source evidence inspection)
```

---

## 3.1. Evidence Layer & Page-Aware Parser (Phase 1)

The foundation of FACTLINE is page-accurate text extraction and immutable provenance tracing:

```text
PDF
 │
 ▼
PyMuPDF (pymupdf)
 │
 ▼
PageText (1-indexed page_number, raw text, char_count, has_text)
 │
 ▼
Evidence / Provenance (document_id, page_number, supporting_text)
```

### Key Invariants:
1. **1-Indexed Page Numbering**: All page references are strictly 1-indexed (`page_number = index + 1`) to remain human-readable and match document physical pages.
2. **Deterministic Source Truth**: Raw text is preserved faithfully without paraphrasing, summarizing, or altering numbers and symbols.
3. **Provenance Traceability**: Every extracted fact links directly to a specific page number and the exact supporting text snippet on that page.
4. **Current Limitation**: The current evidence layer relies on embedded/extractable PDF text and does not yet perform OCR for image-only pages.

---

## 3.2. Structured Fact Extraction Pipeline (Phase 2)

Phase 2 converts extracted page text into structured `FactRecord` candidates grounded in source evidence:

```text
ParsedDocument
      │
      ▼
PageText (1-indexed page_number, raw text)
      │
      ▼
FactExtractor (Gemini structured JSON schema)
      │
      ▼
Raw Candidate Claims (entity, metric, value_raw, time_period, epistemic_status, quote)
      │
      ▼
Deterministic Verification Layer
      ├── Pydantic Schema Validation (FactRecord)
      ├── Epistemic Status Validation
      ├── Confidence Clamping [0.0, 1.0]
      ├── Provenance Assembly (document_id, page_number, supporting_text)
      ├── Evidence Verification (EvidenceVerifier against PageText) ──► Reject if ungrounded
      └── Deterministic Fact ID Generation (SHA-256 digest)
      │
      ▼
Validated Grounded FactRecords
```

### LLM vs Deterministic Application Boundary:
- **LLM Responsibility**: Semantic interpretation of complex natural language in tables and prose, identifying candidate metrics, entity subjects, temporal labels, and locating supporting verbatim text spans.
- **Deterministic Application Responsibility**: Strict schema validation, provenance construction, checking that `supporting_text` exists on the physical page text via `EvidenceVerifier`, computing reproducible `fact_id`s, and rejecting ungrounded or malformed candidates.

### Critical Invariants:
1. **Evidence Verification**: The LLM is NEVER the source of truth for evidence. If candidate `supporting_text` does not exist on the source page text, the candidate fact is strictly **rejected**.
2. **Epistemic Status Preservation**: Distinguishes between `reported`, `estimated`, `projected`, `target`, and `audited`. Estimated/projected/target claims are never silently converted to `reported`.
3. **Deterministic Fact IDs**: Fact IDs are calculated from `(document_id, page_number, entity, metric, value_raw, time_period.label)` via SHA-256 hash.
4. **Context Integrity**: Unmentioned contextual attributes (`scope`, `geography`, `data_vintage`) remain `None` rather than being hallucinated.
5. **No Cross-Document Comparison**: Phase 2 strictly operates page-by-page. It does **not** perform cross-document matching, comparability gating, or contradiction detection.

### LLM Configuration & Failure Modes:
- **Provider**: Google Gemini REST API via `GEMINI_API_KEY` (or `LLM_API_KEY`) and `GEMINI_MODEL` (default: `gemini-1.5-flash`).
- **Empty Pages**: Pages with no extractable text are skipped without invoking the LLM.
- **API / Parsing Errors**: Surface controlled `ExtractionError` rather than silently fabricating fallback facts.

---

## 4. Canonical Fact & Provenance Schema

### Epistemic Status
```python
class EpistemicStatus(str, Enum):
    REPORTED = "reported"      # Audited/historic official statements
    ESTIMATED = "estimated"    # Interim / preliminary estimations
    PROJECTED = "projected"    # Forward-looking forecasts
    TARGET = "target"          # Management goals / budgets
    AUDITED = "audited"        # Explicitly audited final figures
```

### Provenance Model
```python
class Provenance(BaseModel):
    document_id: str
    document_date: Optional[str] = None
    page_number: int
    supporting_text: str
```

### Temporal Distinction
Temporal information strictly differentiates between:
- `time_period`: The interval or point in time the metric actually describes (e.g., `FY2023-24`, `Q4 FY24`).
- `document_date`: The publication or release date of the document (e.g., `2024-05-17`).
- `data_vintage`: The release edition, estimate revision, or publication vintage (e.g., `First Advance Estimate`, `Provisional Actuals`).

```python
class TimePeriod(BaseModel):
    label: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
```

### Canonical Fact Record
```python
class FactRecord(BaseModel):
    fact_id: str

    entity: str
    metric: str

    value_raw: str
    value_numeric: Optional[float] = None
    unit: Optional[str] = None

    time_period: TimePeriod

    scope: Optional[str] = None
    geography: Optional[str] = None

    epistemic_status: EpistemicStatus = EpistemicStatus.REPORTED
    data_vintage: Optional[str] = None

    provenance: Provenance
    extraction_confidence: float = 0.0
```

---

## 5. Normalization Layer (Phase 3)

The normalization layer transforms raw extracted `FactRecord` claims into canonical `NormalizedFact` representations deterministically without mutating or destroying source truth.

```text
FactRecord (Immutable Source Truth)
   │
   ▼
FactNormalizer (Deterministic Pipeline)
   ├── UnitNormalizer (Decimal scaling: crore, lakh, million, billion; explicit currencies)
   ├── DateNormalizer (Bounded ISO intervals; strict fiscal convention validation)
   ├── EntityNormalizer (Presentation-friendly casing; generic legal suffix removal)
   └── MetricNormalizer (Whitespace & case standardization without semantic merging)
   │
   ▼
NormalizedFact (Derived Canonical Representation)
   ├── fact: FactRecord (Original unmodified fact with provenance)
   ├── normalized_value: NormalizedValue (Canonical numeric magnitude, unit, currency)
   ├── canonical_entity: str (e.g. "Delhivery Limited" -> "Delhivery")
   ├── canonical_metric: str (e.g. "Revenue from Operations" -> "revenue from operations")
   ├── normalized_time_period: Optional[TimePeriod] (ISO start_date and end_date)
   └── normalization_warnings: List[str]
```

### Key Invariants & Rules:
1. **Source Preservation**: The original `FactRecord` (`value_raw`, `provenance`, `entity`, `metric`, `time_period.label`) is NEVER altered or overwritten.
2. **Exact Decimal Arithmetic**: Monetary and scale multiplications use exact integer/decimal math (`1 crore = 10,000,000`, `1 million = 1,000,000`) before converting to float at the model boundary to eliminate floating-point precision drift.
   - Example: `₹8,142 Cr` → `81,420,000,000 INR`
   - Example: `₹81,415.38 million` → `81,415,380,000 INR`
3. **Percentage Invariant**: Percentage figures retain percentage point representation (`6.4%` → `numeric_value=6.4, canonical_unit="percent"`). They are never converted to `0.064` to prevent confusion between percentage points and decimal ratios.
4. **Currency Safety**: Explicit currencies (`₹`, `INR`, `Rs.`, `US$`, `USD`, `€`, `EUR`, `£`, `GBP`) are recognized. Bare `$` symbols without explicit country codes are left unassigned (`currency=None`, status `PARTIAL`) to prevent false cross-currency comparisons.
5. **Conservative Fiscal Year Parsing**: Bare `FY24` or `Q4 FY24` labels are NOT blindly assumed to follow an April–March fiscal year unless the fiscal convention is explicitly established. Unresolved fiscal labels return `normalized_time_period=None` with diagnostic warnings.
6. **Conservative Entity Canonicalization**: Generic legal corporate suffixes (`Limited`, `Ltd.`, `Pvt. Ltd.`, `Inc.`, `Corp.`) are stripped deterministically while preserving presentation-friendly casing (`"Delhivery Limited"` → `"Delhivery"`).
7. **No Semantic Equivalence**: Normalization canonicalizes syntax and scales; it does **not** assert that two metrics or periods mean the same thing (e.g., `Revenue` vs. `Revenue from operations` remain separate).

> [!IMPORTANT]
> **Normalization creates canonical representations; it does not establish semantic equivalence or relationships between facts.**

---

## 6. Candidate Matching & Comparability Gate (Phase 4)

The central reasoning boundary of FACTLINE enforces:
> **"FACTLINE never compares numerical values until the claims have passed the Comparability Gate."**

```text
NormalizedFact A  +  NormalizedFact B
              │
              ▼
    CandidateMatcher (Deterministic Signals)
    ├── Strong Entity Alignment (canonical_entity match)
    └── Metric Token Signals (exact match or key token overlap)
              │
              ▼
    CandidatePair (plausible fact pair)
              │
              ▼
    ComparabilityGate (Dimension Evaluation)
    ├── Entity Dimension: Exact canonical entity equality
    ├── Metric Dimension: Canonical metric equivalence (no semantic conflation)
    ├── Unit Dimension: Canonical unit dimensional compatibility (INR vs INR, % vs %)
    ├── Time Period Dimension: Exact bounded ISO interval match (FY24 != Q4 FY24)
    ├── Scope Dimension: Explicit scope compatibility (missing is not same)
    ├── Geography Dimension: Explicit geography compatibility (missing is not same)
    ├── Epistemic Status: Diagnostic recording (reported vs estimated preserved)
    └── Data Vintage: Diagnostic recording (advance estimates preserved)
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
COMPARABLE  INSUFFICIENT  NON_COMPARABLE
            CONTEXT
```

### Decision Semantics:
- **`COMPARABLE`**: All essential dimensions (entity, metric, unit, bounded time interval) are compatible and sufficient context exists. Diagnostic reasons (`EPISTEMIC_STATUS_DIFFERENCE`, `DATA_VINTAGE_DIFFERENCE`) are preserved for subsequent relationship classification.
- **`NON_COMPARABLE`**: A demonstrable incompatibility exists across one or more dimensions (`ENTITY_MISMATCH`, `METRIC_MISMATCH`, `UNIT_MISMATCH`, `TIME_MISMATCH`, `SCOPE_MISMATCH`, `GEOGRAPHY_MISMATCH`).
- **`INSUFFICIENT_CONTEXT`**: Dimensions do not conflict, but missing essential context (`MISSING_TIME_PERIOD`, `MISSING_UNIT`, `MISSING_SCOPE`, `MISSING_GEOGRAPHY`) prevents a defensible comparison.

### Key Rules & Invariants:
1. **Candidate Matching vs. Comparability**: Candidate matching identifies plausible pairs using entity alignment and metric token signals without requiring exact equality as the sole gateway. The Comparability Gate then performs strict dimension evaluation.
2. **Missing is Not Same**: An unspecified dimension (e.g. missing scope or geography on one fact) is never assumed to be equivalent to a specified dimension. It triggers `INSUFFICIENT_CONTEXT`.
3. **Time Granularity Invariant**: Temporal interval containment (e.g. `Q4 FY24` lying inside `FY24`) is NOT temporal equivalence. Comparing an annual total against a quarterly total yields `TIME_MISMATCH` → `NON_COMPARABLE`.
4. **Numeric Independence**: Numerical values are never used to determine candidate status or comparability. Facts are evaluated on their semantic and contextual dimensions regardless of numerical proximity.
5. **No Relationship Inference**: Comparability only establishes whether two claims *can* legitimately be compared; it does **not** evaluate whether they corroborate, contradict, or resolve each other.

---

## 7. Relationship Vocabulary & Engine (Phase 5)

Comparable fact pairs are evaluated by the deterministic relationship engine into one of the following locked categories:

- **`CORROBORATES`**: Two comparable claims describe identical underlying values or states within mathematically derived display precision tolerance (`EXACT_MATCH`).
- **`CONTRADICTS`**: Two comparable claims describe materially conflicting values or states exceeding display precision resolution (`VALUE_CONFLICT`). Epistemic status differences by themselves NEVER produce `CONTRADICTS`.
- **`CONTEXT_RESOLVES`**: Two comparable claims appear different at surface level, but become consistent once unit scales, rounding/precision boundaries, or reporting formats are resolved (`ROUNDING_DIFFERENCE`).
- **`EVOLVES_FROM`**: A verified revision, subsequent advance estimate, or updated data vintage alters an earlier preliminary estimate (`ESTIMATE_REVISED`, `DATA_VINTAGE_EVOLUTION`). Requires explicit revision semantics, not merely different document dates.
- **`SUPERSEDES`**: Explicit restatement, audited replacement, or formal supersession of an earlier claim (`EXPLICIT_RESTATEMENT`, `AUDITED_REVISION`).
- **`UNRESOLVED`**: Insufficient context, non-comparable claims, or unverified relationships. Abstention is strictly enforced over guessing (`NON_COMPARABLE_CLAIMS`, `INSUFFICIENT_CONTEXT_FOR_RELATIONSHIP`, `EPISTEMIC_STATUS_INCOMPATIBLE`, `UNRESOLVED_VINTAGE_RELATIONSHIP`).

### Mathematical Display Precision Derivation for Rounding:
Instead of arbitrary epsilon thresholds (e.g. `0.01` or `1%`), FACTLINE determines precision dynamically:
1. Parse decimal places from `value_raw` (e.g. `8142` -> 0 decimals; `81415.38` -> 2 decimals).
2. Multiply by canonical unit scale (e.g., `crore` = $10^7$, `million` = $10^6$) to derive resolution $R_a = 10^{7 - 0} = 10^7$ and $R_b = 10^{6 - 2} = 10^4$.
3. The display resolution tolerance is $\Delta_{\text{max}} = \frac{1}{2} \max(R_a, R_b)$.
4. If $|\text{canonical\_val}_a - \text{canonical\_val}_b| \le \Delta_{\text{max}}$, the difference is mathematically auditable rounding (`CONTEXT_RESOLVES` + `ROUNDING_DIFFERENCE`).

### Invariant Decision Hierarchy:
```text
1. Comparability Gate Check: If gate != COMPARABLE -> UNRESOLVED (preserve gate reasons; zero numerical evaluation)
2. Epistemic Status Check: If TARGET / PROJECTED vs REPORTED / AUDITED -> UNRESOLVED (EPISTEMIC_STATUS_INCOMPATIBLE)
3. Explicit Supersession Check: If explicit restatement/supersession detected -> SUPERSEDES (SUPERSEDES_PRIOR_ESTIMATE)
4. Data Vintage / Revision Check: If sequential revision context established -> EVOLVES_FROM (DATA_VINTAGE_EVOLUTION)
5. Numerical Evaluation:
   ├── Exact equality -> CORROBORATES (EXACT_MATCH)
   ├── Within implied display precision -> CONTEXT_RESOLVES (ROUNDING_DIFFERENCE)
   └── Exceeds display precision -> CONTRADICTS (VALUE_CONFLICT)
```

---

## 7.1. End-to-End Analysis Orchestration & Persistence (Phase 6)

The Analysis Orchestrator integrates Phases 1–5 into a unified, synchronous, deterministic workflow:

```text
POST /analysis (PDF Files)
      │
      ▼
AnalysisService.analyze_documents()
      │
      ├── Step 1: Parse (PDFParser -> PageText, 1-indexed, content hash)
      ├── Step 2: Extract (FactExtractor -> Page-by-page FactRecord extraction)
      ├── Step 3: Normalize (FactNormalizer -> Derived NormalizedFact representations)
      ├── Step 4: Candidate Match (CandidateMatcher -> Deterministic entity/metric signals)
      ├── Step 5: Comparability Gate (ComparabilityGate -> COMPARABLE / NON_COMPARABLE / INSUFFICIENT_CONTEXT)
      ├── Step 6: Relationship Engine (RelationshipEngine -> CORROBORATES, CONTRADICTS, CONTEXT_RESOLVES, etc.)
      └── Step 7: Atomic SQLite Persistence (DatabaseRepository -> documents, facts, relationships, analyses)
            │
            ▼
POST /analysis Response (analysis_id UUID, summary metrics)
      │
      ▼
GET /analysis/{analysis_id} (Full inspection of documents, facts with provenance, and relationships)
```

### Key Orchestration & Persistence Invariants:
1. **Execution Identity vs Entity Identity**:
   - `analysis_id`: Generated UUID representing an analysis execution run.
   - `document_id`, `fact_id`, `relationship_id`: Strictly deterministic. Analyzing the same PDF inputs maintains identical entity IDs across runs without duplicate row proliferation.
2. **Numeric Isolation**: Numerical values are never used to generate candidates or infer comparability ("Numbers never create candidates").
3. **Loss-Free Persistence**: Full `NormalizedFact` representations, warnings, canonical values, and raw source provenance are stored in SQLite and completely recoverable.
4. **Failure Safety & Rollback**: Extraction errors and persistence exceptions abort the transaction, preventing corrupted or partial analyses.

---

## 8. Division of Responsibilities: Deterministic vs LLM

To guarantee correctness and auditability:

| Responsibility Area | Handled By | Rationale |
| :--- | :--- | :--- |
| **Fact Discovery & Entity Extraction** | LLM (Gemini) | Interprets complex natural language in tables and prose page-by-page |
| **Evidence Verification** | Deterministic Code | Strictly verifies supporting text exists verbatim on the document page |
| **Unit & Scale Normalization** | Deterministic Code | Exact Decimal arithmetic for crore/million/percentage scales |
| **Date & Interval Boundaries** | Deterministic Code | Bounded ISO intervals without assuming April-March blindly |
| **Entity & Metric Canonicalization** | Deterministic Code | Legal suffix stripping, casing, and whitespace normalization |
| **Candidate Fact Matching** | Deterministic Code | Entity alignment and metric token signals; zero numeric heuristics |
| **Comparability Gate Rules** | Deterministic Code | Multi-dimensional evaluation enforcing strict comparison invariants |
| **Relationship Reasoning** | Deterministic Code | Decision hierarchy: Gate -> Epistemic -> Supersedes -> Vintage -> Math |
| **Explanation Generation** | Deterministic Code | Traceable, grounded explanation templates referencing exact provenance |
| **Analysis Orchestration & DB** | Deterministic Code | Synchronous SQLite transactions, idempotency, and REST API |

---

## 9. Evidence-First UI Concept

The eventual user interface will provide an investigative workbench:
- **Document Management**: Upload and inspect indexed PDF documents.
- **Fact Browser**: Search and filter extracted facts with exact document, page number, and quote provenance.
- **Relationship Matrix**: View pairwise relationships (`CORROBORATES`, `CONTRADICTS`, `CONTEXT_RESOLVES`, `EVOLVES_FROM`, `SUPERSEDES`, `UNRESOLVED`).
- **Explanation Pane**: Side-by-side visual comparison showing the claim, source snippet, page number, normalized values, and contextual delta explanation.

---

## 10. Explicit Non-Goals

The following are explicitly **out of scope**:
- Generic conversational chatbot interfaces.
- Unconstrained agent swarms or autonomous multi-agent loops.
- Graph databases (e.g. Neo4j) or heavy distributed vector search infrastructure.
- Complex microservice architectures or container orchestration for prototype stage.
- Hardcoded document-specific parsing rules or synthetic fact overrides.
