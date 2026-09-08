# FACTLINE Phase 10: Local Extraction Benchmark Framework

## Experimental Architecture Overview

This directory contains the isolated experimental benchmark suite for evaluating whether a **fully local semantic information extraction pipeline** can replace the production Gemini extractor while preserving schema fidelity, strict evidence grounding, and numeric accuracy.

```text
CURRENT PRODUCTION:
PDF → PyMuPDF → Gemini 2.5 Flash → FactRecord → Normalization → Matching → Comparability → Persistence

EXPERIMENTAL LOCAL:
PDF → Docling/LocalParser → GLiNER/Local IE → FactRecord → Normalization → Matching → Comparability → Persistence
```

## Core Evaluation Dimensions

1. **Fact Precision & Recall** vs 18 gold-labeled real PDF benchmark cases.
2. **Strict Evidence Grounding Rate** (Must achieve 100% verifiable source-page text).
3. **Numeric and Unit Accuracy** (Preserving qualifiers like `>`, numeric scales `Cr/Mn/Bn`, and semantic units).
4. **Temporal Context & Scope Accuracy** (Fiscal years, calendar years, national vs corporate scopes).
5. **Latency & Computational Resource Footprint** (CPU RAM, throughput per page, first-load time).
6. **Generalization** (Unseen document evaluation).
