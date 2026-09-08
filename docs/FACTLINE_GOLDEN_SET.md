# FACTLINE Golden Evaluation Benchmark Set

This document serves as the ground truth benchmark for evaluating FACTLINE's extraction, comparability gate, and relationship reasoning pipeline across verified document corpora.

> **Note**: Benchmark entries must be manually verified against source PDFs before inclusion. No synthetic or unverified cases should be added.

---

## 1. Category I: Corroboration
*Pairs of facts across documents describing the same underlying metric that confirm each other.*

| Case ID | Entity | Metric | Time Period | Doc A (Page) | Value A | Doc B (Page) | Value B | Expected Relationship | Notes / Ground Truth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *TBD* | | | | | | | | `CORROBORATES` | |

---

## 2. Category II: Genuine or Likely Contradiction
*Comparable claims that materially disagree without a valid contextual explanation.*

| Case ID | Entity | Metric | Time Period | Doc A (Page) | Value A | Doc B (Page) | Value B | Expected Relationship | Notes / Conflict Details |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *TBD* | | | | | | | | `CONTRADICTS` | |

---

## 3. Category III: Apparent Contradiction Explained Through Context
*Claims that appear divergent on the surface but resolve when accounting for units, scale, rounding, time definition, scope, or revisions.*

| Case ID | Entity | Metric | Doc A Context (Value / Unit / Period) | Doc B Context (Value / Unit / Period) | Expected Relationship | Resolution Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *TBD* | | | | | `CONTEXT_RESOLVES` / `EVOLVES_FROM` | |

---

## 4. Category IV: Extraction or Reasoning Failures / Edge Cases
*Challenging real-world cases illustrating ambiguity, missing context, or boundary condition handling.*

| Case ID | Document / Source Text | Challenge Description | Expected System Behavior | Actual Failure / Risk Mode |
| :--- | :--- | :--- | :--- | :--- |
| *TBD* | | | `INSUFFICIENT_CONTEXT` / Explicit Uncertainty | Avoiding hallucinated assumptions |
