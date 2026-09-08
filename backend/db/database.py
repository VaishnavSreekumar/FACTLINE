"""Database initialization, schema management, and atomic persistence for FACTLINE."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.models.document import ParsedDocument
from backend.models.fact import EpistemicStatus, FactRecord, Provenance, TimePeriod
from backend.models.normalization import NormalizationStatus, NormalizedFact, NormalizedValue
from backend.models.relationship import RelationshipResult, RelationshipType

DATABASE_PATH = os.getenv("DATABASE_PATH", "factline.db")


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Returns a connection to the SQLite database with row factory enabled."""
    target_path = db_path or DATABASE_PATH
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initializes the database schema if tables do not exist."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                document_name TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                total_pages INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                entity TEXT NOT NULL,
                metric TEXT NOT NULL,
                value_raw TEXT NOT NULL,
                value_numeric REAL,
                unit TEXT,
                time_period_label TEXT,
                time_period_start TEXT,
                time_period_end TEXT,
                scope TEXT,
                geography TEXT,
                epistemic_status TEXT NOT NULL,
                data_vintage TEXT,
                extraction_confidence REAL NOT NULL,
                provenance_page INTEGER NOT NULL,
                provenance_text TEXT NOT NULL,
                provenance_date TEXT,
                canonical_entity TEXT,
                canonical_metric TEXT,
                canonical_value REAL,
                canonical_unit TEXT,
                scale TEXT,
                currency TEXT,
                normalization_status TEXT,
                normalization_notes TEXT,
                normalized_period_start TEXT,
                normalized_period_end TEXT,
                normalization_warnings TEXT,
                raw_json TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id TEXT PRIMARY KEY,
                fact_a_id TEXT NOT NULL,
                fact_b_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                reason_codes TEXT NOT NULL,
                explanation TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_a TEXT,
                evidence_b TEXT,
                contextual_factors TEXT,
                FOREIGN KEY (fact_a_id) REFERENCES facts (fact_id) ON DELETE CASCADE,
                FOREIGN KEY (fact_b_id) REFERENCES facts (fact_id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                document_ids TEXT NOT NULL,
                fact_ids TEXT NOT NULL,
                relationship_ids TEXT NOT NULL,
                summary TEXT NOT NULL
            );
            """
        )
    conn.close()


class DatabaseRepository:
    """Encapsulates transactional CRUD operations for documents, facts, relationships, and analyses."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DATABASE_PATH
        init_db(self.db_path)

    def save_analysis(
        self,
        analysis_id: str,
        documents: List[ParsedDocument],
        normalized_facts: List[NormalizedFact],
        relationships: List[RelationshipResult],
        candidate_pair_count: int,
        extraction_status: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Atomically persists documents, facts, relationships, and analysis record."""
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = get_db_connection(self.db_path)

        try:
            with conn:
                # 1. Save Documents (Idempotent: INSERT OR REPLACE)
                for doc in documents:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO documents (
                            document_id, document_name, content_hash, total_pages, created_at
                        ) VALUES (?, ?, ?, ?, ?);
                        """,
                        (
                            doc.document_id,
                            doc.document_name,
                            doc.document_id,  # content hash is preserved in document_id
                            doc.total_pages,
                            now_iso,
                        ),
                    )

                # 2. Save Normalized Facts (Idempotent: INSERT OR REPLACE)
                for nf in normalized_facts:
                    fact = nf.fact
                    prov = fact.provenance
                    nv = nf.normalized_value
                    tp = fact.time_period
                    ntp = nf.normalized_time_period

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO facts (
                            fact_id, document_id, entity, metric, value_raw, value_numeric, unit,
                            time_period_label, time_period_start, time_period_end,
                            scope, geography, epistemic_status, data_vintage, extraction_confidence,
                            provenance_page, provenance_text, provenance_date,
                            canonical_entity, canonical_metric, canonical_value, canonical_unit,
                            scale, currency, normalization_status, normalization_notes,
                            normalized_period_start, normalized_period_end, normalization_warnings,
                            raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            fact.fact_id,
                            prov.document_id,
                            fact.entity,
                            fact.metric,
                            fact.value_raw,
                            fact.value_numeric,
                            fact.unit,
                            tp.label if tp else None,
                            tp.start_date if tp else None,
                            tp.end_date if tp else None,
                            fact.scope,
                            fact.geography,
                            fact.epistemic_status.value if isinstance(fact.epistemic_status, EpistemicStatus) else str(fact.epistemic_status),
                            fact.data_vintage,
                            fact.extraction_confidence,
                            prov.page_number,
                            prov.supporting_text,
                            prov.document_date,
                            nf.canonical_entity,
                            nf.canonical_metric,
                            nv.numeric_value if nv else None,
                            nv.canonical_unit if nv else None,
                            nv.scale if nv else None,
                            nv.currency if nv else None,
                            nv.normalization_status.value if nv and isinstance(nv.normalization_status, NormalizationStatus) else str(nv.normalization_status if nv else "unresolved"),
                            nv.normalization_notes if nv else None,
                            ntp.start_date if ntp else None,
                            ntp.end_date if ntp else None,
                            json.dumps(nf.normalization_warnings),
                            nf.model_dump_json(),
                        ),
                    )

                # 3. Save Relationships (Idempotent: INSERT OR REPLACE)
                for rel in relationships:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO relationships (
                            relationship_id, fact_a_id, fact_b_id, relationship_type,
                            reason_codes, explanation, confidence,
                            evidence_a, evidence_b, contextual_factors
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            rel.relationship_id,
                            rel.fact_a_id,
                            rel.fact_b_id,
                            rel.relationship_type.value if isinstance(rel.relationship_type, RelationshipType) else str(rel.relationship_type),
                            json.dumps(rel.reason_codes),
                            rel.explanation,
                            rel.confidence,
                            rel.evidence_a.model_dump_json() if rel.evidence_a else None,
                            rel.evidence_b.model_dump_json() if rel.evidence_b else None,
                            json.dumps(rel.contextual_factors),
                        ),
                    )

                # 4. Save Analysis Record
                doc_ids = [doc.document_id for doc in documents]
                fact_ids = [nf.fact.fact_id for nf in normalized_facts]
                rel_ids = [rel.relationship_id for rel in relationships]
                summary = {
                    "documents_processed": len(documents),
                    "facts_extracted": len(normalized_facts),
                    "candidate_pairs": candidate_pair_count,
                    "relationships_created": len(relationships),
                }
                if extraction_status is not None:
                    summary["extraction_status"] = extraction_status

                conn.execute(
                    """
                    INSERT INTO analyses (
                        analysis_id, created_at, document_ids, fact_ids, relationship_ids, summary
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        analysis_id,
                        now_iso,
                        json.dumps(doc_ids),
                        json.dumps(fact_ids),
                        json.dumps(rel_ids),
                        json.dumps(summary),
                    ),
                )

            return {
                "analysis_id": analysis_id,
                "created_at": now_iso,
                "summary": summary,
            }
        finally:
            conn.close()

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves complete structured analysis data including documents, facts, and relationships."""
        conn = get_db_connection(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM analyses WHERE analysis_id = ?;", (analysis_id,))
            analysis_row = cur.fetchone()
            if not analysis_row:
                return None

            doc_ids = json.loads(analysis_row["document_ids"])
            fact_ids = json.loads(analysis_row["fact_ids"])
            rel_ids = json.loads(analysis_row["relationship_ids"])
            summary = json.loads(analysis_row["summary"])

            # Retrieve Documents
            documents = []
            if doc_ids:
                placeholders = ",".join("?" for _ in doc_ids)
                cur.execute(f"SELECT * FROM documents WHERE document_id IN ({placeholders});", doc_ids)
                for row in cur.fetchall():
                    documents.append({
                        "document_id": row["document_id"],
                        "document_name": row["document_name"],
                        "content_hash": row["content_hash"],
                        "total_pages": row["total_pages"],
                        "created_at": row["created_at"],
                    })

            # Retrieve Facts
            facts = []
            if fact_ids:
                placeholders = ",".join("?" for _ in fact_ids)
                cur.execute(f"SELECT * FROM facts WHERE fact_id IN ({placeholders});", fact_ids)
                for row in cur.fetchall():
                    raw_json = row["raw_json"]
                    if raw_json:
                        nf = NormalizedFact.model_validate_json(raw_json)
                        facts.append(nf.model_dump())
                    else:
                        facts.append(dict(row))

            # Retrieve Relationships
            relationships = []
            if rel_ids:
                placeholders = ",".join("?" for _ in rel_ids)
                cur.execute(f"SELECT * FROM relationships WHERE relationship_id IN ({placeholders});", rel_ids)
                for row in cur.fetchall():
                    ev_a = json.loads(row["evidence_a"]) if row["evidence_a"] else None
                    ev_b = json.loads(row["evidence_b"]) if row["evidence_b"] else None
                    relationships.append({
                        "relationship_id": row["relationship_id"],
                        "fact_a_id": row["fact_a_id"],
                        "fact_b_id": row["fact_b_id"],
                        "relationship_type": row["relationship_type"],
                        "reason_codes": json.loads(row["reason_codes"]),
                        "explanation": row["explanation"],
                        "confidence": row["confidence"],
                        "evidence_a": ev_a,
                        "evidence_b": ev_b,
                        "contextual_factors": json.loads(row["contextual_factors"]),
                    })

            return {
                "analysis_id": analysis_id,
                "created_at": analysis_row["created_at"],
                "summary": summary,
                "documents": documents,
                "facts": facts,
                "relationships": relationships,
            }
        finally:
            conn.close()
