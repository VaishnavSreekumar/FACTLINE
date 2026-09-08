"""Health endpoint and FactRecord schema tests."""

from fastapi.testclient import TestClient
from backend.models.fact import FactRecord, Provenance, TimePeriod, EpistemicStatus


def test_health_endpoint(client: TestClient):
    """Verify GET /health returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_fact_record_schema():
    """Verify FactRecord model validation."""
    fact = FactRecord(
        fact_id="fact-001",
        entity="Delhivery",
        metric="Revenue from Operations",
        value_raw="₹81,415.38 million",
        value_numeric=81415.38,
        unit="INR Million",
        time_period=TimePeriod(label="FY2023-24", start_date="2023-04-01", end_date="2024-03-31"),
        scope="Consolidated",
        geography="India",
        epistemic_status=EpistemicStatus.AUDITED,
        provenance=Provenance(
            document_id="delhivery_fy24_ar.pdf",
            document_date="2024-05-17",
            page_number=142,
            supporting_text="Revenue from operations for the financial year ended March 31, 2024 stood at ₹81,415.38 million."
        ),
        extraction_confidence=0.98
    )
    assert fact.fact_id == "fact-001"
    assert fact.value_numeric == 81415.38
    assert fact.epistemic_status == EpistemicStatus.AUDITED
