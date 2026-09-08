"""Pytest fixtures and configuration."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)
