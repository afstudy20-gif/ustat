import pytest
import pandas as pd
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import app
from services import store


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "simulation: marks tests as simulation-based (deselect with '-m \"not simulation\"')"
    )


@pytest.fixture
def client():
    return TestClient(app)


def make_session(df: pd.DataFrame, session_id: str = "test_session") -> str:
    store.save(session_id, df)
    return session_id
