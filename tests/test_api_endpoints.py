import sys
from unittest.mock import MagicMock, patch

# Patch BEFORE importing app.main to avoid loading missing model
with patch("app.model_loader.get_model", return_value=MagicMock(get_recommendations=lambda user_id, top_n: ([1, 2, 3, 4], 0.002, {"test": "metadata"}))):
    from app.main import app

import pytest

@pytest.fixture
def client():
    """Provide a Flask test client."""
    with app.test_client() as client:
        yield client


def test_api_running(client):
    """Basic test to ensure the API route is alive."""
    res = client.get("/api")
    assert res.status_code == 200
    assert b"Movie Recommendation API" in res.data


def test_recommend_endpoint(client):
    """Test /recommend/<userid> with a mocked model."""
    res = client.get("/recommend/1234")
    assert res.status_code == 200
    body = res.data.decode().strip()
    assert body
    assert all(x.strip().isdigit() for x in body.split(",")), f"Response was: {body}"


@patch("app.main.model", MagicMock(get_recommendations=lambda user_id, top_n: ([], 0.001, {})))
def test_recommend_empty_list(client):
    """Handle edge case where model returns no recommendations."""
    res = client.get("/recommend/9999")
    assert res.status_code == 200
    body = res.data.decode().strip()
    assert body == "" or body == ","