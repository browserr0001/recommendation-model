from unittest.mock import MagicMock, patch

# Patch BEFORE importing app.main to avoid loading missing model
with patch("app.model_loader.get_model", return_value=MagicMock(
    predict=lambda user_id, top_n=20: ([1, 2, 3, 4], 0.002, {"meta": "test"})
)):
    from app.main import app

import pytest

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_api_running(client):
    res = client.get("/api")
    assert res.status_code == 200
    assert b"Movie Recommendation API" in res.data

def test_recommend_endpoint(client):
    res = client.get("/recommend/1234")
    assert res.status_code == 200
    body = res.data.decode().strip()
    assert body
    assert all(x.strip().isdigit() for x in body.split(",")), f"Response was: {body}"

@patch("app.main.model", MagicMock(
    predict=lambda user_id, top_n=20: ([], 0.001, {"meta": "empty"})
))
def test_recommend_empty_list(client):
    res = client.get("/recommend/9999")
    assert res.status_code == 200
    body = res.data.decode().strip()
    assert body == "" or body == ","