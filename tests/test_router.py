import os
from unittest.mock import patch, MagicMock
import pytest

# Set BACKENDS before importing router so it picks these up
os.environ["BACKENDS"] = "http://backend-a:5001,http://backend-b:5001"

import router


@pytest.fixture
def router_client():
    """Flask test client for the router service."""
    with router.app.test_client() as client:
        yield client


def _make_fake_response(body: str, status: int = 200, content_type: str = "text/plain"):
    """Helper to build a fake requests.Response-like object."""
    resp = MagicMock()
    resp.status_code = status
    resp.content = body.encode("utf-8")
    resp.headers = {"Content-Type": content_type}
    return resp


def test_health_endpoint_reports_backends(router_client):
    """The /health endpoint should report the configured backends."""
    res = router_client.get("/health")
    assert res.status_code == 200

    data = res.get_json()
    assert data["status"] == "ok"
    # Should match what router.BACKENDS was initialized with
    assert data["backends"] == router.BACKENDS
    assert data["backends"] == [
        "http://backend-a:5001",
        "http://backend-b:5001",
    ]


@patch("router.requests.get")
def test_recommend_round_robin(mock_get, router_client):
    """
    /recommend/<userid> should forward to backends in round-robin order.
    We don't hit real backends; we patch requests.get to inspect calls.
    """
    # Make the fake backend return something simple
    def fake_get(url, timeout):
        return _make_fake_response(f"response from {url}")

    mock_get.side_effect = fake_get

    # Fire three requests through the router
    res1 = router_client.get("/recommend/1")
    res2 = router_client.get("/recommend/2")
    res3 = router_client.get("/recommend/3")

    # Extract URLs the router called
    called_urls = [args[0] for args, kwargs in mock_get.call_args_list]

    # With two backends and round robin, we expect A, B, A
    assert called_urls == [
        "http://backend-a:5001/recommend/1",
        "http://backend-b:5001/recommend/2",
        "http://backend-a:5001/recommend/3",
    ]

    # And the router should return the backend responses
    assert "response from http://backend-a:5001/recommend/1" in res1.data.decode()
    assert "response from http://backend-b:5001/recommend/2" in res2.data.decode()
    assert "response from http://backend-a:5001/recommend/3" in res3.data.decode()


@patch("router.requests.get")
def test_api_endpoint_proxies_to_backend(mock_get, router_client):
    """
    /api should proxy to one of the backends and return its response.
    """
    mock_get.return_value = _make_fake_response("Movie Recommendation API is running.", status=200)

    res = router_client.get("/api")

    # Router should call GET <backend-url>/api with some backend from BACKENDS
    assert mock_get.call_count == 1
    called_url = mock_get.call_args[0][0]
    assert called_url in [f"{b}/api" for b in router.BACKENDS]

    assert res.status_code == 200
    assert b"Movie Recommendation API" in res.data
