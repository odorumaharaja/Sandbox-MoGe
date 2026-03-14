import pytest
import httpx
import os
from pathlib import Path
import time

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
IMAGE_PATH = Path("Test.jpg")

@pytest.fixture
def client():
    return httpx.Client(base_url=BASE_URL, timeout=120.0)

def test_health_check(client):
    """Test GET /health"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_get_models(client):
    """Test GET /v1/models"""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "v2" in data

def test_get_capabilities(client):
    """Test GET /v1/capabilities"""
    response = client.get("/v1/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "has_normal" in data
    assert "has_scale" in data

def test_inference_flow(client):
    """Test POST /v1/inference and asset accessibility"""
    if not IMAGE_PATH.exists():
        pytest.skip(f"Test image {IMAGE_PATH} not found")

    with open(IMAGE_PATH, "rb") as f:
        files = {"image": ("test.jpg", f, "image/jpeg")}
        data = {
            "max_size": 400,
            "resolution_level": "Low",
            "apply_mask": "true",
            "remove_edge": "true",
            "produce_depth": "true",
            "produce_normal": "true"
        }
        
        response = client.post("/v1/inference", files=files, data=data)
        assert response.status_code == 200
        
        result = response.json()
        assert "task_id" in result
        assert "files" in result
        assert "image_png" in result["files"]

        # Verify asset accessibility
        for field, path in result['files'].items():
            if path:
                asset_resp = client.get(path)
                assert asset_resp.status_code == 200, f"Asset {field} at {path} not accessible"
