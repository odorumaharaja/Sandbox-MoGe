import pytest
import httpx
import os
from pathlib import Path

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
IMAGE_PATH = Path("Test.jpg")

@pytest.fixture(scope="module")
def inference_result():
    """Run inference once and provide result for measurement tests"""
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
            "produce_normal": "false"
        }
        with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
            response = client.post("/v1/inference", files=files, data=data)
            assert response.status_code == 200
            return response.json()

def test_measure_success(inference_result):
    """Test measurement between two valid points"""
    task_id = inference_result["task_id"]
    # Let's pick points that are definitely within bounds for 400px max size
    measure_data = {
        "task_id": task_id,
        "p1": {"x": 50, "y": 50},
        "p2": {"x": 100, "y": 100}
    }
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        response = client.post("/v1/measure", json=measure_data)
        assert response.status_code == 200
        data = response.json()
        assert "distance" in data
        assert isinstance(data["distance"], float)
        assert data["distance"] >= 0

def test_measure_out_of_bounds(inference_result):
    """Test measurement with out-of-bounds coordinates (should return 400)"""
    task_id = inference_result["task_id"]
    measure_data = {
        "task_id": task_id,
        "p1": {"x": 5000, "y": 5000},
        "p2": {"x": 100, "y": 100}
    }
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        response = client.post("/v1/measure", json=measure_data)
        assert response.status_code == 400
        assert "out of bounds" in response.text.lower()

def test_measure_invalid_task_id():
    """Test measurement with non-existent task_id (should return 404)"""
    measure_data = {
        "task_id": "non-existent-task",
        "p1": {"x": 50, "y": 50},
        "p2": {"x": 100, "y": 100}
    }
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        response = client.post("/v1/measure", json=measure_data)
        assert response.status_code == 404
