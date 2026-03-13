import httpx
import os
import time
from pathlib import Path
from typing import Dict, Any

def test_api():
    base_url = "http://localhost:8000"
    image_path = Path("Test.jpg")
    
    if not image_path.exists():
        print(f"Error: {image_path} not found. Please provide a test image.")
        return

    # 0. Test Health Check
    print("\n--- Testing GET /health ---")
    try:
        resp = httpx.get(f"{base_url}/health")
        resp.raise_for_status()
        print(f"Health: {resp.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")

    # 1. Test Models Endpoint
    print("\n--- Testing GET /v1/models ---")
    try:
        resp = httpx.get(f"{base_url}/v1/models")
        resp.raise_for_status()
        print(f"Available Models: {resp.json()}")
    except Exception as e:
        print(f"Failed to get models: {e}")

    # 2. Test Capabilities Endpoint
    print("\n--- Testing GET /v1/capabilities ---")
    try:
        resp = httpx.get(f"{base_url}/v1/capabilities")
        if resp.status_code == 200:
            print(f"Capabilities: {resp.json()}")
        else:
            print(f"Capabilities check returned status {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"Failed to get capabilities: {e}")

    # 3. Test Inference Endpoint
    print("\n--- Testing POST /v1/inference ---")
    with open(image_path, "rb") as f:
        files = {"image": ("test.jpg", f, "image/jpeg")}
        data = {
            "max_size": 400,
            "resolution_level": "Low",
            "apply_mask": "true",
            "remove_edge": "true",
            "produce_depth": "true",
            "produce_normal": "true"
        }
        
        print("Sending inference request (this may take a few seconds)...")
        start_time = time.time()
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{base_url}/v1/inference", files=files, data=data)
            
            elapsed = time.time() - start_time
            print(f"Request took {elapsed:.2f} seconds.")
            
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                return

            result = response.json()
            print("Response received successfully!")
            print(f"Task ID: {result['task_id']}")
            print(f"Image Size: {result['image_size']}")
            print(f"FOV: {result['fov']}")
            print(f"Timings: {result['timings']}")
            
            print("\nGenerated files:")
            for field, path in result['files'].items():
                print(f"  - {field}: {path}")
                
            # 4. Verify Static Assets
            print("\n--- Verifying asset accessibility ---")
            for field, path in result['files'].items():
                if path:
                    asset_url = f"{base_url}{path}"
                    asset_resp = httpx.get(asset_url)
                    if asset_resp.status_code == 200:
                        print(f"  [OK] {path} is accessible")
                    else:
                        print(f"  [FAILED] {path} returned {asset_resp.status_code}")

        except Exception as e:
            print(f"Inference test failed: {e}")

if __name__ == "__main__":
    # Ensure server is running before executing this script
    test_api()
