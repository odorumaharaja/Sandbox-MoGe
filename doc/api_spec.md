# MoGe API Specification

The MoGe API is a FastAPI-based service for 3D reconstruction from single images.

## Base URL
- Local: `http://localhost:8000`
- Docker: `http://backend:8000` (internal)

## Endpoints

### 1. Health Check
`GET /health`
- **Description**: Returns the status of the API and model loading state.
- **Response**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true
  }
  ```

### 2. Run Inference
`POST /v1/inference`
- **Description**: Processes an image and returns 3D reconstruction metadata and asset links.
- **Request (multipart/form-data)**:
  - `image`: File (Required)
  - `max_size`: int (Default: 800)
  - `resolution_level`: str ("Low", "Medium", "High", "Ultra" - Default: "High")
  - `apply_mask`: bool (Default: true)
  - `remove_edge`: bool (Default: true)
  - `produce_depth`: bool (Default: true)
  - `produce_normal`: bool (Default: true)
- **Response**:
  ```json
  {
    "task_id": "uuid-string",
    "fov": [horizontal_fov, vertical_fov],
    "timings": {
      "preprocess_s": 0.0,
      "inference_s": 0.0,
      "visualization_s": 0.0,
      "export_s": 0.0,
      "total_s": 0.0
    },
    "files": {
      "mesh_glb": "/assets/task_id/mesh.glb",
      "pointcloud_ply": "/assets/task_id/pointcloud.ply",
      ...
    },
    "image_size": {
      "width": 800,
      "height": 600
    }
  }
  ```

### 3. List Models
`GET /v1/models`
- **Description**: Lists available MoGe model versions.

### 4. Get Capabilities
`GET /v1/capabilities`
- **Description**: Reports features supported by the currently loaded model (e.g., normal estimation).

### 5. Measure Distance
`POST /v1/measure`
- **Description**: Calculates the Euclidean distance between two 3D points in an image.
- **Note**: The coordinates `(x, y)` should correspond to the indices of the resized image returned in the inference response (`image_png`).
- **Request (application/json)**:
  ```json
  {
    "task_id": "uuid-string",
    "p1": {"x": 100, "y": 150},
    "p2": {"x": 200, "y": 250}
  }
  ```
- **Response**:
  ```json
  {
    "distance": 1.234,
    "unit": "m"
  }
  ```

## Asset Serving
`GET /assets/{task_id}/{filename}`
- **Description**: Static files generated during inference are served from a temporary directory.
- **Note**: Assets are stored in a temporary system directory (`/tmp/moge_assets`) and are not persistent across reboots or container restarts.

## Data Models (Pydantic)
All request/response schemas are defined in `backend/schemas.py`.
