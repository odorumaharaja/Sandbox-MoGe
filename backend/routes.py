import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from . import schemas

router = APIRouter()

# Directory for assets
ASSETS_DIR = Path(tempfile.gettempdir()) / "moge_assets"

@router.post("/v1/inference", response_model=schemas.InferenceResponse)
async def run_inference(
    request: Request,
    image: UploadFile = File(...),
    max_size: int = Form(800),
    resolution_level: str = Form("High"),
    apply_mask: bool = Form(True),
    remove_edge: bool = Form(True),
    produce_depth: bool = Form(True),
    produce_normal: bool = Form(True)
):
    moge_inference = request.app.state.moge_inference
    if moge_inference is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # 1. Read and Decode Image
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

    # 2. Run MoGe Inference
    try:
        result = moge_inference.run(
            image=img_rgb,
            max_size=max_size,
            resolution_level=resolution_level,
            apply_mask=apply_mask,
            remove_edge=remove_edge,
            produce_depth=produce_depth,
            produce_normal=produce_normal,
            enable_download=True
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # 3. Organize Assets
    temp_output_path = Path(result['model_3d_file']).parent
    task_id = temp_output_path.name
    
    task_assets_dir = ASSETS_DIR / task_id
    task_assets_dir.mkdir(exist_ok=True, parents=True)
    
    files_map = {}
    name_to_field = {
        'mesh.glb': 'mesh_glb',
        'pointcloud.ply': 'pointcloud_ply',
        'pointcloud.glb': 'pointcloud_glb',
        'depth.exr': 'depth_exr',
        'points.exr': 'points_exr',
        'mask.png': 'mask_png',
        'normal.exr': 'normal_exr',
        'image.png': 'image_png'
    }

    processed_files = result['output_files']
    if result['model_3d_file'] and result['model_3d_file'] not in processed_files:
        processed_files.append(str(result['model_3d_file']))

    for file_path_str in processed_files:
        src = Path(file_path_str)
        if src.exists():
            dst = task_assets_dir / src.name
            shutil.copy2(src, dst)
            field_name = name_to_field.get(src.name)
            if field_name:
                # Store relative paths for the API response
                files_map[field_name] = f"/assets/{task_id}/{src.name}"

    height, width = result['results']['image'].shape[:2]

    return schemas.InferenceResponse(
        task_id=task_id,
        fov=list(result['fov']),
        timings=schemas.TimingInfo(**result['timings']),
        files=schemas.InferenceFiles(**files_map),
        image_size=schemas.ImageSize(width=width, height=height)
    )

@router.get("/v1/models")
async def get_models() -> Dict[str, str]:
    from .inference import DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION
    return DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION

@router.get("/v1/capabilities")
async def get_capabilities(request: Request) -> Dict[str, bool]:
    moge_inference = request.app.state.moge_inference
    if moge_inference is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return moge_inference.get_model_capabilities()

@router.get("/health", response_model=schemas.HealthCheckResponse)
async def health_check(request: Request):
    moge_inference = request.app.state.moge_inference
    return schemas.HealthCheckResponse(
        status="healthy",
        model_loaded=moge_inference is not None
    )

@router.post("/v1/measure", response_model=schemas.MeasureResponse)
async def measure_distance(
    request: schemas.MeasureRequest
):
    task_assets_dir = ASSETS_DIR / request.task_id
    points_path = task_assets_dir / "points.exr"

    if not points_path.exists():
        raise HTTPException(status_code=404, detail="Points data not found for this task")

    # Load points data
    try:
        points_bgr = cv2.imread(str(points_path), cv2.IMREAD_UNCHANGED)
        if points_bgr is None:
            raise HTTPException(status_code=500, detail="Failed to load points data")
        
        h, w = points_bgr.shape[:2]
        
        # Bounds check
        if not (0 <= request.p1.x < w and 0 <= request.p1.y < h):
            raise HTTPException(status_code=400, detail=f"Point 1 ({request.p1.x}, {request.p1.y}) is out of bounds for image size ({w}, {h})")
        if not (0 <= request.p2.x < w and 0 <= request.p2.y < h):
            raise HTTPException(status_code=400, detail=f"Point 2 ({request.p2.x}, {request.p2.y}) is out of bounds for image size ({w}, {h})")

        # points_bgr is (H, W, 3) in BGR order (because it was saved as RGB->BGR)
        # So we convert back to RGB to get [X, Y, Z]
        points_rgb = cv2.cvtColor(points_bgr, cv2.COLOR_BGR2RGB)
        
        p1_3d = points_rgb[request.p1.y, request.p1.x]
        p2_3d = points_rgb[request.p2.y, request.p2.x]
        
        distance = float(np.linalg.norm(p1_3d - p2_3d))
        
        return schemas.MeasureResponse(distance=distance)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Measurement error: {str(e)}")
