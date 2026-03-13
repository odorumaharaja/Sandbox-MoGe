from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class TimingInfo(BaseModel):
    preprocess_s: float = Field(..., description="Image preprocessing time in seconds")
    inference_s: float = Field(..., description="Model inference time in seconds")
    visualization_s: float = Field(..., description="Visualization time in seconds")
    export_s: float = Field(..., description="File export time in seconds")
    total_s: float = Field(..., description="Total processing time in seconds")

class InferenceFiles(BaseModel):
    mesh_glb: str = Field(..., description="Path to the mesh GLB file")
    pointcloud_ply: str = Field(..., description="Path to the point cloud PLY file")
    pointcloud_glb: str = Field(..., description="Path to the point cloud GLB file")
    depth_exr: Optional[str] = Field(None, description="Path to the depth map EXR file")
    points_exr: str = Field(..., description="Path to the point map EXR file")
    mask_png: str = Field(..., description="Path to the mask PNG file")
    normal_exr: Optional[str] = Field(None, description="Path to the normal map EXR file")

class ImageSize(BaseModel):
    width: int
    height: int

class InferenceResponse(BaseModel):
    task_id: str = Field(..., description="Unique ID for this inference task")
    fov: List[float] = Field(..., description="Horizontal and Vertical FOV in degrees")
    timings: TimingInfo
    files: InferenceFiles
    image_size: ImageSize

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
