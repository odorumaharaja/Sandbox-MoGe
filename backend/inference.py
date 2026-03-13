import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys
from pathlib import Path
import time
import uuid
import tempfile
from typing import *
import atexit
from concurrent.futures import ThreadPoolExecutor
import shutil

import cv2
import torch
import numpy as np
import trimesh
import trimesh.visual
from PIL import Image

try:
    import spaces   # This is for deployment at huggingface.co/spaces
    HUGGINFACE_SPACES_INSTALLED = True
except ImportError:
    HUGGINFACE_SPACES_INSTALLED = False

import utils3d
from moge.utils.io import write_normal
from moge.utils.vis import colorize_depth, colorize_normal
from moge.model import import_model_class_by_version
from moge.utils.geometry_numpy import depth_occlusion_edge_numpy
from moge.utils.tools import timeit

DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION = {
    "v1": "Ruicheng/moge-vitl",
    "v2": "Ruicheng/moge-2-vitl-normal",
}

def gpu_decorator(func):
    if HUGGINFACE_SPACES_INSTALLED:
        return spaces.GPU(func)
    return func

class MoGeInference:
    """MoGe depth estimation inference engine"""
    
    def __init__(self, pretrained_model_name_or_path: str, model_version: str, use_fp16: bool):
        print("Load model...")
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION.get(model_version, DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION['v2'])
        
        self.model = import_model_class_by_version(model_version).from_pretrained(pretrained_model_name_or_path).cuda().eval()
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.model.half()
        
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

    def delete_later(self, path: Union[str, os.PathLike], delay: int = 300):
        """Delete specified file later in background"""
        def _delete():
            try: 
                os.remove(path) 
            except FileNotFoundError:
                pass
        def _wait_and_delete():
            time.sleep(delay)
            _delete()
        self.thread_pool_executor.submit(_wait_and_delete)
        atexit.register(_delete)

    @gpu_decorator
    def run_with_gpu(self, image: np.ndarray, resolution_level: int, apply_mask: bool) -> Dict[str, np.ndarray]:
        """Run MoGe model on GPU"""
        image_tensor = torch.tensor(image, dtype=torch.float32 if not self.use_fp16 else torch.float16, device=torch.device('cuda')).permute(2, 0, 1) / 255
        output = self.model.infer(image_tensor, apply_mask=apply_mask, resolution_level=resolution_level, use_fp16=self.use_fp16)
        output = {k: v.cpu().numpy() for k, v in output.items()}
        return output

    def run(self, image: np.ndarray, max_size: int = 800, resolution_level: str = 'High', apply_mask: bool = True, 
            remove_edge: bool = True, produce_depth: bool = True, produce_normal: bool = True, 
            enable_download: bool = True) -> Dict[str, any]:
        """
        Run MoGe inference and generate depth, normal, and 3D mesh.
        
        Args:
            image: Input image (RGB, numpy array)
            max_size: Maximum image size
            resolution_level: Inference resolution level ('Low', 'Medium', 'High', 'Ultra')
            apply_mask: Apply mask flag
            remove_edge: Remove edge flag
            produce_depth: Produce depth map flag
            produce_normal: Produce normal map flag
            enable_download: Enable download flag
            
        Returns:
            Dictionary containing processing results
        """
        t_start = time.time()
        
        # Image preprocessing
        larger_size = max(image.shape[:2])
        if larger_size > max_size:
            scale = max_size / larger_size
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        t_after_preproc = time.time()

        height, width = image.shape[:2]
        resolution_level_int = {'Low': 0, 'Medium': 5, 'High': 9, 'Ultra': 30}.get(resolution_level, 9)
        
        # Run inference
        output = self.run_with_gpu(image, resolution_level_int, apply_mask)
        t_after_infer = time.time()

        """
        `output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
        The maps are in the same size as the input image. 
        {
            "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
            "depth": (H, W),        # depth map
            "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
            "mask": (H, W),         # a binary mask for valid pixels. 
            "intrinsics": (3, 3),   # normalized camera intrinsics
        }
        """
        points, depth, mask, normal = output['points'], output.get('depth', None), output['mask'], output.get('normal', None)

        # Respect produce depth/normal options
        if not produce_depth:
            depth = None
        if not produce_normal:
            normal = None

        # Remove edges (only when depth is available)
        if remove_edge and depth is not None:
            mask_cleaned = mask & ~utils3d.np.depth_map_edge(depth, rtol=0.04)
        else:
            mask_cleaned = mask
        
        results = {
            **output,
            'mask_cleaned': mask_cleaned,
            'image': image
        }

        # Depth and normal visualization
        depth_vis = colorize_depth(depth) if depth is not None else None
        normal_vis = colorize_normal(normal) if normal is not None else None

        t_after_viz = time.time()

        # Build mesh and point cloud
        if normal is None:
            faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
                points,
                image.astype(np.float32) / 255,
                utils3d.np.uv_map(height, width),
                mask=mask_cleaned,
                tri=True
            )
            vertex_normals = None
        else:
            faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.np.build_mesh_from_map(
                points,
                image.astype(np.float32) / 255,
                utils3d.np.uv_map(height, width),
                normal,
                mask=mask_cleaned,
                tri=True
            )
        
        # Coordinate system transformation
        vertices = vertices * np.array([1, -1, -1], dtype=np.float32) 
        vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32)
        if vertex_normals is not None:
            vertex_normals = vertex_normals * np.array([1, -1, -1], dtype=np.float32)

        # File output processing
        tempdir = Path(tempfile.gettempdir(), 'moge')
        tempdir.mkdir(exist_ok=True)
        output_path = Path(tempdir, str(uuid.uuid4()))
        output_path.mkdir(exist_ok=True, parents=True)
        
        files = []
        model_3d_file = None
        
        if enable_download:
            # Mesh export
            trimesh.Trimesh(
                vertices=vertices,
                faces=faces, 
                visual=trimesh.visual.texture.TextureVisuals(
                    uv=vertex_uvs, 
                    material=trimesh.visual.material.PBRMaterial(
                        baseColorTexture=Image.fromarray(image),
                        metallicFactor=0.5,
                        roughnessFactor=1.0
                    )
                ),
                vertex_normals=vertex_normals,
                process=False
            ).export(output_path / 'mesh.glb')
            
            # Point cloud export
            pointcloud = trimesh.PointCloud(
                vertices=vertices, 
                colors=vertex_colors,
            )
            pointcloud.vertex_normals = vertex_normals
            pointcloud.export(output_path / 'pointcloud.ply', vertex_normal=True)
            
            trimesh.PointCloud(
                vertices=vertices, 
                colors=vertex_colors,
            ).export(output_path / 'pointcloud.glb', include_normals=True)
            
            # Map export
            cv2.imwrite(str(output_path / 'mask.png'), mask.astype(np.uint8) * 255)
            if depth is not None:
                cv2.imwrite(str(output_path / 'depth.exr'), depth.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            cv2.imwrite(str(output_path / 'points.exr'), cv2.cvtColor(points.astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            if normal is not None:
                cv2.imwrite(str(output_path / 'normal.exr'), cv2.cvtColor(normal.astype(np.float32) * np.array([1, -1, -1], dtype=np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])

            files = ['mesh.glb', 'pointcloud.ply', 'depth.exr', 'points.exr', 'mask.png']
            if normal is not None:
                files.append('normal.exr')

            for f in files:
                self.delete_later(output_path / f)

            model_3d_file = output_path / 'pointcloud.glb'

        t_after_export = time.time()

        # FOV calculation
        intrinsics = results['intrinsics']
        fov_x, fov_y = utils3d.np.intrinsics_to_fov(intrinsics)
        fov_x, fov_y = np.rad2deg([fov_x, fov_y])

        # Timing information
        t_total = (t_after_export if 't_after_export' in locals() else time.time()) - t_start
        timings = {
            'preprocess_s': t_after_preproc - t_start,
            'inference_s': t_after_infer - t_after_preproc,
            'visualization_s': t_after_viz - t_after_infer,
            'export_s': (t_after_export - t_after_viz) if 't_after_export' in locals() else 0.0,
            'total_s': t_total
        }

        return {
            'results': results,
            'depth_vis': depth_vis,
            'normal_vis': normal_vis,
            'model_3d_file': model_3d_file,
            'output_files': [(output_path / f).as_posix() for f in files if (output_path / f).exists()] if enable_download else [],
            'fov': (fov_x, fov_y),
            'timings': timings
        }

    def get_model_capabilities(self) -> Dict[str, bool]:
        """Check model capabilities"""
        return {
            'has_normal': hasattr(self.model, 'normal_head'),
            'has_scale': hasattr(self.model, 'scale_head'),
        }
