import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys
from pathlib import Path
import time
import uuid
import tempfile
import itertools
from typing import *
import atexit
from concurrent.futures import ThreadPoolExecutor
import shutil
import click
import cv2
import torch
import numpy as np
import trimesh
import trimesh.visual
from PIL import Image
import gradio as gr

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
        image_tensor = torch.tensor(image, dtype=torch.float32 if not self.use_fp16 else torch.float16, device=torch.device('cuda')).permute(2, 0, 1) / 255
        output = self.model.infer(image_tensor, apply_mask=apply_mask, resolution_level=resolution_level, use_fp16=self.use_fp16)
        output = {k: v.cpu().numpy() for k, v in output.items()}
        return output

    def run(self, image: np.ndarray, max_size: int = 800, resolution_level: str = 'High',  apply_mask: bool = True, remove_edge: bool = True, request: gr.Request = None, produce_depth: bool = True, produce_normal: bool = True, enable_download: bool = True):
        t_start = time.time()
        larger_size = max(image.shape[:2])
        if larger_size > max_size:
            scale = max_size / larger_size
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        t_after_preproc = time.time()

        height, width = image.shape[:2]
        resolution_level_int = {'Low': 0, 'Medium': 5, 'High': 9, 'Ultra': 30}.get(resolution_level, 9)
        
        output = self.run_with_gpu(image, resolution_level_int, apply_mask)
        t_after_infer = time.time()

        points, depth, mask, normal = output['points'], output.get('depth', None), output['mask'], output.get('normal', None)

        # Honor options to skip depth/normal generation for speed (post-infer disabling)
        if not produce_depth:
            depth = None
        if not produce_normal:
            normal = None

        # remove edge only when depth is available
        if remove_edge and depth is not None:
            mask_cleaned = mask & ~utils3d.np.depth_map_edge(depth, rtol=0.04)
        else:
            mask_cleaned = mask
        
        results = {
            **output,
            'mask_cleaned': mask_cleaned,
            'image': image
        }

        # depth & normal visualization (may be disabled)
        if depth is not None:
            depth_vis = colorize_depth(depth)
        else:
            depth_vis = gr.update(label="Depth map (disabled)")

        if normal is not None:
            normal_vis = colorize_normal(normal)
        else:
            normal_vis = gr.update(label="Normal map (not available or disabled)")

        t_after_viz = time.time()

        # mesh & pointcloud
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
        vertices = vertices * np.array([1, -1, -1], dtype=np.float32) 
        vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32)
        if vertex_normals is not None:
            vertex_normals = vertex_normals * np.array([1, -1, -1], dtype=np.float32)

        tempdir = Path(tempfile.gettempdir(), 'moge')
        tempdir.mkdir(exist_ok=True)
        output_path = Path(tempdir, request.session_hash)
        shutil.rmtree(output_path, ignore_errors=True)
        output_path.mkdir(exist_ok=True, parents=True)
        
        files = []
        model_3d_file = None
        if enable_download:
            trimesh.Trimesh(
                vertices=vertices,
                faces=faces, 
                visual = trimesh.visual.texture.TextureVisuals(
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
            
            cv2.imwrite(str(output_path /'mask.png'), mask.astype(np.uint8) * 255)
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

        # FOV
        intrinsics = results['intrinsics']
        fov_x, fov_y = utils3d.np.intrinsics_to_fov(intrinsics)
        fov_x, fov_y = np.rad2deg([fov_x, fov_y])

        # messages
        viewer_message = f'**Note:** Inference has been completed. It may take a few seconds to download the 3D model.'
        if resolution_level != 'Ultra':
            depth_message = f'**Note:** Want sharper depth map? Try increasing the `maximum image size` and setting the `inference resolution level` to `Ultra` in the settings.'
        else:
            depth_message = ""

        model_3d_out = model_3d_file if model_3d_file is not None else None
        downloadable_files = [(output_path / f).as_posix() for f in files if (output_path / f).exists()] if enable_download else []

        # timings
        t_total = (t_after_export if 't_after_export' in locals() else time.time()) - t_start
        timings = {
            'preprocess_s': t_after_preproc - t_start,
            'inference_s': t_after_infer - t_after_preproc,
            'visualization_s': t_after_viz - t_after_infer,
            'export_s': (t_after_export - t_after_viz) if 't_after_export' in locals() else 0.0,
            'total_s': t_total
        }
        timings_text = (
            f"- Preprocess: {timings['preprocess_s']:.3f}s\n"
            f"- Inference: {timings['inference_s']:.3f}s\n"
            f"- Visualization: {timings['visualization_s']:.3f}s\n"
            f"- Export: {timings['export_s']:.3f}s\n"
            f"- Total: {timings['total_s']:.3f}s"
        )

        return (
            results,
            depth_vis,
            normal_vis, 
            model_3d_out, 
            downloadable_files,
            f'- **Horizontal FOV: {fov_x:.1f}°**. \n - **Vertical FOV: {fov_y:.1f}°',
            viewer_message,
            depth_message,
            timings_text
        )

def reset_measure(results: Dict[str, np.ndarray]):
    return [results['image'], [], ""]

def measure(results: Dict[str, np.ndarray], measure_points: List[Tuple[int, int]], enabled: bool, event: gr.SelectData):
    if not enabled:
        if results is None:
            return [None, measure_points, ""]
        return [results['image'], measure_points, ""]
    point2d = event.index[0], event.index[1]
    measure_points.append(point2d)

    image = results['image'].copy()
    for p in measure_points:
        image = cv2.circle(image, p, radius=5, color=(255, 0, 0), thickness=2)

    depth_text = ""
    for i, p in enumerate(measure_points):
        d = results['depth'][p[1], p[0]]
        depth_text += f"- **P{i + 1} depth: {d:.2f}m.**\n"

    if len(measure_points) == 2:
        point1, point2 = measure_points
        image = cv2.line(image, point1, point2, color=(255, 0, 0), thickness=2)
        distance = np.linalg.norm(results['points'][point1[1], point1[0]] - results['points'][point2[1], point2[0]])
        measure_points = []

        distance_text = f"- **Distance: {distance:.2f}m**"

        text = depth_text + distance_text
        return [image, measure_points, text]
    else:
        return [image, measure_points, depth_text]
        
def create_demo(inference: MoGeInference, share: bool):
    print("Create Gradio app...")
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
f'''
<div align="center">
<h1> Estimate Object Estimator </h1>
</div>
''')
        results = gr.State(value=None)
        measure_points = gr.State(value=[])

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", image_mode="RGB", label="Input Image")
                with gr.Accordion(label="Settings", open=False):
                    max_size_input = gr.Number(value=800, label="Maximum Image Size", precision=0, minimum=256, maximum=2048)
                    resolution_level = gr.Dropdown(['Low', 'Medium', 'High', 'Ultra'], label="Inference Resolution Level", value='High')
                    apply_mask = gr.Checkbox(value=True, label="Apply mask")
                    remove_edges = gr.Checkbox(value=True, label="Remove edges")
                    produce_depth = gr.Checkbox(value=True, label="Produce Depth (enable depth map)")
                    produce_normal = gr.Checkbox(value=hasattr(inference.model, 'normal_head'), label="Produce Normal (slower)")
                    enable_measure = gr.Checkbox(value=hasattr(inference.model, 'scale_head'), label="Enable Measure")
                    enable_download = gr.Checkbox(value=True, label="Enable Download (save output files)")
                submit_btn = gr.Button("Submit", variant='primary')

            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("3D View"):
                        viewer_message = gr.Markdown("")
                        model_3d = gr.Model3D(display_mode="solid", label="3D Point Map", clear_color=[1.0, 1.0, 1.0, 1.0], height="60vh")
                        fov = gr.Markdown()
                        timings = gr.Markdown("", label="Timings")
                    with gr.Tab("Depth"):
                        depth_message = gr.Markdown("")
                        depth_map = gr.Image(type="numpy", label="Colorized Depth Map", format='png', interactive=False)
                    with gr.Tab("Normal", interactive=hasattr(inference.model, 'normal_head')):
                        normal_map = gr.Image(type="numpy", label="Normal Map", format='png', interactive=False)
                    with gr.Tab("Measure", interactive=hasattr(inference.model, 'scale_head')):
                        gr.Markdown("### Click on the image to measure the distance between two points. \n"
                         "**Note:** Metric scale is most reliable for typical indoor or street scenes, and may degrade for contents unfamiliar to the model (e.g., stylized or close-up images).")
                        measure_image = gr.Image(type="numpy", show_label=False, format='webp', interactive=False, sources=[])
                        measure_text = gr.Markdown("")
                    with gr.Tab("Download"):
                        files = gr.File(type='filepath', label="Output Files")
        
        if Path('example_images').exists():
            example_image_paths = sorted(list(itertools.chain(*[Path('example_images').glob(f'*.{ext}') for ext in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']])))
            examples = gr.Examples(
                examples = example_image_paths,
                inputs=input_image,
                label="Examples"
            )

        submit_btn.click(
            fn=lambda: [None, None, None, None, None, "", "", "", ""],
            outputs=[results, depth_map, normal_map, model_3d, files, fov, viewer_message, depth_message, timings]
        ).then(
            fn=inference.run,
            inputs=[input_image, max_size_input, resolution_level, apply_mask, remove_edges, produce_depth, produce_normal, enable_download],
            outputs=[results, depth_map, normal_map, model_3d, files, fov, viewer_message, depth_message, timings]
        ).then(
            fn=reset_measure,
            inputs=[results],
            outputs=[measure_image, measure_points, measure_text]
        )

        measure_image.select(
            fn=measure,
            inputs=[results, measure_points, enable_measure],
            outputs=[measure_image, measure_points, measure_text]
        )
    
    return demo

@click.command(help='Web demo')
@click.option('--share', is_flag=True, help='Whether to run the app in shared mode.')
@click.option('--pretrained', 'pretrained_model_name_or_path', default=None, help='The name or path of the pre-trained model.')
@click.option('--version', 'model_version', default='v2', help='The version of the model.')
@click.option('--fp16', 'use_fp16', is_flag=True, help='Whether to use fp16 inference.')
def main(share: bool, pretrained_model_name_or_path: str, model_version: str, use_fp16: bool):
    inference = MoGeInference(pretrained_model_name_or_path, model_version, use_fp16)
    demo = create_demo(inference, share)
    demo.launch(share=share)

if __name__ == '__main__':
    main()