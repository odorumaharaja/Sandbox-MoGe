import gradio as gr
import httpx
import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import io

# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def process_image(image, max_size, resolution_level, apply_mask, remove_edge, produce_depth, produce_normal):
    if image is None:
        return None, None, "Please upload an image."

    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {"image": ("image.jpg", img_byte_arr, "image/jpeg")}
    data = {
        "max_size": max_size,
        "resolution_level": resolution_level,
        "apply_mask": str(apply_mask).lower(),
        "remove_edge": str(remove_edge).lower(),
        "produce_depth": str(produce_depth).lower(),
        "produce_normal": str(produce_normal).lower()
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{BACKEND_URL}/v1/inference", files=files, data=data)
            response.raise_for_status()
            result = response.json()

        task_id = result['task_id']
        files_info = result['files']
        
        # Download the mesh GLB
        mesh_url = f"{BACKEND_URL}{files_info['pointcloud_glb']}"
        mesh_resp = httpx.get(mesh_url)
        mesh_resp.raise_for_status()
        
        temp_dir = Path(tempfile.gettempdir()) / "moge_frontend"
        temp_dir.mkdir(exist_ok=True)
        mesh_path = temp_dir / f"{task_id}_mesh.glb"
        mesh_path.write_bytes(mesh_resp.content)

        # Download visualization images if available (mocked here or use result['depth_vis'] if API returned it)
        # Actually our API returns asset paths. Let's just show the 3D result.
        
        status_msg = f"Task {task_id} completed successfully.\nFOV: {result['fov']}\nTimings: {result['timings']['total_s']:.2f}s"
        
        return str(mesh_path), status_msg

    except Exception as e:
        return None, f"Error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="MoGe 3D Reconstruction") as demo:
    gr.Markdown("# MoGe 3D Reconstruction")
    gr.Markdown("Upload an image to generate a 3D point cloud/mesh using MoGe API.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            with gr.Accordion("Advanced Settings", open=False):
                max_size = gr.Slider(minimum=256, maximum=1024, value=800, step=64, label="Max Size")
                resolution_level = gr.Dropdown(choices=["Low", "Medium", "High", "Ultra"], value="High", label="Resolution Level")
                apply_mask = gr.Checkbox(value=True, label="Apply Mask")
                remove_edge = gr.Checkbox(value=True, label="Remove Edge")
                produce_depth = gr.Checkbox(value=True, label="Produce Depth")
                produce_normal = gr.Checkbox(value=True, label="Produce Normal")
            
            submit_btn = gr.Button("Generate 3D", variant="primary")
        
        with gr.Column():
            output_3d = gr.Model3D(label="3D Result")
            output_msg = gr.Textbox(label="Status")

    submit_btn.click(
        fn=process_image,
        inputs=[input_image, max_size, resolution_level, apply_mask, remove_edge, produce_depth, produce_normal],
        outputs=[output_3d, output_msg]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
