import gradio as gr
import httpx
import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import io
import cv2

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

        # Download the resized image for measurement
        img_url = f"{BACKEND_URL}{files_info['image_png']}"
        img_resp = httpx.get(img_url)
        img_resp.raise_for_status()
        resized_image = Image.open(io.BytesIO(img_resp.content))
        
        status_msg = f"Task {task_id} completed successfully.\nFOV: {result['fov']}\nTimings: {result['timings']['total_s']:.2f}s"
        
        return str(mesh_path), status_msg, task_id, resized_image

    except Exception as e:
        return None, f"Error: {str(e)}", None, None

def on_select_point(task_id, points, image, event: gr.SelectData):
    if task_id is None or image is None:
        return points, image, "Please run inference first."
    
    # event.index is (x, y)
    points.append((event.index[0], event.index[1]))
    
    # Draw selected points on a copy of the image
    img_np = np.array(image)
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    for p in points:
        cv2.circle(img_bgr, (p[0], p[1]), radius=5, color=(255, 0, 0), thickness=-1)
        cv2.circle(img_bgr, (p[0], p[1]), radius=6, color=(255, 255, 255), thickness=1)

    if len(points) == 2:
        p1, p2 = points[0], points[1]
        cv2.line(img_bgr, p1, p2, color=(255, 0, 0), thickness=2)
        
        # Call measure API
        try:
            with httpx.Client(timeout=10.0) as client:
                req_data = {
                    "task_id": task_id,
                    "p1": {"x": p1[0], "y": p1[1]},
                    "p2": {"x": p2[0], "y": p2[1]}
                }
                response = client.post(f"{BACKEND_URL}/v1/measure", json=req_data)
                response.raise_for_status()
                result = response.json()
                distance = result['distance']
                msg = f"Distance between P1({p1[0]}, {p1[1]}) and P2({p2[0]}, {p2[1]}): **{distance:.4f}m**"
        except Exception as e:
            msg = f"Measurement error: {str(e)}"
        
        # Reset points for next measurement
        points = []
        # Convert back to RGB for Gradio
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return points, img_rgb, msg
    else:
        msg = f"Point {len(points)} selected: ({event.index[0]}, {event.index[1]}). Click another point to measure distance."
        # Convert back to RGB for Gradio
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return points, img_rgb, msg

# Gradio Interface
with gr.Blocks(title="MoGe 3D Reconstruction") as demo:
    gr.Markdown("# MoGe 3D Reconstruction")
    gr.Markdown("Upload an image to generate a 3D point cloud/mesh using MoGe API.")
    
    with gr.Tabs():
        with gr.Tab("3D Reconstruction"):
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
                    
                    gr.Markdown("---")
                    gr.Markdown("### Measure Tool")
                    gr.Markdown("Click two points on the image below to measure the real-world distance.")
                    measure_image = gr.Image(type="numpy", label="Measure Image", interactive=False)
                    measure_info = gr.Markdown("Run 3D reconstruction to enable measurement.")
                    
                    current_task_id = gr.State(None)
                    measure_points = gr.State([])
                    processed_image_state = gr.State(None)

    submit_btn.click(
        fn=process_image,
        inputs=[input_image, max_size, resolution_level, apply_mask, remove_edge, produce_depth, produce_normal],
        outputs=[output_3d, output_msg, current_task_id, processed_image_state]
    ).then(
        fn=lambda img: img,
        inputs=[processed_image_state],
        outputs=[measure_image]
    )

    measure_image.select(
        fn=on_select_point,
        inputs=[current_task_id, measure_points, processed_image_state],
        outputs=[measure_points, measure_image, measure_info]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
