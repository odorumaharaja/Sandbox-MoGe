"""
UI definition and display module
Gradio application to display MoGe inference results
"""

from pathlib import Path
from typing import Dict, List, Tuple
import itertools

import cv2
import numpy as np
import gradio as gr

from inference import MoGeInference


def reset_measure(results: Dict[str, np.ndarray]):
    """Reset measurement display"""
    if results is None:
        return [None, [], ""]
    return [results['results']['image'], [], ""]


def measure(results: Dict[str, np.ndarray], measure_points: List[Tuple[int, int]], enabled: bool, event: gr.SelectData):
    """
    Measure distance between two points on image
    """
    if not enabled:
        if results is None:
            return [None, measure_points, ""]
        return [results['results']['image'], measure_points, ""]
    
    point2d = event.index[0], event.index[1]
    measure_points.append(point2d)

    image = results['results']['image'].copy()
    for p in measure_points:
        image = cv2.circle(image, p, radius=5, color=(255, 0, 0), thickness=2)

    depth_text = ""
    for i, p in enumerate(measure_points):
        d = results['results']['depth'][p[1], p[0]]
        depth_text += f"- **P{i + 1} depth: {d:.2f}m.**\n"

    if len(measure_points) == 2:
        point1, point2 = measure_points
        image = cv2.line(image, point1, point2, color=(255, 0, 0), thickness=2)
        distance = np.linalg.norm(results['results']['points'][point1[1], point1[0]] - results['results']['points'][point2[1], point2[0]])
        measure_points = []

        distance_text = f"- **Distance: {distance:.2f}m**"

        text = depth_text + distance_text
        return [image, measure_points, text]
    else:
        return [image, measure_points, depth_text]


def create_demo(inference: MoGeInference, share: bool) -> gr.Blocks:
    """
    UI definition for MoGe inference application
    
    Args:
        inference: MoGeInference instance
        share: Gradio share mode
        
    Returns:
        Gradio Blocks instance
    """
    # Check model capabilities
    capabilities = inference.get_model_capabilities()
    
    print("Create Gradio app...")
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
f'''
<div align="center">
<h1>Monocular Object Volume Estimator</h1>
</div>
''')
        results = gr.State(value=None)
        measure_points = gr.State(value=[])

        with gr.Row():
            # Left side: Input and settings
            with gr.Column():
                input_image = gr.Image(type="numpy", image_mode="RGB", label="Input Image", width=800, height=600)
                submit_btn = gr.Button("Submit", variant='primary')
                with gr.Accordion(label="Settings", open=False):
                    max_size_input = gr.Number(value=800, label="Maximum Image Size", precision=0, minimum=256, maximum=2048)
                    resolution_level = gr.Dropdown(['Low', 'Medium', 'High', 'Ultra'], label="Inference Resolution Level", value='High')
                    apply_mask = gr.Checkbox(value=True, label="Apply mask")
                    remove_edges = gr.Checkbox(value=True, label="Remove edges")
                    produce_depth = gr.Checkbox(value=True, label="Produce Depth (enable depth map)")
                    produce_normal = gr.Checkbox(value=capabilities['has_normal'], label="Produce Normal (slower)")
                    enable_measure = gr.Checkbox(value=capabilities['has_scale'], label="Enable Measure")
                    enable_download = gr.Checkbox(value=True, label="Enable Download (save output files)")

            # Right side: Output
            with gr.Column():
                with gr.Tabs():
                    # 3D View tab
                    with gr.Tab("3D View"):
                        viewer_message = gr.Markdown("")
                        model_3d = gr.Model3D(display_mode="solid", label="3D Point Map", clear_color=[1.0, 1.0, 1.0, 1.0], height="60vh")
                        fov = gr.Markdown()
                        timings = gr.Markdown("", label="Timings")
                    
                    # Depth tab
                    with gr.Tab("Depth"):
                        depth_message = gr.Markdown("")
                        depth_map = gr.Image(type="numpy", label="Colorized Depth Map", format='png', interactive=False)
                    
                    # Normal map tab
                    with gr.Tab("Normal", interactive=capabilities['has_normal']):
                        normal_map = gr.Image(type="numpy", label="Normal Map", format='png', interactive=False)
                    
                    # Measure tab
                    with gr.Tab("Measure", interactive=capabilities['has_scale']):
                        gr.Markdown("### Click on the image to measure the distance between two points. \n"
                         "**Note:** Metric scale is most reliable for typical indoor or street scenes, and may degrade for contents unfamiliar to the model (e.g., stylized or close-up images).")
                        measure_image = gr.Image(type="numpy", show_label=False, format='webp', interactive=False, sources=[])
                        measure_text = gr.Markdown("")
                    
                    # Download tab
                    with gr.Tab("Download"):
                        files = gr.File(type='filepath', label="Output Files")
        
        # Sample images
        if Path('example_images').exists():
            example_image_paths = sorted(list(itertools.chain(*[Path('example_images').glob(f'*.{ext}') for ext in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']])))
            examples = gr.Examples(
                examples=example_image_paths,
                inputs=input_image,
                label="Examples"
            )

        # Inference execution flow
        submit_btn.click(
            fn=lambda: [None, None, None, None, None, "", "", "", ""],
            outputs=[results, depth_map, normal_map, model_3d, files, fov, viewer_message, depth_message, timings]
        ).then(
            fn=lambda img, max_sz, res_lvl, apply_m, rm_edge, prod_depth, prod_norm, en_dl: _run_inference(
                inference, img, max_sz, res_lvl, apply_m, rm_edge, prod_depth, prod_norm, en_dl
            ),
            inputs=[input_image, max_size_input, resolution_level, apply_mask, remove_edges, produce_depth, produce_normal, enable_download],
            outputs=[results, depth_map, normal_map, model_3d, files, fov, viewer_message, depth_message, timings]
        ).then(
            fn=reset_measure,
            inputs=[results],
            outputs=[measure_image, measure_points, measure_text]
        )

        # Measurement functionality
        measure_image.select(
            fn=measure,
            inputs=[results, measure_points, enable_measure],
            outputs=[measure_image, measure_points, measure_text]
        )
    
    return demo


def _run_inference(inference: MoGeInference, image: np.ndarray, max_size: int, 
                   resolution_level: str, apply_mask: bool, remove_edge: bool, 
                   produce_depth: bool, produce_normal: bool, enable_download: bool):
    """
    Run inference and format results for UI display
    
    Returns:
        Tuple corresponding to each Gradio output
    """
    # Run inference
    result = inference.run(
        image,
        max_size=int(max_size),
        resolution_level=resolution_level,
        apply_mask=apply_mask,
        remove_edge=remove_edge,
        produce_depth=produce_depth,
        produce_normal=produce_normal,
        enable_download=enable_download
    )

    # Format for UI
    depth_vis_display = result['depth_vis'] if result['depth_vis'] is not None else gr.update(label="Depth map (disabled)")
    normal_vis_display = result['normal_vis'] if result['normal_vis'] is not None else gr.update(label="Normal map (not available or disabled)")

    # Generate messages
    viewer_message = '**Note:** Inference has been completed. It may take a few seconds to download the 3D model.'
    if resolution_level != 'Ultra':
        depth_message = f'**Note:** Want sharper depth map? Try increasing the `maximum image size` and setting the `inference resolution level` to `Ultra` in the settings.'
    else:
        depth_message = ""

    # Format timing information
    timings = result['timings']
    timings_text = (
        f"- Preprocess: {timings['preprocess_s']:.3f}s\n"
        f"- Inference: {timings['inference_s']:.3f}s\n"
        f"- Visualization: {timings['visualization_s']:.3f}s\n"
        f"- Export: {timings['export_s']:.3f}s\n"
        f"- Total: {timings['total_s']:.3f}s"
    )

    # Display FOV
    fov_x, fov_y = result['fov']
    fov_text = f'- **Horizontal FOV: {fov_x:.1f}°**. \n - **Vertical FOV: {fov_y:.1f}°'

    return (
        result,
        depth_vis_display,
        normal_vis_display,
        result['model_3d_file'],
        result['output_files'],
        fov_text,
        viewer_message,
        depth_message,
        timings_text
    )
