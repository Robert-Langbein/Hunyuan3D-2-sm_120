# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import argparse
import os
import time
import uuid
from typing import List

import gradio as gr
import torch
import trimesh
from PIL import Image

from hy3dgen.config import DeviceConfig, get_texture_quality_config
from hy3dgen.texgen import Hunyuan3DPaintPipeline


SAVE_DIR = None
TEXTURE_DEVICE = None
TEXTURE_QUALITY_CONFIG = None
texgen_worker: Hunyuan3DPaintPipeline


def ensure_dir() -> str:
    os.makedirs(SAVE_DIR, exist_ok=True)
    return SAVE_DIR


def load_images(image_paths: List[str]) -> List[Image.Image]:
    outputs: List[Image.Image] = []
    for path in image_paths:
        outputs.append(Image.open(path))
    return outputs


def apply_runtime_quality(pipeline: Hunyuan3DPaintPipeline, max_num_view: int, texture_resolution: int):
    pipeline.config.max_num_view = int(max_num_view)
    pipeline.config._limit_views(int(max_num_view))

    pipeline.config.texture_size = texture_resolution
    pipeline.config.render_size = texture_resolution
    pipeline.render.set_default_texture_resolution(texture_resolution)
    pipeline.render.set_default_render_resolution(texture_resolution)


def texture_mesh(mesh_file, reference_images, max_num_view, texture_resolution):
    if mesh_file is None:
        raise gr.Error("Please upload a mesh (glb/obj/ply/stl).")
    if reference_images is None or len(reference_images) == 0:
        raise gr.Error("Please provide at least one reference image.")

    mesh_path = mesh_file if isinstance(mesh_file, str) else mesh_file.name
    mesh = trimesh.load(mesh_path)

    image_paths = reference_images if isinstance(reference_images, list) else [reference_images]
    images = load_images(image_paths)

    apply_runtime_quality(texgen_worker, int(max_num_view), int(texture_resolution))

    start_time = time.time()
    textured_mesh = texgen_worker(mesh, image=images)
    elapsed = time.time() - start_time

    ensure_dir()
    run_dir = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, "textured_mesh.glb")
    textured_mesh.export(output_path, include_normals=True)

    if TEXTURE_QUALITY_CONFIG.low_vram_mode and torch.cuda.is_available():
        torch.cuda.empty_cache()

    stats = {
        "device": TEXTURE_DEVICE,
        "max_num_view": int(max_num_view),
        "texture_resolution": int(texture_resolution),
        "render_resolution": int(texture_resolution),
        "time_sec": elapsed,
        "output_path": output_path,
    }
    return output_path, output_path, stats


def build_app():
    with gr.Blocks(title="Hunyuan3D Texture Only", analytics_enabled=False) as demo:
        gr.Markdown("# Hunyuan3D Texture Generation\nUpload an existing mesh and one or more reference images.")
        with gr.Row():
            with gr.Column(scale=3):
                mesh_input = gr.File(label="Mesh (glb/obj/ply/stl)", file_types=[".glb", ".obj", ".ply", ".stl"],
                                     type="filepath")
                image_input = gr.File(label="Reference Images", file_count="multiple", type="filepath",
                                      file_types=[".png", ".jpg", ".jpeg"])
                max_view = gr.Slider(minimum=1, maximum=6, step=1,
                                     value=TEXTURE_QUALITY_CONFIG.max_num_view, label="Max Views")
                texture_res = gr.Slider(minimum=256, maximum=2048, step=64,
                                        value=TEXTURE_QUALITY_CONFIG.texture_size, label="Texture Resolution")
                run_btn = gr.Button("Generate Texture", variant="primary")
            with gr.Column(scale=5):
                preview = gr.Model3D(label="Textured Preview")
                download = gr.File(label="Download Textured Mesh")
                stats = gr.Json(label="Run Info")

        run_btn.click(
            texture_mesh,
            inputs=[mesh_input, image_input, max_view, texture_res],
            outputs=[preview, download, stats],
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--texgen_model_path", type=str, default="tencent/Hunyuan3D-2")
    parser.add_argument("--texture_device", type=str, default="cuda:0")
    parser.add_argument("--texture_quality", type=str, default="standard",
                        choices=["standard", "balanced", "low_vram", "high"])
    parser.add_argument("--max_num_view", type=int, default=None)
    parser.add_argument("--texture_resolution", type=int, default=None)
    parser.add_argument("--render_resolution", type=int, default=None)
    parser.add_argument("--low_vram_mode", action="store_true")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--cache-path", type=str, default="texture_cache")
    args = parser.parse_args()

    device_config = DeviceConfig(
        shape_device=args.texture_device,
        texture_device=args.texture_device,
    )
    TEXTURE_DEVICE = device_config.resolved_texture()
    TEXTURE_QUALITY_CONFIG = get_texture_quality_config(
        preset=args.texture_quality,
        max_num_view=args.max_num_view,
        texture_size=args.texture_resolution,
        render_size=args.render_resolution,
        low_vram_mode=args.low_vram_mode,
    )

    texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(
        args.texgen_model_path,
        device=TEXTURE_DEVICE,
        texture_quality=args.texture_quality,
        max_num_view=args.max_num_view,
        texture_size=args.texture_resolution,
        render_size=args.render_resolution,
        low_vram_mode=args.low_vram_mode,
    )
    if args.low_vram_mode:
        texgen_worker.enable_model_cpu_offload(device=TEXTURE_DEVICE)

    SAVE_DIR = args.cache_path
    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=False)

