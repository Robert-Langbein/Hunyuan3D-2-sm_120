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
from typing import List

import trimesh
from PIL import Image

from hy3dgen.config import get_texture_quality_config
from hy3dgen.texgen import Hunyuan3DPaintPipeline


def load_images(image_paths: List[str]) -> List[Image.Image]:
    outputs: List[Image.Image] = []
    for path in image_paths:
        outputs.append(Image.open(path))
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Texture an existing mesh with Hunyuan3D-Paint.")
    parser.add_argument("--mesh_path", type=str, required=True, help="Input mesh path (glb/obj/ply/stl).")
    parser.add_argument("--reference_image", action="append", required=True,
                        help="Reference image path. Can be passed multiple times.")
    parser.add_argument("--output_path", type=str, default="textured_output.glb", help="Where to save the textured mesh.")
    parser.add_argument("--texgen_model_path", type=str, default="tencent/Hunyuan3D-2")
    parser.add_argument("--texture_device", type=str, default="cuda:0")
    parser.add_argument("--texture_quality", type=str, default="standard",
                        choices=["standard", "balanced", "low_vram", "high"])
    parser.add_argument("--max_num_view", type=int, default=None)
    parser.add_argument("--texture_resolution", type=int, default=None)
    parser.add_argument("--render_resolution", type=int, default=None)
    parser.add_argument("--low_vram_mode", action="store_true")
    args = parser.parse_args()

    quality = get_texture_quality_config(
        preset=args.texture_quality,
        max_num_view=args.max_num_view,
        texture_size=args.texture_resolution,
        render_size=args.render_resolution,
        low_vram_mode=args.low_vram_mode,
    )

    pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(
        args.texgen_model_path,
        device=args.texture_device,
        texture_quality=args.texture_quality,
        max_num_view=args.max_num_view,
        texture_size=args.texture_resolution,
        render_size=args.render_resolution,
        low_vram_mode=args.low_vram_mode,
    )
    if args.low_vram_mode:
        pipeline_texgen.enable_model_cpu_offload(device=args.texture_device)

    mesh = trimesh.load(args.mesh_path)
    images = load_images(args.reference_image)

    textured_mesh = pipeline_texgen(mesh, image=images)
    textured_mesh.export(args.output_path, include_normals=True)

    print(f"Textured mesh saved to {os.path.abspath(args.output_path)}")
    print(f"Device (texture): {args.texture_device}")
    print(f"Quality preset: {args.texture_quality}, views: {quality.max_num_view}, "
          f"resolution: {quality.texture_size}")


if __name__ == "__main__":
    main()

