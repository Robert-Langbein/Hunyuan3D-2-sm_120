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

"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import traceback
import uuid
from io import BytesIO

import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

from hy3dgen.config import DeviceConfig, get_texture_quality_config
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer, \
    MeshSimplifier
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline

LOGDIR = '.'

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


class ModelWorker:
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2mini',
                 tex_model_path='tencent/Hunyuan3D-2',
                 subfolder='hunyuan3d-dit-v2-mini-turbo',
                 device='cuda',
                 shape_device=None,
                 texture_device=None,
                 enable_tex=False,
                 enable_shape=True,
                 texture_quality: str = "standard",
                 max_num_view: int = None,
                 texture_resolution: int = None,
                 render_resolution: int = None,
                 low_vram_mode: bool = False):
        self.model_path = model_path
        self.worker_id = worker_id
        self.enable_shape = enable_shape

        device_config = DeviceConfig(
            shape_device=shape_device or device,
            texture_device=texture_device or device,
        )
        self.shape_device = device_config.resolved_shape()
        self.texture_device = device_config.resolved_texture()
        self.tex_quality = get_texture_quality_config(
            preset=texture_quality,
            max_num_view=max_num_view,
            texture_size=texture_resolution,
            render_size=render_resolution,
            low_vram_mode=low_vram_mode,
        )
        logger.info(f"Loading the model {model_path} on worker {worker_id} ...")

        self.rembg = BackgroundRemover()
        if self.enable_shape:
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                model_path,
                subfolder=subfolder,
                use_safetensors=True,
                device=self.shape_device,
            )
            self.pipeline.enable_flashvdm(mc_algo='mc')
            # self.pipeline_t2i = HunyuanDiTPipeline(
            #     'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
            #     device=self.shape_device
            # )
        if enable_tex:
            # Remote-Texturierung im Texture-Container: nutze stabiles Paint-Modell,
            # lade jedoch Binär-Gewichte (pytorch_model.bin) statt safetensors,
            # da der Text-Encoder nur als .bin vorliegt.
            self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(
                tex_model_path,
                subfolder="hunyuan3d-paint-v2-0",
                device=self.texture_device,
                texture_quality=texture_quality,
                max_num_view=max_num_view,
                texture_size=texture_resolution,
                render_size=render_resolution,
                low_vram_mode=low_vram_mode,
                use_safetensors=False,
            )
            if low_vram_mode:
                self.pipeline_tex.enable_model_cpu_offload(device=self.texture_device)

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        if 'image' in params:
            image = params["image"]
            image = load_image_from_base64(image)
        else:
            if 'text' in params:
                text = params["text"]
                image = self.pipeline_t2i(text)
            else:
                raise ValueError("No input image or text provided")

        image = self.rembg(image)
        params['image'] = image

        if 'mesh' in params:
            mesh = trimesh.load(BytesIO(base64.b64decode(params["mesh"])), file_type='glb')
        else:
            if not self.enable_shape:
                raise ValueError("Shape generation disabled; please provide a mesh.")
            seed = params.get("seed", 1234)
            generator_device = "cpu"
            if self.shape_device.startswith("cuda") and torch.cuda.is_available():
                generator_device = self.shape_device
            elif self.shape_device == "mps":
                generator_device = "mps"
            params['generator'] = torch.Generator(device=generator_device).manual_seed(seed)
            params['octree_resolution'] = params.get("octree_resolution", 128)
            params['num_inference_steps'] = params.get("num_inference_steps", 5)
            params['guidance_scale'] = params.get('guidance_scale', 5.0)
            params['mc_algo'] = 'mc'
            import time
            start_time = time.time()
            mesh = self.pipeline(**params)[0]
            logger.info("--- %s seconds ---" % (time.time() - start_time))

        if params.get('texture', False):
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh, max_facenum=params.get('face_count', 40000))
            if not hasattr(self, "pipeline_tex"):
                raise ValueError("Texture generation disabled; please enable_tex or disable `texture` flag.")
            mesh = self.pipeline_tex(mesh, image)

        type = params.get('type', 'glb')
        with tempfile.NamedTemporaryFile(suffix=f'.{type}', delete=False) as temp_file:
            mesh.export(temp_file.name)
            mesh = trimesh.load(temp_file.name)
            save_path = os.path.join(SAVE_DIR, f'{str(uid)}.{type}')
            mesh.export(save_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return save_path, uid


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 你可以指定允许的来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)


@app.post("/generate")
async def generate(request: Request):
    logger.info("Worker generating...")
    params = await request.json()
    uid = uuid.uuid4()
    try:
        file_path, uid = worker.generate(uid, params)
        return FileResponse(file_path)
    except ValueError as e:
        traceback.print_exc()
        print("Caught ValueError:", e)
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except torch.cuda.CudaError as e:
        print("Caught torch.cuda.CudaError:", e)
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except Exception as e:
        print("Caught Unknown Error", e)
        traceback.print_exc()
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)


@app.post("/send")
async def generate(request: Request):
    logger.info("Worker send...")
    params = await request.json()
    uid = uuid.uuid4()
    threading.Thread(target=worker.generate, args=(uid, params,)).start()
    ret = {"uid": str(uid)}
    return JSONResponse(ret, status_code=200)


@app.get("/status/{uid}")
async def status(uid: str):
    save_file_path = os.path.join(SAVE_DIR, f'{uid}.glb')
    print(save_file_path, os.path.exists(save_file_path))
    if not os.path.exists(save_file_path):
        response = {'status': 'processing'}
        return JSONResponse(response, status_code=200)
    else:
        base64_str = base64.b64encode(open(save_file_path, 'rb').read()).decode()
        response = {'status': 'completed', 'model_base64': base64_str}
        return JSONResponse(response, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--tex_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Deprecated fallback. Prefer --shape_device/--texture_device.")
    parser.add_argument("--shape_device", type=str, default=None,
                        help="Device for shape generation (e.g. cuda:0).")
    parser.add_argument("--texture_device", type=str, default=None,
                        help="Device for texture generation (e.g. cuda:1).")
    parser.add_argument("--texture_quality", type=str, default="standard",
                        choices=["standard", "balanced", "low_vram", "high"])
    parser.add_argument("--max_num_view", type=int, default=None)
    parser.add_argument("--texture_resolution", type=int, default=None)
    parser.add_argument("--render_resolution", type=int, default=None)
    parser.add_argument("--low_vram_mode", action="store_true")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument('--enable_tex', action='store_true')
    parser.add_argument('--disable_shape', action='store_true')
    parser.add_argument("--cache-path", type=str, default="gradio_cache",
                        help="Directory for temporary outputs returned by API.")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    # Update SAVE_DIR based on provided cache-path
    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    worker = ModelWorker(
        model_path=args.model_path,
        tex_model_path=args.tex_model_path,
        subfolder='hunyuan3d-dit-v2-mini-turbo',
        device=args.device,
        shape_device=args.shape_device,
        texture_device=args.texture_device,
        enable_tex=args.enable_tex,
        enable_shape=not args.disable_shape,
        texture_quality=args.texture_quality,
        max_num_view=args.max_num_view,
        texture_resolution=args.texture_resolution,
        render_resolution=args.render_resolution,
        low_vram_mode=args.low_vram_mode,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
