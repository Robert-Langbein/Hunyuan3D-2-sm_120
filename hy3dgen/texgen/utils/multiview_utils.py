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

import os
import random
import importlib
import logging

import numpy as np
import torch
from typing import List
from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, LCMScheduler


class Multiview_Diffusion_Net():
    def __init__(self, config) -> None:
        logger = logging.getLogger(__name__)

        self.device = config.device
        self.view_size = 512
        multiview_ckpt_path = config.multiview_ckpt_path
        current_file_path = os.path.abspath(__file__)
        custom_pipeline_path = os.path.join(os.path.dirname(current_file_path), '..', 'hunyuanpaint')

        logger.info(
            "Init Multiview_Diffusion_Net | device=%s, multiview_ckpt_path=%s, custom_pipeline_path=%s, pipe_name=%s",
            self.device,
            multiview_ckpt_path,
            custom_pipeline_path,
            getattr(config, "pipe_name", "unknown"),
        )

        # Workaround für ältere Hunyuan-HF-Module: dort wird in Basic2p5DTransformerBlock
        # auf ein Attribut `dim` zugegriffen, das es in neueren Diffusers-BasicTransformerBlock-
        # Implementierungen nicht mehr gibt. Wir patchen gezielt __getattr__ im HF-Modul,
        # damit für "dim" eine sinnvolle Dimension aus der Attention abgeleitet wird.
        try:
            hf_modules = importlib.import_module("diffusers_modules.local.modules")
            Basic2p5D = getattr(hf_modules, "Basic2p5DTransformerBlock", None)

            if Basic2p5D is not None:
                logger.info(
                    "Patching Basic2p5DTransformerBlock.__getattr__ in diffusers_modules.local.modules "
                    "für dynamische 'dim'-Berechnung."
                )
                original_getattr = Basic2p5D.__getattr__

                def _patched_getattr(self, name: str):
                    if name == "dim":
                        attn1 = getattr(self, "attn1", getattr(self, "transformer", None))
                        # Versuche, Kopfanzahl und Head-Dimension robust abzuleiten
                        num_heads = getattr(self, "num_attention_heads", getattr(attn1, "heads", 8))
                        q_proj = getattr(attn1, "to_q", None)
                        if q_proj is not None and hasattr(q_proj, "in_features"):
                            return q_proj.in_features
                        # Fallback: Kopfanzahl * Head-Dimension, wenn vorhanden
                        head_dim = getattr(attn1, "head_dim", None)
                        if head_dim is not None:
                            return num_heads * head_dim
                        # Letzter Fallback: eine konservative Standarddimension
                        return 1024

                    try:
                        return original_getattr(self, name)
                    except AttributeError:
                        # Wie im Original: weiter auf den wrapped Transformer durchreichen
                        return getattr(self.transformer, name)

                Basic2p5D.__getattr__ = _patched_getattr
        except ImportError:
            # Wenn das HF-Modul noch nicht existiert, wird es beim ersten Laden
            # von DiffusionPipeline.from_pretrained erzeugt; in diesem Fall gibt
            # es auch kein altes Basic2p5D, das Probleme machen könnte.
            logger.info("HF-Modul 'diffusers_modules.local.modules' noch nicht vorhanden; kein Patch nötig.")
        except Exception as exc:  # defensive: Logging für unerwartete Fehler
            logger.exception("Fehler beim Patchen von Basic2p5DTransformerBlock: %s", exc)

        logger.info(
            "Lade Multiview DiffusionPipeline.from_pretrained | path=%s, torch_dtype=float16, use_safetensors=False",
            multiview_ckpt_path,
        )

        pipeline = DiffusionPipeline.from_pretrained(
            multiview_ckpt_path,
            custom_pipeline=custom_pipeline_path,
            torch_dtype=torch.float16,
            use_safetensors=False,
            local_files_only=False,
        )

        if config.pipe_name in ['hunyuanpaint']:
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config,
                                                                             timestep_spacing='trailing')
        elif config.pipe_name in ['hunyuanpaint-turbo']:
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config,
                                                        timestep_spacing='trailing')
            pipeline.set_turbo(True)
            # pipeline.prepare() 

        pipeline.set_progress_bar_config(disable=True)
        self.pipeline = pipeline.to(self.device)

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    def __call__(self, input_images, control_images, camera_info):

        self.seed_everything(0)

        if not isinstance(input_images, List):
            input_images = [input_images]

        input_images = [input_image.resize((self.view_size, self.view_size)) for input_image in input_images]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((self.view_size, self.view_size))
            if control_images[i].mode == 'L':
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode='1')

        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        camera_info_gen = [camera_info]
        camera_info_ref = [[0]]
        kwargs['width'] = self.view_size
        kwargs['height'] = self.view_size
        kwargs['num_in_batch'] = num_view
        kwargs['camera_info_gen'] = camera_info_gen
        kwargs['camera_info_ref'] = camera_info_ref
        kwargs["normal_imgs"] = normal_image
        kwargs["position_imgs"] = position_image

        mvd_image = self.pipeline(input_images, num_inference_steps=30, **kwargs).images

        return mvd_image
