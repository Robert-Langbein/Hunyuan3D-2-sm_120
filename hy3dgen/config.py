from dataclasses import dataclass
from typing import Optional

import torch


def resolve_device(device: Optional[str], allow_cpu_fallback: bool = True) -> str:
    """
    Resolve a requested device string into something usable by torch, optionally
    falling back to CPU when the target accelerator is not available.
    """
    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = device or ""

    normalized = device_str.lower().replace(" ", "")

    if normalized in ("", "auto"):
        if torch.cuda.is_available():
            return "cuda:0" if torch.cuda.device_count() > 0 else "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if normalized.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                index = torch.device(normalized).index
            except Exception:
                index = None
            if index is None:
                return "cuda"
            if index < torch.cuda.device_count():
                return f"cuda:{index}"
        return "cpu" if allow_cpu_fallback else normalized

    if normalized.startswith("mps"):
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu" if allow_cpu_fallback else normalized

    return device_str or "cpu"


@dataclass
class DeviceConfig:
    shape_device: str = "cuda:0"
    texture_device: Optional[str] = None
    allow_cpu_fallback: bool = True

    def resolved_shape(self) -> str:
        return resolve_device(self.shape_device, allow_cpu_fallback=self.allow_cpu_fallback)

    def resolved_texture(self) -> str:
        target = self.texture_device or self.shape_device
        resolved = resolve_device(target, allow_cpu_fallback=self.allow_cpu_fallback)
        return resolved


@dataclass
class TextureQualityConfig:
    max_num_view: int
    texture_size: int
    render_size: int
    low_vram_mode: bool


def _base_texture_quality() -> TextureQualityConfig:
    return TextureQualityConfig(
        max_num_view=6,
        texture_size=2048,
        render_size=2048,
        low_vram_mode=False,
    )


TEXTURE_QUALITY_PRESETS = {
    "standard": _base_texture_quality(),
    "balanced": TextureQualityConfig(
        max_num_view=4,
        texture_size=1536,
        render_size=1536,
        low_vram_mode=False,
    ),
    "low_vram": TextureQualityConfig(
        max_num_view=3,
        texture_size=768,
        render_size=768,
        low_vram_mode=True,
    ),
    "high": TextureQualityConfig(
        max_num_view=6,
        texture_size=2048,
        render_size=2048,
        low_vram_mode=False,
    ),
}


def get_texture_quality_config(
    preset: str = "standard",
    max_num_view: Optional[int] = None,
    texture_size: Optional[int] = None,
    render_size: Optional[int] = None,
    low_vram_mode: bool = False,
) -> TextureQualityConfig:
    base = TEXTURE_QUALITY_PRESETS.get(preset, TEXTURE_QUALITY_PRESETS["standard"])
    config = TextureQualityConfig(
        max_num_view=max_num_view or base.max_num_view,
        texture_size=texture_size or base.texture_size,
        render_size=render_size or base.render_size,
        low_vram_mode=low_vram_mode or base.low_vram_mode,
    )
    config.max_num_view = max(1, min(config.max_num_view, 6))
    return config

