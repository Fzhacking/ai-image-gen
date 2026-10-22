"""
Video and music generation (the "video" and "music" model families).

Video (diffusers):
    ltx      LTX-Video 2B (Lightricks)        text-to-video and image-to-video, 24 fps, 768x512 and friends,
                                              about 1-2 minutes for 4 s on an RTX 3090; ~28 GB download
    wan      Wan 2.1 T2V 1.3B (Wan-AI)        text-to-video, 16 fps, 832x480, better motion, several minutes

Music:
    musicgen        MusicGen stereo medium (Meta, transformers)   text-to-music, 32 kHz stereo, up to 30 s
    musicgen-small  MusicGen stereo small                          same, faster, lower quality
    stable-audio    Stable Audio Open 1.0 (diffusers, gated)       44.1 kHz stereo, up to 47 s  (untested here)

All of them are ordinary ModelSpec entries (models.py / registry.json) so the Admin panel lists, downloads,
edits and loads them like any image model. MediaManager holds one loaded video or music pipeline; the image
ModelManager and the MediaManager share one GPU, so prepare_for() unloads whatever would not fit next to the
model about to be used (video and image models never share the card; a small music model may).

    from media import MEDIA_FAMILIES, MediaManager, VIDEO_SIZES
"""

from __future__ import annotations

import gc
import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

import models
from models import ModelSpec

# bitsandbytes logs "inputs will be cast from bfloat16 to float16" on every 8-bit matmul (hundreds of lines per clip)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

MEDIA_FAMILIES = {"video": "video", "music": "music"}     # family -> kind

# Sizes every video model here accepts (multiples of 32 for LTX, 16 for Wan).
VIDEO_SIZES = [
    ("768 x 512  ·  3:2 landscape (default)", "768x512"),
    ("512 x 768  ·  2:3 portrait", "512x768"),
    ("832 x 480  ·  16:9-ish, Wan native", "832x480"),
    ("480 x 832  ·  9:16 portrait", "480x832"),
    ("1024 x 576  ·  16:9, slower", "1024x576"),
    ("576 x 1024  ·  9:16 tall, slower", "576x1024"),
]
VIDEO_UPSCALE_CHOICES = [("Full HD (1920 px)", 1920), ("2K (2560 px)", 2560), ("4K (3840 px)", 3840)]

WAN_NEGATIVE = ("bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
                "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
                "still picture, messy background, three legs, many people in the background, walking backwards")
LTX_NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted, watermark, text"

# Per-architecture behaviour. "arch" comes from the repo's model_index.json / config.json (see arch_of).
PROFILES: dict[str, dict] = {
    "ltx": dict(kind="video", label="LTX-Video", fps=24, frame_mult=8, max_seconds=10, steps=40, guidance=3.0,
                negative=LTX_NEGATIVE, image_to_video=True, vram_gb=16,
                note="fast; describe the motion and the camera in one or two sentences"),
    "wan": dict(kind="video", label="Wan 2.1", fps=16, frame_mult=4, max_seconds=6, steps=30, guidance=5.0,
                negative=WAN_NEGATIVE, image_to_video=False, vram_gb=17,
                note="slower, smoother motion; 832x480 is its native size"),
    "musicgen": dict(kind="music", label="MusicGen", max_seconds=30, min_seconds=3, steps=0, guidance=3.0,
                     negative="", vram_gb=5, sample_rate=32000, note="describe genre, mood, instruments and tempo"),
    "stable_audio": dict(kind="music", label="Stable Audio Open", max_seconds=47, min_seconds=3, steps=100, guidance=7.0,
                         negative="low quality, average quality, noise, distortion", vram_gb=8, sample_rate=44100,
                         note="gated on Hugging Face; good for sound effects and loops"),
}

PIPELINE_ARCH = {"LTXPipeline": "ltx", "LTXImageToVideoPipeline": "ltx", "LTXConditionPipeline": "ltx",
                 "WanPipeline": "wan", "WanImageToVideoPipeline": "wan", "StableAudioPipeline": "stable_audio"}
CONFIG_ARCH = {"musicgen": "musicgen", "musicgen_melody": "musicgen"}

_arch_cache: dict[str, str | None] = {}


def kind_of(family: str) -> str:
    return MEDIA_FAMILIES.get(family, "image")


def arch_of(spec: ModelSpec) -> str | None:
    """Which loader a video/music repo needs: read from model_index.json (diffusers) or config.json (transformers).
    Built-in keys are known without touching the network; others are read from the HF cache or downloaded (tiny files)."""
    builtin = {"ltx": "ltx", "wan": "wan", "musicgen": "musicgen", "musicgen-small": "musicgen", "stable-audio": "stable_audio"}
    if spec.key in builtin:
        return builtin[spec.key]
    if spec.source in _arch_cache:
        return _arch_cache[spec.source]
    arch = None
    try:
        from huggingface_hub import hf_hub_download
        from downloads import hf_token
        token = hf_token() or None
        try:
            idx = json.loads(Path(hf_hub_download(spec.source, "model_index.json", token=token)).read_text(encoding="utf-8"))
            arch = PIPELINE_ARCH.get(idx.get("_class_name", ""))
        except Exception:
            cfg = json.loads(Path(hf_hub_download(spec.source, "config.json", token=token)).read_text(encoding="utf-8"))
            arch = CONFIG_ARCH.get(cfg.get("model_type", ""))
    except Exception:
        arch = None
    _arch_cache[spec.source] = arch
    return arch


def profile_for(spec: ModelSpec) -> dict:
    """Profile with the model's own steps/guidance substituted (admins edit those in the registry)."""
    arch = arch_of(spec) or ("ltx" if spec.family == "video" else "musicgen")
    p = dict(PROFILES[arch])
    p["arch"] = arch
    if spec.steps:
        p["steps"] = spec.steps
    if spec.guidance is not None:
        p["guidance"] = spec.guidance
    return p


def repo_allow_patterns(spec: ModelSpec) -> list[str]:
    """What to fetch from a video/music repo: diffusers sub-folders only (LTX's repo also carries 250 GB of
    single-file checkpoints at its root), or the transformers files for MusicGen."""
    if arch_of(spec) == "musicgen":
        return ["*.json", "model.safetensors", "*.txt", "*.model"]
    return ["model_index.json", "*/*"]


def frames_for(seconds: float, profile: dict) -> int:
    m = profile["frame_mult"]
    n = max(1, round(seconds * profile["fps"]))
    return max(m + 1, round((n - 1) / m) * m + 1)    # nearest valid count: LTX 8k+1 (4 s -> 97), Wan 4k+1 (4 s -> 65)


def parse_size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def free_vram_gb() -> float:
    import torch
    free, _total = torch.cuda.mem_get_info()
    return free / 1024**3


class MediaManager:
    """One loaded video or music model at a time."""

    def __init__(self, lock: threading.RLock, device: str = "cuda"):
        self.lock = lock
        self.device = device
        self.spec: ModelSpec | None = None
        self.pipe = None
        self.arch: str | None = None
        self.kind: str | None = None
        self.profile: dict = {}
        self._i2v = None            # LTX image-to-video pipeline sharing the loaded components
        self._processor = None      # MusicGen processor

    # --- loading -------------------------------------------------------------

    def load(self, model: str | ModelSpec, log=print):
        spec = models.resolve(model)
        if self.pipe is not None and self.spec and self.spec.key == spec.key:
            return self.pipe
        self.unload()
        prof = profile_for(spec)
        t0 = time.time()
        if not models.is_downloaded(spec):
            log(f"Downloading {spec.name} (~{spec.download_gb:g} GB, first time only) ...")
        log(f"Loading {spec.name} ({prof['label']}) ...")
        loader = {"ltx": self._load_ltx, "wan": self._load_wan, "musicgen": self._load_musicgen,
                  "stable_audio": self._load_stable_audio}[prof["arch"]]
        pipe = loader(spec, log)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        self.spec, self.pipe, self.arch, self.kind, self.profile = spec, pipe, prof["arch"], prof["kind"], prof
        log(f"{spec.name} ready in {time.time() - t0:.1f}s")
        return pipe

    def _text_encoder(self, spec: ModelSpec, cls, log, eight_bit: bool = False):
        """Text encoder straight onto the GPU in bf16. The repos ship it in fp32 (19-23 GB, more than this machine's
        free RAM), so the first load converts shard by shard and saves a bf16 copy under models/cache/ for later loads.
        eight_bit=True quantises with bitsandbytes (half the VRAM; used for Wan's 11 GB UMT5-XXL on 24 GB cards)."""
        import torch
        bf16_dir = models.MODELS_DIR / "cache" / f"{spec.key}-text_encoder-bf16"
        kw: dict = dict(torch_dtype=torch.bfloat16, device_map=self.device)
        if eight_bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        if (bf16_dir / "config.json").exists():
            log(f"  text encoder (bf16 copy{', 8-bit' if eight_bit else ''}) ...")
            return cls.from_pretrained(bf16_dir, **kw)
        log("  text encoder (converting fp32 -> bf16, first load only) ...")
        te = cls.from_pretrained(spec.source, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map=self.device)
        try:
            te.save_pretrained(bf16_dir, max_shard_size="4GB")
            log(f"  saved a bf16 copy to {bf16_dir} for faster loads")
        except Exception as e:   # the copy is an optimisation only
            log(f"  could not save the bf16 copy ({e})")
        if eight_bit:
            # quantise from the fresh copy so this load is as small as later ones
            del te
            gc.collect()
            torch.cuda.empty_cache()
            if (bf16_dir / "config.json").exists():
                return cls.from_pretrained(bf16_dir, **kw)
            return cls.from_pretrained(spec.source, subfolder="text_encoder", **kw)
        return te

    def _load_ltx(self, spec: ModelSpec, log):
        import torch
        from diffusers import AutoencoderKLLTXVideo, LTXPipeline, LTXVideoTransformer3DModel
        from transformers import T5EncoderModel
        dtype = torch.bfloat16
        te = self._text_encoder(spec, T5EncoderModel, log)
        log("  transformer ...")
        tr = LTXVideoTransformer3DModel.from_pretrained(spec.source, subfolder="transformer", torch_dtype=dtype, device_map=self.device)
        vae = AutoencoderKLLTXVideo.from_pretrained(spec.source, subfolder="vae", torch_dtype=dtype, device_map=self.device)
        pipe = LTXPipeline.from_pretrained(spec.source, text_encoder=te, transformer=tr, vae=vae, torch_dtype=dtype)
        pipe.to(self.device)
        pipe.vae.enable_tiling()
        return pipe

    def _load_wan(self, spec: ModelSpec, log):
        import torch
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
        from transformers import UMT5EncoderModel
        dtype = torch.bfloat16
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # UMT5-XXL is 11 GB in bf16; in 8-bit it is 6 GB, which leaves room for the fp32 VAE decode on a 24 GB card
        # (bf16 encoder + fp32 decode of 49 frames paged VRAM and took over 10 minutes).
        te = self._text_encoder(spec, UMT5EncoderModel, log, eight_bit=vram_gb < 32)
        log("  transformer ...")
        tr = WanTransformer3DModel.from_pretrained(spec.source, subfolder="transformer", torch_dtype=dtype, device_map=self.device)
        vae = AutoencoderKLWan.from_pretrained(spec.source, subfolder="vae", torch_dtype=torch.float32, device_map=self.device)
        pipe = WanPipeline.from_pretrained(spec.source, text_encoder=te, transformer=tr, vae=vae, torch_dtype=dtype)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)   # 3.0 for 480p
        pipe.to(self.device)
        pipe.vae.enable_tiling()
        return pipe

    def _load_musicgen(self, spec: ModelSpec, log):
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(spec.source)
        model = MusicgenForConditionalGeneration.from_pretrained(spec.source, torch_dtype=torch.float16, device_map=self.device)
        model.eval()
        return model

    def _load_stable_audio(self, spec: ModelSpec, log):
        import torch
        from diffusers import StableAudioPipeline
        pipe = StableAudioPipeline.from_pretrained(spec.source, torch_dtype=torch.float16)
        pipe.to(self.device)
        return pipe

    def unload(self) -> None:
        if self.pipe is None:
            return
        import torch
        self.pipe = None
        self._i2v = None
        self._processor = None
        self.spec, self.arch, self.kind, self.profile = None, None, None, {}
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def loaded_name(self) -> str | None:
        return self.spec.name if self.spec else None

    # --- video ---------------------------------------------------------------

    def generate_video(self, prompt: str, negative: str, width: int, height: int, frames: int, steps: int,
                       guidance: float, seed: int, image=None, on_step: Callable[[int, int], None] | None = None) -> list:
        """Returns a list of PIL frames."""
        import torch
        if self.kind != "video":
            raise RuntimeError("no video model is loaded")
        gen = torch.Generator(device=self.device).manual_seed(int(seed))

        def cb(pipe, step, timestep, kwargs):
            if on_step:
                on_step(step + 1, steps)
            return kwargs

        kw = dict(prompt=prompt, negative_prompt=negative or None, width=width, height=height, num_frames=frames,
                  num_inference_steps=steps, guidance_scale=float(guidance), generator=gen, callback_on_step_end=cb)
        if self.arch == "ltx":
            kw.update(frame_rate=self.profile["fps"], decode_timestep=0.03, decode_noise_scale=0.025)
            if image is not None:
                if self._i2v is None:
                    # share the loaded components (no copies, nothing moved); the VAE keeps its tiling setting
                    from diffusers import LTXImageToVideoPipeline
                    self._i2v = LTXImageToVideoPipeline(**self.pipe.components)
                    self._i2v.set_progress_bar_config(disable=True)
                torch.cuda.empty_cache()
                from diffusers.utils import load_image
                img = load_image(image) if isinstance(image, str) else image
                img = img.convert("RGB").resize((width, height))
                out = self._i2v(image=img, **kw)
            else:
                out = self.pipe(**kw)
        elif self.arch == "wan":
            if image is not None:
                raise RuntimeError("Wan 2.1 1.3B is text-to-video only; use LTX-Video to animate an image")
            out = self.pipe(**kw, output_type="pil")
        else:
            raise RuntimeError(f"{self.arch} does not generate video")
        return list(out.frames[0])

    # --- music ---------------------------------------------------------------

    def generate_music(self, prompt: str, negative: str, seconds: float, guidance: float, temperature: float, seed: int,
                       on_progress: Callable[[float], None] | None = None) -> tuple[np.ndarray, int]:
        """Returns (samples (n, channels) float32 in -1..1, sample_rate)."""
        import torch
        if self.kind != "music":
            raise RuntimeError("no music model is loaded")
        torch.manual_seed(int(seed))
        if self.arch == "musicgen":
            from transformers import StoppingCriteria, StoppingCriteriaList
            model = self.pipe
            fr = model.config.audio_encoder.frame_rate            # 50 tokens per second
            sr = model.config.audio_encoder.sampling_rate
            max_tokens = int(round(seconds * fr))
            inputs = self._processor(text=[prompt], padding=True, return_tensors="pt").to(self.device)

            class _Progress(StoppingCriteria):
                def __init__(self):
                    super().__init__()
                    self.n = 0

                def __call__(self, input_ids, scores, **kwargs):
                    self.n += 1
                    if on_progress:
                        on_progress(min(1.0, self.n / max_tokens))
                    return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

            with torch.inference_mode():
                audio = model.generate(**inputs, do_sample=True, guidance_scale=float(guidance), temperature=float(temperature),
                                       max_new_tokens=max_tokens, stopping_criteria=StoppingCriteriaList([_Progress()]))
            wav = audio[0].float().cpu().numpy().T          # (samples, channels)
            return wav, int(sr)
        if self.arch == "stable_audio":
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
            steps = int(self.profile["steps"]) or 100
            out = self.pipe(prompt=prompt, negative_prompt=negative or None, num_inference_steps=steps,
                            audio_end_in_s=float(seconds), num_waveforms_per_prompt=1, guidance_scale=float(guidance), generator=gen)
            wav = out.audios[0].T.float().cpu().numpy()
            return wav, int(self.pipe.vae.sampling_rate)
        raise RuntimeError(f"{self.arch} does not generate music")


# ----------------------------------------------------------------------------- GPU coordination

def prepare_for(spec: ModelSpec, image_mgr, media_mgr: MediaManager, log=print) -> list[str]:
    """Free the GPU for `spec`: video and image models never share the card; a music model stays next to an
    image model only while enough VRAM is free. Returns notes about what was unloaded."""
    notes = []
    kind = kind_of(spec.family)
    if kind == "image":
        if media_mgr.spec and media_mgr.kind == "video":
            notes.append(f"unloaded {media_mgr.spec.name} to make room for {spec.name}")
            media_mgr.unload()
        elif media_mgr.spec and media_mgr.kind == "music" and spec.family in getattr(image_mgr, "HEAVY_FAMILIES", ()):
            notes.append(f"unloaded {media_mgr.spec.name} (the big image models need the whole card)")
            media_mgr.unload()
    else:
        need = profile_for(spec).get("vram_gb", 16)
        if media_mgr.spec and media_mgr.spec.key != spec.key:
            notes.append(f"unloaded {media_mgr.spec.name}")
            media_mgr.unload()
        if image_mgr.spec and (kind == "video" or free_vram_gb() < need + 2):
            notes.append(f"unloaded {image_mgr.spec.name} to make room for {spec.name}")
            image_mgr.unload()
    for n in notes:
        log(n)
    return notes


if __name__ == "__main__":
    for m in models.all_models(include_disabled=True):
        if kind_of(m.family) != "image":
            p = profile_for(m)
            print(f"{m.key:14} {m.name:28} {p['kind']:5} {p['arch']:12} {'on disk' if models.is_downloaded(m) else 'not downloaded'}")
