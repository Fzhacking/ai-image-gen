"""
Model registry and loader.

Built-in models (Hugging Face repos, downloaded on first use into the HF cache):

    sdxl         Stable Diffusion XL base 1.0        general purpose baseline
    juggernaut   Juggernaut XL v9                    photoreal people, products, scenes
    realvis      RealVisXL V5.0                      photoreal, skin and lighting detail
    dreamshaper  DreamShaper XL v2 Turbo             fast (6 steps) illustration / fantasy
    playground   Playground v2.5                     highly aesthetic, painterly-real
    animagine    Animagine XL 3.1                    anime
    zimage       Z-Image Turbo                       6B flow model, 8 steps, photoreal + text,
                                                     Apache 2.0, ~20 GB, runs on 24 GB with offload
    qwen         Qwen-Image + Lightning              20B model in 4-bit, 8 steps, best text
                                                     rendering; ~48 GB, slow first load
    flux         FLUX.1 [schnell]                    best prompt understanding, 4 steps
                                                     (gated: `huggingface-cli login` + accept
                                                     terms; ~33 GB; runs 4-bit on 24 GB cards)

Local models:
    models/checkpoints/*.safetensors    any SDXL single-file checkpoint (Civitai style)
    models/loras/*.safetensors          SDXL LoRAs, used with <lora:name:0.8> in the prompt

Prompt weighting (SDXL family, via compel):
    (word)1.3  (word)++  (word)--   or Automatic1111 style  (word:1.3)
Long prompts (>77 tokens) are not truncated.
"""

from __future__ import annotations

import gc
import re
import time
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LORAS_DIR = MODELS_DIR / "loras"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    source: str                       # Hugging Face repo id or local file path
    family: str = "sdxl"              # "sdxl" | "flux"
    steps: int = 30
    guidance: float = 7.0
    scheduler: str | None = "dpmpp_2m_karras"   # None = keep the scheduler shipped with the model
    variant: str | None = "fp16"      # HF weight variant to download
    negative: bool = True             # supports a negative prompt
    aliases: tuple[str, ...] = ()
    styles: tuple[str, ...] = ()      # prompt_parser style names this model is auto-picked for
    description: str = ""
    download_gb: float = 7.0
    local: bool = False


BUILTIN: list[ModelSpec] = [
    ModelSpec(
        key="sdxl", name="SDXL base 1.0", source="stabilityai/stable-diffusion-xl-base-1.0",
        steps=30, guidance=7.0, aliases=("base", "sdxl-base"),
        description="general purpose baseline",
    ),
    ModelSpec(
        key="juggernaut", name="Juggernaut XL v9", source="RunDiffusion/Juggernaut-XL-v9",
        steps=30, guidance=5.0, aliases=("jugg", "juggernautxl", "juggernaut-xl"),
        styles=("photorealistic", "cinematic"),
        description="photoreal people, products and scenes; best all-round upgrade",
    ),
    ModelSpec(
        key="realvis", name="RealVisXL V5.0", source="SG161222/RealVisXL_V5.0",
        steps=30, guidance=4.0, scheduler="dpmpp_sde_karras",
        aliases=("realvisxl", "realistic", "realvision"),
        description="photoreal with strong skin, fabric and lighting detail",
    ),
    ModelSpec(
        key="dreamshaper", name="DreamShaper XL v2 Turbo", source="Lykon/dreamshaper-xl-v2-turbo",
        steps=6, guidance=2.0, scheduler="dpmpp_sde_karras",
        aliases=("turbo", "dreamshaper-turbo", "dreamshaperxl"),
        styles=("digital painting", "comic book style", "cyberpunk"),
        description="fast 6-step model for illustration, fantasy and stylised art",
    ),
    ModelSpec(
        key="playground", name="Playground v2.5", source="playgroundai/playground-v2.5-1024px-aesthetic",
        steps=40, guidance=3.0, scheduler=None,
        aliases=("playground2.5", "aesthetic", "pg"),
        styles=("oil painting", "watercolor painting"),
        description="highly aesthetic, painterly-real; strong composition and colour",
    ),
    ModelSpec(
        key="animagine", name="Animagine XL 3.1", source="cagliostrolab/animagine-xl-3.1",
        steps=28, guidance=6.0, scheduler="euler_a", variant=None,
        aliases=("anime-xl", "animagine-xl", "animaginexl"),
        styles=("anime style",),
        description="anime and manga characters",
    ),
    ModelSpec(
        key="zimage", name="Z-Image Turbo", source="Tongyi-MAI/Z-Image-Turbo",
        family="zimage", steps=8, guidance=0.0, scheduler=None, variant=None, negative=False,
        aliases=("z-image", "z-image-turbo", "zimage-turbo", "zit"),
        description="6B flow model, 8 steps, photoreal and text rendering; Apache 2.0; ~20 GB",
        download_gb=20,
    ),
    ModelSpec(
        key="qwen", name="Qwen-Image (Lightning 8-step)", source="Qwen/Qwen-Image",
        family="qwen", steps=8, guidance=1.0, scheduler=None, variant=None, negative=False,
        aliases=("qwen-image", "qwenimage", "qwen-lightning"),
        description="20B Qwen-Image in 4-bit with the Lightning LoRA: best text rendering and layout; "
                    "~48 GB download, first load quantises for several minutes",
        download_gb=48,
    ),
    ModelSpec(
        key="flux", name="FLUX.1 [schnell]", source="black-forest-labs/FLUX.1-schnell",
        family="flux", steps=4, guidance=0.0, scheduler=None, variant=None, negative=False,
        aliases=("flux-schnell", "schnell", "flux1"),
        description="best prompt understanding (text, counting, complex scenes); 4 steps; "
                    "needs Hugging Face login, ~33 GB download",
        download_gb=33,
    ),
]

SCHEDULER_NAMES = {
    "dpmpp_2m_karras": "DPM++ 2M Karras",
    "dpmpp_sde_karras": "DPM++ SDE Karras",
    "euler_a": "Euler a",
    "euler": "Euler",
}


# ----------------------------------------------------------------------------- registry

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def local_checkpoints() -> list[ModelSpec]:
    """SDXL single-file checkpoints dropped into models/checkpoints/."""
    specs = []
    for path in sorted(CHECKPOINTS_DIR.glob("*.safetensors")):
        stem = path.stem
        fast = bool(re.search(r"turbo|lightning|hyper", stem, re.IGNORECASE))
        specs.append(ModelSpec(
            key=_slug(stem), name=stem, source=str(path), local=True,
            steps=6 if fast else 30, guidance=2.0 if fast else 6.0,
            scheduler="dpmpp_sde_karras" if fast else "dpmpp_2m_karras", variant=None,
            description=f"local checkpoint ({path.stat().st_size / 1024**3:.1f} GB)",
        ))
    return specs


def all_models() -> list[ModelSpec]:
    return BUILTIN + local_checkpoints()


def model_words() -> list[str]:
    """Every word that names a model (keys and aliases), for the prompt parser."""
    words: list[str] = []
    for m in all_models():
        words.append(m.key)
        words.extend(m.aliases)
    return sorted(set(words), key=len, reverse=True)


def resolve(name: str | ModelSpec | None) -> ModelSpec:
    """Find a model by key, alias, repo id, or local path. Raises KeyError if unknown."""
    if isinstance(name, ModelSpec):
        return name
    if not name:
        return BUILTIN[0]
    needle = name.strip().lower()
    for m in all_models():
        if needle in (m.key, m.name.lower(), m.source.lower(), *m.aliases) or needle == _slug(m.name):
            return m
    p = Path(name)
    if p.is_file() and p.suffix == ".safetensors":
        return local_checkpoints_from(p)
    if "/" in name:  # arbitrary HF repo in diffusers layout, assume SDXL
        return ModelSpec(key=_slug(name), name=name, source=name, variant=None,
                         description="custom Hugging Face repo")
    raise KeyError(name)


def local_checkpoints_from(path: Path) -> ModelSpec:
    return ModelSpec(key=_slug(path.stem), name=path.stem, source=str(path), local=True,
                     variant=None, guidance=6.0, description="local checkpoint")


def display_name(source: str) -> str:
    for m in all_models():
        if m.source == source:
            return m.name
    return Path(source).name


def model_for_style(style: str | None) -> ModelSpec | None:
    """The built-in model that specialises in a prompt_parser style, if any."""
    if not style:
        return None
    for m in BUILTIN:
        if style in m.styles:
            return m
    return None


def is_downloaded(spec: ModelSpec) -> bool:
    """True if the weights are already on disk (so loading will not trigger a download)."""
    if spec.local or Path(spec.source).exists():
        return Path(spec.source).exists()
    from huggingface_hub.constants import HF_HUB_CACHE
    snaps = Path(HF_HUB_CACHE) / f"models--{spec.source.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return False
    big = "unet" if spec.family == "sdxl" else "transformer"
    return any((snap / big).is_dir() and any((snap / big).glob("*.safetensors")) for snap in snaps.iterdir())


def find_lora(name: str) -> Path | None:
    """models/loras/<name>.safetensors, matched case-insensitively and ignoring punctuation."""
    want = _slug(name)
    for p in LORAS_DIR.glob("*.safetensors"):
        if _slug(p.stem) == want:
            return p
    return None


def available_loras() -> list[str]:
    return sorted(p.stem for p in LORAS_DIR.glob("*.safetensors"))


# ----------------------------------------------------------------------------- prompt weighting

_A1111_WEIGHT = re.compile(r"\(([^()]+?):(\d+(?:\.\d+)?)\)")


def a1111_to_compel(prompt: str) -> str:
    """Convert '(red hair:1.3)' into compel's '(red hair)1.3'."""
    return _A1111_WEIGHT.sub(r"(\1)\2", prompt)


# ----------------------------------------------------------------------------- loader

def _make_scheduler(name: str, config):
    from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
    if name == "dpmpp_2m_karras":
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
    if name == "dpmpp_sde_karras":
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    if name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(config)
    if name == "euler":
        return EulerDiscreteScheduler.from_config(config)
    raise ValueError(f"unknown scheduler {name}")


class ModelManager:
    """Holds one loaded pipeline at a time and swaps it on demand (VRAM is freed on swap)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.spec: ModelSpec | None = None
        self.pipe = None
        self.scheduler_name: str = ""
        self._compel = None
        self._compel_failed = False
        self._empty_chunk = None
        self._loaded_loras: dict[str, Path] = {}
        self._loras_active = False
        self._base_adapters: list[str] = []   # adapters a model always runs with (Qwen-Image Lightning)

    # --- loading -------------------------------------------------------------

    def load(self, model: str | ModelSpec, log=print):
        spec = resolve(model)
        if self.pipe is not None and self.spec and self.spec.key == spec.key:
            return self.pipe
        self.unload()
        t0 = time.time()
        if not is_downloaded(spec):
            log(f"Downloading {spec.name} (~{spec.download_gb:g} GB, first time only) ...")
        log(f"Loading {spec.name} ...")
        if spec.family == "flux":
            pipe = self._load_flux(spec, log)
        elif spec.family == "zimage":
            pipe = self._load_zimage(spec, log)
        elif spec.family == "qwen":
            pipe = self._load_qwen(spec, log)
        else:
            pipe = self._load_sdxl(spec)
        pipe.set_progress_bar_config(disable=True)
        self.spec, self.pipe = spec, pipe
        self.scheduler_name = SCHEDULER_NAMES.get(spec.scheduler or "", type(pipe.scheduler).__name__)
        log(f"{spec.name} ready in {time.time() - t0:.1f}s  (scheduler: {self.scheduler_name})")
        return pipe

    def _load_sdxl(self, spec: ModelSpec):
        import torch
        from diffusers import StableDiffusionXLPipeline
        if Path(spec.source).is_file():
            pipe = StableDiffusionXLPipeline.from_single_file(spec.source, torch_dtype=torch.float16)
        else:
            kwargs = dict(torch_dtype=torch.float16, use_safetensors=True)
            if spec.variant:
                kwargs["variant"] = spec.variant
            pipe = StableDiffusionXLPipeline.from_pretrained(spec.source, **kwargs)
        pipe.to(self.device)
        pipe.vae.enable_slicing()
        if spec.scheduler:
            pipe.scheduler = _make_scheduler(spec.scheduler, pipe.scheduler.config)
        return pipe

    def _load_flux(self, spec: ModelSpec, log=print):
        import torch
        from huggingface_hub.errors import GatedRepoError
        try:
            from diffusers import FluxPipeline
            dtype = torch.bfloat16
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram_gb >= 40:
                return FluxPipeline.from_pretrained(spec.source, torch_dtype=dtype).to(self.device)
            # 24 GB cards: 4-bit transformer + 8-bit T5, with per-model CPU offload.
            from diffusers import BitsAndBytesConfig as DiffBnb, FluxTransformer2DModel
            from transformers import BitsAndBytesConfig as TfBnb, T5EncoderModel
            log("FLUX on <40 GB VRAM: quantising transformer to 4-bit and T5 to 8-bit ...")
            transformer = FluxTransformer2DModel.from_pretrained(
                spec.source, subfolder="transformer", torch_dtype=dtype,
                quantization_config=DiffBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=dtype),
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                spec.source, subfolder="text_encoder_2", torch_dtype=dtype,
                quantization_config=TfBnb(load_in_8bit=True),
            )
            pipe = FluxPipeline.from_pretrained(
                spec.source, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=dtype,
            )
            pipe.enable_model_cpu_offload()
            return pipe
        except (GatedRepoError, OSError) as e:  # diffusers wraps the 401 in an OSError
            raise RuntimeError(
                "FLUX.1 [schnell] is gated on Hugging Face and no login token is stored. Accept the terms at "
                "https://huggingface.co/black-forest-labs/FLUX.1-schnell, then run "
                "`.venv\\Scripts\\huggingface-cli login` and try again. Until then use Z-Image Turbo or an SDXL model."
            ) from e

    def _load_zimage(self, spec: ModelSpec, log=print):
        """Z-Image Turbo: 6B flow transformer + Qwen3 text encoder, bf16, about 18 GB resident.
        Fits a 24 GB card with per-model CPU offload (the text encoder leaves the GPU after encoding)."""
        import torch
        from diffusers import ZImagePipeline, ZImageTransformer2DModel
        dtype = torch.bfloat16
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # The repo stores the transformer in fp32 (23 GB). It is loaded shard by shard straight onto the GPU
        # in bf16 (CPU offload on a 16 GB machine crashed), and the bf16 copy is saved next to the project
        # once so later loads read 12 GB instead of converting 23 GB.
        bf16_dir = MODELS_DIR / "cache" / f"{spec.key}-transformer-bf16"
        if (bf16_dir / "config.json").exists():
            transformer = ZImageTransformer2DModel.from_pretrained(bf16_dir, torch_dtype=dtype, device_map=self.device)
        else:
            transformer = ZImageTransformer2DModel.from_pretrained(
                spec.source, subfolder="transformer", torch_dtype=dtype, device_map=self.device,
            )
            log(f"Saving a bf16 copy of the transformer to {bf16_dir} for faster loads ...")
            try:
                transformer.save_pretrained(bf16_dir, max_shard_size="4GB")
            except Exception as e:  # cache is an optimisation only
                log(f"could not save bf16 copy ({e})")
        if vram_gb < 32:
            # 8-bit Qwen3 text encoder: 3.8 GB instead of 7.5 GB, so transformer + encoder + VAE fit a 24 GB card.
            from transformers import BitsAndBytesConfig, Qwen3Model
            log("Z-Image on <32 GB VRAM: loading the text encoder in 8-bit")
            text_encoder = Qwen3Model.from_pretrained(
                spec.source, subfolder="text_encoder", torch_dtype=dtype,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map=self.device,
            )
            pipe = ZImagePipeline.from_pretrained(
                spec.source, transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype,
            )
        else:
            pipe = ZImagePipeline.from_pretrained(spec.source, transformer=transformer, torch_dtype=dtype)
        pipe.to(self.device)
        return pipe

    QWEN_TEXT_ENCODER = "diffusers/qwen-image-nf4"          # pre-quantised 4-bit Qwen2.5-VL-7B
    QWEN_LIGHTNING = ("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors")

    def _load_qwen(self, spec: ModelSpec, log=print):
        """Qwen-Image (20B MMDiT) on a 24 GB card: transformer quantised to NF4 on first load and cached,
        4-bit text encoder from the diffusers NF4 repo, and the Lightning 8-step LoRA as a permanent adapter."""
        import math
        import torch
        from diffusers import (BitsAndBytesConfig as DiffBnb, FlowMatchEulerDiscreteScheduler, QwenImagePipeline,
                               QwenImageTransformer2DModel)
        from huggingface_hub import hf_hub_download
        from transformers import Qwen2_5_VLForConditionalGeneration
        dtype = torch.bfloat16

        nf4_dir = MODELS_DIR / "cache" / f"{spec.key}-transformer-nf4"
        if (nf4_dir / "config.json").exists():
            transformer = QwenImageTransformer2DModel.from_pretrained(nf4_dir, torch_dtype=dtype, device_map=self.device)
        else:
            log("First load: quantising the 20B transformer to 4-bit (NF4) shard by shard, this takes several minutes ...")
            transformer = QwenImageTransformer2DModel.from_pretrained(
                spec.source, subfolder="transformer", torch_dtype=dtype, device_map=self.device,
                quantization_config=DiffBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype),
            )
            log(f"Saving the 4-bit transformer to {nf4_dir} for faster loads ...")
            try:
                transformer.save_pretrained(nf4_dir, max_shard_size="4GB")
            except Exception as e:
                log(f"could not save 4-bit copy ({e})")

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.QWEN_TEXT_ENCODER, subfolder="text_encoder", torch_dtype=dtype, device_map=self.device,
        )
        pipe = QwenImagePipeline.from_pretrained(
            spec.source, transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype,
        )
        pipe.to(self.device)
        # 17.6 GB sit resident; keep the VAE decode peak small so the whole run stays under 24 GB.
        for name in ("enable_slicing", "enable_tiling"):
            try:
                getattr(pipe.vae, name)()
            except Exception:
                pass

        # Lightning recipe from lightx2v: exponential time shift of log(3), 8 steps, no CFG.
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, **{
            "base_image_seq_len": 256, "max_image_seq_len": 8192, "base_shift": math.log(3), "max_shift": math.log(3),
            "shift": 1.0, "shift_terminal": None, "use_dynamic_shifting": True, "time_shift_type": "exponential",
        })
        lora_path = hf_hub_download(*self.QWEN_LIGHTNING)
        log("Attaching the Lightning 8-step LoRA ...")
        pipe.load_lora_weights(lora_path, adapter_name="lightning")
        self._base_adapters = ["lightning"]
        self._loaded_loras["lightning"] = Path(lora_path)
        pipe.set_adapters(["lightning"], adapter_weights=[1.0])
        return pipe

    def unload(self) -> None:
        if self.pipe is None:
            return
        import torch
        self._compel = None
        self._compel_failed = False
        self._empty_chunk = None
        self._loaded_loras.clear()
        self._loras_active = False
        self._base_adapters = []
        self.pipe = None
        self.spec = None
        gc.collect()
        torch.cuda.empty_cache()

    # --- LoRA ----------------------------------------------------------------

    def apply_loras(self, loras: list[tuple[str, float]], log=print) -> list[str]:
        """Activate the given (name, weight) LoRAs; returns a note per LoRA for the record."""
        notes: list[str] = []
        if loras and not hasattr(self.pipe, "load_lora_weights"):
            notes.append(f"{self.spec.name} does not support LoRAs; ignored")
            loras = []
        names, weights = [], []
        for name, weight in loras:
            path = find_lora(name)
            if not path:
                notes.append(f"LoRA '{name}' not found in {LORAS_DIR} (have: {', '.join(available_loras()) or 'none'})")
                continue
            adapter = _slug(path.stem).replace("-", "_")
            if adapter not in self._loaded_loras:
                log(f"Loading LoRA {path.name} ...")
                try:
                    self.pipe.load_lora_weights(str(path), adapter_name=adapter)
                except Exception as e:  # an SDXL LoRA on Z-Image, or a broken file
                    log(f"LoRA {path.name} not loaded: {e}")
                    notes.append(f"LoRA {path.stem} is not compatible with {self.spec.name}; ignored")
                    continue
                self._loaded_loras[adapter] = path
            names.append(adapter)
            weights.append(weight)
            notes.append(f"LoRA {path.stem} x{weight:g}")
        base = [a for a in self._base_adapters if a not in names]
        if names:
            self.pipe.enable_lora()
            self.pipe.set_adapters(base + names, adapter_weights=[1.0] * len(base) + weights)
            self._loras_active = True
        elif self._loras_active:
            if base:
                self.pipe.set_adapters(base, adapter_weights=[1.0] * len(base))
            else:
                self.pipe.disable_lora()
            self._loras_active = False
        return notes

    # --- prompt encoding -----------------------------------------------------

    def encode(self, prompt: str, negative: str) -> dict:
        """Pipeline kwargs for the prompt: weighted/long-prompt embeddings for SDXL, plain text otherwise."""
        if self.spec is None or self.spec.family != "sdxl":
            return {"prompt": prompt}
        if self._compel is None and not self._compel_failed:
            try:
                import transformers
                from compel import Compel, ReturnedEmbeddingsType
                transformers.logging.set_verbosity_error()  # compel chunks long prompts; silence the >77 warning
                p = self.pipe
                self._compel = Compel(
                    tokenizer=[p.tokenizer, p.tokenizer_2],
                    text_encoder=[p.text_encoder, p.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False,
                    device=self.device,
                )
            except Exception as e:  # compel optional: fall back to the pipeline's own encoder
                print(f"compel unavailable ({e}); prompt weighting disabled")
                self._compel_failed = True
        if self._compel is None:
            return {"prompt": prompt, "negative_prompt": negative}
        try:
            import torch
            with torch.no_grad():
                cond, pooled = self._compel(a1111_to_compel(prompt))
                neg, neg_pooled = self._compel(a1111_to_compel(negative or ""))
                if cond.shape[1] != neg.shape[1]:
                    # Long prompts come back as several 77-token chunks; pad the shorter side with
                    # chunks of the empty prompt so both have the same length (compel's own helper
                    # does not support SDXL's dual encoder).
                    if self._empty_chunk is None:
                        self._empty_chunk = self._compel("")[0]
                    cond, neg = self._pad_to_same(cond, neg)
            return {
                "prompt_embeds": cond, "pooled_prompt_embeds": pooled,
                "negative_prompt_embeds": neg, "negative_pooled_prompt_embeds": neg_pooled,
            }
        except Exception as e:
            print(f"compel failed on this prompt ({e}); using plain prompt")
            return {"prompt": prompt, "negative_prompt": negative}

    def _pad_to_same(self, a, b):
        import torch
        chunk = self._empty_chunk
        target = max(a.shape[1], b.shape[1])

        def pad(t):
            while t.shape[1] < target:
                t = torch.cat([t, chunk.to(t.device, t.dtype)], dim=1)
            return t

        return pad(a), pad(b)

    HEAVY_FAMILIES = ("qwen", "zimage", "flux")

    def release_cache(self) -> None:
        """Return freed activations to the driver after an image. Matters for the big flow models, where the
        resident weights leave only a few GB of headroom and fragmentation can push Windows into paging VRAM."""
        if self.spec and self.spec.family in self.HEAVY_FAMILIES:
            import torch
            torch.cuda.empty_cache()

    def generation_kwargs(self, steps: int, guidance: float, width: int, height: int, generator, callback=None) -> dict:
        kw = dict(num_inference_steps=steps, guidance_scale=guidance, width=width, height=height, generator=generator)
        if callback is not None:
            kw["callback_on_step_end"] = callback
        if self.spec and self.spec.family == "flux":
            kw["max_sequence_length"] = 256
        if self.spec and self.spec.family == "qwen":
            # Qwen-Image drives classifier-free guidance with true_cfg_scale; 1.0 (Lightning) means none.
            kw["true_cfg_scale"] = max(1.0, guidance)
            kw.pop("guidance_scale")
        return kw


STARTER_LORAS = {
    # file name in models/loras/  ->  (Hugging Face repo, file in repo, trigger word)
    "pixel-art-xl.safetensors": ("nerijs/pixel-art-xl", "pixel-art-xl.safetensors", "pixel art"),
    "papercut.safetensors": ("TheLastBen/Papercut_SDXL", "papercut.safetensors", "papercut"),
}


def fetch_starter_loras(log=print) -> list[Path]:
    """Download the two SDXL LoRAs the documentation refers to (they are not kept in git)."""
    import shutil
    from huggingface_hub import hf_hub_download
    LORAS_DIR.mkdir(parents=True, exist_ok=True)
    got = []
    for name, (repo, filename, trigger) in STARTER_LORAS.items():
        dst = LORAS_DIR / name
        if dst.exists():
            log(f"{name} already present")
        else:
            log(f"Downloading {repo}/{filename} ...")
            shutil.copy(hf_hub_download(repo, filename), dst)
        got.append(dst)
        log(f"  {name}  trigger word: '{trigger}'  use: <lora:{dst.stem}:0.8>")
    return got


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "fetch-loras":
        fetch_starter_loras()
        raise SystemExit(0)
    for m in all_models():
        where = "downloaded" if is_downloaded(m) else f"not downloaded (~{m.download_gb:g} GB)"
        print(f"{m.key:12} {m.name:26} {m.steps:>3} steps  cfg {m.guidance:<4g} {where}")
        print(f"{'':12} {m.description}")
    print("LoRAs:", ", ".join(available_loras()) or "none in models/loras/")
