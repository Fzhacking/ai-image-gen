"""
Frazile AI Image Studio: local web UI for AI image generation on your GPU (diffusers + Gradio).

Run:
    .venv/Scripts/python.exe app.py
Then open http://127.0.0.1:7860 in your browser.

Environment (used by the Docker image, optional otherwise):
    GRADIO_SERVER_NAME   bind address, default 127.0.0.1 (0.0.0.0 inside Docker)
    GRADIO_SERVER_PORT   port, default 7860
    FRAZILE_NO_BROWSER   set to 1 to not open a browser tab on start

Everything is controlled from the prompt. Examples:
    "a cozy cabin in a snowy forest, 16:9, 4k"
    "portrait of a cyberpunk samurai, 3 images, seed 42"
    "desktop wallpaper of a mountain lake, photorealistic"        -> auto-picks Juggernaut XL
    "a knight in silver armour, model realvis, (glowing sword:1.3)"
    "neon street at night using flux"
See prompt_parser.py for the full list of understood words and models.py for the models.

Tabs:
    Generate   prompt, model, results, upscale bar (2K/4K/8K with Real-ESRGAN after generation)
    History    every generation from generations.db with its settings, timings and GPU telemetry
    Admin      live system panel: GPU, CPU, memory, storage, network, power, software, app
"""

import os
import random
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from PIL import Image

import db
import gpu_info
import models
import sysinfo
from models import ModelManager, ModelSpec
from prompt_parser import Settings, parse
from sysinfo import human_bytes
from upscaler import TARGETS, Upscaler

STARTUP_MODEL = "zimage"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_NEGATIVE = "blurry, low quality, deformed, watermark, text"
UPSCALE_CHOICES = [("2K (2560 px)", "2k"), ("4K (3840 px)", "4k"), ("8K (7680 px)", "8k")]

HELP = """
**Write everything in the prompt.** The app reads these hints:

| Want | Say |
|---|---|
| A specific model | `model juggernaut`, `model: flux`, `using realvis` (or pick it in the Model box) |
| Emphasis | `(glowing sword:1.3)`, `(word)++`, `(word)--` (SDXL models; long prompts are not cut off) |
| LoRA | `<lora:name:0.8>` for `models/loras/name.safetensors` (SDXL models) |
| Aspect ratio | `16:9`, `9:16`, `4:3`, `3:2`, `1:1`, `21:9` |
| Exact size | `1920x1080`, `1024x768` (scaled to the GPU-friendly equivalent) |
| Orientation | `portrait`, `landscape`, `square`, `wide`, `tall`, `wallpaper`, `phone wallpaper`, `poster`, `banner` |
| Real 4K output | `4k` (3840 px long side), `8k` (7680 px), `2k` (2560 px), or a big size like `3840x2160`. Generated at native size, then upscaled with Real-ESRGAN. You can also upscale afterwards with the Upscale bar. |
| Quality | `highly detailed`, `masterpiece` (more steps) or `quick`, `draft` (fewer steps) |
| Fidelity | `strict` (follow prompt closely) or `creative` (looser) |
| Style | `photorealistic`, `anime`, `cinematic`, `watercolor`, `oil painting`, `3d render`, `pixel art`, `sketch`, `comic`, `cyberpunk`, `low poly` |
| Several images | `3 images`, `4 variations` |
| Reproducible | `seed 42` |
| Manual override | `30 steps`, `cfg 8` |
"""

HISTORY_COLUMNS = ["id", "when", "prompt", "model", "size", "steps", "seconds", "it/s", "peak VRAM GB", "temp C", "seed"]


# ----------------------------------------------------------------------------- branding

PINK = gr.themes.Color(
    name="frazile-pink", c50="#FDF2F8", c100="#FCE7F3", c200="#FBCFE8", c300="#F9A8D4", c400="#F472B6",
    c500="#EC4899", c600="#DB2777", c700="#BE185D", c800="#9D174D", c900="#831843", c950="#500724",
)
YELLOW = gr.themes.Color(
    name="frazile-yellow", c50="#FEFCE8", c100="#FEF9C3", c200="#FEF08A", c300="#FDE047", c400="#FACC15",
    c500="#EAB308", c600="#CA8A04", c700="#A16207", c800="#854D0E", c900="#713F12", c950="#422006",
)
BG, SURFACE, SURFACE_2, BORDER, TEXT, MUTED = "#0B0F1A", "#141A2B", "#0F1422", "#232B40", "#F5F7FF", "#9AA3B8"
GRADIENT = "linear-gradient(90deg, #EC4899 0%, #F59E0B 55%, #FACC15 100%)"
GRADIENT_HOVER = "linear-gradient(90deg, #F472B6 0%, #FBBF24 55%, #FDE047 100%)"
# Status colours are reserved for state and always come with an icon and a word, never colour alone.
STATUS = {"good": ("#0ca30c", "●", "OK"), "warning": ("#fab219", "▲", "Warm"),
          "serious": ("#ec835a", "▲", "Hot"), "critical": ("#d03b3b", "■", "Critical")}


def build_theme() -> gr.themes.Base:
    import inspect
    theme = gr.themes.Base(
        primary_hue=PINK,
        secondary_hue=YELLOW,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_lg,
    )
    valid = set(inspect.signature(gr.themes.Base.set).parameters)
    both = {}  # apply every value to light and dark mode so the studio is always dark
    for key, value in {
        "body_background_fill": BG,
        "body_text_color": TEXT,
        "body_text_color_subdued": MUTED,
        "background_fill_primary": SURFACE,
        "background_fill_secondary": SURFACE_2,
        "border_color_primary": BORDER,
        "block_background_fill": SURFACE,
        "block_border_color": BORDER,
        "block_label_background_fill": SURFACE,
        "block_label_text_color": MUTED,
        "block_title_text_color": TEXT,
        "block_info_text_color": MUTED,
        "panel_background_fill": SURFACE,
        "panel_border_color": BORDER,
        "input_background_fill": SURFACE_2,
        "input_background_fill_focus": SURFACE_2,
        "input_border_color": BORDER,
        "input_border_color_focus": "#EC4899",
        "input_placeholder_color": "#5B6478",
        "button_primary_background_fill": GRADIENT,
        "button_primary_background_fill_hover": GRADIENT_HOVER,
        "button_primary_text_color": "#1A0A12",
        "button_primary_border_color": "transparent",
        "button_secondary_background_fill": "#1C2436",
        "button_secondary_background_fill_hover": "#273149",
        "button_secondary_text_color": TEXT,
        "button_secondary_border_color": "#2A3350",
        "table_even_background_fill": "#121829",
        "table_odd_background_fill": SURFACE_2,
        "table_border_color": BORDER,
        "checkbox_background_color": SURFACE_2,
        "checkbox_background_color_selected": "#EC4899",
        "checkbox_border_color": "#2A3350",
        "checkbox_label_background_fill": SURFACE_2,
        "checkbox_label_background_fill_selected": "#2A1A2E",
        "checkbox_label_text_color": TEXT,
        "checkbox_label_border_color": "#2A3350",
        "color_accent": "#EC4899",
        "color_accent_soft": "#2A1A2E",
        "link_text_color": "#FACC15",
        "link_text_color_hover": "#FDE047",
        "slider_color": "#EC4899",
        "shadow_drop": "0 1px 2px rgba(0,0,0,.35)",
        "shadow_drop_lg": "0 12px 32px -12px rgba(236,72,153,.35)",
        "block_shadow": "none",
    }.items():
        for k in (key, f"{key}_dark"):
            if k in valid:
                both[k] = value
    return theme.set(**both)


CSS = f"""
.gradio-container {{ max-width: none !important; padding: 0 28px 28px !important; }}
footer {{ display: none !important; }}

.fz-header {{ display: flex; align-items: center; justify-content: space-between; gap: 16px;
              padding: 14px 4px 14px; border-bottom: 1px solid {BORDER}; margin-bottom: 12px; }}
.fz-brand {{ display: flex; align-items: center; gap: 12px; }}
.fz-mark {{ width: 42px; height: 42px; border-radius: 12px; background: {GRADIENT};
            box-shadow: 0 8px 24px -8px rgba(236,72,153,.7); display: grid; place-items: center;
            font-weight: 900; font-size: 21px; color: #1A0A12; letter-spacing: -.04em; }}
.fz-logo {{ font-weight: 800; font-size: 26px; letter-spacing: -.02em; line-height: 1; color: {TEXT}; }}
.fz-logo span {{ background: {GRADIENT}; -webkit-background-clip: text; background-clip: text; color: transparent; }}
.fz-tag {{ color: {MUTED}; font-size: 13px; margin-top: 5px; }}
.fz-sys {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
.fz-pill {{ font-family: var(--font-mono); font-size: 11.5px; color: {MUTED}; border: 1px solid {BORDER};
            background: {SURFACE}; border-radius: 999px; padding: 5px 10px; white-space: nowrap; }}
.fz-pill b {{ color: #FACC15; font-weight: 600; }}

.fz-interp {{ border-left: 3px solid #EC4899; background: #17122A; border-radius: 10px;
              padding: 8px 12px; font-size: 13px; margin: 2px 0 8px; }}
.fz-interp p {{ margin: 0; }}
.fz-interp small {{ color: {MUTED}; }}
.fz-loaded {{ font-size: 12px; color: {MUTED}; margin: -4px 0 4px 2px; }}
.fz-loaded b {{ color: #F9A8D4; font-weight: 600; }}

.fz-upscale {{ border: 1px dashed #3A2C4A; border-radius: 14px; padding: 10px 12px;
               background: linear-gradient(180deg, #151225 0%, {SURFACE} 100%); margin-top: 8px; }}
.fz-hint {{ color: {MUTED}; font-size: 12px; }}

button.primary {{ font-weight: 700; letter-spacing: .01em;
                  box-shadow: 0 10px 28px -10px rgba(236,72,153,.75); transition: transform .08s ease; }}
button.primary:hover {{ transform: translateY(-1px); }}
button.secondary {{ font-weight: 600; }}

.tabs > .tab-nav button.selected, .tab-nav button.selected {{ color: #FACC15 !important; border-color: #EC4899 !important; }}
.prose h3 {{ color: #FACC15; }}
.prose h4 {{ color: #F9A8D4; margin-top: 4px; }}
.prose table th {{ color: #F9A8D4; }}
.prose code {{ background: #1C1530; color: #FDE68A; border: 1px solid #3A2C4A; }}
.gallery-item, .thumbnail-item {{ border-radius: 12px !important; }}
.progress-bar {{ background: {GRADIENT} !important; }}

/* admin panel */
.fz-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); gap: 10px; margin: 4px 0 12px; }}
.fz-tile {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 12px 14px 10px; min-height: 92px; }}
.fz-tile .l {{ font-size: 12px; color: {MUTED}; }}
.fz-tile .v {{ font-size: 24px; font-weight: 600; color: {TEXT}; line-height: 1.15; margin-top: 4px; }}
.fz-tile .v small {{ font-size: 13px; color: {MUTED}; font-weight: 500; margin-left: 2px; }}
.fz-tile .s {{ font-size: 11.5px; color: {MUTED}; margin-top: 4px; }}
.fz-tile .st {{ font-size: 11.5px; font-weight: 600; margin-top: 4px; }}
.fz-meter {{ height: 6px; border-radius: 3px; background: #3B2238; margin-top: 8px; overflow: hidden; }}
.fz-meter > div {{ height: 100%; border-radius: 3px; background: #EC4899; }}
.fz-cores {{ display: flex; gap: 3px; margin-top: 8px; align-items: flex-end; height: 22px; }}
.fz-cores > div {{ flex: 1; background: #3B2238; border-radius: 2px; position: relative; height: 100%; }}
.fz-cores > div > div {{ position: absolute; bottom: 0; left: 0; right: 0; background: #EC4899; border-radius: 2px; }}
.fz-admin-head {{ display: flex; justify-content: space-between; align-items: center; color: {MUTED}; font-size: 12px; }}
"""


def header_html(static_gpu: dict) -> str:
    return f"""
<div class="fz-header">
  <div class="fz-brand">
    <div class="fz-mark">F</div>
    <div>
      <div class="fz-logo">Frazile <span>AI</span></div>
      <div class="fz-tag">Image Studio · SDXL fine-tunes, FLUX, LoRAs and Real-ESRGAN on your own GPU</div>
    </div>
  </div>
  <div class="fz-sys">
    <span class="fz-pill"><b>{static_gpu['gpu_name']}</b> · {static_gpu['vram_total_gb']} GB</span>
    <span class="fz-pill">CUDA {static_gpu['cuda_version']}</span>
    <span class="fz-pill">torch {static_gpu['torch_version']}</span>
    <span class="fz-pill">diffusers {static_gpu['diffusers_version']}</span>
  </div>
</div>"""


def models_markdown() -> str:
    rows = ["| Say | Model | Steps | CFG | On disk | Best for |", "|---|---|---|---|---|---|"]
    for m in models.all_models():
        on_disk = "yes" if models.is_downloaded(m) else f"no (~{m.download_gb:g} GB)"
        rows.append(f"| `model {m.key}` | {m.name} | {m.steps} | {m.guidance:g} | {on_disk} | {m.description} |")
    loras = ", ".join(f"`<lora:{n}>`" for n in models.available_loras()) or "none yet"
    rows += [
        "",
        "**Auto** keeps Z-Image Turbo (loaded at start) or FLUX when one of them is active. When an SDXL-family model "
        "is loaded it picks a specialist from the style words (photo/cinematic → Juggernaut, anime → Animagine, "
        "painting → Playground, digital art/comic → DreamShaper) if that model is already on disk.",
        "",
        f"**Add your own:** drop any SDXL `.safetensors` checkpoint (for example from Civitai) into `models/checkpoints/` and "
        f"SDXL LoRAs into `models/loras/`, then restart the app. LoRAs available now: {loras}.",
        "",
        "**FLUX.1 [schnell]** needs a one-time Hugging Face login: accept the terms on its model page, run "
        "`.venv\\Scripts\\huggingface-cli login`, then say `model flux`. On a 24 GB card it runs 4-bit with CPU offload.",
    ]
    return "\n".join(rows)


MANAGER = ModelManager()
MANAGER.load(STARTUP_MODEL)
UPSCALER = Upscaler()
print(f"Upscaler ready: {UPSCALER.name}")
STATIC_GPU = gpu_info.static_info()

MODEL_CHOICES = [("Auto (pick by style words in the prompt)", "auto")] + [
    (f"{m.name}  -  {m.description}", m.key) for m in models.all_models()
]


def loaded_line() -> str:
    spec = MANAGER.spec
    if not spec:
        return ""
    return f"Loaded now: **{spec.name}** · {MANAGER.scheduler_name} · upscaler {UPSCALER.name}"


# ----------------------------------------------------------------------------- model selection

def pick_model(s: Settings, choice: str) -> tuple[ModelSpec, str | None]:
    """Which model to use for this prompt, and an optional note explaining an automatic choice."""
    if s.model:
        try:
            return models.resolve(s.model), None
        except KeyError:
            raise gr.Error(
                f"Unknown model '{s.model}'. Available: {', '.join(m.key for m in models.all_models())}"
            )
    if choice and choice != "auto":
        return models.resolve(choice), None
    current = MANAGER.spec or models.resolve(STARTUP_MODEL)
    suited = models.model_for_style(s.style)
    # Z-Image and FLUX handle every style themselves; Auto only swaps between SDXL-family specialists.
    if suited and suited.key != current.key and current.family == "sdxl":
        if models.is_downloaded(suited):
            return suited, f"auto: {suited.name} for '{s.style}'"
        return current, f"{suited.name} suits '{s.style}' but is not downloaded; say 'model {suited.key}' to fetch it"
    return current, None


# ----------------------------------------------------------------------------- generate tab

def preview(prompt: str, choice: str) -> str:
    """Live 'interpreted as' line shown under the prompt box."""
    if not prompt.strip():
        return ""
    s = parse(prompt)
    try:
        spec, note = pick_model(s, choice)
    except gr.Error as e:
        return f"**{e}**"
    s.apply_defaults(spec.steps, spec.guidance)
    if note:
        s.notes.append(note)
    line = f"**Will generate:** {s.summary()}  ·  model **{spec.name}**"
    if not spec.negative and s.loras:
        s.notes.append("LoRAs are ignored by FLUX")
    if s.notes:
        line += f"  \n<small>{' · '.join(s.notes)}</small>"
    return line


def details_markdown(rec: dict) -> str:
    """Human-readable report for one generation record."""
    g = rec.get
    lines = [
        f"### Generation #{g('id', '?')}  ·  {g('created_at', '')[:19]}",
        "",
        "**Timing**",
        f"- Inference: **{g('inference_seconds'):.2f} s**  ({g('steps_per_second'):.2f} steps/s)",
        f"- Save to disk: {g('save_seconds'):.2f} s  ·  Total: {g('total_seconds'):.2f} s",
        "",
        "**Image**",
        f"- Model: **{models.display_name(g('model_id') or '')}**",
        f"- {g('width')}x{g('height')}, {g('steps')} steps, guidance {g('guidance'):g}, seed {g('seed')}",
        f"- File: `{Path(g('image_path', '')).name}` ({g('file_size_kb'):.0f} KB)",
        f"- Style: {g('style') or 'none'}  ·  Image {g('batch_index')}/{g('batch_size')}",
    ]
    if g("upscaled_path"):
        lines += [
            "",
            "**Upscale**",
            f"- {g('width')}x{g('height')} → **{g('upscaled_width')}x{g('upscaled_height')}** "
            f"in {g('upscale_seconds'):.2f} s ({g('upscale_passes')} pass{'es' if g('upscale_passes') != 1 else ''})",
            f"- {g('upscaler')}  ·  file `{Path(g('upscaled_path')).name}` ({g('upscaled_size_kb') / 1024:.1f} MB)",
        ]
    lines += [
        "",
        "**GPU**",
        f"- {g('gpu_name')}  ·  {g('vram_total_gb')} GB VRAM  ·  compute {g('compute_capability')}",
        f"- Peak VRAM used: **{g('vram_peak_gb')} GB**  ·  reserved {g('vram_reserved_gb')} GB",
        f"- Temperature: {g('temp_before_c')} °C → **{g('temp_after_c')} °C**",
        f"- Power: {g('power_before_w')} W → **{g('power_after_w')} W** (limit {g('power_limit_w')} W)",
        f"- Clocks after run: SM {g('sm_clock_mhz')} MHz, memory {g('mem_clock_mhz')} MHz  ·  "
        f"fan {g('fan_after_pct')} %  ·  load {g('gpu_util_after_pct')} %  ·  {g('pstate')}",
        f"- Driver {g('driver_version')}  ·  CUDA {g('cuda_version')}  ·  torch {g('torch_version')}  ·  "
        f"diffusers {g('diffusers_version')}",
        "",
        "**Prompt sent to model**",
        f"> {g('final_prompt')}",
        "",
        f"**Negative:** {g('negative_prompt') or '(not supported by this model)'}",
    ]
    if g("parser_notes"):
        lines.append(f"**Notes:** {g('parser_notes')}")
    return "\n".join(lines)


def upscale_record(rec: dict, target_px: int, exact_size: tuple[int, int] | None = None) -> dict:
    """Upscale one generation's native image to target_px long side; save and return the DB fields."""
    img = Image.open(rec["image_path"]).convert("RGB")
    up_img, up_info = UPSCALER.upscale_to(img, target_px)
    if exact_size:
        tw, th = exact_size
        left, top = max(0, (up_img.width - tw) // 2), max(0, (up_img.height - th) // 2)
        up_img = up_img.crop((left, top, left + min(tw, up_img.width), top + min(th, up_img.height)))
    up_path = Path(rec["image_path"]).with_name(f"{Path(rec['image_path']).stem}_{target_px}px.png")
    up_img.save(up_path)
    return {
        "upscale_target": target_px,
        "upscaled_width": up_img.width,
        "upscaled_height": up_img.height,
        "upscaled_path": str(up_path),
        "upscaled_size_kb": round(up_path.stat().st_size / 1024, 1),
        "upscale_seconds": up_info["seconds"],
        "upscale_passes": up_info["passes"],
        "upscaler": up_info["upscaler"],
    }


def generate(prompt: str, negative: str, model_choice: str, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    s = parse(prompt)
    spec, auto_note = pick_model(s, model_choice)
    if auto_note:
        s.notes.append(auto_note)
    s.apply_defaults(spec.steps, spec.guidance)
    negative = (negative.strip() or DEFAULT_NEGATIVE) if spec.negative else ""

    if MANAGER.spec is None or MANAGER.spec.key != spec.key:
        progress(0, desc=f"Loading {spec.name}" + ("" if models.is_downloaded(spec) else f" (downloading ~{spec.download_gb:g} GB)"))
    try:
        pipe = MANAGER.load(spec)
    except RuntimeError as e:
        raise gr.Error(str(e))
    s.notes.extend(MANAGER.apply_loras(s.loras))
    s.notes.append(f"scheduler {MANAGER.scheduler_name}")
    encoded = MANAGER.encode(s.prompt, negative)

    base_seed = s.seed if s.seed is not None else random.randint(0, 2**32 - 1)
    total_steps = s.steps * s.count
    images, reports, records = [], [], []

    for i in range(s.count):
        seed = base_seed + i
        gen = torch.Generator(device="cuda").manual_seed(seed)

        def on_step(pipe, step, timestep, kwargs, i=i):
            progress((i * s.steps + step + 1) / total_steps, desc=f"Image {i + 1}/{s.count}  ·  {spec.name}")
            return kwargs

        torch.cuda.reset_peak_memory_stats()
        before = gpu_info.snapshot()
        t_start = time.perf_counter()

        img = pipe(
            **encoded,
            **MANAGER.generation_kwargs(s.steps, s.guidance, s.width, s.height, gen, on_step),
        ).images[0]
        torch.cuda.synchronize()
        t_infer = time.perf_counter()
        after = gpu_info.snapshot()
        mem = gpu_info.torch_memory()

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = OUTPUT_DIR / f"{stamp}_seed{seed}.png"
        img.save(path)
        t_end = time.perf_counter()

        rec = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image_path": str(path),
            "file_size_kb": round(path.stat().st_size / 1024, 1),
            "user_prompt": prompt.strip(),
            "final_prompt": s.prompt,
            "negative_prompt": negative,
            "parser_notes": " · ".join(s.notes),
            "model_id": spec.source,
            "width": s.width, "height": s.height, "steps": s.steps, "guidance": s.guidance,
            "seed": seed, "style": s.style, "batch_index": i + 1, "batch_size": s.count,
            "inference_seconds": round(t_infer - t_start, 3),
            "save_seconds": round(t_end - t_infer, 3),
            "total_seconds": round(t_end - t_start, 3),
            "steps_per_second": round(s.steps / (t_infer - t_start), 2),
            "gpu_name": STATIC_GPU["gpu_name"],
            "vram_total_gb": STATIC_GPU["vram_total_gb"],
            "vram_peak_gb": mem["vram_peak_gb"],
            "vram_reserved_gb": mem["vram_reserved_gb"],
            "compute_capability": STATIC_GPU["compute_capability"],
            "driver_version": after.get("driver_version"),
            "cuda_version": STATIC_GPU["cuda_version"],
            "torch_version": STATIC_GPU["torch_version"],
            "diffusers_version": STATIC_GPU["diffusers_version"],
            "temp_before_c": before.get("temperature_c"),
            "temp_after_c": after.get("temperature_c"),
            "power_before_w": before.get("power_w"),
            "power_after_w": after.get("power_w"),
            "power_limit_w": after.get("power_limit_w"),
            "gpu_util_after_pct": after.get("gpu_util_pct"),
            "sm_clock_mhz": after.get("sm_clock_mhz"),
            "mem_clock_mhz": after.get("mem_clock_mhz"),
            "fan_after_pct": after.get("fan_pct"),
            "pstate": after.get("pstate"),
        }

        final_path = str(path)
        if s.upscale_to:
            progress(1.0, desc=f"Upscaling image {i + 1}/{s.count} to {s.upscale_to} px")
            rec.update(upscale_record(rec, s.upscale_to, s.exact_size))
            rec["vram_peak_gb"] = gpu_info.torch_memory()["vram_peak_gb"]  # peak now includes the upscaler
            final_path = rec["upscaled_path"]

        rec["id"] = db.insert(rec)
        print(f"#{rec['id']} {spec.key} saved {path.name}  {s.width}x{s.height}  {rec['inference_seconds']}s  "
              f"peak {rec['vram_peak_gb']} GB  {rec['temp_after_c']} C"
              + (f"  -> {rec['upscaled_width']}x{rec['upscaled_height']} in {rec['upscale_seconds']}s" if rec.get("upscaled_path") else ""))
        images.append(final_path)
        reports.append(details_markdown(rec))
        records.append({"id": rec["id"], "image_path": str(path), "upscaled_path": rec.get("upscaled_path"),
                        "width": s.width, "height": s.height})

    return (
        images,
        "\n\n---\n\n".join(reports),
        records,
        None,
        gr.update(visible=True),
        upscale_hint(records, None),
        loaded_line(),
    )


def clear_all():
    return "", None, "", [], None, gr.update(visible=False), "", ""


def upscale_hint(records: list[dict], selected: int | None) -> str:
    if not records:
        return ""
    if selected is not None and 0 <= selected < len(records):
        r = records[selected]
        done = f" · already {Path(r['upscaled_path']).stem.rsplit('_', 1)[-1]}" if r.get("upscaled_path") else ""
        return f"Image {selected + 1} of {len(records)} selected ({r['width']}x{r['height']}{done}). Click another image to change."
    return (f"All {len(records)} images" if len(records) > 1 else "The generated image") + \
        " will be upscaled. Click an image to upscale only that one."


def on_select(evt: gr.SelectData, records: list[dict]):
    return evt.index, upscale_hint(records, evt.index)


def upscale(records: list[dict], selected: int | None, target: str, progress=gr.Progress()):
    if not records:
        raise gr.Error("Generate an image first.")
    target_px = TARGETS[target]
    todo = [selected] if selected is not None and 0 <= selected < len(records) else list(range(len(records)))
    images, reports = [], []
    for n, idx in enumerate(todo):
        r = records[idx]
        progress(n / len(todo), desc=f"Upscaling image {idx + 1} to {target_px} px")
        rec = db.get(r["id"]) or {"image_path": r["image_path"]}
        fields = upscale_record(rec, target_px)
        db.update(r["id"], fields)
        r["upscaled_path"] = fields["upscaled_path"]
        print(f"#{r['id']} upscaled -> {fields['upscaled_width']}x{fields['upscaled_height']} in {fields['upscale_seconds']}s")
    for r in records:
        images.append(r.get("upscaled_path") or r["image_path"])
        full = db.get(r["id"])
        if full:
            reports.append(details_markdown(full))
    return images, "\n\n---\n\n".join(reports), records, upscale_hint(records, selected)


# ----------------------------------------------------------------------------- history tab

def history_table(search: str = "", limit: int = 50) -> pd.DataFrame:
    rows = db.fetch(limit=int(limit), search=search)
    data = [
        [r["id"], r["created_at"][:16].replace("T", " "), r["user_prompt"][:70],
         models.display_name(r.get("model_id") or ""),
         (f"{r['width']}x{r['height']} → {r['upscaled_width']}x{r['upscaled_height']}"
          if r.get("upscaled_width") else f"{r['width']}x{r['height']}"),
         r["steps"], r["inference_seconds"], r["steps_per_second"],
         r["vram_peak_gb"], r["temp_after_c"], r["seed"]]
        for r in rows
    ]
    return pd.DataFrame(data, columns=HISTORY_COLUMNS)


def history_stats() -> str:
    st = db.stats()
    if not st["images"]:
        return "No generations recorded yet."
    return (
        f"**{st['images']} images**  ·  {st['total_seconds']} s of GPU time  ·  "
        f"avg {st['avg_inference_seconds']} s (fastest {st['fastest_seconds']} s, slowest {st['slowest_seconds']} s)  ·  "
        f"avg {st['avg_steps_per_second']} steps/s  ·  avg peak VRAM {st['avg_vram_peak_gb']} GB "
        f"(max {st['max_vram_peak_gb']} GB)  ·  hottest {st['max_temp_c']} °C  ·  peak power {st['max_power_w']} W"
    )


def refresh_history(search: str, limit: int):
    return history_table(search, limit), history_stats()


def select_history(evt: gr.SelectData, table: pd.DataFrame):
    if table is None or len(table) == 0:
        return None, ""
    row_id = int(table.iloc[evt.index[0]]["id"])
    rec = db.get(row_id)
    if not rec:
        return None, f"Generation #{row_id} not found."
    shown = rec.get("upscaled_path") or rec["image_path"]
    img = shown if Path(shown).exists() else None
    return img, details_markdown(rec)


# ----------------------------------------------------------------------------- admin tab

THROTTLE_BITS = {
    0x1: "GPU idle", 0x2: "application clock setting", 0x4: "software power cap", 0x8: "hardware slowdown",
    0x10: "sync boost", 0x20: "software thermal slowdown", 0x40: "hardware thermal slowdown",
    0x80: "hardware power brake", 0x100: "display clock setting",
}


def _throttle_text(hex_value: str | None) -> str:
    try:
        v = int(hex_value or "0", 16)
    except ValueError:
        return hex_value or "n/a"
    names = [n for bit, n in THROTTLE_BITS.items() if v & bit]
    return ", ".join(names) if names else "none"


def _meter_color(pct: float | None) -> str:
    if pct is None:
        return "#EC4899"
    if pct >= 95:
        return STATUS["critical"][0]
    if pct >= 85:
        return STATUS["warning"][0]
    return "#EC4899"


def _temp_status(temp: float | None) -> str:
    if temp is None:
        return "good"
    return "good" if temp < 70 else "warning" if temp < 83 else "serious" if temp < 90 else "critical"


def tile(label: str, value: str, unit: str = "", sub: str = "", meter: float | None = None,
         status: str | None = None, cores: list[float] | None = None) -> str:
    html = [f'<div class="fz-tile"><div class="l">{label}</div>',
            f'<div class="v">{value}<small>{unit}</small></div>']
    if status:
        color, icon, word = STATUS[status]
        html.append(f'<div class="st" style="color:{color}">{icon} {word}</div>')
    if sub:
        html.append(f'<div class="s">{sub}</div>')
    if meter is not None:
        pct = max(0.0, min(100.0, meter))
        html.append(f'<div class="fz-meter"><div style="width:{pct:.1f}%;background:{_meter_color(pct)}"></div></div>')
    if cores:
        bars = "".join(f'<div title="core {i}: {c:.0f}%"><div style="height:{max(3, c):.0f}%"></div></div>' for i, c in enumerate(cores))
        html.append(f'<div class="fz-cores">{bars}</div>')
    html.append("</div>")
    return "".join(html)


def _kv(rows: list[tuple[str, object]]) -> str:
    out = ["| | |", "|---|---|"]
    for k, v in rows:
        out.append(f"| {k} | {'n/a' if v in (None, '') else v} |")
    return "\n".join(out)


def gpu_energy_estimate() -> tuple[float, int]:
    """Rough GPU energy of all recorded generations: inference seconds × mean of before/after power."""
    wh, n = 0.0, 0
    for r in db.fetch(limit=10**6):
        p = [x for x in (r.get("power_before_w"), r.get("power_after_w")) if x]
        if p and r.get("inference_seconds"):
            wh += r["inference_seconds"] * (sum(p) / len(p)) / 3600
            n += 1
    return wh, n


def admin_refresh(check_internet: bool):
    d = sysinfo.everything(loaded_model=MANAGER.spec.name if MANAGER.spec else None, check_internet=check_internet)
    c, m, dk, n, g, p, sw, a = d["cpu"], d["memory"], d["disks"], d["network"], d["gpu"], d["power"], d["software"], d["app"]
    live, st, tm, ex = g["live"], g["static"], g["torch"], g["extra"]

    vram_used, vram_total = live.get("mem_used_mb"), live.get("mem_total_mb")
    vram_pct = vram_used / vram_total * 100 if vram_used and vram_total else None
    sysdisk = next((x for x in dk["partitions"] if x["mount"].upper().startswith("C")), dk["partitions"][0] if dk["partitions"] else None)
    up, down = n.get("up_rate"), n.get("down_rate")
    temp = live.get("temperature_c")

    tiles = "".join([
        tile("GPU load", f"{live.get('gpu_util_pct', 0):.0f}", "%", f"{live.get('sm_clock_mhz', 0):.0f} MHz · {live.get('pstate', '')}",
             meter=live.get("gpu_util_pct")),
        tile("VRAM used", f"{(vram_used or 0) / 1024:.1f}", f" / {(vram_total or 0) / 1024:.0f} GB",
             f"torch {tm.get('vram_allocated_gb', 0)} GB allocated", meter=vram_pct),
        tile("GPU temperature", f"{temp if temp is not None else 'n/a'}", " °C", f"fan {live.get('fan_pct', 0):.0f} %",
             status=_temp_status(temp)),
        tile("GPU power", f"{p.get('gpu_draw_w') or 0:.0f}", f" / {p.get('gpu_limit_w') or 0:.0f} W",
             f"{p.get('gpu_pct_of_limit') or 0:.0f} % of limit", meter=p.get("gpu_pct_of_limit")),
        tile("CPU", f"{c.get('usage_pct') or 0:.0f}", "%", f"{c['logical_cores']} threads · {c.get('freq_current_mhz') or 0} MHz",
             cores=c.get("per_core_pct")),
        tile("Memory", f"{m['used'] / 1024**3:.1f}", f" / {m['total'] / 1024**3:.0f} GB", f"{m['percent']:.0f} % used",
             meter=m["percent"]),
        tile("System disk", f"{(sysdisk['free'] if sysdisk else 0) / 1024**3:.0f}", " GB free",
             f"{sysdisk['percent']:.0f} % used of {sysdisk['total'] / 1024**3:.0f} GB" if sysdisk else "", meter=sysdisk["percent"] if sysdisk else None),
        tile("Network", f"↓ {human_bytes(down or 0)}/s", "", f"↑ {human_bytes(up or 0)}/s · {n.get('established') or 0} connections"),
        tile("App uptime", a["uptime"], "", f"PID {a['pid']} · RSS {human_bytes(a['rss'])}"),
    ])
    tiles_html = f'<div class="fz-admin-head"><span>Collected {d["collected_at"][11:]}</span>' \
                 f'<span>{n["hostname"]} · {n.get("primary_ip") or "no IP"} · {sw["os"]}</span></div><div class="fz-tiles">{tiles}</div>'

    gpu_md = "#### GPU\n" + _kv([
        ("Card", f"{st.get('gpu_name')} · {st.get('vram_total_gb')} GB · compute {st.get('compute_capability')} · {st.get('multiprocessors')} SMs"),
        ("Driver / CUDA / cuDNN", f"{live.get('driver_version')} / {st.get('cuda_version')} / {st.get('cudnn_version')}"),
        ("Clocks (current / max)", f"SM {live.get('sm_clock_mhz'):.0f} / {ex.get('max_sm_clock_mhz')} MHz · memory {live.get('mem_clock_mhz'):.0f} / {ex.get('max_mem_clock_mhz')} MHz"),
        ("Utilisation", f"GPU {live.get('gpu_util_pct'):.0f} % · memory bus {live.get('mem_util_pct'):.0f} %"),
        ("VRAM (nvidia-smi)", f"{(vram_used or 0):.0f} / {(vram_total or 0):.0f} MiB"),
        ("VRAM (torch)", f"allocated {tm.get('vram_allocated_gb')} GB · reserved {tm.get('vram_reserved_gb')} GB · session peak {tm.get('vram_peak_gb')} GB"),
        ("PCIe link", f"gen {ex.get('pcie_gen')} x{ex.get('pcie_width')} (drops to gen 1 x1 at idle)"),
        ("Throttle reasons", _throttle_text(ex.get("throttle_reasons"))),
        ("VBIOS / bus", f"{ex.get('vbios')} · {ex.get('pci_bus')}"),
        ("Loaded model", f"{a.get('loaded_model') or 'none'} · {MANAGER.scheduler_name}"),
    ])
    if g["processes"]:
        gpu_md += "\n\n**Processes on the GPU**\n\n| PID | VRAM | Name |\n|---|---|---|\n" + "\n".join(
            f"| {x['pid']} | {x['vram_mb']} MiB | {x['name']} |" for x in g["processes"][:12])

    cpu_md = "#### CPU\n" + _kv([
        ("Processor", c["name"]),
        ("Architecture", c["architecture"]),
        ("Cores", f"{c['physical_cores']} physical · {c['logical_cores']} logical"),
        ("Frequency", f"{c.get('freq_current_mhz')} MHz" + (f" (max {c['freq_max_mhz']} MHz)" if c.get("freq_max_mhz") else "")),
        ("Usage", f"{c.get('usage_pct')} %  ·  per core: " + " ".join(f"{x:.0f}" for x in c.get("per_core_pct") or [])),
        ("Load average (1/5/15 min)", f"{c.get('load_1m')} / {c.get('load_5m')} / {c.get('load_15m')}"),
        ("Processes / threads", f"{c['processes']} / {c['threads']}"),
        ("Booted", f"{c['boot_time'].replace('T', ' ')} · up {c['uptime']}"),
    ])

    parts = "\n".join(
        f"| {x['mount']} | {x['fs']} | {human_bytes(x['total'])} | {human_bytes(x['used'])} | {human_bytes(x['free'])} | {x['percent']:.0f} % |"
        for x in dk["partitions"])
    pj = dk["project"]
    mem_md = "#### Memory and storage\n" + _kv([
        ("RAM", f"{human_bytes(m['used'])} used · {human_bytes(m['available'])} available · {human_bytes(m['total'])} total ({m['percent']:.0f} %)"),
        ("Swap / page file", f"{human_bytes(m['swap_used'])} / {human_bytes(m['swap_total'])} ({m['swap_percent']:.0f} %)"),
        ("Disk I/O since boot", f"read {human_bytes(dk.get('read_total'))} · written {human_bytes(dk.get('write_total'))}"),
        ("Outputs folder", f"{human_bytes(pj['outputs'])} in {pj['outputs_files']} PNGs"),
        ("Models folder", human_bytes(pj["models_dir"])),
        ("Hugging Face cache", f"{human_bytes(pj['hf_cache'])} · `{pj['hf_cache_path']}`"),
        ("Database", human_bytes(pj["database"])),
    ]) + f"\n\n| Mount | FS | Total | Used | Free | Used % |\n|---|---|---|---|---|---|\n{parts}"

    ifaces = "\n".join(
        f"| {x['name']} | {'up' if x['up'] else 'down'} | {x['ipv4'] or ''} | {x['mac'] or ''} | "
        f"{str(x['speed_mbps']) + ' Mb/s' if x['speed_mbps'] else ''} | {human_bytes(x['sent'])} | {human_bytes(x['recv'])} |"
        for x in n["interfaces"] if x["up"] or x["ipv4"])
    net_rows = [
        ("Hostname / primary IP", f"{n['hostname']} · {n.get('primary_ip') or 'n/a'}"),
        ("Throughput now", f"↓ {human_bytes(down or 0)}/s · ↑ {human_bytes(up or 0)}/s" + ("" if up is not None else " (refresh again for rates)")),
        ("Totals since boot", f"sent {human_bytes(n['bytes_sent'])} ({n['packets_sent']:,} packets) · received {human_bytes(n['bytes_recv'])} ({n['packets_recv']:,} packets)"),
        ("Errors / drops", f"in {n['errors_in']} / {n['drops_in']} · out {n['errors_out']} / {n['drops_out']}"),
        ("Connections", f"{n.get('established')} established · listening on {', '.join(map(str, n['listening_ports'])) or 'n/a'}"),
    ]
    if d.get("internet"):
        net_rows.append(("Reachability (TCP 443)", " · ".join(f"{h}: {ms} ms" if ms is not None else f"{h}: unreachable" for h, ms in d["internet"].items())))
    net_md = "#### Network\n" + _kv(net_rows) + \
        f"\n\n| Interface | State | IPv4 | MAC | Speed | Sent | Received |\n|---|---|---|---|---|---|---|\n{ifaces}"

    wh, n_rec = gpu_energy_estimate()
    bat = p.get("battery")
    power_md = "#### Power\n" + _kv([
        ("GPU draw", f"{p.get('gpu_draw_w')} W of {p.get('gpu_limit_w')} W limit ({p.get('gpu_pct_of_limit')} %)"),
        ("GPU power limits", f"default {ex.get('power_default_limit_w')} W · max {ex.get('power_max_limit_w')} W"),
        ("GPU performance state", f"{p.get('gpu_pstate')} · fan {p.get('gpu_fan_pct')} %"),
        ("Windows power plan", p.get("windows_power_plan")),
        ("Battery", f"{bat['percent']} % · {'plugged in' if bat['plugged'] else 'on battery'}" if bat else "none (desktop)"),
        ("GPU energy, all generations", f"≈ {wh:.1f} Wh over {n_rec} images (inference time × mean power)"),
    ])

    stats = db.stats()
    soft_md = "#### Software and app\n" + _kv([
        ("OS", sw["os"]),
        ("Python", sw["python"]),
        ("torch / CUDA / cuDNN", f"{sw['torch']} / {sw['cuda']} / {sw['cudnn']}"),
        ("diffusers / transformers / gradio", f"{sw['diffusers']} / {sw['transformers']} / {sw['gradio']}"),
        ("Running in Docker", "yes" if sw["in_container"] else "no"),
        ("App process", f"PID {a['pid']} · started {a['started'].replace('T', ' ')} · up {a['uptime']} · {human_bytes(a['rss'])} RSS · {a['threads']} threads"),
        ("Working directory", f"`{a['working_dir']}`"),
        ("Generations recorded", f"{stats.get('images') or 0} images · {stats.get('total_seconds') or 0} s GPU time · hottest {stats.get('max_temp_c') or 'n/a'} °C · peak {stats.get('max_power_w') or 'n/a'} W"),
    ])
    return tiles_html, gpu_md, cpu_md, mem_md, net_md, power_md, soft_md


# ----------------------------------------------------------------------------- ui

with gr.Blocks(title="Frazile AI · Image Studio", fill_width=True) as demo:
    gr.HTML(header_html(STATIC_GPU))

    with gr.Tabs():
        with gr.Tab("Generate"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=380):
                    prompt = gr.Textbox(
                        label="Prompt", lines=5,
                        placeholder="a cozy cabin in a snowy forest at dusk, cinematic, 16:9, 4k",
                        info="Enter generates · Shift+Enter adds a line",
                    )
                    interpreted = gr.Markdown("", elem_classes=["fz-interp"])
                    model_choice = gr.Dropdown(
                        MODEL_CHOICES, value="auto", label="Model",
                        info="Or name it in the prompt: 'model juggernaut'. Switching takes ~25 s; the first use of a model downloads it.",
                    )
                    loaded_md = gr.Markdown(loaded_line(), elem_classes=["fz-loaded"])
                    negative = gr.Textbox(label="Negative prompt (what to avoid)", lines=2, value=DEFAULT_NEGATIVE)
                    with gr.Row():
                        run = gr.Button("Generate", variant="primary", size="lg", scale=3)
                        clear = gr.Button("Clear", variant="secondary", size="lg", scale=1, min_width=90)
                    with gr.Accordion("What can I write in the prompt?", open=False):
                        gr.Markdown(HELP)
                    with gr.Accordion("Models", open=False):
                        gr.Markdown(models_markdown())

                with gr.Column(scale=5):
                    gallery = gr.Gallery(label="Results", columns=2, height=640, object_fit="contain", preview=False)
                    with gr.Row(visible=False, elem_classes=["fz-upscale"]) as upscale_bar:
                        upscale_target = gr.Radio(UPSCALE_CHOICES, value="4k", label="Upscale with Real-ESRGAN", scale=3)
                        upscale_btn = gr.Button("Upscale", variant="primary", scale=1, min_width=140)
                    upscale_note = gr.Markdown("", elem_classes=["fz-hint"])
                    with gr.Accordion("Generation details (GPU, timing, settings)", open=True):
                        details = gr.Markdown("")

            last_records = gr.State([])
            selected_index = gr.State(None)

            prompt.change(preview, [prompt, model_choice], interpreted, show_progress="hidden")
            model_choice.change(preview, [prompt, model_choice], interpreted, show_progress="hidden")
            gen_outputs = [gallery, details, last_records, selected_index, upscale_bar, upscale_note, loaded_md]
            run.click(generate, [prompt, negative, model_choice], gen_outputs)
            prompt.submit(generate, [prompt, negative, model_choice], gen_outputs)
            clear.click(clear_all, None, [prompt, gallery, details, last_records, selected_index, upscale_bar, upscale_note, interpreted])
            gallery.select(on_select, [last_records], [selected_index, upscale_note])
            upscale_btn.click(upscale, [last_records, selected_index, upscale_target],
                              [gallery, details, last_records, upscale_note])

            gr.Examples(
                examples=[
                    ["a cozy cabin in a snowy forest at dusk, cinematic, 16:9, 4k"],
                    ["close-up portrait of an old fisherman, weathered skin, golden hour, photorealistic"],
                    ["a knight in silver armour in a misty forest, (glowing blue sword:1.3), model realvis"],
                    ["anime girl with a red umbrella in the rain, tall, 2 images"],
                    ["a corgi astronaut floating in space, 3d render, square, seed 7, using dreamshaper"],
                    ["watercolor of a paris street cafe, tall, quick"],
                ],
                inputs=[prompt],
            )

        with gr.Tab("History") as history_tab:
            stats_md = gr.Markdown(history_stats())
            with gr.Row():
                search = gr.Textbox(label="Search prompts", placeholder="cabin", scale=3)
                limit = gr.Number(label="Show last", value=50, precision=0, scale=1)
                refresh = gr.Button("Refresh", scale=1)
            table = gr.Dataframe(
                value=history_table(), headers=HISTORY_COLUMNS, interactive=False, wrap=True,
                label="Click a row to see the image and its full details",
            )
            with gr.Row():
                hist_image = gr.Image(label="Image", type="filepath", height=520)
                hist_details = gr.Markdown("")

            history_tab.select(refresh_history, [search, limit], [table, stats_md])
            refresh.click(refresh_history, [search, limit], [table, stats_md])
            search.submit(refresh_history, [search, limit], [table, stats_md])
            table.select(select_history, [table], [hist_image, hist_details])

        with gr.Tab("Admin") as admin_tab:
            with gr.Row():
                admin_refresh_btn = gr.Button("Refresh", variant="primary", scale=1, min_width=120)
                auto_refresh = gr.Checkbox(label="Auto-refresh every 5 s", value=False, scale=1)
                internet_check = gr.Checkbox(label="Check internet reachability (Hugging Face, GitHub, PyPI)", value=False, scale=2)
            admin_tiles = gr.HTML("")
            with gr.Row():
                with gr.Column():
                    admin_gpu = gr.Markdown("")
                    admin_power = gr.Markdown("")
                    admin_soft = gr.Markdown("")
                with gr.Column():
                    admin_cpu = gr.Markdown("")
                    admin_mem = gr.Markdown("")
                    admin_net = gr.Markdown("")
            admin_outputs = [admin_tiles, admin_gpu, admin_cpu, admin_mem, admin_net, admin_power, admin_soft]
            timer = gr.Timer(5, active=False)

            admin_tab.select(admin_refresh, [internet_check], admin_outputs)
            admin_refresh_btn.click(admin_refresh, [internet_check], admin_outputs)
            timer.tick(admin_refresh, [internet_check], admin_outputs, show_progress="hidden")
            auto_refresh.change(lambda on: gr.Timer(active=on), auto_refresh, timer)

if __name__ == "__main__":
    in_docker = os.environ.get("FRAZILE_IN_DOCKER") == "1"
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        inbrowser=not in_docker and os.environ.get("FRAZILE_NO_BROWSER") != "1",
        theme=build_theme(),
        css=CSS,
    )
