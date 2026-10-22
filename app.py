"""
Frazile AI Image Studio: local web UI for AI image generation on your GPU (diffusers + Gradio).

Run:
    .venv/Scripts/python.exe app.py
Then open http://127.0.0.1:7860 in your browser.

Everything is controlled from the prompt. Examples:
    "a cozy cabin in a snowy forest, 16:9, 4k"
    "portrait of a cyberpunk samurai, 3 images, seed 42"
    "desktop wallpaper of a mountain lake, photorealistic"        -> auto-picks Juggernaut XL
    "a knight in silver armour, model realvis, (glowing sword:1.3)"
    "neon street at night using flux"
See prompt_parser.py for the full list of understood words and models.py for the models.

After a generation an Upscale bar appears under the results: pick 2K / 4K / 8K and upscale the
selected image (or all of them) with Real-ESRGAN. Upscales are recorded on the same database row.

Every generation is recorded in generations.db (SQLite) with its settings, timings and GPU
telemetry. The History tab browses that database.
"""

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
from models import ModelManager, ModelSpec
from prompt_parser import Settings, parse
from upscaler import TARGETS, Upscaler

STARTUP_MODEL = "sdxl"
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


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=PINK,
        secondary_hue=YELLOW,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_lg,
    )
    import inspect
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
.gradio-container {{ max-width: 1440px !important; }}
footer {{ display: none !important; }}

.fz-header {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 16px;
              padding: 10px 4px 18px; border-bottom: 1px solid {BORDER}; margin-bottom: 14px; }}
.fz-brand {{ display: flex; align-items: center; gap: 12px; }}
.fz-mark {{ width: 40px; height: 40px; border-radius: 12px; background: {GRADIENT};
            box-shadow: 0 8px 24px -8px rgba(236,72,153,.7); display: grid; place-items: center;
            font-weight: 900; font-size: 20px; color: #1A0A12; letter-spacing: -.04em; }}
.fz-logo {{ font-weight: 800; font-size: 26px; letter-spacing: -.02em; line-height: 1; color: {TEXT}; }}
.fz-logo span {{ background: {GRADIENT}; -webkit-background-clip: text; background-clip: text; color: transparent; }}
.fz-tag {{ color: {MUTED}; font-size: 13px; margin-top: 4px; }}
.fz-sys {{ color: {MUTED}; font-size: 12px; font-family: var(--font-mono); text-align: right; line-height: 1.6; }}
.fz-sys b {{ color: #FACC15; font-weight: 600; }}

.fz-interp {{ border-left: 3px solid #EC4899; background: #17122A; border-radius: 10px;
              padding: 8px 12px; font-size: 13px; margin: 2px 0 8px; }}
.fz-interp p {{ margin: 0; }}
.fz-interp small {{ color: {MUTED}; }}

.fz-upscale {{ border: 1px dashed #3A2C4A; border-radius: 14px; padding: 10px 12px;
               background: linear-gradient(180deg, #151225 0%, {SURFACE} 100%); margin-top: 8px; }}
.fz-upscale .fz-hint {{ color: {MUTED}; font-size: 12px; }}

button.primary {{ font-weight: 700; letter-spacing: .01em;
                  box-shadow: 0 10px 28px -10px rgba(236,72,153,.75); transition: transform .08s ease; }}
button.primary:hover {{ transform: translateY(-1px); }}
button.secondary {{ font-weight: 600; }}
.fz-ghost {{ background: transparent !important; border: 1px solid #3A2C4A !important; color: #FACC15 !important; }}

.tabs > .tab-nav button.selected, .tab-nav button.selected {{ color: #FACC15 !important; border-color: #EC4899 !important; }}
.prose h3 {{ color: #FACC15; }}
.prose table th {{ color: #F9A8D4; }}
.prose code {{ background: #1C1530; color: #FDE68A; border: 1px solid #3A2C4A; }}
.gallery-item, .thumbnail-item {{ border-radius: 12px !important; }}
.progress-bar {{ background: {GRADIENT} !important; }}
"""


def header_html(static_gpu: dict, upscaler_name: str) -> str:
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
    <b>{static_gpu['gpu_name']}</b> · {static_gpu['vram_total_gb']} GB VRAM<br>
    CUDA {static_gpu['cuda_version']} · torch {static_gpu['torch_version']} · {upscaler_name}
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
        "**Auto** picks a model from the style words in the prompt (photo/cinematic → Juggernaut, anime → Animagine, "
        "painting → Playground, digital art/comic → DreamShaper) when that model is already on disk; otherwise it keeps "
        "the model currently loaded.",
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
    if suited and suited.key != current.key:
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
    """Upscale one generation's native image to target_px long side; save, record in the DB, return fields."""
    img = Image.open(rec["image_path"]).convert("RGB")
    up_img, up_info = UPSCALER.upscale_to(img, target_px)
    if exact_size:
        tw, th = exact_size
        left, top = max(0, (up_img.width - tw) // 2), max(0, (up_img.height - th) // 2)
        up_img = up_img.crop((left, top, left + min(tw, up_img.width), top + min(th, up_img.height)))
    up_path = Path(rec["image_path"]).with_name(f"{Path(rec['image_path']).stem}_{target_px}px.png")
    up_img.save(up_path)
    fields = {
        "upscale_target": target_px,
        "upscaled_width": up_img.width,
        "upscaled_height": up_img.height,
        "upscaled_path": str(up_path),
        "upscaled_size_kb": round(up_path.stat().st_size / 1024, 1),
        "upscale_seconds": up_info["seconds"],
        "upscale_passes": up_info["passes"],
        "upscaler": up_info["upscaler"],
    }
    return fields


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
            up = upscale_record(rec, s.upscale_to, s.exact_size)
            rec.update(up)
            rec["vram_peak_gb"] = gpu_info.torch_memory()["vram_peak_gb"]  # peak now includes the upscaler
            final_path = up["upscaled_path"]

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
    )


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
        torch.cuda.reset_peak_memory_stats()
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


# ----------------------------------------------------------------------------- ui

with gr.Blocks(title="Frazile AI · Image Studio") as demo:
    gr.HTML(header_html(STATIC_GPU, UPSCALER.name))

    with gr.Tabs():
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1, min_width=360):
                    prompt = gr.Textbox(
                        label="Prompt", lines=4,
                        placeholder="a cozy cabin in a snowy forest at dusk, cinematic, 16:9, 4k",
                    )
                    interpreted = gr.Markdown("", elem_classes=["fz-interp"])
                    model_choice = gr.Dropdown(
                        MODEL_CHOICES, value="auto", label="Model",
                        info="Or name it in the prompt: 'model juggernaut'. Switching takes ~25 s; the first use of a model downloads it.",
                    )
                    negative = gr.Textbox(label="Negative prompt (what to avoid)", lines=2, value=DEFAULT_NEGATIVE)
                    run = gr.Button("Generate", variant="primary", size="lg")
                    with gr.Accordion("What can I write in the prompt?", open=False):
                        gr.Markdown(HELP)
                    with gr.Accordion("Models", open=False):
                        gr.Markdown(models_markdown())

                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="Results", columns=2, height=560, object_fit="contain")
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
            gen_outputs = [gallery, details, last_records, selected_index, upscale_bar, upscale_note]
            run.click(generate, [prompt, negative, model_choice], gen_outputs)
            prompt.submit(generate, [prompt, negative, model_choice], gen_outputs)
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
                hist_image = gr.Image(label="Image", type="filepath", height=480)
                hist_details = gr.Markdown("")

            history_tab.select(refresh_history, [search, limit], [table, stats_md])
            refresh.click(refresh_history, [search, limit], [table, stats_md])
            search.submit(refresh_history, [search, limit], [table, stats_md])
            table.select(select_history, [table], [hist_image, hist_details])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=build_theme(), css=CSS)
