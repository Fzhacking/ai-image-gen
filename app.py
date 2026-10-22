"""
Frazile AI Image Studio: local web UI for AI image generation on your GPU (diffusers + Gradio).

Run:
    .venv/Scripts/python.exe app.py
Then open http://127.0.0.1:7860 in your browser.

Environment (used by the Docker image, optional otherwise):
    GRADIO_SERVER_NAME       bind address, default 127.0.0.1 (0.0.0.0 inside Docker)
    GRADIO_SERVER_PORT       port, default 7860
    FRAZILE_NO_BROWSER       set to 1 to not open a browser tab on start
    FRAZILE_STARTUP_MODEL    model key to load at start (overrides the Admin setting; "none" = load on first use)
    FRAZILE_ADMIN_PASSWORD   if set, destructive Admin actions need this password (Admin > Settings)

Everything is controlled from the prompt. Examples:
    "a cozy cabin in a snowy forest, 16:9, 4k"
    "portrait of a cyberpunk samurai, 3 images, seed 42"
    "desktop wallpaper of a mountain lake, photorealistic"        -> auto-picks Juggernaut XL
    "a knight in silver armour, model realvis, (glowing sword:1.3)"
    "neon street at night using flux"
See prompt_parser.py for the full list of understood words and models.py for the models.

Tabs:
    Studio     prompt, quick chips, model and LoRA boxes, results, upscale bar (2K/4K/8K with Real-ESRGAN)
    Video      text/image-to-video with LTX-Video or Wan 2.1, frame-wise upscale to Full HD/2K/4K, soundtrack (media_ui.py)
    Music      text-to-music with MusicGen (media_ui.py)
    History    every generation (images, videos, tracks) from generations.db: filters, thumbnails, reuse, delete
    Admin      models and LoRAs management, downloads, system stats, settings, logs (admin_ui.py)
"""

import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import applog

applog.install()  # capture every print into the Admin > Logs buffer (console output continues)

# Less fragmentation in the CUDA allocator; set before torch is imported. Keeps the 20B models under 24 GB.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr
import pandas as pd
import torch
from PIL import Image

import admin_ui
import db
import gpu_info
import media
import media_ui
import models
import registry
import video_io
from downloads import JOBS
from media import MediaManager
from models import ModelManager, ModelSpec
from prompt_parser import Settings, parse
from ui_common import (header_html, loaded_line, lora_choice_update, lora_choices, model_cards_html, model_choice_update,
                       model_choices)
from ui_theme import CSS, build_theme
from upscaler import TARGETS, Upscaler

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

UPSCALE_CHOICES = [("2K (2560 px)", "2k"), ("4K (3840 px)", "4k"), ("8K (7680 px)", "8k")]
HISTORY_COLUMNS = ["id", "kind", "when", "prompt", "model", "size / length", "steps", "seconds", "it/s", "peak VRAM GB", "temp C", "seed"]
KIND_CHOICES = [("Everything", ""), ("Images", "image"), ("Videos", "video"), ("Music", "music")]

HELP = """
**Write everything in the prompt.** The app reads these hints:

| Want | Say |
|---|---|
| A specific model | `model juggernaut`, `model: flux`, `using realvis` (or pick it in the Model box) |
| Emphasis | `(glowing sword:1.3)`, `(word)++`, `(word)--` (SDXL models; long prompts are not cut off) |
| LoRA | `<lora:name:0.8>` for `models/loras/name.safetensors`, or tick it in the LoRAs box |
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

EMPTY_STATE = """
<div class="fz-empty"><b>Your images appear here</b>
Describe what you want, add hints like <code>16:9</code>, <code>4k</code>, <code>3 images</code> or <code>model juggernaut</code>,
then press <b>Generate</b> or Enter.<br>The chips above the prompt add the common hints for you.</div>"""


# ----------------------------------------------------------------------------- start-up

MANAGER = ModelManager()
MEDIA = MediaManager(MANAGER.lock)       # video and music models; shares the GPU lock with the image manager
STARTUP_MODEL = os.environ.get("FRAZILE_STARTUP_MODEL") or registry.setting("startup_model") or "zimage"
if STARTUP_MODEL != "none":
    try:
        _startup_spec = models.resolve(STARTUP_MODEL)
        if models.kind_of(_startup_spec.family) == "image":
            MANAGER.load(_startup_spec)
        else:
            MEDIA.load(_startup_spec)
    except Exception as e:  # a missing download or a bad registry entry must not stop the app
        print(f"Could not load the start-up model '{STARTUP_MODEL}': {e}")
UPSCALER = Upscaler()
print(f"Upscaler ready: {UPSCALER.name}")
STATIC_GPU = gpu_info.static_info()
media_ui.CTX.manager, media_ui.CTX.media, media_ui.CTX.upscaler, media_ui.CTX.static_gpu, media_ui.CTX.output_dir = (
    MANAGER, MEDIA, UPSCALER, STATIC_GPU, OUTPUT_DIR)


def ui_lora_weight(stem: str) -> float:
    info = models.lora_info(stem)
    return float(info.get("weight") or registry.setting("ui_lora_weight") or 0.8)


def default_negative() -> str:
    return registry.setting("default_negative") or ""


# ----------------------------------------------------------------------------- model selection

def _image_only(spec: ModelSpec) -> ModelSpec:
    kind = models.kind_of(spec.family)
    if kind != "image":
        raise gr.Error(f"{spec.name} makes {kind}: use the {kind.capitalize()} tab for it.")
    return spec


def pick_model(s: Settings, choice: str) -> tuple[ModelSpec, str | None]:
    """Which model to use for this prompt, and an optional note explaining an automatic choice."""
    if s.model:
        try:
            return _image_only(models.resolve(s.model)), None
        except KeyError:
            raise gr.Error(f"Unknown model '{s.model}'. Available: {', '.join(m.key for m in models.all_models(kind='image'))}")
    if choice and choice != "auto":
        return _image_only(models.resolve(choice)), None
    fallback = models.resolve(STARTUP_MODEL) if STARTUP_MODEL != "none" else None
    if fallback is None or models.kind_of(fallback.family) != "image":
        fallback = models.all_models(kind="image")[0]
    current = MANAGER.spec or fallback
    if s.loras:
        # Switch to a model of the family the LoRAs were trained for, so they take effect.
        families = sorted({models.lora_info(n).get("family") or "sdxl" for n, _ in s.loras})
        if current.family not in families:
            for fam in families:
                cands = models.family_models(fam)
                if cands:
                    return cands[0], f"LoRAs need a {fam} model: using {cands[0].name}"
    suited = models.model_for_style(s.style)
    # Z-Image, Qwen and FLUX handle every style themselves; Auto only swaps between SDXL-family specialists.
    if suited and suited.key != current.key and current.family == "sdxl":
        if models.is_downloaded(suited):
            return suited, f"auto: {suited.name} for '{s.style}'"
        return current, f"{suited.name} suits '{s.style}' but is not downloaded; say 'model {suited.key}' to fetch it"
    return current, None


# ----------------------------------------------------------------------------- studio tab

def add_ui_loras(s: Settings, selected: list[str] | None) -> None:
    """LoRAs ticked in the LoRAs box join the prompt's <lora:...> tags (prompt tags win on weight)."""
    named = {models._slug(n) for n, _ in s.loras}
    for name in selected or []:
        if models._slug(name) not in named:
            s.loras.append((name, ui_lora_weight(name)))


def trigger_hints(s: Settings) -> list[str]:
    """'add pixel art to the prompt' for LoRAs whose trigger words are missing."""
    hints = []
    low = s.prompt.lower()
    for name, _ in s.loras:
        trig = models.lora_info(name).get("trigger") or ""
        if trig and trig.split(",")[0].strip().lower() not in low:
            hints.append(f"{name}: add '{trig.split(',')[0].strip()}' to the prompt")
    return hints


def preview(prompt: str, choice: str, loras_ui: list[str] | None = None) -> str:
    """Live 'interpreted as' line shown under the prompt box."""
    if not prompt.strip():
        return ""
    s = parse(prompt)
    add_ui_loras(s, loras_ui)
    try:
        spec, note = pick_model(s, choice)
    except gr.Error as e:
        return f"**{e}**"
    s.apply_defaults(spec.steps, spec.guidance)
    if note:
        s.notes.append(note)
    line = f"**Will generate:** {s.summary()}  ·  model **{spec.name}**"
    for name, _ in s.loras:
        fam = models.lora_info(name).get("family")
        if fam and fam != spec.family:
            s.notes.append(f"LoRA {name} is for {fam} models and will be ignored on {spec.name}")
    s.notes.extend(trigger_hints(s))
    if MANAGER.spec is None or MANAGER.spec.key != spec.key:
        s.notes.append(f"{spec.name} is not loaded: the first run loads it" + ("" if models.is_downloaded(spec) else f" (downloads ~{spec.download_gb:g} GB)"))
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


def generate(prompt: str, negative: str, model_choice: str, loras_ui: list[str] | None = None, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    s = parse(prompt)
    add_ui_loras(s, loras_ui)
    spec, auto_note = pick_model(s, model_choice)
    if auto_note:
        s.notes.append(auto_note)
        gr.Info(auto_note)
    s.apply_defaults(spec.steps, spec.guidance)
    negative = (negative.strip() or default_negative()) if spec.negative else ""

    with MANAGER.lock:
        if MANAGER.spec is None or MANAGER.spec.key != spec.key:
            progress(0, desc=f"Loading {spec.name}" + ("" if models.is_downloaded(spec) else f" (downloading ~{spec.download_gb:g} GB)"))
        s.notes.extend(media.prepare_for(spec, MANAGER, MEDIA))   # a loaded video model gives the card back first
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
                            "width": s.width, "height": s.height, "seed": seed})
            if registry.setting("auto_release_vram", True):
                MANAGER.release_cache()

    return (
        images,
        "\n\n---\n\n".join(reports),
        records,
        None,
        gr.update(visible=True),
        upscale_hint(records, None),
        loaded_line(MANAGER, UPSCALER.name, MEDIA),
        gr.update(visible=False),
    )


def clear_all():
    return "", None, "", [], None, gr.update(visible=False), "", "", gr.update(visible=True)


def upscale_hint(records: list[dict], selected: int | None) -> str:
    if not records:
        return ""
    if selected is not None and 0 <= selected < len(records):
        r = records[selected]
        done = f" · already {Path(r['upscaled_path']).stem.rsplit('_', 1)[-1]}" if r.get("upscaled_path") else ""
        return f"Image {selected + 1} of {len(records)} selected ({r['width']}x{r['height']}, seed {r['seed']}{done}). Click another image to change."
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
    with MANAGER.lock:
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


# ----------------------------------------------------------------------------- prompt chips

RATIO_CHIPS = ["1:1", "16:9", "9:16", "4:3", "3:2", "21:9"]
SIZE_CHIPS = ["2k", "4k", "8k"]
COUNT_CHIPS = ["2 images", "4 images"]
QUALITY_CHIPS = ["quick", "highly detailed"]
STYLE_CHIPS = ["photorealistic", "cinematic", "anime", "watercolor", "3d render", "pixel art"]
CHIP_GROUPS = {t: RATIO_CHIPS for t in RATIO_CHIPS}
CHIP_GROUPS.update({t: SIZE_CHIPS for t in SIZE_CHIPS})
CHIP_GROUPS.update({t: COUNT_CHIPS for t in COUNT_CHIPS})
CHIP_GROUPS.update({t: QUALITY_CHIPS for t in QUALITY_CHIPS})
CHIP_GROUPS.update({t: STYLE_CHIPS for t in STYLE_CHIPS})


def toggle_chip(prompt: str, token: str) -> str:
    """Add a hint to the prompt, or remove it if present; hints in the same group replace each other."""
    parts = [p.strip() for p in (prompt or "").split(",")]
    parts = [p for p in parts if p]
    low = [p.lower() for p in parts]
    if token.lower() in low:
        return ", ".join(p for p in parts if p.lower() != token.lower())
    group = {t.lower() for t in CHIP_GROUPS.get(token, [])}
    parts = [p for p in parts if p.lower() not in group]
    return ", ".join(parts + [token])


def chip_seed(prompt: str) -> str:
    cleaned = re.sub(r"\s*,?\s*\bseed\s*[:=]?\s*\d+", "", prompt or "", flags=re.IGNORECASE).strip(" ,")
    return f"{cleaned}, seed {random.randint(1, 99999)}" if cleaned else f"seed {random.randint(1, 99999)}"


def reuse_seed(prompt: str, records: list[dict], selected: int | None) -> str:
    if not records:
        raise gr.Error("Generate an image first.")
    r = records[selected if selected is not None and 0 <= selected < len(records) else 0]
    cleaned = re.sub(r"\s*,?\s*\bseed\s*[:=]?\s*\d+", "", prompt or "", flags=re.IGNORECASE).strip(" ,")
    return f"{cleaned}, seed {r['seed']}"


def drop_seed(prompt: str) -> str:
    return re.sub(r"\s*,?\s*\bseed\s*[:=]?\s*\d+", "", prompt or "", flags=re.IGNORECASE).strip(" ,")


# ----------------------------------------------------------------------------- history tab

def history_model_choices():
    return [("All models", "")] + [(models.display_name(m), m) for m in db.models_used()]


def history_rows(search: str = "", limit: int = 50, model_id: str = "", upscaled_only: bool = False, kind: str = "") -> list[dict]:
    return db.fetch_filtered(limit=int(limit or 50), search=search or "", model_id=model_id or None, upscaled_only=bool(upscaled_only),
                             kind=kind or None)


def _size_text(r: dict) -> str:
    kind = r.get("kind") or "image"
    if kind == "music":
        return f"{r.get('duration_s') or 0:.0f} s · {(r.get('sample_rate') or 0) // 1000} kHz"
    base = f"{r['width']}x{r['height']}"
    if kind == "video":
        base += f" · {r.get('duration_s') or 0:.1f} s"
    if r.get("upscaled_width"):
        base += f" → {r['upscaled_width']}x{r['upscaled_height']}"
    return base


def history_table(rows: list[dict]) -> pd.DataFrame:
    data = [
        [r["id"], r.get("kind") or "image", r["created_at"][:16].replace("T", " "), r["user_prompt"][:70],
         models.display_name(r.get("model_id") or ""), _size_text(r),
         r["steps"], r["inference_seconds"], r["steps_per_second"],
         r["vram_peak_gb"], r["temp_after_c"], r["seed"]]
        for r in rows
    ]
    return pd.DataFrame(data, columns=HISTORY_COLUMNS)


THUMB_DIR = OUTPUT_DIR / ".thumbs"
THUMB_SIZE = 384


def thumbnail_for(path: str) -> str | None:
    """A small JPEG next to the outputs (outputs/.thumbs/), built once per image, so the History grid
    never decodes full-size or upscaled PNGs."""
    src = Path(path)
    if not src.exists():
        return None
    THUMB_DIR.mkdir(exist_ok=True)
    ext = src.suffix.lower()
    thumb = THUMB_DIR / (f"{src.stem}.png" if ext == ".wav" else f"{src.stem}.jpg")
    if not thumb.exists() or thumb.stat().st_mtime < src.stat().st_mtime:
        try:
            if ext == ".mp4":
                video_io.video_thumbnail(src, thumb, THUMB_SIZE)       # first frame
            elif ext == ".wav":
                video_io.waveform_png(src, thumb)                     # waveform picture
            else:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    im.thumbnail((THUMB_SIZE, THUMB_SIZE))
                    im.save(thumb, "JPEG", quality=82)
        except (OSError, RuntimeError):
            return str(src) if ext not in (".mp4", ".wav") else None
    return str(thumb)


def history_gallery(rows: list[dict]) -> list[tuple[str, str]]:
    out = []
    icons = {"video": "▶ ", "music": "♪ "}
    for r in rows[:60]:
        t = thumbnail_for(r["image_path"])
        if t:
            out.append((t, f"#{r['id']} {icons.get(r.get('kind') or 'image', '')}· {r['user_prompt'][:60]}"))
    return out


def history_stats() -> str:
    st = db.stats()
    rc = db.recent_counts()
    if not st["images"]:
        return "No generations recorded yet."
    k = db.counts_by_kind()
    return (
        f"**{st['images']} generations** ({k['image']['count']} images, {k['video']['count']} videos, {k['music']['count']} tracks; "
        f"{rc['today']} today, {rc['week']} this week)  ·  {st['total_seconds']} s of GPU time  ·  "
        f"avg {st['avg_inference_seconds']} s (fastest {st['fastest_seconds']} s, slowest {st['slowest_seconds']} s)  ·  "
        f"avg {st['avg_steps_per_second']} steps/s  ·  avg peak VRAM {st['avg_vram_peak_gb']} GB "
        f"(max {st['max_vram_peak_gb']} GB)  ·  hottest {st['max_temp_c']} °C  ·  peak power {st['max_power_w']} W  ·  "
        f"{st['upscaled_images']} upscaled"
    )


def refresh_history(search: str, limit: int, model_id: str, upscaled_only: bool, kind: str = ""):
    rows = history_rows(search, limit, model_id, upscaled_only, kind)
    return (history_table(rows), history_gallery(rows), history_stats(),
            gr.update(choices=history_model_choices(), value=model_id if model_id in db.models_used() else ""))


_NO_PREVIEW = (gr.update(value=None, visible=True), gr.update(value=None, visible=False), gr.update(value=None, visible=False))


def show_record(row_id: int | None):
    """(image, video, audio, details, id): the player matching the record's kind is shown, the others hidden."""
    if not row_id:
        return (*_NO_PREVIEW, "", None)
    rec = db.get(int(row_id))
    if not rec:
        return (*_NO_PREVIEW, f"Generation #{row_id} not found.", None)
    kind = rec.get("kind") or "image"
    shown = rec.get("soundtrack_path") if kind == "video" and rec.get("soundtrack_path") else None
    shown = shown or rec.get("upscaled_path") or rec["image_path"]
    shown = shown if Path(shown).exists() else None
    img = gr.update(value=shown if kind == "image" else None, visible=kind == "image")
    vid = gr.update(value=shown if kind == "video" else None, visible=kind == "video")
    aud = gr.update(value=shown if kind == "music" else None, visible=kind == "music")
    details = details_markdown(rec) if kind == "image" else media_ui.media_details_markdown(rec)
    return img, vid, aud, details, int(row_id)


def select_history(evt: gr.SelectData, table: pd.DataFrame):
    if table is None or len(table) == 0:
        return show_record(None)
    return show_record(int(table.iloc[evt.index[0]]["id"]))


def select_history_thumb(evt: gr.SelectData):
    caption = evt.value.get("caption", "") if isinstance(evt.value, dict) else ""
    m = re.match(r"#(\d+)", caption or "")
    return show_record(int(m.group(1)) if m else None)


def reuse_prompt(row_id: int | None, with_seed: bool):
    """Put the prompt back into the tab that made it (Studio, Video or Music) and switch there."""
    if not row_id:
        raise gr.Error("Select a generation first.")
    rec = db.get(int(row_id))
    if not rec:
        raise gr.Error("That generation no longer exists.")
    kind = rec.get("kind") or "image"
    prompt = drop_seed(rec["user_prompt"])
    if with_seed and kind == "image":
        prompt = f"{prompt}, seed {rec['seed']}"
    if with_seed and kind != "image":
        gr.Info(f"Seed {rec['seed']}: enter it in the Seed box of the {kind.capitalize()} tab")
    tab = {"image": "studio", "video": "video", "music": "music"}[kind]
    keep = gr.update()
    return (prompt if kind == "image" else keep, prompt if kind == "video" else keep, prompt if kind == "music" else keep,
            gr.Tabs(selected=tab))


def delete_history(row_id: int | None, remove_files: bool, search: str, limit: int, model_id: str, upscaled_only: bool, kind: str = ""):
    if not row_id:
        raise gr.Error("Select a generation first.")
    removed = db.delete(int(row_id), remove_files=remove_files)
    gr.Info(f"Deleted generation #{row_id}" + (f" and {len(removed)} file(s)" if removed else ""))
    rows = history_rows(search, limit, model_id, upscaled_only, kind)
    return history_table(rows), history_gallery(rows), history_stats(), *_NO_PREVIEW, "", None


# ----------------------------------------------------------------------------- ui

def refresh_header():
    return header_html(STATIC_GPU, MANAGER, len(JOBS.active()), MEDIA)


with gr.Blocks(title="Frazile AI · Image Studio (images, video, music)", fill_width=True) as demo:
    header = gr.HTML(refresh_header())

    with gr.Tabs() as tabs:
        with gr.Tab("Studio", id="studio") as studio_tab:
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=400):
                    gr.HTML('<div class="fz-chips-label">Quick hints · click to add or remove</div>')
                    with gr.Row(elem_classes=["fz-chips"]):
                        chip_buttons = [gr.Button(t, size="sm", variant="secondary") for t in RATIO_CHIPS + SIZE_CHIPS]
                    with gr.Row(elem_classes=["fz-chips"]):
                        chip_buttons += [gr.Button(t, size="sm", variant="secondary") for t in COUNT_CHIPS + QUALITY_CHIPS]
                        seed_chip = gr.Button("random seed", size="sm", variant="secondary")
                    with gr.Row(elem_classes=["fz-chips"]):
                        chip_buttons += [gr.Button(t, size="sm", variant="secondary") for t in STYLE_CHIPS]
                    prompt = gr.Textbox(
                        label="Prompt", lines=5,
                        placeholder="a cozy cabin in a snowy forest at dusk, cinematic, 16:9, 4k",
                        info="Enter generates · Shift+Enter adds a line",
                    )
                    interpreted = gr.Markdown("", elem_classes=["fz-interp"])
                    model_choice = gr.Dropdown(
                        model_choices(), value="auto", label="Model", allow_custom_value=True,
                        info="Or name it in the prompt: 'model juggernaut'. Switching takes ~25 s (longer for Z-Image and Qwen); the first use of a model downloads it.",
                    )
                    loaded_md = gr.Markdown(loaded_line(MANAGER, UPSCALER.name, MEDIA), elem_classes=["fz-loaded"])
                    lora_choice = gr.Dropdown(
                        lora_choices(), multiselect=True, label="LoRAs", allow_custom_value=True,
                        info="Each LoRA applies at its own default weight (Admin › LoRAs); write <lora:name:1.2> in the prompt for another. "
                             "Auto switches to a model of the LoRA's family.",
                    )
                    negative = gr.Textbox(label="Negative prompt (what to avoid)", lines=2, value=default_negative())
                    with gr.Row():
                        run = gr.Button("Generate", variant="primary", size="lg", scale=3)
                        clear = gr.Button("Clear", variant="secondary", size="lg", scale=1, min_width=90)
                    with gr.Accordion("What can I write in the prompt?", open=False):
                        gr.Markdown(HELP)
                    with gr.Accordion("Models and LoRAs available", open=False) as models_acc:
                        model_cards = gr.HTML(model_cards_html(MANAGER))

                with gr.Column(scale=5):
                    empty_state = gr.HTML(EMPTY_STATE)
                    gallery = gr.Gallery(label="Results", columns=2, height=640, object_fit="contain", preview=False)
                    with gr.Row(visible=False, elem_classes=["fz-upscale"]) as upscale_bar:
                        upscale_target = gr.Radio(UPSCALE_CHOICES, value="4k", label="Upscale with Real-ESRGAN", scale=3)
                        upscale_btn = gr.Button("Upscale", variant="primary", scale=1, min_width=140)
                        reuse_seed_btn = gr.Button("Reuse this seed", variant="secondary", scale=1, min_width=140)
                        variations_btn = gr.Button("New variation", variant="secondary", scale=1, min_width=140)
                    upscale_note = gr.Markdown("", elem_classes=["fz-hint"])
                    with gr.Accordion("Generation details (GPU, timing, settings)", open=True):
                        details = gr.Markdown("")

            last_records = gr.State([])
            selected_index = gr.State(None)

            preview_inputs = [prompt, model_choice, lora_choice]
            prompt.change(preview, preview_inputs, interpreted, show_progress="hidden")
            model_choice.change(preview, preview_inputs, interpreted, show_progress="hidden")
            lora_choice.change(preview, preview_inputs, interpreted, show_progress="hidden")
            gen_inputs = [prompt, negative, model_choice, lora_choice]
            gen_outputs = [gallery, details, last_records, selected_index, upscale_bar, upscale_note, loaded_md, empty_state]
            run.click(generate, gen_inputs, gen_outputs, concurrency_id="gpu")
            prompt.submit(generate, gen_inputs, gen_outputs, concurrency_id="gpu")
            variations_btn.click(drop_seed, prompt, prompt).then(generate, gen_inputs, gen_outputs, concurrency_id="gpu")
            clear.click(clear_all, None, [prompt, gallery, details, last_records, selected_index, upscale_bar, upscale_note, interpreted, empty_state])
            gallery.select(on_select, [last_records], [selected_index, upscale_note])
            upscale_btn.click(upscale, [last_records, selected_index, upscale_target],
                              [gallery, details, last_records, upscale_note], concurrency_id="gpu")
            reuse_seed_btn.click(reuse_seed, [prompt, last_records, selected_index], prompt)
            for b in chip_buttons:
                b.click(toggle_chip, [prompt, gr.State(b.value)], prompt, show_progress="hidden")
            seed_chip.click(chip_seed, prompt, prompt, show_progress="hidden")
            studio_tab.select(lambda c, l: (model_choice_update(c), lora_choice_update(l), model_cards_html(MANAGER)),
                              [model_choice, lora_choice], [model_choice, lora_choice, model_cards], show_progress="hidden")

            gr.Examples(
                examples=[
                    ["a cozy cabin in a snowy forest at dusk, cinematic, 16:9, 4k"],
                    ["close-up portrait of an old fisherman, weathered skin, golden hour, photorealistic"],
                    ["a neon sign that says FRAZILE AI above a rainy street at night, cinematic, 16:9"],
                    ["a knight in silver armour in a misty forest, (glowing blue sword:1.3), model realvis"],
                    ["anime girl with a red umbrella in the rain, tall, 2 images"],
                    ["a corgi astronaut floating in space, 3d render, square, seed 7, using dreamshaper"],
                    ["pixel art of a knight guarding a castle gate, <lora:pixel-art-xl:0.9>, model juggernaut, seed 3"],
                    ["papercut landscape of mountains and a lake at sunset, <lora:papercut>, model juggernaut, 16:9"],
                    ["a vintage poster for a coffee shop with the headline GRAND OPENING, model qwen"],
                    ["watercolor of a paris street cafe, tall, quick"],
                ],
                inputs=[prompt],
            )

        with gr.Tab("Video", id="video") as video_tab:
            video_ui = media_ui.build_video_tab(loaded_md)
            video_tab.select(media_ui.refresh_video_choices, [video_ui.model, video_ui.track], [video_ui.model, video_ui.track],
                             show_progress="hidden")

        with gr.Tab("Music", id="music") as music_tab:
            music_ui = media_ui.build_music_tab(loaded_md)
            music_tab.select(media_ui.refresh_music_choices, [music_ui.model], [music_ui.model], show_progress="hidden")

        with gr.Tab("History", id="history") as history_tab:
            stats_md = gr.Markdown(history_stats())
            with gr.Row():
                search = gr.Textbox(label="Search prompts", placeholder="cabin", scale=3)
                h_kind = gr.Dropdown(KIND_CHOICES, value="", label="Kind", scale=1)
                h_model = gr.Dropdown(history_model_choices(), value="", label="Model", scale=2)
                h_upscaled = gr.Checkbox(label="Upscaled only", value=False, scale=1)
                limit = gr.Number(label="Show last", value=50, precision=0, scale=1)
                refresh = gr.Button("Refresh", scale=1)
            h_filters = [search, limit, h_model, h_upscaled, h_kind]
            with gr.Tabs():
                with gr.Tab("Table"):
                    table = gr.Dataframe(
                        value=history_table(history_rows()), headers=HISTORY_COLUMNS, interactive=False, wrap=True,
                        label="Click a row to see the image and its full details",
                    )
                with gr.Tab("Thumbnails"):
                    thumbs = gr.Gallery(value=history_gallery(history_rows()), columns=6, height=420, object_fit="cover",
                                        preview=False, label="Click a thumbnail to see its details (▶ video, ♪ music)")
            with gr.Row():
                with gr.Column(scale=1):
                    hist_image = gr.Image(label="Image", type="filepath", height=520)
                    hist_video = gr.Video(label="Video", height=520, visible=False)
                    hist_audio = gr.Audio(label="Track", type="filepath", visible=False)
                    with gr.Row():
                        h_reuse = gr.Button("Reuse prompt", variant="primary")
                        h_reuse_seed = gr.Button("Reuse prompt + seed")
                    with gr.Row():
                        h_del_files = gr.Checkbox(label="Also delete the files", value=False)
                        h_delete = gr.Button("Delete this generation", variant="stop")
                with gr.Column(scale=1):
                    hist_details = gr.Markdown("")
            h_selected = gr.State(None)

            h_outputs = [table, thumbs, stats_md, h_model]
            h_preview = [hist_image, hist_video, hist_audio, hist_details, h_selected]
            history_tab.select(refresh_history, h_filters, h_outputs)
            refresh.click(refresh_history, h_filters, h_outputs)
            search.submit(refresh_history, h_filters, h_outputs)
            h_model.input(refresh_history, h_filters, h_outputs)
            h_upscaled.input(refresh_history, h_filters, h_outputs)
            h_kind.input(refresh_history, h_filters, h_outputs)
            table.select(select_history, [table], h_preview)
            thumbs.select(select_history_thumb, None, h_preview)
            reuse_targets = [prompt, video_ui.prompt, music_ui.prompt, tabs]
            h_reuse.click(lambda i: reuse_prompt(i, False), [h_selected], reuse_targets)
            h_reuse_seed.click(lambda i: reuse_prompt(i, True), [h_selected], reuse_targets)
            h_delete.click(delete_history, [h_selected, h_del_files, *h_filters], [table, thumbs, stats_md, *h_preview])

        with gr.Tab("Admin", id="admin"):
            admin = admin_ui.build_admin(MANAGER, UPSCALER, STATIC_GPU, prompt, tabs, model_choice, lora_choice, loaded_md, media_mgr=MEDIA)

    header_timer = gr.Timer(5, active=True)
    header_timer.tick(refresh_header, None, header, show_progress="hidden", concurrency_id="stats", concurrency_limit=3)

if __name__ == "__main__":
    in_docker = os.environ.get("FRAZILE_IN_DOCKER") == "1"
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        inbrowser=not in_docker and os.environ.get("FRAZILE_NO_BROWSER") != "1",
        theme=build_theme(),
        css=CSS,
        max_file_size="8gb",
        allowed_paths=[str(OUTPUT_DIR), str(THUMB_DIR)],
    )
