"""
Helpers shared by the Studio (app.py) and the Admin panel (admin_ui.py): dropdown choices,
header, stat tiles, key/value tables and small HTML builders.
"""

from __future__ import annotations

from html import escape
from typing import Any

import gradio as gr

import db
import gpu_info
import models
import registry
from downloads import JOBS, human_bytes
from ui_theme import ACCENT, MUTED, STATUS, TEXT, badge, family_badge

HEAVY_NOTE = {"zimage": "no negative prompt, no LoRA weighting", "qwen": "4-bit, Lightning 8 steps", "flux": "gated on Hugging Face"}


# ----------------------------------------------------------------------------- choices

def model_choices() -> list[tuple[str, str]]:
    out = [("Auto (pick by style words in the prompt)", "auto")]
    for m in models.all_models():
        disk = "" if models.is_downloaded(m) else f"  ·  not downloaded (~{m.download_gb:g} GB)"
        out.append((f"{m.name}  -  {m.description}{disk}", m.key))
    return out


def lora_choices(family: str | None = None) -> list[tuple[str, str]]:
    out = []
    for stem, e in registry.lora_entries(include_missing=False).items():
        if not e.get("enabled", True) or (family and e.get("family") != family):
            continue
        trig = f"  ·  say '{e['trigger']}'" if e.get("trigger") else ""
        out.append((f"{e['name']}  [{e['family']}]{trig}", stem))
    return out


def model_choice_update(value: str | None = None):
    choices = model_choices()
    keys = {v for _, v in choices}
    return gr.update(choices=choices, value=value if value in keys else "auto")


def lora_choice_update(value: list[str] | None = None):
    choices = lora_choices()
    keys = {v for _, v in choices}
    return gr.update(choices=choices, value=[v for v in (value or []) if v in keys])


# ----------------------------------------------------------------------------- header

def header_html(static_gpu: dict, manager, jobs_active: int = 0) -> str:
    live = gpu_info.snapshot()
    spec = manager.spec
    used, total = live.get("mem_used_mb"), live.get("mem_total_mb")
    temp = live.get("temperature_c")
    temp_status = "good" if temp is None or temp < 70 else "warning" if temp < 83 else "serious" if temp < 90 else "critical"
    dot = STATUS[temp_status][0]
    pills = [
        f'<span class="fz-pill live"><span class="dot" style="background:{dot}"></span>'
        f'<b>{escape(spec.name) if spec else "no model loaded"}</b></span>',
        f'<span class="fz-pill live">VRAM <b>{(used or 0) / 1024:.1f}</b> / {(total or 0) / 1024:.0f} GB'
        f' · <b>{temp if temp is not None else "?"}</b> °C · {live.get("gpu_util_pct") or 0:.0f} %</span>',
    ]
    if jobs_active:
        pills.append(f'<span class="fz-pill live">⬇ <b>{jobs_active}</b> download{"s" if jobs_active != 1 else ""}</span>')
    pills += [
        f'<span class="fz-pill"><b>{escape(str(static_gpu["gpu_name"]))}</b> · {static_gpu["vram_total_gb"]} GB</span>',
        f'<span class="fz-pill">CUDA {static_gpu["cuda_version"]} · torch {static_gpu["torch_version"]} · diffusers {static_gpu["diffusers_version"]}</span>',
    ]
    return f"""
<div class="fz-header">
  <div class="fz-brand">
    <div class="fz-mark">F</div>
    <div>
      <div class="fz-logo">Frazile <span>AI</span></div>
      <div class="fz-tag">Image Studio · Z-Image, SDXL fine-tunes, Qwen-Image, FLUX, LoRAs and Real-ESRGAN on your own GPU</div>
    </div>
  </div>
  <div class="fz-sys">{''.join(pills)}</div>
</div>"""


def loaded_line(manager, upscaler_name: str) -> str:
    spec = manager.spec
    if not spec:
        return f"Loaded now: **nothing** (the first generation loads a model) · upscaler {upscaler_name}"
    return f"Loaded now: **{spec.name}** · {manager.scheduler_name} · upscaler {upscaler_name}"


# ----------------------------------------------------------------------------- model cards

def model_cards_html(manager) -> str:
    by_model = {r["model_id"]: r for r in db.stats_by_model()}
    cards = []
    for m in models.all_models():
        on_disk = models.is_downloaded(m)
        st = by_model.get(m.source)
        active = manager.spec is not None and manager.spec.key == m.key
        corner = family_badge(m.family) + (badge("loaded", "ok") if active else badge("on disk", "ok") if on_disk else badge(f"~{m.download_gb:g} GB", "warn"))
        meta = f"<b>{m.steps}</b> steps · cfg <b>{m.guidance:g}</b> · {models.SCHEDULER_NAMES.get(m.scheduler or '', 'model sampler')}"
        if st:
            meta += f" · <b>{st['images']}</b> images · avg <b>{st['avg_seconds']}</b> s"
        cards.append(
            f'<div class="fz-card{" active" if active else ""}"><div class="corner">{corner}</div>'
            f'<div class="t">{escape(m.name)}</div><div class="d">{escape(m.description)}</div>'
            f'<div class="m">say <b>model {m.key}</b> · {meta}</div></div>'
        )
    loras = registry.lora_entries(include_missing=False)
    lora_line = ", ".join(f"<code>&lt;lora:{escape(s)}&gt;</code>" + (f" ({escape(e['trigger'])})" if e.get("trigger") else "") for s, e in loras.items() if e.get("enabled", True)) or "none yet"
    return (f'<div class="fz-cards">{"".join(cards)}</div>'
            f'<div class="fz-note"><b>LoRAs:</b> {lora_line}. Add models and LoRAs in <b>Admin › Add</b>; '
            f'their trigger words appear in the LoRAs box.</div>')


# ----------------------------------------------------------------------------- tiles & tables

def meter_color(pct: float | None) -> str:
    if pct is None:
        return ACCENT
    if pct >= 95:
        return STATUS["critical"][0]
    if pct >= 85:
        return STATUS["warning"][0]
    return ACCENT


def temp_status(temp: float | None) -> str:
    if temp is None:
        return "good"
    return "good" if temp < 70 else "warning" if temp < 83 else "serious" if temp < 90 else "critical"


def tile(label: str, value: str, unit: str = "", sub: str = "", meter: float | None = None,
         status: str | None = None, cores: list[float] | None = None, status_word: str | None = None) -> str:
    html = [f'<div class="fz-tile"><div class="l">{label}</div>',
            f'<div class="v">{value}<small>{unit}</small></div>']
    if status:
        color, icon, word = STATUS[status]
        html.append(f'<div class="st" style="color:{color}">{icon} {status_word or word}</div>')
    if sub:
        html.append(f'<div class="s">{sub}</div>')
    if meter is not None:
        pct = max(0.0, min(100.0, meter))
        html.append(f'<div class="fz-meter"><div style="width:{pct:.1f}%;background:{meter_color(pct)}"></div></div>')
    if cores:
        bars = "".join(f'<div title="core {i}: {c:.0f}%"><div style="height:{max(3, c):.0f}%"></div></div>' for i, c in enumerate(cores))
        html.append(f'<div class="fz-cores">{bars}</div>')
    html.append("</div>")
    return "".join(html)


def kv(rows: list[tuple[str, Any]]) -> str:
    out = ["| | |", "|---|---|"]
    for k, v in rows:
        out.append(f"| {k} | {'n/a' if v in (None, '') else v} |")
    return "\n".join(out)


def jobs_html() -> str:
    jobs = JOBS.jobs()
    if not jobs:
        return '<div class="fz-note">No downloads yet. Add a model or LoRA in <b>Admin › Add</b>, or press <b>Download</b> on a model.</div>'
    parts = []
    for j in jobs:
        d = j.to_dict()
        pct = d["pct"]
        state = {"queued": badge("queued", "off"), "running": badge("running", "warn"), "done": badge("done", "ok"),
                 "error": badge("error", "bad"), "cancelled": badge("cancelled", "off")}[d["status"]]
        right = []
        if d["total"]:
            right.append(f'{human_bytes(d["done"])} / {human_bytes(d["total"])}')
        elif d["done"]:
            right.append(human_bytes(d["done"]))
        if d["status"] == "running" and d["speed"]:
            right.append(f'{human_bytes(d["speed"])}/s')
        if d["eta_s"] is not None:
            right.append(f'{d["eta_s"] / 60:.0f} min left' if d["eta_s"] > 90 else f'{d["eta_s"]:.0f} s left')
        if d["status"] in ("done", "error", "cancelled"):
            right.append(f'{d["seconds"]:.0f} s')
        meter = ""
        if pct is not None and d["status"] in ("running", "queued", "done"):
            meter = f'<div class="fz-meter"><div style="width:{pct:.1f}%"></div></div>'
        msg = escape(d["error"] or d["message"] or "")
        parts.append(
            f'<div class="fz-job"><div class="r"><div>#{d["id"]} {state} <b>{escape(d["label"])}</b></div>'
            f'<span>{" · ".join(right)}</span></div>{meter}<div class="msg">{msg}</div></div>'
        )
    return f'<div class="fz-jobs">{"".join(parts)}</div>'
