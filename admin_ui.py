"""
Admin panel: models, LoRAs, downloads, system stats, settings and logs.

Built into the Studio's Admin tab by build_admin(). Sub-tabs:

    Overview    stat tiles, activity, per-model generation statistics
    Models      every model (built-in, local checkpoint, admin-added): edit defaults, enable/disable,
                load, download, set as start-up model, delete
    LoRAs       every LoRA file with its family, trigger words and default weight; try it in the Studio
    Add         add a model or LoRA from a Hugging Face repo/file, a Civitai link, a URL, a local
                path or an upload; downloads run in the background
    Downloads   progress of background downloads
    System      GPU, CPU, memory, storage, network, power, software and app details
    Settings    start-up model, default LoRA weight, negative prompt, tokens, housekeeping
    Logs        the app's own log lines

Destructive actions (deleting files, cleaning outputs) need the confirmation box ticked, and if the
environment variable FRAZILE_ADMIN_PASSWORD is set they also need the panel unlocked in Settings.
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pandas as pd

import applog
import db
import downloads
import media
import models
import registry
import sysinfo
from downloads import JOBS, human_bytes
from models import CHECKPOINTS_DIR, LORAS_DIR, ModelSpec
from ui_common import (jobs_html, kv, loaded_line, lora_choice_update, model_choice_update, temp_status, tile)
from ui_theme import badge, family_badge, status_span

CTX = SimpleNamespace(manager=None, media=None, upscaler=None, static_gpu=None)

MODEL_COLUMNS = ["key", "name", "family", "type", "steps", "cfg", "sampler", "on disk", "size", "uses", "avg s", "enabled", "startup"]
LORA_COLUMNS = ["name", "file", "family", "trigger words", "weight", "size", "enabled", "uses", "source"]
MODEL_STATS_COLUMNS = ["model", "uses", "avg s", "min s", "max s", "steps/s", "avg VRAM GB", "max VRAM GB", "max °C", "upscaled", "last used"]
SCHED_CHOICES = [(v, k) for k, v in registry.SCHEDULERS.items()]
FAMILY_CHOICES = [(f"{k} — {v}", k) for k, v in registry.FAMILIES.items()]
LORA_FAMILY_CHOICES = [(f"{k} — {v}", k) for k, v in registry.FAMILIES.items() if k in registry.IMAGE_FAMILIES]
ADD_TYPES = [("Model (diffusers repo)", "model"), ("Checkpoint file (SDXL .safetensors)", "checkpoint"), ("LoRA file", "lora")]

THROTTLE_BITS = {
    0x1: "GPU idle", 0x2: "application clock setting", 0x4: "software power cap", 0x8: "hardware slowdown",
    0x10: "sync boost", 0x20: "software thermal slowdown", 0x40: "hardware thermal slowdown",
    0x80: "hardware power brake", 0x100: "display clock setting",
}


# ----------------------------------------------------------------------------- guards

def admin_password() -> str:
    return os.environ.get("FRAZILE_ADMIN_PASSWORD", "")


def guard(unlocked: bool, confirm: bool = True) -> None:
    if not confirm:
        raise gr.Error("Tick the confirmation box first.")
    if admin_password() and not unlocked:
        raise gr.Error("The Admin panel is locked: enter the admin password in Settings.")


def unlock(password: str, unlocked: bool):
    if not admin_password():
        return True, '<div class="fz-note">No admin password is set (FRAZILE_ADMIN_PASSWORD); destructive actions only need the confirmation box.</div>'
    if password == admin_password():
        return True, f'<div class="fz-note">{status_span("good", "Unlocked")} for this browser session.</div>'
    return unlocked, f'<div class="fz-note">{status_span("critical", "Wrong password")}</div>'


# ----------------------------------------------------------------------------- overview

def _model_stats_df() -> pd.DataFrame:
    rows = [[models.display_name(r["model_id"] or ""), r["images"], r["avg_seconds"], r["min_seconds"], r["max_seconds"],
             r["avg_steps_per_second"], r["avg_vram_peak_gb"], r["max_vram_peak_gb"], r["max_temp_c"], r["upscaled"],
             (r["last_used"] or "")[:16].replace("T", " ")] for r in db.stats_by_model()]
    return pd.DataFrame(rows, columns=MODEL_STATS_COLUMNS)


def _tiles(d: dict) -> str:
    c, m, dk, n, g, p, a = d["cpu"], d["memory"], d["disks"], d["network"], d["gpu"], d["power"], d["app"]
    live, tm = g["live"], g["torch"]
    vram_used, vram_total = live.get("mem_used_mb"), live.get("mem_total_mb")
    vram_pct = vram_used / vram_total * 100 if vram_used and vram_total else None
    sysdisk = next((x for x in dk["partitions"] if x["mount"].upper().startswith("C")), dk["partitions"][0] if dk["partitions"] else None)
    up, down = n.get("up_rate"), n.get("down_rate")
    temp = live.get("temperature_c")
    tiles = "".join([
        tile("GPU load", f"{live.get('gpu_util_pct', 0) or 0:.0f}", "%", f"{live.get('sm_clock_mhz', 0) or 0:.0f} MHz · {live.get('pstate', '')}",
             meter=live.get("gpu_util_pct")),
        tile("VRAM used", f"{(vram_used or 0) / 1024:.1f}", f" / {(vram_total or 0) / 1024:.0f} GB",
             f"torch {tm.get('vram_allocated_gb', 0)} GB allocated", meter=vram_pct),
        tile("GPU temperature", f"{temp if temp is not None else 'n/a'}", " °C", f"fan {live.get('fan_pct', 0) or 0:.0f} %",
             status=temp_status(temp)),
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
    return (f'<div class="fz-admin-head"><span>Collected {d["collected_at"][11:]}</span>'
            f'<span>{n["hostname"]} · {n.get("primary_ip") or "no IP"} · {d["software"]["os"]}</span></div><div class="fz-tiles">{tiles}</div>')


def _activity_html(d: dict) -> str:
    mgr = CTX.manager
    rc = db.recent_counts()
    st = db.stats()
    reg = registry.summary()
    pj = d["disks"]["project"]
    active = JOBS.active()
    startup = registry.setting("startup_model")
    hf = downloads.hf_token()
    kinds = db.counts_by_kind()
    med = CTX.media
    loaded = " + ".join(x for x in [mgr.spec.name if mgr.spec else "", med.spec.name if med and med.spec else ""] if x) or "none"
    tiles = "".join([
        tile("Generations today", f"{rc['today']}", "", f"{rc['week']} this week · {rc['total']} total"),
        tile("Images · videos · tracks", f"{kinds['image']['count']} · {kinds['video']['count']} · {kinds['music']['count']}", "",
             f"{kinds['video']['duration']:.0f} s of video · {kinds['music']['duration']:.0f} s of music"),
        tile("GPU time, 24 h", f"{rc['gpu_seconds_24h'] / 60:.1f}", " min", f"{st.get('total_seconds') or 0:.0f} s all time"),
        tile("Loaded now", escape(loaded), "", f"{mgr.scheduler_name or ''} · start-up: {escape(startup)}"),
        tile("Models", f"{len(models.all_models())}", "", f"{sum(1 for m in models.all_models() if models.is_downloaded(m))} on disk · "
             f"{len(models.all_models(kind='video'))} video · {len(models.all_models(kind='music'))} music · {reg['custom_models']} added"),
        tile("LoRAs", f"{reg['loras']}", "", f"{sum(db.lora_usage().values())} uses recorded"),
        tile("Downloads", f"{len(active)}", " active", f"{len(JOBS.jobs()) - len(active)} finished", status="warning" if active else None, status_word="running"),
        tile("Outputs", human_bytes(pj["outputs"]), "", f"{pj['outputs_files']} files (png, mp4, wav) · DB {human_bytes(pj['database'])}"),
        tile("Hugging Face", "signed in" if hf else "no token", "", "FLUX and gated repos available" if hf else "needed for FLUX / gated repos",
             status="good" if hf else "warning", status_word="OK" if hf else "Missing"),
    ])
    return f'<div class="fz-tiles">{tiles}</div>'


def overview_refresh():
    d = sysinfo.everything(loaded_model=CTX.manager.spec.name if CTX.manager.spec else None)
    return _tiles(d), _activity_html(d), _model_stats_df()


# ----------------------------------------------------------------------------- models tab

def _model_type(m: ModelSpec) -> str:
    if registry.is_custom(m.key):
        return "added"
    if m.local:
        return "local file"
    return "built-in"


def models_df() -> pd.DataFrame:
    by_model = {r["model_id"]: r for r in db.stats_by_model()}
    startup = registry.setting("startup_model")
    rows = []
    for m in models.all_models(include_disabled=True):
        st = by_model.get(m.source, {})
        disk = models.is_downloaded(m)
        rows.append([m.key, m.name, m.family, _model_type(m), m.steps, m.guidance,
                     models.SCHEDULER_NAMES.get(m.scheduler or "", "model default"),
                     "yes" if disk else "no", human_bytes(models.on_disk_bytes(m)) if disk else f"~{m.download_gb:g} GB",
                     st.get("images", 0), st.get("avg_seconds", ""), "yes" if registry.is_enabled(m.key) else "no",
                     "★" if m.key == startup else ""])
    return pd.DataFrame(rows, columns=MODEL_COLUMNS)


def _find(key: str) -> ModelSpec:
    for m in models.all_models(include_disabled=True):
        if m.key == key:
            return m
    raise gr.Error(f"Unknown model '{key}'. Select one in the table.")


def model_status_html(key: str) -> str:
    if not key:
        return '<div class="fz-note">Select a model in the table to edit it.</div>'
    m = _find(key)
    mgr = CTX.manager
    disk = models.is_downloaded(m)
    st = next((r for r in db.stats_by_model() if r["model_id"] == m.source), None)
    loaded_keys = {x.spec.key for x in (mgr, CTX.media) if x is not None and x.spec}
    bits = [family_badge(m.family), badge(_model_type(m), "off"),
            badge("loaded now", "ok") if key in loaded_keys else "",
            badge(f"on disk · {human_bytes(models.on_disk_bytes(m))}", "ok") if disk else badge(f"not downloaded · ~{m.download_gb:g} GB", "warn"),
            badge("start-up model", "ok") if registry.setting("startup_model") == key else "",
            badge("enabled", "ok") if registry.is_enabled(key) else badge("disabled", "bad")]
    src = m.source
    if "/" in src and not Path(src).exists():
        src = f'<a href="https://huggingface.co/{escape(src)}" target="_blank">{escape(src)}</a>'
    else:
        src = f"<code>{escape(src)}</code>"
    lines = [f"<div>{' '.join(b for b in bits if b)}</div>",
             f"<div style='margin-top:6px'><b>Source:</b> {src}</div>"]
    if st:
        lines.append(f"<div><b>Usage:</b> {st['images']} generations · avg {st['avg_seconds']} s · peak VRAM up to {st['max_vram_peak_gb']} GB · last {str(st['last_used'])[:16].replace('T', ' ')}</div>")
    if models.kind_of(m.family) != "image":
        p = media.profile_for(m)
        extra = (f"{p['fps']} fps · clips up to {p['max_seconds']} s · " + ("image-to-video supported" if p.get("image_to_video") else "text-to-video only")
                 if p["kind"] == "video" else f"{p.get('sample_rate', 0) // 1000} kHz · tracks up to {p['max_seconds']} s")
        lines.append(f"<div><b>{p['label']}</b> ({p['kind']} model, {p['kind'].capitalize()} tab): {extra}. {escape(p.get('note', ''))}</div>")
    if m.aliases:
        lines.append(f"<div><b>Also answers to:</b> {escape(', '.join(m.aliases))}</div>")
    if m.styles:
        lines.append(f"<div><b>Auto-picked for styles:</b> {escape(', '.join(m.styles))}</div>")
    return f'<div class="fz-note">{"".join(lines)}</div>'


def model_editor_values(key: str):
    """Values for every editor component, in EDITOR order."""
    if not key:
        return ("", "", "", gr.update(value="sdxl", interactive=False), gr.update(value="", interactive=False),
                30, 7.0, "dpmpp_2m_karras", "", True, "", "", True, model_status_html(""))
    m = _find(key)
    custom = registry.is_custom(key)
    return (m.key, m.name, m.description, gr.update(value=m.family, interactive=custom),
            gr.update(value=m.source, interactive=custom), m.steps, m.guidance, m.scheduler or "", m.variant or "",
            m.negative, ", ".join(m.aliases), ", ".join(m.styles), registry.is_enabled(key), model_status_html(key))


def select_model(evt: gr.SelectData, table: pd.DataFrame):
    if table is None or len(table) == 0:
        return model_editor_values("")
    return model_editor_values(str(table.iloc[evt.index[0]]["key"]))


def save_model(key, name, desc, family, source, steps, guidance, sched, variant, negative, aliases, styles, enabled):
    if not key:
        raise gr.Error("Select a model first.")
    custom = registry.is_custom(key)
    values = dict(name=name.strip() or key, description=desc.strip(), steps=int(max(1, min(150, steps or 1))),
                  guidance=float(max(0.0, min(30.0, guidance or 0))), scheduler=sched or "", variant=(variant or "").strip(),
                  negative=bool(negative), aliases=aliases, styles=styles, enabled=bool(enabled))
    if custom:
        values.update(source=source.strip(), family=family)
    registry.upsert_model(key, custom=custom, **values)
    gr.Info(f"Saved {values['name']}. Changes apply to the next load.")
    return models_df(), model_status_html(key), model_choice_update(), gr.update()


def load_model(key, progress=gr.Progress()):
    if not key:
        raise gr.Error("Select a model first.")
    spec = _find(key)
    mgr, med = CTX.manager, CTX.media
    progress(0, desc=f"Loading {spec.name}" + ("" if models.is_downloaded(spec) else f" (downloading ~{spec.download_gb:g} GB)"))
    t0 = time.time()
    with mgr.lock:
        try:
            media.prepare_for(spec, mgr, med, log=applog.log)
            if models.kind_of(spec.family) == "image":
                mgr.load(spec)
            else:
                med.load(spec)
        except (RuntimeError, KeyError, OSError) as e:
            raise gr.Error(str(e))
    gr.Info(f"{spec.name} loaded in {time.time() - t0:.0f} s")
    return models_df(), model_status_html(key), loaded_line(mgr, CTX.upscaler.name, CTX.media)


def unload_model():
    mgr, med = CTX.manager, CTX.media
    names = [x.spec.name for x in (mgr, med) if x is not None and x.spec]
    with mgr.lock:
        mgr.unload()
        if med is not None:
            med.unload()
    gr.Info(f"Unloaded {', '.join(names)}; VRAM freed" if names else "Nothing was loaded")
    return models_df(), model_status_html(""), loaded_line(mgr, CTX.upscaler.name, CTX.media)


def _unload_if_loaded(key: str) -> None:
    mgr, med = CTX.manager, CTX.media
    with mgr.lock:
        if mgr.spec and mgr.spec.key == key:
            mgr.unload()
        if med is not None and med.spec and med.spec.key == key:
            med.unload()


def download_model(key):
    if not key:
        raise gr.Error("Select a model first.")
    spec = _find(key)
    if models.is_downloaded(spec):
        gr.Info(f"{spec.name} is already on disk")
        return jobs_html(), model_status_html(key)
    if spec.local or "/" not in spec.source:
        raise gr.Error(f"{spec.name} points at a local file that is missing: {spec.source}")
    if any(j.active and j.label == spec.name for j in JOBS.jobs()):
        gr.Info(f"{spec.name} is already downloading")
        return jobs_html(), model_status_html(key)
    JOBS.submit("repo", spec.name, downloads.download_repo(spec))
    gr.Info(f"Downloading {spec.name} in the background (see Downloads)")
    return jobs_html(), model_status_html(key)


def set_startup(key):
    if not key:
        raise gr.Error("Select a model first.")
    spec = _find(key)
    registry.set_settings(startup_model=key)
    gr.Info(f"{spec.name} loads at the next start")
    return models_df(), model_status_html(key)


def reset_or_remove_model(key, unlocked):
    if not key:
        raise gr.Error("Select a model first.")
    guard(unlocked)
    spec = _find(key)
    custom = registry.is_custom(key)
    if custom:
        _unload_if_loaded(key)
    registry.remove_model(key)
    gr.Info(f"Removed {spec.name} from the list" if custom else f"Reset {spec.name} to its built-in defaults")
    new_key = "" if custom else key
    return (models_df(), *model_editor_values(new_key), model_choice_update(), loaded_line(CTX.manager, CTX.upscaler.name, CTX.media))


def delete_model_files(key, confirm, unlocked):
    if not key:
        raise gr.Error("Select a model first.")
    guard(unlocked, confirm)
    spec = _find(key)
    mgr = CTX.manager
    _unload_if_loaded(key)
    removed = downloads.delete_model_files(spec)
    if spec.local or registry.is_custom(key) and Path(spec.source).suffix == ".safetensors":
        registry.remove_model(key)
    gr.Info(f"Deleted {len(removed)} item(s) for {spec.name}" if removed else f"Nothing on disk for {spec.name}")
    applog.log(f"admin: deleted files of {spec.key}: {removed}")
    still = any(m.key == key for m in models.all_models(include_disabled=True))
    return (models_df(), *model_editor_values(key if still else ""), model_choice_update(), loaded_line(mgr, CTX.upscaler.name, CTX.media), False)


# ----------------------------------------------------------------------------- loras tab

def loras_df() -> pd.DataFrame:
    uses = db.lora_usage()
    rows = []
    for stem, e in registry.lora_entries().items():
        rows.append([e["name"], f"{stem}.safetensors" + (" (missing)" if e["missing"] else ""), e["family"], e["trigger"],
                     e["weight"], human_bytes(e["size"]) if not e["missing"] else "", "yes" if e.get("enabled", True) else "no",
                     uses.get(stem, 0), e.get("source", "")])
    return pd.DataFrame(rows, columns=LORA_COLUMNS)


def lora_status_html(stem: str) -> str:
    if not stem:
        return '<div class="fz-note">Select a LoRA in the table to edit it, or add one in <b>Add</b>.</div>'
    e = registry.lora(stem)
    if not e:
        return '<div class="fz-note">Unknown LoRA.</div>'
    bits = [family_badge(e["family"]), badge("missing file", "bad") if e["missing"] else badge(human_bytes(e["size"]), "ok"),
            badge("enabled", "ok") if e.get("enabled", True) else badge("disabled", "bad")]
    uses = db.lora_usage().get(stem, 0)
    cands = models.family_models(e["family"])
    with_model = f"works with {', '.join(m.name for m in cands[:4])}" if cands else f"no {e['family']} model is on disk yet"
    return (f'<div class="fz-note"><div>{" ".join(bits)}</div>'
            f'<div style="margin-top:6px"><b>Prompt tag:</b> <code>&lt;lora:{escape(stem)}:{e["weight"]:g}&gt;</code>'
            f' · <b>trigger:</b> {escape(e["trigger"]) or "none recorded"} · {uses} uses · {with_model}</div>'
            f'<div><b>File:</b> <code>{escape(e["path"])}</code></div></div>')


def lora_editor_values(stem: str):
    if not stem:
        return "", "", "sdxl", "", 0.8, "", "", True, "", lora_status_html("")
    e = registry.lora(stem) or {}
    return (stem, e.get("name", stem), e.get("family", "sdxl"), e.get("trigger", ""), float(e.get("weight", 0.8)),
            e.get("description", ""), e.get("source", ""), bool(e.get("enabled", True)), ", ".join(e.get("tags", [])),
            lora_status_html(stem))


def select_lora(evt: gr.SelectData, table: pd.DataFrame):
    if table is None or len(table) == 0:
        return lora_editor_values("")
    file = str(table.iloc[evt.index[0]]["file"]).replace(" (missing)", "")
    return lora_editor_values(Path(file).stem)


def save_lora(stem, name, family, trigger, weight, desc, source, enabled, tags):
    if not stem:
        raise gr.Error("Select a LoRA first.")
    registry.set_lora(stem, name=name.strip() or stem, family=family, trigger=trigger.strip(), weight=float(weight),
                      description=desc.strip(), source=source.strip(), enabled=bool(enabled), tags=tags)
    gr.Info(f"Saved LoRA {name or stem}")
    return loras_df(), lora_status_html(stem), lora_choice_update()


def try_lora(stem):
    if not stem:
        raise gr.Error("Select a LoRA first.")
    e = registry.lora(stem)
    if not e or e["missing"]:
        raise gr.Error("That LoRA file is missing.")
    cands = models.family_models(e["family"])
    model_word = f", model {cands[0].key}" if cands else ""
    subject = {"sdxl": "a fox in an autumn forest", "zimage": "a fox in an autumn forest", "qwen": "a fox in an autumn forest",
               "flux": "a fox in an autumn forest"}.get(e["family"], "a fox")
    prompt = f"{e['trigger'] + ' of ' if e['trigger'] else ''}{subject}, <lora:{stem}:{e['weight']:g}>{model_word}, seed 7"
    return prompt, gr.Tabs(selected="studio")


def delete_lora(stem, confirm, unlocked):
    if not stem:
        raise gr.Error("Select a LoRA first.")
    guard(unlocked, confirm)
    e = registry.lora(stem)
    path = Path(e["path"]) if e else None
    if path and path.exists():
        path.unlink()
    registry.forget_lora(stem)
    mgr = CTX.manager
    if mgr.pipe is not None and stem.replace("-", "_") in mgr._loaded_loras:
        with mgr.lock:
            mgr.unload()   # the adapter weights are inside the pipeline; reload cleanly next time
    applog.log(f"admin: deleted LoRA {stem}")
    gr.Info(f"Deleted LoRA {stem}")
    return (loras_df(), *lora_editor_values(""), lora_choice_update(), False)


# ----------------------------------------------------------------------------- add tab

def _info_html(info: dict) -> str:
    if not info.get("input"):
        return ""
    if info.get("error") and not info.get("ok"):
        return f'<div class="fz-note">{status_span("critical", "Cannot use this source")}: {escape(info["error"])}</div>'
    bits = [status_span("good", "Found"), f"<b>{escape(info.get('summary', ''))}</b>"]
    kind = {"hf_repo": "Hugging Face repo", "hf_file": "Hugging Face file", "civitai": "Civitai", "url": "direct URL", "local": "local file"}.get(info.get("kind"), "")
    rows = [f"<div>{' · '.join(bits)}</div>", f"<div style='margin-top:4px'>{kind}"]
    if info.get("family"):
        rows[-1] += f" · family {family_badge(info['family'])}"
    if info.get("type"):
        rows[-1] += f" · type <b>{escape(info['type'])}</b>"
    if info.get("gated"):
        rows[-1] += " · " + badge("gated", "warn")
    if info.get("nsfw"):
        rows[-1] += " · " + badge("marked NSFW on Civitai", "warn")
    rows[-1] += "</div>"
    if info.get("trigger"):
        rows.append(f"<div>trigger words: <b>{escape(info['trigger'])}</b></div>")
    for w in info.get("warnings", []):
        rows.append(f"<div class='fz-warn'>▲ {escape(w)}</div>")
    return f'<div class="fz-note">{"".join(rows)}</div>'


def check_source(text: str):
    info = downloads.inspect_source(text)
    add_type = info.get("type") or "model"
    if add_type == "model" and info.get("layout") == "single-file":
        add_type = "checkpoint"
    name = info.get("name") or ""
    key = models._slug(name)[:40]
    family = info.get("family") or "sdxl"
    steps, guidance, sched = 30, 7.0, "dpmpp_2m_karras"
    low = (name + " " + (info.get("filename") or "")).lower()
    if family == "sdxl" and any(w in low for w in ("turbo", "lightning", "hyper")):
        steps, guidance, sched = 6, 2.0, "dpmpp_sde_karras"
    elif family == "zimage":
        steps, guidance, sched = 8, 0.0, ""
    elif family == "qwen":
        steps, guidance, sched = 8, 1.0, ""
    elif family == "flux":
        steps, guidance, sched = 4, 0.0, ""
    elif family == "video":
        steps, guidance, sched = (30, 5.0, "") if "wan" in (info.get("pipeline") or "").lower() else (40, 3.0, "")
    elif family == "music":
        steps, guidance, sched = (100, 7.0, "") if "stableaudio" in (info.get("pipeline") or "").lower() else (0, 3.0, "")
    cands = info.get("candidates") or ([info["filename"]] if info.get("filename") else [])
    return (info, _info_html(info), gr.update(value=add_type), key, name, family,
            gr.update(choices=cands, value=cands[0] if cands else None, visible=len(cands) > 1),
            info.get("trigger") or "", steps, guidance, sched, (info.get("description") or "")[:200],
            gr.update(interactive=bool(info.get("ok"))))


def add_source(info, add_type, key, name, family, file_choice, trigger, steps, guidance, sched, desc, weight, download_now):
    if not info or not info.get("ok"):
        raise gr.Error("Check the source first.")
    key = models._slug(key or name or "model")
    name = (name or key).strip()
    kind = info["kind"]
    if file_choice and kind == "hf_repo":
        info = dict(info, filename=file_choice)
    if add_type == "model":
        if kind != "hf_repo" or info.get("layout") not in ("diffusers", "transformers"):
            raise gr.Error("Only a Hugging Face repo in diffusers (or MusicGen) layout can be added as a model; files become checkpoints or LoRAs.")
        if any(m.key == key for m in models.all_models(include_disabled=True)):
            raise gr.Error(f"A model with key '{key}' exists already; choose another key.")
        registry.upsert_model(key, custom=True, name=name, source=info["repo"], family=family, steps=int(steps), guidance=float(guidance),
                              scheduler=sched or "", variant=info.get("variant") or "", negative=family in ("sdxl", "video"), description=desc.strip(),
                              download_gb=round((info.get("download_size") or info.get("total_size") or 0) / 1024**3, 1), enabled=True)
        where = {"video": "Video tab", "music": "Music tab"}.get(family, f"say 'model {key}'")
        msg = f"Added model {name} ({where})."
        if download_now:
            JOBS.submit("repo", name, downloads.download_repo(models.resolve(key)))
            msg += " Downloading in the background."
        applog.log(f"admin: added model {key} from {info['repo']}")
        gr.Info(msg)
        return (f'<div class="fz-note">{status_span("good", msg)}</div>', jobs_html(), models_df(), loras_df(),
                model_choice_update(), lora_choice_update())

    if kind == "hf_repo" and info.get("layout") != "single-file":
        raise gr.Error("That repo is a full model: add it as a Model instead.")
    rename = key
    weight = float(weight)

    if add_type == "lora":
        def on_done(job, _name=name, _family=family, _trigger=trigger, _desc=desc, _src=info.get("input"), _w=weight):
            stem = Path(job.target).stem
            registry.set_lora(stem, name=_name, family=_family, trigger=_trigger.strip(), weight=_w, description=_desc.strip(),
                              source=_src, enabled=True)
            applog.log(f"admin: LoRA {stem} ready ({_family}, trigger {_trigger!r})")
        JOBS.submit("lora", f"LoRA {name}", downloads.download_file(info, "lora", rename), on_done=on_done)
        msg = f"Fetching LoRA {name}; it appears in the LoRAs box when the download finishes."
    else:
        if family != "sdxl":
            raise gr.Error("Single-file checkpoints are loaded with the SDXL loader; FLUX, Qwen and Z-Image need a diffusers repo.")
        def on_done(job, _name=name, _desc=desc, _steps=int(steps), _g=float(guidance), _s=sched or "dpmpp_2m_karras", _src=info.get("input")):
            stem = Path(job.target).stem
            registry.upsert_model(models._slug(stem), custom=False, name=_name, description=_desc.strip() or f"checkpoint from {_src}",
                                  steps=_steps, guidance=_g, scheduler=_s, enabled=True)
            applog.log(f"admin: checkpoint {stem} ready")
        JOBS.submit("model", f"Checkpoint {name}", downloads.download_file(info, "checkpoint", rename), on_done=on_done)
        msg = f"Fetching checkpoint {name} into models/checkpoints; it appears in the Model box when the download finishes."
    gr.Info(msg)
    return (f'<div class="fz-note">{status_span("good", msg)}</div>', jobs_html(), models_df(), loras_df(),
            model_choice_update(), lora_choice_update())


def upload_file(path, add_type, name, family, trigger, weight):
    if not path:
        raise gr.Error("Choose a .safetensors file first.")
    src = Path(path)
    if src.suffix.lower() != ".safetensors":
        raise gr.Error("Only .safetensors files are accepted.")
    stem = downloads.safe_stem(name or src.stem)
    folder = LORAS_DIR if add_type == "lora" else CHECKPOINTS_DIR
    dest = downloads.unique_path(folder, f"{stem}.safetensors")
    shutil.move(str(src), dest)
    if add_type == "lora":
        registry.set_lora(dest.stem, name=name or dest.stem, family=family, trigger=trigger.strip(), weight=float(weight),
                          source="upload", enabled=True)
    else:
        registry.upsert_model(models._slug(dest.stem), custom=False, name=name or dest.stem, description="uploaded checkpoint", enabled=True)
    applog.log(f"admin: uploaded {add_type} {dest.name} ({human_bytes(dest.stat().st_size)})")
    gr.Info(f"Saved {dest.name}")
    return (f'<div class="fz-note">{status_span("good", f"Saved {dest.name}")} ({human_bytes(dest.stat().st_size)})</div>',
            models_df(), loras_df(), model_choice_update(), lora_choice_update(), None)


# ----------------------------------------------------------------------------- downloads tab

def downloads_refresh():
    return jobs_html()


def cancel_job(job_id):
    try:
        ok = JOBS.cancel(int(job_id))
    except (TypeError, ValueError):
        ok = False
    gr.Info("Cancel requested" if ok else "No active download with that number")
    return jobs_html()


def clear_jobs():
    n = JOBS.clear_finished()
    gr.Info(f"Cleared {n} finished download(s)")
    return jobs_html()


# ----------------------------------------------------------------------------- system tab

def _throttle_text(hex_value: str | None) -> str:
    try:
        v = int(hex_value or "0", 16)
    except ValueError:
        return hex_value or "n/a"
    names = [n for bit, n in THROTTLE_BITS.items() if v & bit]
    return ", ".join(names) if names else "none"


def gpu_energy_estimate() -> tuple[float, int]:
    wh, n = 0.0, 0
    for r in db.fetch(limit=10**6):
        p = [x for x in (r.get("power_before_w"), r.get("power_after_w")) if x]
        if p and r.get("inference_seconds"):
            wh += r["inference_seconds"] * (sum(p) / len(p)) / 3600
            n += 1
    return wh, n


def system_refresh(check_internet: bool):
    mgr = CTX.manager
    d = sysinfo.everything(loaded_model=mgr.spec.name if mgr.spec else None, check_internet=check_internet)
    c, m, dk, n, g, p, sw, a = d["cpu"], d["memory"], d["disks"], d["network"], d["gpu"], d["power"], d["software"], d["app"]
    live, st, tm, ex = g["live"], g["static"], g["torch"], g["extra"]
    vram_used, vram_total = live.get("mem_used_mb"), live.get("mem_total_mb")
    up, down = n.get("up_rate"), n.get("down_rate")

    gpu_md = "#### GPU\n" + kv([
        ("Card", f"{st.get('gpu_name')} · {st.get('vram_total_gb')} GB · compute {st.get('compute_capability')} · {st.get('multiprocessors')} SMs"),
        ("Driver / CUDA / cuDNN", f"{live.get('driver_version')} / {st.get('cuda_version')} / {st.get('cudnn_version')}"),
        ("Clocks (current / max)", f"SM {live.get('sm_clock_mhz') or 0:.0f} / {ex.get('max_sm_clock_mhz')} MHz · memory {live.get('mem_clock_mhz') or 0:.0f} / {ex.get('max_mem_clock_mhz')} MHz"),
        ("Utilisation", f"GPU {live.get('gpu_util_pct') or 0:.0f} % · memory bus {live.get('mem_util_pct') or 0:.0f} %"),
        ("VRAM (nvidia-smi)", f"{(vram_used or 0):.0f} / {(vram_total or 0):.0f} MiB"),
        ("VRAM (torch)", f"allocated {tm.get('vram_allocated_gb')} GB · reserved {tm.get('vram_reserved_gb')} GB · session peak {tm.get('vram_peak_gb')} GB"),
        ("PCIe link", f"gen {ex.get('pcie_gen')} x{ex.get('pcie_width')} (drops to gen 1 x1 at idle)"),
        ("Throttle reasons", _throttle_text(ex.get("throttle_reasons"))),
        ("VBIOS / bus", f"{ex.get('vbios')} · {ex.get('pci_bus')}"),
        ("Loaded model", f"{a.get('loaded_model') or 'none'} · {mgr.scheduler_name}"),
    ])
    if g["processes"]:
        gpu_md += "\n\n**Processes on the GPU**\n\n| PID | VRAM | Name |\n|---|---|---|\n" + "\n".join(
            f"| {x['pid']} | {x['vram_mb']} MiB | {x['name']} |" for x in g["processes"][:12])

    cpu_md = "#### CPU\n" + kv([
        ("Processor", c["name"]), ("Architecture", c["architecture"]),
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
    mem_md = "#### Memory and storage\n" + kv([
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
    net_md = "#### Network\n" + kv(net_rows) + \
        f"\n\n| Interface | State | IPv4 | MAC | Speed | Sent | Received |\n|---|---|---|---|---|---|---|\n{ifaces}"

    wh, n_rec = gpu_energy_estimate()
    bat = p.get("battery")
    power_md = "#### Power\n" + kv([
        ("GPU draw", f"{p.get('gpu_draw_w')} W of {p.get('gpu_limit_w')} W limit ({p.get('gpu_pct_of_limit')} %)"),
        ("GPU power limits", f"default {ex.get('power_default_limit_w')} W · max {ex.get('power_max_limit_w')} W"),
        ("GPU performance state", f"{p.get('gpu_pstate')} · fan {p.get('gpu_fan_pct')} %"),
        ("Windows power plan", p.get("windows_power_plan")),
        ("Battery", f"{bat['percent']} % · {'plugged in' if bat['plugged'] else 'on battery'}" if bat else "none (desktop)"),
        ("GPU energy, all generations", f"≈ {wh:.1f} Wh over {n_rec} images (inference time × mean power)"),
    ])

    stats = db.stats()
    soft_md = "#### Software and app\n" + kv([
        ("OS", sw["os"]), ("Python", sw["python"]),
        ("torch / CUDA / cuDNN", f"{sw['torch']} / {sw['cuda']} / {sw['cudnn']}"),
        ("diffusers / transformers / gradio", f"{sw['diffusers']} / {sw['transformers']} / {sw['gradio']}"),
        ("Running in Docker", "yes" if sw["in_container"] else "no"),
        ("App process", f"PID {a['pid']} · started {a['started'].replace('T', ' ')} · up {a['uptime']} · {human_bytes(a['rss'])} RSS · {a['threads']} threads"),
        ("Working directory", f"`{a['working_dir']}`"),
        ("Registry", f"`{registry.REGISTRY_PATH}`" + (f" · updated {registry.load().get('updated_at', '')[:16].replace('T', ' ')}" if registry.REGISTRY_PATH.exists() else " (not created yet)")),
        ("Generations recorded", f"{stats.get('images') or 0} images · {stats.get('total_seconds') or 0} s GPU time · hottest {stats.get('max_temp_c') or 'n/a'} °C · peak {stats.get('max_power_w') or 'n/a'} W"),
    ])
    return _tiles(d), gpu_md, cpu_md, mem_md, net_md, power_md, soft_md


# ----------------------------------------------------------------------------- settings tab

def startup_choices():
    return [("Nothing: load the model on the first generation", "none")] + [(m.name, m.key) for m in models.all_models()]


def settings_values():
    return (registry.setting("startup_model"), float(registry.setting("ui_lora_weight")), registry.setting("default_negative"),
            registry.setting("civitai_token"), bool(registry.setting("auto_release_vram")), int(registry.setting("outputs_keep_days") or 0))


def save_settings(startup, weight, negative, civitai, auto_release, keep_days):
    registry.set_settings(startup_model=startup or "none", ui_lora_weight=float(weight), default_negative=negative.strip(),
                          civitai_token=(civitai or "").strip(), auto_release_vram=bool(auto_release), outputs_keep_days=int(keep_days or 0))
    gr.Info("Settings saved")
    return models_df(), hf_status_html()


def hf_status_html(check: bool = False) -> str:
    tok = downloads.hf_token()
    if not tok:
        return f'<div class="fz-note">{status_span("warning", "No Hugging Face token")} · FLUX and other gated repos cannot be downloaded. Paste a token from huggingface.co/settings/tokens.</div>'
    who = ""
    if check:
        try:
            from huggingface_hub import HfApi
            who = f" · signed in as <b>{escape(HfApi().whoami()['name'])}</b>"
        except Exception as e:
            who = f" · token rejected ({escape(str(e))[:80]})"
    return f'<div class="fz-note">{status_span("good", "Hugging Face token stored")}{who}</div>'


def save_hf_token(token: str):
    token = (token or "").strip()
    if not token:
        raise gr.Error("Paste a token first.")
    from huggingface_hub import login
    try:
        login(token=token, add_to_git_credential=False)
    except Exception as e:
        raise gr.Error(f"Token rejected: {e}")
    applog.log("admin: Hugging Face token saved")
    return hf_status_html(check=True), ""


def cleanup_outputs(days, confirm, unlocked):
    guard(unlocked, confirm)
    days = int(days or 0)
    if days < 1:
        raise gr.Error("Give a number of days (1 or more).")
    cutoff = datetime.now() - timedelta(days=days)
    removed_rows, removed_files = 0, 0
    for r in db.fetch_filtered(limit=10**6):
        if r["created_at"] < cutoff.isoformat(timespec="seconds"):
            removed_files += len(db.delete(r["id"], remove_files=True))
            removed_rows += 1
    out = Path(__file__).parent / "outputs"
    for p in out.glob("*.png"):
        if datetime.fromtimestamp(p.stat().st_mtime) < cutoff:
            try:
                p.unlink()
                removed_files += 1
            except OSError:
                pass
    applog.log(f"admin: cleanup removed {removed_rows} records and {removed_files} files older than {days} days")
    gr.Info(f"Removed {removed_rows} records and {removed_files} files older than {days} days")
    return f'<div class="fz-note">{status_span("good", "Done")}: {removed_rows} records, {removed_files} files removed.</div>', False


# ----------------------------------------------------------------------------- logs tab

def logs_text(n, contains, level):
    return applog.tail_text(int(n or 200), contains or "", None if level == "all" else level)


# ----------------------------------------------------------------------------- build

def build_admin(manager, upscaler, static_gpu, prompt_box, tabs, model_choice, lora_choice, loaded_md, media_mgr=None):
    CTX.manager, CTX.media, CTX.upscaler, CTX.static_gpu = manager, media_mgr, upscaler, static_gpu
    unlocked = gr.State(not admin_password())
    CTX.unlocked = unlocked
    stats_kw = dict(concurrency_id="stats", concurrency_limit=3)
    gpu_kw = dict(concurrency_id="gpu")

    with gr.Tabs():
        # ---------------------------------------------------------------- overview
        with gr.Tab("Overview") as t_overview:
            with gr.Row():
                ov_btn = gr.Button("Refresh", variant="primary", scale=0, min_width=120)
                ov_auto = gr.Checkbox(label="Auto-refresh every 5 s", value=False, scale=0)
            ov_tiles = gr.HTML("")
            gr.Markdown("#### Activity")
            ov_activity = gr.HTML("")
            gr.Markdown("#### Generation statistics by model")
            ov_models = gr.Dataframe(headers=MODEL_STATS_COLUMNS, interactive=False, wrap=True)
            ov_timer = gr.Timer(5, active=False)
            ov_out = [ov_tiles, ov_activity, ov_models]
            t_overview.select(overview_refresh, None, ov_out, **stats_kw)
            ov_btn.click(overview_refresh, None, ov_out, **stats_kw)
            ov_timer.tick(overview_refresh, None, ov_out, show_progress="hidden", **stats_kw)
            ov_auto.change(lambda on: gr.Timer(active=on), ov_auto, ov_timer)

        # ---------------------------------------------------------------- models
        with gr.Tab("Models") as t_models:
            gr.Markdown("Click a model to edit its defaults. Changes are saved to `models/registry.json` and apply to the next load. "
                        "Add new models in the **Add** tab. Image, **video** and **music** models all live here; the family decides "
                        "which tab uses them.")
            m_table = gr.Dataframe(value=models_df, headers=MODEL_COLUMNS, interactive=False, wrap=True, max_height=360)
            m_status = gr.HTML(model_status_html(""))
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        ed_key = gr.Textbox(label="Key (say 'model <key>')", interactive=False, scale=1)
                        ed_name = gr.Textbox(label="Name", scale=2)
                        ed_family = gr.Dropdown(FAMILY_CHOICES, value="sdxl", label="Family", interactive=False, scale=2)
                    ed_desc = gr.Textbox(label="Description (shown in the Model box)")
                    ed_source = gr.Textbox(label="Source (Hugging Face repo id or local file)", interactive=False)
                    with gr.Row():
                        ed_steps = gr.Number(label="Default steps", value=30, precision=0)
                        ed_guidance = gr.Number(label="Default guidance (CFG)", value=7.0)
                        ed_sched = gr.Dropdown(SCHED_CHOICES, value="dpmpp_2m_karras", label="Sampler")
                        ed_variant = gr.Textbox(label="Weight variant", placeholder="fp16 or empty")
                    with gr.Row():
                        ed_aliases = gr.Textbox(label="Aliases (comma separated)", placeholder="jugg, juggernautxl")
                        ed_styles = gr.Textbox(label="Auto-pick for styles", placeholder="photorealistic, cinematic")
                    with gr.Row():
                        ed_negative = gr.Checkbox(label="Supports a negative prompt", value=True)
                        ed_enabled = gr.Checkbox(label="Enabled (shown in the Model box and usable from the prompt)", value=True)
                with gr.Column(scale=1, min_width=220):
                    m_save = gr.Button("Save changes", variant="primary")
                    m_load = gr.Button("Load now")
                    m_download = gr.Button("Download in background")
                    m_startup = gr.Button("Set as start-up model")
                    m_unload = gr.Button("Unload current model (free VRAM)")
                    m_reset = gr.Button("Remove from list / reset built-in", variant="stop")
                    with gr.Group():
                        m_confirm = gr.Checkbox(label="I want to delete this model's files from disk", value=False)
                        m_delete = gr.Button("Delete files from disk", variant="stop")
            EDITOR = [ed_key, ed_name, ed_desc, ed_family, ed_source, ed_steps, ed_guidance, ed_sched, ed_variant,
                      ed_negative, ed_aliases, ed_styles, ed_enabled, m_status]
            t_models.select(lambda: models_df(), None, m_table, **stats_kw)
            m_table.select(select_model, [m_table], EDITOR)
            m_save.click(save_model, EDITOR[:-1], [m_table, m_status, model_choice, loaded_md])
            m_load.click(load_model, [ed_key], [m_table, m_status, loaded_md], **gpu_kw)
            m_unload.click(unload_model, None, [m_table, m_status, loaded_md], **gpu_kw)
            m_startup.click(set_startup, [ed_key], [m_table, m_status])
            m_reset.click(reset_or_remove_model, [ed_key, unlocked], [m_table, *EDITOR, model_choice, loaded_md])

        # ---------------------------------------------------------------- loras
        with gr.Tab("LoRAs") as t_loras:
            gr.Markdown("Every `.safetensors` file in `models/loras/`. Record the **family** the LoRA was trained for and its "
                        "**trigger words**: Auto then switches to a matching model and the Studio shows the words to use.")
            l_table = gr.Dataframe(value=loras_df, headers=LORA_COLUMNS, interactive=False, wrap=True, max_height=300)
            l_status = gr.HTML(lora_status_html(""))
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        le_stem = gr.Textbox(label="File (tag: <lora:file:weight>)", interactive=False)
                        le_name = gr.Textbox(label="Display name")
                        le_family = gr.Dropdown(LORA_FAMILY_CHOICES, value="sdxl", label="Trained for")
                    with gr.Row():
                        le_trigger = gr.Textbox(label="Trigger words (put them in the prompt)", scale=2)
                        le_weight = gr.Slider(-1.0, 2.0, value=0.8, step=0.05, label="Default weight", scale=1)
                    le_desc = gr.Textbox(label="Description / notes")
                    with gr.Row():
                        le_source = gr.Textbox(label="Source (where it came from)", scale=2)
                        le_tags = gr.Textbox(label="Tags", placeholder="style, pixel, retro", scale=1)
                    le_enabled = gr.Checkbox(label="Enabled (listed in the LoRAs box)", value=True)
                with gr.Column(scale=1, min_width=220):
                    l_save = gr.Button("Save changes", variant="primary")
                    l_try = gr.Button("Try it in the Studio")
                    with gr.Group():
                        l_confirm = gr.Checkbox(label="I want to delete this LoRA file", value=False)
                        l_delete = gr.Button("Delete LoRA file", variant="stop")
            LEDITOR = [le_stem, le_name, le_family, le_trigger, le_weight, le_desc, le_source, le_enabled, le_tags, l_status]
            t_loras.select(lambda: loras_df(), None, l_table, **stats_kw)
            l_table.select(select_lora, [l_table], LEDITOR)
            l_save.click(save_lora, LEDITOR[:-1], [l_table, l_status, lora_choice])
            l_try.click(try_lora, [le_stem], [prompt_box, tabs])
            l_delete.click(delete_lora, [le_stem, l_confirm, unlocked], [l_table, *LEDITOR, lora_choice, l_confirm])

        # ---------------------------------------------------------------- add
        with gr.Tab("Add") as t_add:
            gr.Markdown(
                "Paste a **Hugging Face repo** (`RunDiffusion/Juggernaut-XL-v9`, `Lightricks/LTX-Video`, `facebook/musicgen-stereo-medium`), "
                "a **file in a repo** (`nerijs/pixel-art-xl/pixel-art-xl.safetensors`), a **Civitai link**, a **direct URL** to a `.safetensors` file "
                "or a **local path**, then press *Check*. Video and music repos are recognised from their pipeline class. "
                "Downloads run in the background; models become available as soon as they finish."
            )
            with gr.Row():
                a_source = gr.Textbox(label="Source", placeholder="owner/repo · owner/repo/file.safetensors · https://civitai.com/models/… · D:\\models\\file.safetensors", scale=5)
                a_check = gr.Button("Check", variant="primary", scale=1, min_width=110)
            a_info = gr.HTML("")
            a_state = gr.State({})
            with gr.Row():
                a_type = gr.Radio(ADD_TYPES, value="model", label="Add as")
                a_file = gr.Dropdown([], label="File in repo", visible=False, allow_custom_value=True)
            with gr.Row():
                a_key = gr.Textbox(label="Key (how to name it in the prompt)", scale=1)
                a_name = gr.Textbox(label="Display name", scale=2)
                a_family = gr.Dropdown(FAMILY_CHOICES, value="sdxl", label="Family / trained for", scale=2)
            with gr.Row():
                a_trigger = gr.Textbox(label="Trigger words (LoRA)", scale=2)
                a_weight = gr.Slider(-1.0, 2.0, value=0.8, step=0.05, label="Default weight (LoRA)", scale=1)
            with gr.Row():
                a_steps = gr.Number(label="Default steps", value=30, precision=0)
                a_guidance = gr.Number(label="Default guidance", value=7.0)
                a_sched = gr.Dropdown(SCHED_CHOICES, value="dpmpp_2m_karras", label="Sampler")
            a_desc = gr.Textbox(label="Description")
            with gr.Row():
                a_download = gr.Checkbox(label="Download now (models only; files always download)", value=True)
                a_add = gr.Button("Add", variant="primary", interactive=False, scale=0, min_width=140)
            a_result = gr.HTML("")
            with gr.Accordion("Upload a file from this computer instead", open=False):
                gr.Markdown("Uploads go through the browser, so they suit LoRAs (a few hundred MB). Fill in name, family and trigger words above first.")
                with gr.Row():
                    u_file = gr.File(label=".safetensors file", file_types=[".safetensors"], scale=3)
                    u_btn = gr.Button("Save upload", scale=1)

            a_check.click(check_source, [a_source], [a_state, a_info, a_type, a_key, a_name, a_family, a_file, a_trigger,
                                                     a_steps, a_guidance, a_sched, a_desc, a_add], **stats_kw)
            a_source.submit(check_source, [a_source], [a_state, a_info, a_type, a_key, a_name, a_family, a_file, a_trigger,
                                                       a_steps, a_guidance, a_sched, a_desc, a_add], **stats_kw)

        # ---------------------------------------------------------------- downloads
        with gr.Tab("Downloads") as t_downloads:
            with gr.Row():
                d_refresh = gr.Button("Refresh", variant="primary", scale=0, min_width=120)
                d_id = gr.Number(label="Download #", precision=0, scale=0, min_width=130)
                d_cancel = gr.Button("Cancel that download", scale=0)
                d_clear = gr.Button("Clear finished", scale=0)
            d_html = gr.HTML(jobs_html())
            d_timer = gr.Timer(3, active=True)
            t_downloads.select(downloads_refresh, None, d_html, **stats_kw)
            d_refresh.click(downloads_refresh, None, d_html, **stats_kw)
            d_timer.tick(downloads_refresh, None, d_html, show_progress="hidden", **stats_kw)
            d_cancel.click(cancel_job, [d_id], d_html)
            d_clear.click(clear_jobs, None, d_html)

        # ---------------------------------------------------------------- system
        with gr.Tab("System") as t_system:
            with gr.Row():
                s_btn = gr.Button("Refresh", variant="primary", scale=0, min_width=120)
                s_auto = gr.Checkbox(label="Auto-refresh every 5 s", value=False, scale=0)
                s_net = gr.Checkbox(label="Check internet reachability (Hugging Face, GitHub, PyPI)", value=False, scale=0)
            s_tiles = gr.HTML("")
            with gr.Row():
                with gr.Column():
                    s_gpu = gr.Markdown("", elem_classes=["fz-kv"])
                    s_power = gr.Markdown("", elem_classes=["fz-kv"])
                    s_soft = gr.Markdown("", elem_classes=["fz-kv"])
                with gr.Column():
                    s_cpu = gr.Markdown("", elem_classes=["fz-kv"])
                    s_mem = gr.Markdown("", elem_classes=["fz-kv"])
                    s_net_md = gr.Markdown("", elem_classes=["fz-kv"])
            s_out = [s_tiles, s_gpu, s_cpu, s_mem, s_net_md, s_power, s_soft]
            s_timer = gr.Timer(5, active=False)
            t_system.select(system_refresh, [s_net], s_out, **stats_kw)
            s_btn.click(system_refresh, [s_net], s_out, **stats_kw)
            s_timer.tick(system_refresh, [s_net], s_out, show_progress="hidden", **stats_kw)
            s_auto.change(lambda on: gr.Timer(active=on), s_auto, s_timer)

        # ---------------------------------------------------------------- settings
        with gr.Tab("Settings") as t_settings:
            st_vals = settings_values()
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Studio defaults")
                    st_startup = gr.Dropdown(startup_choices(), value=st_vals[0], label="Model loaded at start-up")
                    st_weight = gr.Slider(0.1, 1.5, value=st_vals[1], step=0.05, label="Weight for LoRAs ticked in the LoRAs box (when the LoRA has no weight of its own)")
                    st_negative = gr.Textbox(label="Default negative prompt", value=st_vals[2])
                    st_release = gr.Checkbox(label="Release cached VRAM after each image on the big flow models (Z-Image, Qwen, FLUX)", value=st_vals[4])
                    st_save = gr.Button("Save settings", variant="primary", scale=0, min_width=140)
                with gr.Column():
                    gr.Markdown("#### Accounts and tokens")
                    st_hf_status = gr.HTML(hf_status_html())
                    with gr.Row():
                        st_hf = gr.Textbox(label="Hugging Face token (stored by huggingface_hub, needed for FLUX and gated repos)", type="password", scale=3)
                        st_hf_save = gr.Button("Save token", scale=0, min_width=110)
                        st_hf_check = gr.Button("Check login", scale=0, min_width=110)
                    st_civitai = gr.Textbox(label="Civitai API token (optional; some downloads require it)", value=st_vals[3], type="password")
                    gr.Markdown("#### Housekeeping")
                    with gr.Row():
                        st_days = gr.Number(label="Delete generations and images older than (days)", value=30, precision=0, scale=1)
                        st_clean_confirm = gr.Checkbox(label="Yes, delete them", value=False, scale=1)
                        st_clean = gr.Button("Clean up outputs", variant="stop", scale=0, min_width=150)
                    st_clean_result = gr.HTML("")
                    gr.Markdown("#### Admin lock")
                    with gr.Row():
                        st_pw = gr.Textbox(label="Admin password" + ("" if admin_password() else " (none set: FRAZILE_ADMIN_PASSWORD)"), type="password", scale=3)
                        st_unlock = gr.Button("Unlock", scale=0, min_width=110)
                    st_lock_status = gr.HTML("" if admin_password() else '<div class="fz-note">No admin password is set; destructive actions only need their confirmation box. '
                                             'Set <code>FRAZILE_ADMIN_PASSWORD</code> when the app is reachable from other machines (Docker, 0.0.0.0).</div>')
            st_keep_hidden = gr.Number(value=st_vals[5], visible=False)
            st_save.click(save_settings, [st_startup, st_weight, st_negative, st_civitai, st_release, st_keep_hidden], [m_table, st_hf_status])
            st_hf_save.click(save_hf_token, [st_hf], [st_hf_status, st_hf])
            st_hf_check.click(lambda: hf_status_html(check=True), None, st_hf_status, **stats_kw)
            st_clean.click(cleanup_outputs, [st_days, st_clean_confirm, unlocked], [st_clean_result, st_clean_confirm])
            st_unlock.click(unlock, [st_pw, unlocked], [unlocked, st_lock_status])
            t_settings.select(lambda: gr.update(choices=startup_choices(), value=registry.setting("startup_model")), None, st_startup)

        # ---------------------------------------------------------------- logs
        with gr.Tab("Logs") as t_logs:
            with gr.Row():
                lg_filter = gr.Textbox(label="Contains", placeholder="LoRA, download, error ...", scale=2)
                lg_level = gr.Dropdown([("all", "all"), ("errors only", "error")], value="all", label="Level", scale=1)
                lg_n = gr.Number(label="Lines", value=200, precision=0, scale=1)
                lg_btn = gr.Button("Refresh", variant="primary", scale=0, min_width=120)
                lg_auto = gr.Checkbox(label="Auto-refresh every 3 s", value=False, scale=0)
            lg_text = gr.Textbox(value=logs_text(200, "", "all"), lines=28, max_lines=28, show_label=False, interactive=False, elem_classes=["fz-log"])
            lg_timer = gr.Timer(3, active=False)
            lg_in = [lg_n, lg_filter, lg_level]
            t_logs.select(logs_text, lg_in, lg_text, **stats_kw)
            lg_btn.click(logs_text, lg_in, lg_text, **stats_kw)
            lg_filter.submit(logs_text, lg_in, lg_text, **stats_kw)
            lg_level.change(logs_text, lg_in, lg_text, **stats_kw)
            lg_timer.tick(logs_text, lg_in, lg_text, show_progress="hidden", **stats_kw)
            lg_auto.change(lambda on: gr.Timer(active=on), lg_auto, lg_timer)

    # events that touch several tabs are wired after every component exists
    m_download.click(download_model, [ed_key], [d_html, m_status])
    m_delete.click(delete_model_files, [ed_key, m_confirm, unlocked], [m_table, *EDITOR, model_choice, loaded_md, m_confirm])
    a_add.click(add_source, [a_state, a_type, a_key, a_name, a_family, a_file, a_trigger, a_steps, a_guidance, a_sched, a_desc, a_weight, a_download],
                [a_result, d_html, m_table, l_table, model_choice, lora_choice])
    u_btn.click(upload_file, [u_file, a_type, a_name, a_family, a_trigger, a_weight], [a_result, m_table, l_table, model_choice, lora_choice, u_file])
    return SimpleNamespace(models_table=m_table, loras_table=l_table, downloads_html=d_html)
