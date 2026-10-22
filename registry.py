"""
Persistent registry of models, LoRAs and app settings, edited from the Admin panel.

File: models/registry.json (created on first use). It holds three things:

    settings   startup_model, ui_lora_weight, default_negative, civitai_token ...
    models     user-added models (key -> full spec) and overrides for built-in models
               (steps, guidance, scheduler, description, enabled, ...)
    loras      metadata per LoRA file in models/loras/: display name, family the LoRA was
               trained for, trigger words, default weight, source, enabled

models.py reads this registry when it lists models (all_models) and LoRAs (lora_entries), so
anything added here is usable from the prompt, the Model box and the LoRAs box right away.

    python registry.py           # print the registry
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from models import BUILTIN, LORAS_DIR, MODELS_DIR, STARTER_LORAS, ModelSpec, _slug

REGISTRY_PATH = MODELS_DIR / "registry.json"

FAMILIES = {
    "sdxl": "SDXL (Stable Diffusion XL and fine-tunes: Juggernaut, Pony, Illustrious ...)",
    "zimage": "Z-Image (Tongyi Z-Image Turbo and fine-tunes)",
    "qwen": "Qwen-Image",
    "flux": "FLUX.1 (dev / schnell and fine-tunes)",
    "video": "Video (LTX-Video, Wan 2.1 in diffusers layout) — Video tab",
    "music": "Music (MusicGen, Stable Audio Open) — Music tab",
}
IMAGE_FAMILIES = ("sdxl", "zimage", "qwen", "flux")      # families a LoRA can be trained for
SCHEDULERS = {
    "dpmpp_2m_karras": "DPM++ 2M Karras",
    "dpmpp_sde_karras": "DPM++ SDE Karras",
    "euler_a": "Euler a",
    "euler": "Euler",
    "": "model default",
}
# Which models to prefer when a LoRA needs a family the current model is not in (first one on disk wins).
FAMILY_PREFERRED = {
    "sdxl": ("juggernaut", "realvis", "sdxl", "dreamshaper", "playground", "animagine"),
    "zimage": ("zimage",),
    "qwen": ("qwen",),
    "flux": ("flux",),
    "video": ("ltx", "wan"),
    "music": ("musicgen", "musicgen-small", "stable-audio"),
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "startup_model": "zimage",
    "ui_lora_weight": 0.8,
    "default_negative": "blurry, low quality, deformed, watermark, text",
    "civitai_token": "",
    "auto_release_vram": True,
    "outputs_keep_days": 0,          # 0 = keep forever
}

# Fields of ModelSpec an admin may override on a built-in model.
OVERRIDABLE = ("name", "steps", "guidance", "scheduler", "variant", "negative", "aliases", "styles",
               "description", "download_gb")

_lock = threading.RLock()
_cache: dict[str, Any] | None = None
_cache_mtime: float | None = None


# ----------------------------------------------------------------------------- file

def _empty() -> dict[str, Any]:
    return {"version": 1, "settings": dict(DEFAULT_SETTINGS), "models": {}, "loras": {}}


def load(force: bool = False) -> dict[str, Any]:
    """The registry dict (cached; re-read when the file changes on disk)."""
    global _cache, _cache_mtime
    with _lock:
        mtime = REGISTRY_PATH.stat().st_mtime if REGISTRY_PATH.exists() else None
        if _cache is not None and not force and mtime == _cache_mtime:
            return _cache
        data = _empty()
        if REGISTRY_PATH.exists():
            try:
                loaded = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
                for k in ("settings", "models", "loras"):
                    if isinstance(loaded.get(k), dict):
                        data[k].update(loaded[k])
            except (OSError, json.JSONDecodeError) as e:
                print(f"registry: could not read {REGISTRY_PATH} ({e}); starting empty")
        _cache, _cache_mtime = data, mtime
        return data


def save(data: dict[str, Any] | None = None) -> None:
    global _cache, _cache_mtime
    with _lock:
        data = data if data is not None else load()
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = REGISTRY_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(REGISTRY_PATH)
        _cache, _cache_mtime = data, REGISTRY_PATH.stat().st_mtime


# ----------------------------------------------------------------------------- settings

def setting(key: str, default: Any = None) -> Any:
    return load()["settings"].get(key, DEFAULT_SETTINGS.get(key, default))


def set_settings(**values: Any) -> None:
    with _lock:
        data = load()
        data["settings"].update(values)
        save(data)


# ----------------------------------------------------------------------------- models

def _entry_to_spec(key: str, e: dict[str, Any]) -> ModelSpec:
    known = {f.name for f in fields(ModelSpec)}
    kw = {k: v for k, v in e.items() if k in known}
    kw["key"] = key
    kw["aliases"] = tuple(kw.get("aliases") or ())
    kw["styles"] = tuple(kw.get("styles") or ())
    if kw.get("scheduler") == "":
        kw["scheduler"] = None
    if kw.get("variant") == "":
        kw["variant"] = None
    kw.setdefault("name", key)
    kw.setdefault("source", "")
    src = kw["source"]
    kw["local"] = bool(src) and (Path(src).suffix == ".safetensors" or Path(src).exists())
    return ModelSpec(**kw)


def apply_overrides(spec: ModelSpec) -> ModelSpec:
    """A built-in or local spec with the admin's overrides applied."""
    e = load()["models"].get(spec.key)
    if not e or e.get("custom"):
        return spec
    changes: dict[str, Any] = {}
    for k in OVERRIDABLE:
        if k in e and e[k] is not None:
            v = e[k]
            if k in ("aliases", "styles"):
                v = tuple(v)
            if k in ("scheduler", "variant") and v == "":
                v = None
            changes[k] = v
    return replace(spec, **changes) if changes else spec


def custom_models() -> list[ModelSpec]:
    out = []
    for key, e in load()["models"].items():
        if e.get("custom"):
            try:
                out.append(_entry_to_spec(key, e))
            except TypeError as ex:
                print(f"registry: bad model entry {key}: {ex}")
    return out


def model_entry(key: str) -> dict[str, Any]:
    return dict(load()["models"].get(key, {}))


def is_enabled(key: str) -> bool:
    return bool(load()["models"].get(key, {}).get("enabled", True))


def is_custom(key: str) -> bool:
    return bool(load()["models"].get(key, {}).get("custom"))


def upsert_model(key: str, custom: bool, **values: Any) -> str:
    """Create or update a model entry. For built-ins only the overridable fields are stored."""
    key = _slug(key)
    if not key:
        raise ValueError("The model needs a key (letters, digits, dashes).")
    with _lock:
        data = load()
        e = data["models"].get(key, {})
        if custom:
            if not values.get("source"):
                raise ValueError("A custom model needs a source (Hugging Face repo id, local file or URL).")
            if values.get("family") not in FAMILIES:
                raise ValueError(f"Unknown family {values.get('family')!r}; one of {', '.join(FAMILIES)}.")
            e.update({k: v for k, v in values.items() if v is not None})
            e["custom"] = True
            e.setdefault("added_at", datetime.now().isoformat(timespec="seconds"))
        else:
            allowed = set(OVERRIDABLE) | {"enabled", "notes"}
            e.update({k: v for k, v in values.items() if k in allowed})
            e["custom"] = False
        if "aliases" in e:
            e["aliases"] = _clean_words(e["aliases"])
        if "styles" in e:
            e["styles"] = _clean_words(e["styles"])
        data["models"][key] = e
        save(data)
    return key


def remove_model(key: str) -> dict[str, Any] | None:
    """Delete a custom entry, or reset a built-in's overrides. Returns the removed entry."""
    with _lock:
        data = load()
        e = data["models"].pop(key, None)
        if e is not None:
            save(data)
        return e


def set_enabled(key: str, enabled: bool) -> None:
    with _lock:
        data = load()
        e = data["models"].setdefault(key, {"custom": False})
        e["enabled"] = bool(enabled)
        save(data)


def _clean_words(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = re.split(r"[,\n;]+", value)
    else:
        parts = list(value or [])
    return [p.strip().lower() for p in parts if p and p.strip()]


# ----------------------------------------------------------------------------- loras

LORA_DEFAULTS = {"name": "", "family": "sdxl", "trigger": "", "weight": 0.8, "description": "",
                 "source": "", "enabled": True, "tags": []}


def lora_entries(include_missing: bool = True) -> dict[str, dict[str, Any]]:
    """One record per LoRA: files in models/loras/ merged with their registry metadata.
    Keys are file stems; records with 'missing': True have metadata but no file."""
    data = load()
    out: dict[str, dict[str, Any]] = {}
    files = {p.stem: p for p in LORAS_DIR.glob("*.safetensors")} if LORAS_DIR.exists() else {}
    for stem, path in sorted(files.items()):
        e = dict(LORA_DEFAULTS)
        e.update(data["loras"].get(stem, {}))
        starter = STARTER_LORAS.get(path.name)
        if not e["trigger"] and starter:
            e["trigger"] = starter[2]
        if not e["source"] and starter:
            e["source"] = f"{starter[0]}/{starter[1]}"
        e["name"] = e["name"] or stem
        e["stem"] = stem
        e["path"] = str(path)
        e["size"] = path.stat().st_size
        e["missing"] = False
        out[stem] = e
    if include_missing:
        for stem, meta in data["loras"].items():
            if stem not in out:
                e = dict(LORA_DEFAULTS)
                e.update(meta)
                e.update({"name": e["name"] or stem, "stem": stem, "path": str(LORAS_DIR / f"{stem}.safetensors"),
                          "size": 0, "missing": True})
                out[stem] = e
    return out


def lora(stem: str) -> dict[str, Any] | None:
    return lora_entries().get(stem)


def set_lora(stem: str, **values: Any) -> None:
    with _lock:
        data = load()
        e = data["loras"].setdefault(stem, {})
        for k, v in values.items():
            if k in LORA_DEFAULTS and v is not None:
                e[k] = v
        if "tags" in e:
            e["tags"] = _clean_words(e["tags"])
        if "weight" in e:
            e["weight"] = float(max(-2.0, min(2.0, e["weight"])))
        save(data)


def forget_lora(stem: str) -> None:
    with _lock:
        data = load()
        if data["loras"].pop(stem, None) is not None:
            save(data)


# ----------------------------------------------------------------------------- summary

def summary() -> dict[str, Any]:
    data = load()
    return {
        "path": str(REGISTRY_PATH),
        "exists": REGISTRY_PATH.exists(),
        "updated_at": data.get("updated_at"),
        "custom_models": sum(1 for e in data["models"].values() if e.get("custom")),
        "overrides": sum(1 for e in data["models"].values() if not e.get("custom")),
        "loras": len(lora_entries(include_missing=False)),
        "settings": data["settings"],
    }


if __name__ == "__main__":
    print(json.dumps(load(), indent=2))
    print("\nLoRAs:")
    for stem, e in lora_entries().items():
        print(f"  {stem:20} {e['family']:6} trigger={e['trigger']!r} weight={e['weight']} "
              f"{'MISSING' if e['missing'] else f'{e['size'] / 1024**2:.0f} MB'}")
