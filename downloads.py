"""
Background downloads and source inspection for the Admin panel.

Sources understood by inspect_source():
    Hugging Face repo        "RunDiffusion/Juggernaut-XL-v9"  (diffusers layout -> model family detected
                             from model_index.json; a repo of single .safetensors files -> checkpoint/LoRA)
    Hugging Face file        "nerijs/pixel-art-xl/pixel-art-xl.safetensors" or the full https URL
    Civitai                  "https://civitai.com/models/133005?modelVersionId=..." or an api download URL
                             (type, base model, trigger words and the primary file come from the Civitai API)
    Direct URL               any https link to a .safetensors file
    Local path               "D:\\downloads\\model.safetensors" (copied into models/)

Downloads run in a thread pool; the JobManager keeps progress (bytes, speed, ETA) and a log per
job so the Admin > Downloads tab can poll them. Files land in models/checkpoints/ or
models/loras/, repos in the Hugging Face cache (same place the loader reads from).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

import requests

import models
import registry
from models import CHECKPOINTS_DIR, LORAS_DIR, ModelSpec

CHUNK = 1 << 20   # 1 MiB
USER_AGENT = "FrazileAI-ImageStudio/2.1"

PIPELINE_FAMILY = {
    "StableDiffusionXLPipeline": "sdxl", "StableDiffusionXLImg2ImgPipeline": "sdxl",
    "ZImagePipeline": "zimage", "QwenImagePipeline": "qwen", "FluxPipeline": "flux",
    "LTXPipeline": "video", "LTXImageToVideoPipeline": "video", "LTXConditionPipeline": "video",
    "WanPipeline": "video", "WanImageToVideoPipeline": "video",
    "StableAudioPipeline": "music",
}
TRANSFORMERS_FAMILY = {"musicgen": "music", "musicgen_melody": "music"}   # config.json model_type -> family
SUPPORTED_TEXT = "SDXL, Z-Image, Qwen-Image, FLUX, LTX-Video, Wan 2.1, MusicGen or Stable Audio"
CIVITAI_FAMILY = {  # Civitai baseModel -> our family (None = unsupported here)
    "sdxl 1.0": "sdxl", "sdxl 0.9": "sdxl", "sdxl turbo": "sdxl", "sdxl lightning": "sdxl", "sdxl hyper": "sdxl",
    "pony": "sdxl", "illustrious": "sdxl", "noobai": "sdxl", "sdxl distilled": "sdxl",
    "flux.1 s": "flux", "flux.1 d": "flux", "flux.1": "flux", "qwen": "qwen", "z-image": "zimage",
}


# ----------------------------------------------------------------------------- jobs

@dataclass
class Job:
    id: int
    kind: str                 # "model" | "lora" | "file" | "repo"
    label: str
    status: str = "queued"    # queued | running | done | error | cancelled
    done: int = 0
    total: int | None = None
    speed: float = 0.0        # bytes / s (recent)
    message: str = ""
    error: str | None = None
    target: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel_requested: bool = False
    log: list[str] = field(default_factory=list)
    on_done: Callable[["Job"], None] | None = None
    _rate_mark: tuple[float, int] | None = None

    def say(self, text: str) -> None:
        self.message = text
        self.log.append(f"{time.strftime('%H:%M:%S')} {text}")
        print(f"[download #{self.id}] {text}", flush=True)

    def progress(self, done: int, total: int | None = None) -> None:
        now = time.time()
        if self._rate_mark and now - self._rate_mark[0] >= 1.0:
            self.speed = (done - self._rate_mark[1]) / (now - self._rate_mark[0])
            self._rate_mark = (now, done)
        elif not self._rate_mark:
            self._rate_mark = (now, done)
        self.done = done
        if total:
            self.total = total

    @property
    def pct(self) -> float | None:
        return min(100.0, self.done / self.total * 100) if self.total else None

    @property
    def eta_s(self) -> float | None:
        if self.total and self.speed > 0 and self.status == "running":
            return max(0.0, (self.total - self.done) / self.speed)
        return None

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "label": self.label, "status": self.status, "done": self.done,
            "total": self.total, "pct": self.pct, "speed": self.speed, "eta_s": self.eta_s, "message": self.message,
            "error": self.error, "target": self.target, "started": self.started, "finished": self.finished,
            "seconds": (self.finished or time.time()) - self.started if self.started else 0.0,
        }


class JobManager:
    def __init__(self, workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="download")
        self._jobs: dict[int, Job] = {}
        self._next = 1
        self._lock = threading.Lock()

    def submit(self, kind: str, label: str, fn: Callable[[Job], str | None], on_done=None) -> Job:
        with self._lock:
            job = Job(id=self._next, kind=kind, label=label, on_done=on_done)
            self._next += 1
            self._jobs[job.id] = job
        job.say("queued")

        def run():
            job.status, job.started = "running", time.time()
            try:
                result = fn(job)
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.say("cancelled")
                else:
                    job.status, job.target = "done", result
                    if job.total and job.done < job.total:
                        job.done = job.total
                    job.say(f"done in {time.time() - job.started:.0f} s" + (f": {result}" if result else ""))
                    if job.on_done:
                        try:
                            job.on_done(job)
                        except Exception as e:  # a failing hook must not hide a finished download
                            job.say(f"post-step failed: {e}")
            except Cancelled:
                job.status = "cancelled"
                job.say("cancelled")
            except Exception as e:
                job.status, job.error = "error", f"{type(e).__name__}: {e}"
                job.say(f"failed: {job.error}")
                traceback.print_exc()
            finally:
                job.finished = time.time()

        self._pool.submit(run)
        return job

    def jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: -j.id)

    def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: int) -> bool:
        job = self._jobs.get(job_id)
        if job and job.active:
            job.cancel_requested = True
            job.say("cancel requested")
            return True
        return False

    def clear_finished(self) -> int:
        with self._lock:
            gone = [i for i, j in self._jobs.items() if not j.active]
            for i in gone:
                del self._jobs[i]
        return len(gone)

    def active(self) -> list[Job]:
        return [j for j in self.jobs() if j.active]


class Cancelled(Exception):
    pass


JOBS = JobManager()


# ----------------------------------------------------------------------------- helpers

def human_bytes(n: float | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def hf_token() -> str | None:
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def civitai_token() -> str:
    return (registry.setting("civitai_token") or os.environ.get("CIVITAI_TOKEN") or "").strip()


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.")
    return stem[:80] or "model"


def unique_path(folder: Path, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / filename
    n = 2
    while p.exists():
        p = folder / f"{Path(filename).stem}-{n}{Path(filename).suffix}"
        n += 1
    return p


# ----------------------------------------------------------------------------- inspection

def inspect_source(text: str) -> dict[str, Any]:
    """Figure out what a source string is and what it would give us. Never raises; see 'error'."""
    text = (text or "").strip().strip('"').strip("'")
    info: dict[str, Any] = {"input": text, "kind": None, "ok": False, "warnings": []}
    if not text:
        info["error"] = "Enter a Hugging Face repo id, a Civitai link, a direct .safetensors URL or a local path."
        return info
    try:
        p = Path(text)
        if (len(text) > 3 and text[1:3] in (":\\", ":/")) or text.startswith(("/", "\\\\", "./", "~")):
            return _inspect_local(p, info)
        if "civitai.com" in text:
            return _inspect_civitai(text, info)
        if text.startswith(("http://", "https://")):
            u = urlparse(text)
            if u.netloc.endswith("huggingface.co"):
                m = re.match(r"^/([^/]+/[^/]+)(?:/(?:resolve|blob)/[^/]+/(.+))?", u.path)
                if m and m.group(2):
                    return _inspect_hf_file(m.group(1), unquote(m.group(2)), info)
                if m:
                    return _inspect_hf_repo(m.group(1), info)
            return _inspect_url(text, info)
        parts = text.split("/")
        if len(parts) == 2 and all(parts):
            return _inspect_hf_repo(text, info)
        if len(parts) >= 3 and text.lower().endswith((".safetensors", ".ckpt", ".pt", ".bin")):
            return _inspect_hf_file("/".join(parts[:2]), "/".join(parts[2:]), info)
        info["error"] = "Not recognised. Use 'owner/repo', 'owner/repo/file.safetensors', a civitai.com link, a URL or a path."
    except requests.RequestException as e:
        info["error"] = f"network error: {e}"
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def _inspect_local(p: Path, info: dict) -> dict:
    p = p.expanduser()
    info.update({"kind": "local", "path": str(p)})
    if not p.is_file():
        info["error"] = f"File not found: {p}"
        return info
    size = p.stat().st_size
    info.update({"ok": True, "filename": p.name, "size": size, "name": p.stem,
                 "type": "lora" if size < 1_500_000_000 else "checkpoint", "family": "sdxl",
                 "summary": f"Local file {p.name} ({human_bytes(size)})"})
    info["warnings"].append("Type guessed from the file size; change it below if this is wrong.")
    return info


def _inspect_url(url: str, info: dict) -> dict:
    info.update({"kind": "url", "url": url})
    r = requests.head(url, allow_redirects=True, timeout=20, headers={"User-Agent": USER_AGENT})
    if r.status_code >= 400:
        info["error"] = f"HTTP {r.status_code} for {url}"
        return info
    filename = _filename_from_response(r) or Path(urlparse(r.url).path).name or "model.safetensors"
    size = int(r.headers.get("Content-Length") or 0) or None
    info.update({"ok": True, "filename": filename, "size": size, "name": safe_stem(filename),
                 "type": "lora" if (size or 0) < 1_500_000_000 else "checkpoint", "family": "sdxl",
                 "summary": f"{filename} ({human_bytes(size)}) from {urlparse(url).netloc}"})
    if not filename.lower().endswith(".safetensors"):
        info["warnings"].append("Only .safetensors files can be loaded; this link does not look like one.")
    return info


def _filename_from_response(r: requests.Response) -> str | None:
    cd = r.headers.get("Content-Disposition", "")
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd) or re.search(r'filename="?([^";]+)"?', cd)
    return unquote(m.group(1)).strip() if m else None


def _inspect_hf_repo(repo: str, info: dict) -> dict:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    info.update({"kind": "hf_repo", "repo": repo})
    api = HfApi(token=hf_token() or None)
    try:
        mi = api.model_info(repo, files_metadata=True)
    except GatedRepoError:
        info["error"] = (f"{repo} is gated. Accept its terms on huggingface.co and save a Hugging Face token "
                         "in Admin > Settings, then check again.")
        return info
    except RepositoryNotFoundError:
        info["error"] = f"No repo called {repo} on Hugging Face (private repos need a token)."
        return info
    files = {s.rfilename: (s.size or 0) for s in (mi.siblings or [])}
    total = sum(files.values())
    info.update({"gated": bool(mi.gated), "files": len(files), "total_size": total,
                 "downloads": getattr(mi, "downloads", None), "likes": getattr(mi, "likes", None),
                 "tags": list(mi.tags or [])[:20]})
    if "model_index.json" in files:
        cls = None
        try:
            from huggingface_hub import hf_hub_download
            idx = json.loads(Path(hf_hub_download(repo, "model_index.json", token=hf_token() or None)).read_text())
            cls = idx.get("_class_name")
        except Exception:
            pass
        family = PIPELINE_FAMILY.get(cls or "")
        has_fp16 = any(".fp16." in f for f in files)
        weights = [f for f in files if f.endswith(".safetensors") and (f.count(".fp16.") > 0) == (has_fp16 and family == "sdxl")]
        info.update({"ok": family is not None, "type": "model", "layout": "diffusers", "pipeline": cls,
                     "family": family or "sdxl", "variant": "fp16" if (has_fp16 and family == "sdxl") else None,
                     "name": repo.split("/")[-1].replace("-", " ").replace("_", " "),
                     "download_size": sum(files[f] for f in weights) or total,
                     "summary": f"diffusers pipeline {cls or '?'} · {len(files)} files · {human_bytes(total)} total"})
        if family is None:
            info["error"] = f"{cls or 'this pipeline'} is not supported here ({SUPPORTED_TEXT} only)."
        elif not weights:
            info["ok"] = False
            info["error"] = "The repo has no .safetensors weights (only .bin or .ckpt); the loaders read safetensors only."
        if family in ("video", "music"):
            # the sub-folders are what gets downloaded; LTX's repo also carries huge single-file checkpoints at its root
            info["download_size"] = sum(files[f] for f in weights if "/" in f) or info["download_size"]
            info["summary"] += f" · {family} model, used from the {family.capitalize()} tab"
        if mi.gated:
            info["warnings"].append("Gated repo: a Hugging Face token with accepted terms is required to download.")
        return info
    if "config.json" in files:
        # transformers layout (MusicGen): config.json with a model_type, weights in model.safetensors
        try:
            from huggingface_hub import hf_hub_download
            cfg = json.loads(Path(hf_hub_download(repo, "config.json", token=hf_token() or None)).read_text())
            family = TRANSFORMERS_FAMILY.get(cfg.get("model_type") or "")
        except Exception:
            family = None
        if family:
            weights = [f for f in files if f.endswith(".safetensors")]
            info.update({"ok": bool(weights), "type": "model", "layout": "transformers", "pipeline": cfg.get("model_type"),
                         "family": family, "variant": None, "name": repo.split("/")[-1].replace("-", " ").replace("_", " "),
                         "download_size": sum(files[f] for f in weights) or total,
                         "summary": f"transformers model {cfg.get('model_type')} · {len(files)} files · {human_bytes(total)} total · "
                                    f"used from the {family.capitalize()} tab"})
            if not weights:
                info["error"] = "The repo has no .safetensors weights (only .bin); the loaders read safetensors only."
            if mi.gated:
                info["warnings"].append("Gated repo: a Hugging Face token with accepted terms is required to download.")
            return info
    sts = sorted((f for f in files if f.lower().endswith(".safetensors")), key=lambda f: -files[f])
    if sts:
        biggest = sts[0]
        info.update({"ok": True, "layout": "single-file", "filename": biggest, "size": files[biggest],
                     "candidates": sts[:12], "name": safe_stem(biggest), "family": "sdxl",
                     "type": "lora" if files[biggest] < 1_500_000_000 else "checkpoint",
                     "summary": f"{len(sts)} .safetensors file(s); largest {biggest} ({human_bytes(files[biggest])})"})
        if len(sts) > 1:
            info["warnings"].append("Several files in this repo; the largest is selected, change the file below if needed.")
        return info
    info["error"] = "The repo has neither a diffusers model_index.json nor any .safetensors file."
    return info


def _inspect_hf_file(repo: str, filename: str, info: dict) -> dict:
    from huggingface_hub import HfApi
    info.update({"kind": "hf_file", "repo": repo, "filename": filename})
    api = HfApi(token=hf_token() or None)
    mi = api.model_info(repo, files_metadata=True)
    sizes = {s.rfilename: (s.size or 0) for s in (mi.siblings or [])}
    if filename not in sizes:
        info["error"] = f"{filename} is not in {repo}. Files: {', '.join(list(sizes)[:10])}"
        return info
    size = sizes[filename]
    info.update({"ok": True, "size": size, "name": safe_stem(filename), "family": "sdxl",
                 "type": "lora" if size < 1_500_000_000 else "checkpoint", "gated": bool(mi.gated),
                 "summary": f"{filename} from {repo} ({human_bytes(size)})"})
    return info


def _inspect_civitai(text: str, info: dict) -> dict:
    info.update({"kind": "civitai"})
    u = urlparse(text)
    q = parse_qs(u.query)
    version_id = model_id = None
    m = re.search(r"/models/(\d+)", u.path)
    if m:
        model_id = int(m.group(1))
    if "modelVersionId" in q:
        version_id = int(q["modelVersionId"][0])
    m = re.search(r"/api/download/models/(\d+)", u.path)
    if m:
        version_id = int(m.group(1))
    m = re.search(r"/model-versions/(\d+)", u.path)
    if m:
        version_id = int(m.group(1))
    headers = {"User-Agent": USER_AGENT}
    if civitai_token():
        headers["Authorization"] = f"Bearer {civitai_token()}"
    if version_id:
        r = requests.get(f"https://civitai.com/api/v1/model-versions/{version_id}", headers=headers, timeout=20)
    elif model_id:
        r = requests.get(f"https://civitai.com/api/v1/models/{model_id}", headers=headers, timeout=20)
    else:
        info["error"] = "Could not find a model id in that Civitai link."
        return info
    if r.status_code >= 400:
        info["error"] = f"Civitai API returned HTTP {r.status_code}" + (" (a Civitai API token may be required)" if r.status_code in (401, 403) else "")
        return info
    data = r.json()
    if version_id:
        ver, model_name, model_type = data, (data.get("model") or {}).get("name"), (data.get("model") or {}).get("type")
    else:
        versions = data.get("modelVersions") or []
        if not versions:
            info["error"] = "That Civitai model has no versions."
            return info
        ver, model_name, model_type = versions[0], data.get("name"), data.get("type")
    files = ver.get("files") or []
    primary = next((f for f in files if f.get("primary")), files[0] if files else None)
    if not primary:
        info["error"] = "No downloadable file on that Civitai version."
        return info
    base = (ver.get("baseModel") or "").lower()
    family = next((fam for key, fam in CIVITAI_FAMILY.items() if base.startswith(key)), None)
    kind = (model_type or "").lower()
    our_type = "lora" if kind in ("lora", "locon", "lycoris", "dora") else "checkpoint" if kind == "checkpoint" else None
    size = int((primary.get("sizeKB") or 0) * 1024) or None
    info.update({
        "ok": family is not None and our_type is not None,
        "model_id": model_id, "version_id": ver.get("id"), "url": primary.get("downloadUrl"),
        "filename": primary.get("name"), "size": size, "civitai_type": model_type, "base_model": ver.get("baseModel"),
        "family": family or "sdxl", "type": our_type or "checkpoint",
        "name": f"{model_name} {ver.get('name') or ''}".strip(),
        "trigger": ", ".join(ver.get("trainedWords") or []),
        "description": (model_name or ""), "nsfw": bool((data.get("model") or data).get("nsfw")),
        "summary": f"Civitai {model_type} · {model_name} / {ver.get('name')} · base {ver.get('baseModel')} · {human_bytes(size)}",
    })
    if family is None:
        info["error"] = f"Base model '{ver.get('baseModel')}' is not supported here (SDXL-family, FLUX, Qwen-Image or Z-Image only)."
    elif our_type is None:
        info["error"] = f"Civitai type '{model_type}' is not supported (Checkpoint or LoRA only)."
    if primary.get("metadata", {}).get("format", "SafeTensor") != "SafeTensor":
        info["warnings"].append("The primary file is not a .safetensors file; it cannot be loaded here.")
    if not civitai_token():
        info["warnings"].append("Some Civitai downloads require an API token (Admin > Settings).")
    return info


# ----------------------------------------------------------------------------- downloading

def _stream_to_file(url: str, dest: Path, job: Job, headers: dict | None = None, params: dict | None = None) -> Path:
    headers = {"User-Agent": USER_AGENT, **(headers or {})}
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    if done:
        headers["Range"] = f"bytes={done}-"
    with requests.get(url, stream=True, allow_redirects=True, timeout=60, headers=headers, params=params) as r:
        if r.status_code == 416:      # the .part file is already complete
            done = 0
            tmp.unlink(missing_ok=True)
            return _stream_to_file(url, dest, job, headers={k: v for k, v in headers.items() if k != "Range"}, params=params)
        if r.status_code in (401, 403):
            raise RuntimeError(f"HTTP {r.status_code}: the server refused the download (token required or terms not accepted)")
        r.raise_for_status()
        if r.status_code != 206 and done:
            done = 0  # server ignored the Range header; start over
        length = r.headers.get("Content-Length")
        total = (int(length) + done) if length else None
        ctype = r.headers.get("Content-Type", "")
        if "text/html" in ctype:
            raise RuntimeError("The link returned a web page, not a file (login or token needed?)")
        job.progress(done, total)
        job.say(f"downloading {dest.name} ({human_bytes(total)})")
        mode = "ab" if done else "wb"
        with open(tmp, mode) as f:
            for chunk in r.iter_content(CHUNK):
                if job.cancel_requested:
                    raise Cancelled()
                f.write(chunk)
                done += len(chunk)
                job.progress(done, total)
    if dest.exists():
        dest.unlink()
    tmp.rename(dest)
    return dest


def download_file(info: dict, as_type: str, rename: str | None = None) -> Callable[[Job], str]:
    """Job function that fetches one .safetensors file into checkpoints/ or loras/."""
    folder = LORAS_DIR if as_type == "lora" else CHECKPOINTS_DIR
    filename = (safe_stem(rename) + ".safetensors") if rename else Path(info.get("filename") or "model.safetensors").name
    if not filename.lower().endswith(".safetensors"):
        filename = Path(filename).stem + ".safetensors"

    def run(job: Job) -> str:
        dest = unique_path(folder, filename)
        kind = info["kind"]
        if kind == "local":
            src = Path(info["path"])
            job.say(f"copying {src.name}")
            total = src.stat().st_size
            with open(src, "rb") as fi, open(dest, "wb") as fo:
                done = 0
                while chunk := fi.read(CHUNK * 8):
                    if job.cancel_requested:
                        fo.close()
                        dest.unlink(missing_ok=True)
                        raise Cancelled()
                    fo.write(chunk)
                    done += len(chunk)
                    job.progress(done, total)
            return str(dest)
        if kind in ("hf_file", "hf_repo"):
            from huggingface_hub import hf_hub_url
            url = hf_hub_url(info["repo"], info["filename"])
            headers = {"Authorization": f"Bearer {hf_token()}"} if hf_token() else {}
            return str(_stream_to_file(url, dest, job, headers=headers))
        if kind == "civitai":
            params = {"token": civitai_token()} if civitai_token() else None
            return str(_stream_to_file(info["url"], dest, job, params=params))
        if kind == "url":
            return str(_stream_to_file(info["url"], dest, job))
        raise RuntimeError(f"cannot download a {kind} source as a file")

    return run


def _repo_patterns(spec: ModelSpec) -> tuple[list[str] | None, list[str]]:
    allow = None
    ignore = ["*.bin", "*.ckpt", "*.onnx", "*.msgpack", "*.h5", "*openvino*", "*.gguf", "*.md"]
    if spec.family == "sdxl":
        if spec.variant == "fp16":
            ignore += ["*/diffusion_pytorch_model.safetensors", "*/model.safetensors"]
        else:
            ignore += ["*.fp16.*"]
    if spec.family == "qwen":
        ignore += ["text_encoder/*"]   # the 4-bit text encoder comes from diffusers/qwen-image-nf4 instead
    if spec.family == "flux":
        # the loader reads the diffusers folders; skip the duplicate single-file checkpoint and VAE (~24 GB)
        ignore += ["flux1-*.safetensors", "ae.safetensors"]
    if spec.family in ("video", "music"):
        import media
        allow = media.repo_allow_patterns(spec)
    return allow, ignore


def _cache_bytes(repo: str) -> int:
    d = models.hf_cache_dir(repo)
    total = 0
    if d.is_dir():
        for f in (d / "blobs").glob("*") if (d / "blobs").is_dir() else []:
            if f.is_file():
                total += f.stat().st_size
    return total


def download_repo(spec: ModelSpec) -> Callable[[Job], str]:
    """Job function that fetches a diffusers repo into the HF cache, with byte progress from the cache folder."""

    def run(job: Job) -> str:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.errors import GatedRepoError
        token = hf_token() or None
        allow, ignore = _repo_patterns(spec)
        try:
            mi = HfApi(token=token).model_info(spec.source, files_metadata=True)
        except GatedRepoError as e:
            raise RuntimeError(f"{spec.source} is gated: accept the terms on huggingface.co and save a token in Admin > Settings") from e
        from fnmatch import fnmatch
        wanted = [s for s in (mi.siblings or []) if not any(fnmatch(s.rfilename, pat) or fnmatch(Path(s.rfilename).name, pat) for pat in ignore)]
        if allow:
            wanted = [s for s in wanted if any(fnmatch(s.rfilename, pat) for pat in allow)]
        expected = sum(s.size or 0 for s in wanted)
        already = _cache_bytes(spec.source)
        job.progress(already, max(expected, already))
        job.say(f"fetching {spec.source}: {len(wanted)} files, {human_bytes(expected)}")

        stop = threading.Event()

        def watch():
            while not stop.wait(1.0):
                job.progress(_cache_bytes(spec.source), max(expected, already))

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        try:
            path = snapshot_download(spec.source, allow_patterns=allow, ignore_patterns=ignore, token=token, max_workers=4)
        finally:
            stop.set()
            t.join(timeout=2)
        extras = []
        if spec.family == "qwen":
            job.say("fetching the 4-bit text encoder and the Lightning LoRA")
            from huggingface_hub import hf_hub_download
            snapshot_download(models.ModelManager.QWEN_TEXT_ENCODER, allow_patterns=["text_encoder/*", "*.json"], token=token)
            hf_hub_download(*models.ModelManager.QWEN_LIGHTNING, token=token)
            extras.append("text encoder + Lightning LoRA")
        job.progress(job.total or expected, job.total or expected)
        return path

    return run


def delete_model_files(spec: ModelSpec) -> list[str]:
    """Remove a model's weights from disk: the local file, or its repo in the HF cache, and our converted caches."""
    removed = []
    if spec.local or Path(spec.source).is_file():
        p = Path(spec.source)
        if p.exists():
            p.unlink()
            removed.append(str(p))
    elif spec.source and "/" in spec.source:
        d = models.hf_cache_dir(spec.source)
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(str(d))
    cache = models.MODELS_DIR / "cache"
    if cache.is_dir():
        for sub in cache.glob(f"{spec.key}-*"):
            shutil.rmtree(sub, ignore_errors=True)
            removed.append(str(sub))
    return removed


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:] or ["RunDiffusion/Juggernaut-XL-v9", "nerijs/pixel-art-xl/pixel-art-xl.safetensors"]:
        print(arg)
        print(json.dumps(inspect_source(arg), indent=2, default=str))
