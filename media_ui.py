"""
Video and Music tabs of the web app (generation handlers + layout). app.py calls build_video_tab() and
build_music_tab() inside its gr.Tabs; the handlers share the image ModelManager's lock so only one job uses
the GPU at a time, and media.prepare_for() unloads whatever would not fit next to the model being used.

Video:  prompt -> LTX-Video / Wan 2.1 -> H.264 MP4 in outputs/ (+ optional start image, Real-ESRGAN upscale of every
        frame to Full HD / 2K / 4K, soundtrack from a generated music track)
Music:  prompt -> MusicGen / Stable Audio -> 16-bit WAV in outputs/
Both are recorded in generations.db (kind = video | music) with the same GPU telemetry as images.
"""

from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import torch
from PIL import Image, ImageOps

import applog
import db
import gpu_info
import media
import models
import registry
import video_io
from media import VIDEO_SIZES, VIDEO_UPSCALE_CHOICES
from ui_common import loaded_line, media_choice_update, media_choices

CTX = SimpleNamespace(manager=None, media=None, upscaler=None, static_gpu=None, output_dir=None)

VIDEO_CHIPS = ["slow camera pan", "tracking shot", "drone shot", "close-up", "time-lapse", "slow motion",
               "golden hour", "cinematic lighting", "rain", "fog"]
MUSIC_CHIPS = ["lo-fi hip hop", "cinematic orchestral", "synthwave", "acoustic folk", "jazz trio", "techno",
               "ambient pads", "epic trailer", "upbeat", "melancholic", "relaxing", "120 bpm"]

VIDEO_HELP = """
**Describe the scene, the motion and the camera** in one or two sentences, as you would brief a camera operator:
*"A red fox walks through an autumn forest, leaves falling, soft sunlight; the camera tracks alongside at eye level."*

- **Start image** (LTX-Video): drop in a picture, or a Studio result, and the model animates it. It is cropped to the chosen size.
- **Length**: LTX runs at 24 fps and Wan at 16 fps; the frame count is rounded to what the model accepts.
- **Upscale video**: every frame goes through Real-ESRGAN (two passes for 4K) and is re-encoded; 4K takes about a minute for a 4 s clip.
- **Add soundtrack**: pick a track made in the Music tab; it is cut to the clip length with a short fade-out.
- Video models need the whole card, so the image model is unloaded while a video model is in use (and comes back on the next image).
"""

MUSIC_HELP = """
**Describe genre, mood, instruments and tempo**: *"upbeat synthwave with punchy drums and a warm bass line, 120 bpm"*,
*"calm solo piano, slow, melancholic, reverb"*, *"epic orchestral trailer music with choir and big percussion"*.

- MusicGen writes **instrumental** music only (no vocals) at 32 kHz stereo, up to 30 s per track.
- **Guidance** pushes the result towards the text (3 is the sweet spot); **temperature** adds variety (1.0 = normal).
- Same **seed** + same prompt = same track. Tracks land in `outputs/*.wav` and in History, and can be laid under a video in the Video tab.
"""


# ----------------------------------------------------------------------------- shared helpers

def _spec(key: str | None, kind: str) -> models.ModelSpec:
    cands = models.all_models(kind=kind)
    if not cands:
        raise gr.Error(f"No {kind} model is enabled. Add or enable one in Admin > Models.")
    if key:
        try:
            spec = models.resolve(key)
        except KeyError:
            raise gr.Error(f"Unknown {kind} model '{key}'.")
        if models.kind_of(spec.family) != kind:
            raise gr.Error(f"{spec.name} is not a {kind} model.")
        return spec
    return cands[0]


def _seed(value) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = -1
    return v if v >= 0 else random.randint(0, 2**32 - 1)


def _telemetry(before: dict, after: dict, mem: dict) -> dict:
    sg = CTX.static_gpu
    return {
        "gpu_name": sg["gpu_name"], "vram_total_gb": sg["vram_total_gb"], "vram_peak_gb": mem["vram_peak_gb"],
        "vram_reserved_gb": mem["vram_reserved_gb"], "compute_capability": sg["compute_capability"],
        "driver_version": after.get("driver_version"), "cuda_version": sg["cuda_version"], "torch_version": sg["torch_version"],
        "diffusers_version": sg["diffusers_version"], "temp_before_c": before.get("temperature_c"), "temp_after_c": after.get("temperature_c"),
        "power_before_w": before.get("power_w"), "power_after_w": after.get("power_w"), "power_limit_w": after.get("power_limit_w"),
        "gpu_util_after_pct": after.get("gpu_util_pct"), "sm_clock_mhz": after.get("sm_clock_mhz"), "mem_clock_mhz": after.get("mem_clock_mhz"),
        "fan_after_pct": after.get("fan_pct"), "pstate": after.get("pstate"),
    }


def _load(spec: models.ModelSpec, progress) -> list[str]:
    """Load a video/music model under the GPU lock (caller holds it); returns notes about unloaded models."""
    med = CTX.media
    if med.spec is None or med.spec.key != spec.key:
        progress(0, desc=f"Loading {spec.name}" + ("" if models.is_downloaded(spec) else f" (downloading ~{spec.download_gb:g} GB first)"))
    notes = media.prepare_for(spec, CTX.manager, med, log=applog.log)
    try:
        med.load(spec, log=applog.log)
    except (RuntimeError, OSError, KeyError, ValueError) as e:
        raise gr.Error(f"Could not load {spec.name}: {e}")
    return notes


def media_details_markdown(rec: dict) -> str:
    g = rec.get
    kind = g("kind") or "image"
    lines = [f"### {kind.capitalize()} #{g('id', '?')}  ·  {g('created_at', '')[:19]}", "", "**Timing**",
             f"- Generation: **{g('inference_seconds') or 0:.1f} s**" + (f"  ({g('steps_per_second'):.2f} steps/s)" if g("steps_per_second") else ""),
             f"- Encode and save: {g('save_seconds') or 0:.1f} s  ·  Total: {g('total_seconds') or 0:.1f} s", "",
             f"**{kind.capitalize()}**", f"- Model: **{models.display_name(g('model_id') or '')}**"]
    if kind == "video":
        lines.append(f"- {g('width')}x{g('height')}, {g('frames')} frames at {g('fps'):g} fps = **{g('duration_s'):.1f} s**, "
                     f"{g('steps')} steps, guidance {g('guidance'):g}, seed {g('seed')}")
    else:
        lines.append(f"- **{g('duration_s'):.1f} s** at {(g('sample_rate') or 0) // 1000} kHz stereo, guidance {g('guidance'):g}, seed {g('seed')}"
                     + (f", {g('steps')} steps" if g("steps") else ""))
    lines.append(f"- File: `{Path(g('image_path', '')).name}` ({(g('file_size_kb') or 0) / 1024:.1f} MB)")
    if g("soundtrack_path"):
        lines.append(f"- With soundtrack: `{Path(g('soundtrack_path')).name}`")
    if g("upscaled_path"):
        lines += ["", "**Upscale**",
                  f"- {g('width')}x{g('height')} → **{g('upscaled_width')}x{g('upscaled_height')}** in {g('upscale_seconds') or 0:.1f} s "
                  f"({g('upscale_passes')} pass{'es' if g('upscale_passes') != 1 else ''} per frame)",
                  f"- {g('upscaler')}  ·  file `{Path(g('upscaled_path')).name}` ({(g('upscaled_size_kb') or 0) / 1024:.1f} MB)"]
    lines += ["", "**GPU**",
              f"- {g('gpu_name')}  ·  peak VRAM **{g('vram_peak_gb')} GB**  ·  reserved {g('vram_reserved_gb')} GB",
              f"- Temperature {g('temp_before_c')} °C → **{g('temp_after_c')} °C**  ·  power {g('power_before_w')} W → **{g('power_after_w')} W**",
              f"- Driver {g('driver_version')}  ·  CUDA {g('cuda_version')}  ·  torch {g('torch_version')}  ·  diffusers {g('diffusers_version')}",
              "", "**Prompt**", f"> {g('final_prompt') or g('user_prompt')}", ""]
    if g("negative_prompt"):
        lines.append(f"**Negative:** {g('negative_prompt')}")
    if g("parser_notes"):
        lines.append(f"**Notes:** {g('parser_notes')}")
    return "\n".join(lines)


def toggle_token(prompt: str, token: str) -> str:
    parts = [p.strip() for p in (prompt or "").split(",") if p.strip()]
    if token.lower() in [p.lower() for p in parts]:
        return ", ".join(p for p in parts if p.lower() != token.lower())
    return ", ".join(parts + [token])


def _loaded():
    return loaded_line(CTX.manager, CTX.upscaler.name, CTX.media)


# ----------------------------------------------------------------------------- video

def video_defaults(key: str | None):
    """Model-specific defaults when the Video model box changes: steps, guidance, negative, max length, image box, note."""
    try:
        spec = _spec(key, "video")
    except gr.Error:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "No video model available."
    p = media.profile_for(spec)
    disk = "" if models.is_downloaded(spec) else f" · **not downloaded yet** (~{spec.download_gb:g} GB, fetched on first use or from Admin)"
    note = (f"**{p['label']}** · {p['fps']} fps · up to {p['max_seconds']} s · "
            f"{'image-to-video supported' if p['image_to_video'] else 'text-to-video only'} · {p['note']}{disk}")
    return (gr.update(value=p["steps"]), gr.update(value=p["guidance"]), gr.update(value=p["negative"]),
            gr.update(maximum=p["max_seconds"]), gr.update(visible=p["image_to_video"]), note)


def generate_video(prompt, negative, model_key, image, size, seconds, steps, guidance, seed, progress=gr.Progress()):
    if not (prompt or "").strip():
        raise gr.Error("Describe the clip first.")
    spec = _spec(model_key, "video")
    prof = media.profile_for(spec)
    w, h = media.parse_size(size or "768x512")
    frames = media.frames_for(float(seconds or 4), prof)
    steps = int(steps or prof["steps"])
    guidance = float(guidance if guidance is not None else prof["guidance"])
    seed = _seed(seed)
    start = None
    if image:
        if not prof["image_to_video"]:
            raise gr.Error(f"{spec.name} is text-to-video only; use LTX-Video to animate an image.")
        start = ImageOps.fit(Image.open(image).convert("RGB"), (w, h), Image.LANCZOS)
    out_dir = CTX.output_dir
    mgr, med = CTX.manager, CTX.media
    with mgr.lock:
        notes = _load(spec, progress)
        torch.cuda.reset_peak_memory_stats()
        before = gpu_info.snapshot()
        t0 = time.perf_counter()
        try:
            frame_list = med.generate_video(
                prompt.strip(), (negative or "").strip(), w, h, frames, steps, guidance, seed, image=start,
                on_step=lambda s, n: progress(s / n, desc=f"{spec.name} · step {s}/{n} · {w}x{h} · {frames} frames"))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise gr.Error(f"Out of VRAM at {w}x{h} x {frames} frames. Try a smaller size or a shorter clip.")
        except RuntimeError as e:
            raise gr.Error(str(e))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        after = gpu_info.snapshot()
        mem = gpu_info.torch_memory()
        progress(1.0, desc="Encoding MP4")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"{stamp}_seed{seed}.mp4"
        info = video_io.write_mp4(frame_list, path, prof["fps"])
        t2 = time.perf_counter()
        if registry.setting("auto_release_vram", True):
            torch.cuda.empty_cache()
    rec = {
        "created_at": datetime.now().isoformat(timespec="seconds"), "kind": "video",
        "image_path": str(path), "file_size_kb": round(path.stat().st_size / 1024, 1),
        "user_prompt": prompt.strip(), "final_prompt": prompt.strip(), "negative_prompt": (negative or "").strip(),
        "parser_notes": " · ".join(notes + ([f"start image {Path(image).name}"] if image else [])),
        "model_id": spec.source, "width": info["width"], "height": info["height"], "steps": steps, "guidance": guidance,
        "seed": seed, "style": prof["arch"], "batch_index": 1, "batch_size": 1,
        "frames": info["frames"], "fps": prof["fps"], "duration_s": round(info["frames"] / prof["fps"], 2),
        "inference_seconds": round(t1 - t0, 3), "save_seconds": round(t2 - t1, 3), "total_seconds": round(t2 - t0, 3),
        "steps_per_second": round(steps / (t1 - t0), 3),
        **_telemetry(before, after, mem),
    }
    rec["id"] = db.insert(rec)
    applog.log(f"#{rec['id']} {spec.key} video {path.name}  {w}x{h} x {frames} frames  {rec['inference_seconds']}s  peak {rec['vram_peak_gb']} GB")
    state = {"id": rec["id"], "path": str(path), "current": str(path), "width": info["width"], "height": info["height"],
             "fps": prof["fps"], "seed": seed, "frames": info["frames"], "soundtrack": None}
    return (str(path), media_details_markdown(rec), state, gr.update(visible=True), gr.update(visible=False), _loaded(),
            f"Clip #{rec['id']} · {w}x{h} · {rec['duration_s']:.1f} s · seed {seed}. Upscale it or add a soundtrack below.")


VIDEO_UPSCALER_FILE = models.MODELS_DIR / "realesr-general-x4v3.pth"
VIDEO_UPSCALER_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
_video_upscaler = None


def video_upscaler():
    """Real-ESRGAN's compact 'general x4 v3' network (1.2M parameters, 4.7 MB): the model the Real-ESRGAN project
    ships for video, about 20x faster per frame than the x4plus image network. Fetched on first use; falls back to
    the image upscaler if the download fails."""
    global _video_upscaler
    if _video_upscaler is not None:
        return _video_upscaler
    if not VIDEO_UPSCALER_FILE.exists():
        try:
            import requests
            applog.log(f"downloading the video upscaler weights ({VIDEO_UPSCALER_URL}) ...")
            with requests.get(VIDEO_UPSCALER_URL, stream=True, timeout=60) as r:
                r.raise_for_status()
                tmp = VIDEO_UPSCALER_FILE.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            for attempt in range(10):          # Windows: a virus scanner may hold the file for a moment after close
                try:
                    tmp.replace(VIDEO_UPSCALER_FILE)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.5)
        except Exception as e:
            applog.log(f"video upscaler download failed ({e}); using {CTX.upscaler.name} for video frames")
            _video_upscaler = CTX.upscaler
            return _video_upscaler
    try:
        from upscaler import Upscaler
        _video_upscaler = Upscaler(model_path=VIDEO_UPSCALER_FILE)
        applog.log(f"video upscaler ready: {_video_upscaler.name}")
    except Exception as e:
        applog.log(f"could not load the video upscaler ({e}); using {CTX.upscaler.name}")
        _video_upscaler = CTX.upscaler
    return _video_upscaler


def upscale_video(state, target_px, progress=gr.Progress()):
    if not state:
        raise gr.Error("Generate a video first.")
    target_px = int(target_px or 3840)
    src = Path(state["current"])
    if not src.exists():
        raise gr.Error("The video file is gone.")
    info, frames = video_io.read_frames(src)
    total = state.get("frames") or max(1, round(info["duration"] * info["fps"]))
    up = video_upscaler()
    passes = [0]
    done = [0]

    def gen():
        for fr in frames:
            # one network pass (x4), then a Lanczos resample to the exact target: 768 px -> 3072 -> 3840
            img, i = up.upscale_to(fr, target_px, max_passes=1)
            passes[0] = i["passes"]
            done[0] += 1
            progress(done[0] / total, desc=f"Upscaling frame {done[0]}/{total} to {target_px} px")
            yield img

    dest = src.with_name(f"{src.stem}_up{target_px}.mp4")
    with CTX.manager.lock:
        t0 = time.perf_counter()
        out = video_io.write_mp4(gen(), dest, info["fps"] or state["fps"], crf=18, audio=src if state.get("soundtrack") else None)
        secs = round(time.perf_counter() - t0, 2)
        torch.cuda.empty_cache()
    fields = {"upscale_target": target_px, "upscaled_width": out["width"], "upscaled_height": out["height"], "upscaled_path": str(dest),
              "upscaled_size_kb": round(dest.stat().st_size / 1024, 1), "upscale_seconds": secs, "upscale_passes": passes[0],
              "upscaler": up.name}
    db.update(state["id"], fields)
    state = dict(state, current=str(dest), upscaled=str(dest))
    applog.log(f"#{state['id']} video upscaled -> {out['width']}x{out['height']} ({out['frames']} frames) in {secs}s")
    rec = db.get(state["id"]) or {}
    return str(dest), media_details_markdown(rec), state, f"Upscaled to {out['width']}x{out['height']} in {secs:.0f} s · `{dest.name}`"


def music_track_choices() -> list[tuple[str, str]]:
    rows = db.fetch_filtered(limit=40, kind="music")
    return [(f"#{r['id']} · {r['user_prompt'][:48]} · {r.get('duration_s') or 0:.0f} s", r["image_path"]) for r in rows if Path(r["image_path"]).exists()]


def add_soundtrack(state, track):
    if not state:
        raise gr.Error("Generate a video first.")
    if not track or not Path(track).exists():
        raise gr.Error("Pick a track made in the Music tab (generate one there first).")
    src = Path(state["current"])
    dest = src.with_name(f"{src.stem}_music.mp4")
    t0 = time.perf_counter()
    video_io.mux_audio(src, track, dest)
    db.update(state["id"], {"soundtrack_path": str(dest)})
    state = dict(state, current=str(dest), soundtrack=track)
    applog.log(f"#{state['id']} soundtrack {Path(track).name} -> {dest.name} in {time.perf_counter() - t0:.1f}s")
    rec = db.get(state["id"]) or {}
    return str(dest), media_details_markdown(rec), state, f"Soundtrack added · `{dest.name}`"


def clear_video():
    return "", None, "", None, gr.update(visible=False), gr.update(visible=True), ""


# ----------------------------------------------------------------------------- music

def music_defaults(key: str | None):
    try:
        spec = _spec(key, "music")
    except gr.Error:
        return gr.update(), gr.update(), gr.update(), "No music model available."
    p = media.profile_for(spec)
    disk = "" if models.is_downloaded(spec) else f" · **not downloaded yet** (~{spec.download_gb:g} GB)"
    note = f"**{p['label']}** · {p.get('sample_rate', 0) // 1000} kHz stereo · {p['min_seconds']}-{p['max_seconds']} s per track · {p['note']}{disk}"
    return (gr.update(value=p["guidance"]), gr.update(maximum=p["max_seconds"]), gr.update(value=p["negative"], visible=p["arch"] != "musicgen"), note)


def generate_music(prompt, negative, model_key, seconds, guidance, temperature, seed, progress=gr.Progress()):
    if not (prompt or "").strip():
        raise gr.Error("Describe the music first.")
    spec = _spec(model_key, "music")
    prof = media.profile_for(spec)
    seconds = float(min(max(float(seconds or 15), prof["min_seconds"]), prof["max_seconds"]))
    guidance = float(guidance if guidance is not None else prof["guidance"])
    seed = _seed(seed)
    mgr, med = CTX.manager, CTX.media
    with mgr.lock:
        notes = _load(spec, progress)
        torch.cuda.reset_peak_memory_stats()
        before = gpu_info.snapshot()
        t0 = time.perf_counter()
        try:
            wav, sr = med.generate_music(prompt.strip(), (negative or "").strip(), seconds, guidance, float(temperature or 1.0), seed,
                                         on_progress=lambda p: progress(p, desc=f"{spec.name} · {p * seconds:.0f}/{seconds:.0f} s"))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise gr.Error("Out of VRAM. Unload the image model in Admin > Models and try again.")
        except RuntimeError as e:
            raise gr.Error(str(e))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        after = gpu_info.snapshot()
        mem = gpu_info.torch_memory()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = CTX.output_dir / f"{stamp}_seed{seed}.wav"
        info = video_io.write_wav(wav, sr, path)
        t2 = time.perf_counter()
    rec = {
        "created_at": datetime.now().isoformat(timespec="seconds"), "kind": "music",
        "image_path": str(path), "file_size_kb": round(path.stat().st_size / 1024, 1),
        "user_prompt": prompt.strip(), "final_prompt": prompt.strip(), "negative_prompt": (negative or "").strip() if prof["arch"] != "musicgen" else "",
        "parser_notes": " · ".join(notes + [f"temperature {float(temperature or 1.0):g}"]),
        "model_id": spec.source, "width": None, "height": None, "steps": prof["steps"] or None, "guidance": guidance,
        "seed": seed, "style": prof["arch"], "batch_index": 1, "batch_size": 1,
        "duration_s": info["seconds"], "sample_rate": info["sample_rate"],
        "inference_seconds": round(t1 - t0, 3), "save_seconds": round(t2 - t1, 3), "total_seconds": round(t2 - t0, 3),
        "steps_per_second": None,
        **_telemetry(before, after, mem),
    }
    rec["id"] = db.insert(rec)
    applog.log(f"#{rec['id']} {spec.key} music {path.name}  {info['seconds']} s  {rec['inference_seconds']}s  peak {rec['vram_peak_gb']} GB")
    return (str(path), media_details_markdown(rec), {"id": rec["id"], "path": str(path), "seed": seed}, gr.update(visible=False), _loaded(),
            f"Track #{rec['id']} · {info['seconds']:.0f} s · seed {seed} · ready for the Video tab's *Add soundtrack*.")


def clear_music():
    return "", None, "", None, gr.update(visible=True), ""


# ----------------------------------------------------------------------------- layout

VIDEO_EMPTY = """
<div class="fz-empty"><b>Your video appears here</b>
Describe the scene, the motion and the camera, pick a size and a length, then press <b>Generate video</b>.<br>
A 4 s clip at 768x512 takes about a minute with LTX-Video on an RTX 3090; 4K upscaling adds about a minute.</div>"""

MUSIC_EMPTY = """
<div class="fz-empty"><b>Your track appears here</b>
Describe genre, mood, instruments and tempo, choose a length, then press <b>Generate music</b>.<br>
MusicGen writes a 15 s stereo track in roughly 30-60 s.</div>"""


def build_video_tab(loaded_md):
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=400):
            gr.HTML('<div class="fz-chips-label">Motion and camera hints · click to add or remove</div>')
            with gr.Row(elem_classes=["fz-chips"]):
                chips = [gr.Button(t, size="sm", variant="secondary") for t in VIDEO_CHIPS[:5]]
            with gr.Row(elem_classes=["fz-chips"]):
                chips += [gr.Button(t, size="sm", variant="secondary") for t in VIDEO_CHIPS[5:]]
            v_prompt = gr.Textbox(label="Describe the clip", lines=4,
                                  placeholder="A red fox walks through an autumn forest, leaves falling, soft sunlight; the camera tracks alongside")
            v_model = gr.Dropdown(media_choices("video"), value=(media_choices("video") or [(None, None)])[0][1], label="Video model",
                                  allow_custom_value=True, info="Add more in Admin > Add (any LTX or Wan repo in diffusers layout).")
            v_note = gr.Markdown(video_defaults(v_model.value)[5], elem_classes=["fz-interp"])
            with gr.Row():
                v_size = gr.Dropdown(VIDEO_SIZES, value="768x512", label="Size", scale=3)
                v_seconds = gr.Slider(2, 10, value=4, step=1, label="Length (s)", scale=2)
            with gr.Row():
                v_steps = gr.Slider(8, 60, value=40, step=1, label="Steps")
                v_guidance = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Guidance")
                v_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
            v_image = gr.Image(label="Start image (optional: image-to-video)", type="filepath", height=200)
            v_negative = gr.Textbox(label="Negative prompt", lines=2, value=media.LTX_NEGATIVE)
            with gr.Row():
                v_run = gr.Button("Generate video", variant="primary", size="lg", scale=3)
                v_clear = gr.Button("Clear", variant="secondary", size="lg", scale=1, min_width=90)
            with gr.Accordion("How do I get good clips?", open=False):
                gr.Markdown(VIDEO_HELP)
        with gr.Column(scale=3):
            v_empty = gr.HTML(VIDEO_EMPTY)
            v_video = gr.Video(label="Result", autoplay=True, height=520)
            with gr.Row(visible=False, elem_classes=["fz-upscale"]) as v_bar:
                v_up_target = gr.Radio(VIDEO_UPSCALE_CHOICES, value=3840, label="Upscale every frame with Real-ESRGAN", scale=3)
                v_up_btn = gr.Button("Upscale video", variant="primary", scale=1, min_width=140)
                v_track = gr.Dropdown(music_track_choices(), label="Soundtrack (from the Music tab)", scale=3, allow_custom_value=True)
                v_track_btn = gr.Button("Add soundtrack", variant="secondary", scale=1, min_width=140)
            v_hint = gr.Markdown("", elem_classes=["fz-hint"])
            with gr.Accordion("Clip details (GPU, timing, settings)", open=True):
                v_details = gr.Markdown("")
    v_state = gr.State(None)

    v_model.change(video_defaults, v_model, [v_steps, v_guidance, v_negative, v_seconds, v_image, v_note], show_progress="hidden")
    v_in = [v_prompt, v_negative, v_model, v_image, v_size, v_seconds, v_steps, v_guidance, v_seed]
    v_out = [v_video, v_details, v_state, v_bar, v_empty, loaded_md, v_hint]
    v_run.click(generate_video, v_in, v_out, concurrency_id="gpu")
    v_prompt.submit(generate_video, v_in, v_out, concurrency_id="gpu")
    v_up_btn.click(upscale_video, [v_state, v_up_target], [v_video, v_details, v_state, v_hint], concurrency_id="gpu")
    v_track_btn.click(add_soundtrack, [v_state, v_track], [v_video, v_details, v_state, v_hint])
    v_clear.click(clear_video, None, [v_prompt, v_video, v_details, v_state, v_bar, v_empty, v_hint])
    for b in chips:
        b.click(toggle_token, [v_prompt, gr.State(b.value)], v_prompt, show_progress="hidden")
    gr.Examples(
        examples=[
            ["A red fox walks through an autumn forest, leaves falling, soft sunlight; the camera tracks alongside at eye level"],
            ["Aerial drone shot over a turquoise coastline with white cliffs at golden hour, slow forward motion"],
            ["Close-up of a steaming cup of coffee on a wooden table, rain on the window behind, shallow depth of field, slow camera pan"],
            ["A neon-lit street in Tokyo at night, rain, reflections, people with umbrellas, cinematic tracking shot"],
            ["Time-lapse of clouds rolling over snowy mountain peaks, dramatic light"],
        ],
        inputs=[v_prompt],
    )
    return SimpleNamespace(prompt=v_prompt, model=v_model, track=v_track, image=v_image)


def build_music_tab(loaded_md):
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=400):
            gr.HTML('<div class="fz-chips-label">Genre and mood hints · click to add or remove</div>')
            with gr.Row(elem_classes=["fz-chips"]):
                chips = [gr.Button(t, size="sm", variant="secondary") for t in MUSIC_CHIPS[:6]]
            with gr.Row(elem_classes=["fz-chips"]):
                chips += [gr.Button(t, size="sm", variant="secondary") for t in MUSIC_CHIPS[6:]]
            m_prompt = gr.Textbox(label="Describe the music", lines=3,
                                  placeholder="upbeat synthwave with punchy drums and a warm bass line, 120 bpm")
            m_model = gr.Dropdown(media_choices("music"), value=(media_choices("music") or [(None, None)])[0][1], label="Music model",
                                  allow_custom_value=True, info="MusicGen is instrumental only. Add more models in Admin > Add.")
            m_note = gr.Markdown(music_defaults(m_model.value)[3], elem_classes=["fz-interp"])
            with gr.Row():
                m_seconds = gr.Slider(3, 30, value=15, step=1, label="Length (s)", scale=2)
                m_guidance = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Guidance", scale=1)
                m_temperature = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Temperature", scale=1)
                m_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0, scale=1)
            m_negative = gr.Textbox(label="Negative prompt (Stable Audio only)", lines=1, value="", visible=False)
            with gr.Row():
                m_run = gr.Button("Generate music", variant="primary", size="lg", scale=3)
                m_clear = gr.Button("Clear", variant="secondary", size="lg", scale=1, min_width=90)
            with gr.Accordion("How do I get good tracks?", open=False):
                gr.Markdown(MUSIC_HELP)
        with gr.Column(scale=3):
            m_empty = gr.HTML(MUSIC_EMPTY)
            m_audio = gr.Audio(label="Result", type="filepath", autoplay=True)
            m_hint = gr.Markdown("", elem_classes=["fz-hint"])
            with gr.Accordion("Track details (GPU, timing, settings)", open=True):
                m_details = gr.Markdown("")
    m_state = gr.State(None)

    m_model.change(music_defaults, m_model, [m_guidance, m_seconds, m_negative, m_note], show_progress="hidden")
    m_in = [m_prompt, m_negative, m_model, m_seconds, m_guidance, m_temperature, m_seed]
    m_out = [m_audio, m_details, m_state, m_empty, loaded_md, m_hint]
    m_run.click(generate_music, m_in, m_out, concurrency_id="gpu")
    m_prompt.submit(generate_music, m_in, m_out, concurrency_id="gpu")
    m_clear.click(clear_music, None, [m_prompt, m_audio, m_details, m_state, m_empty, m_hint])
    for b in chips:
        b.click(toggle_token, [m_prompt, gr.State(b.value)], m_prompt, show_progress="hidden")
    gr.Examples(
        examples=[
            ["upbeat synthwave with punchy drums and a warm bass line, 120 bpm"],
            ["calm solo piano, slow and melancholic, soft reverb"],
            ["epic orchestral trailer music with choir and big percussion"],
            ["lo-fi hip hop beat with vinyl crackle, mellow electric piano, relaxing"],
            ["energetic funk with slap bass, brass stabs and a tight drum groove"],
        ],
        inputs=[m_prompt],
    )
    return SimpleNamespace(prompt=m_prompt, model=m_model)


def refresh_video_choices(model_value, track_value):
    return media_choice_update("video", model_value), gr.update(choices=music_track_choices(), value=track_value)


def refresh_music_choices(model_value):
    return media_choice_update("music", model_value)
