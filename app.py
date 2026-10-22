"""
Local web UI for AI image generation on your GPU (Stable Diffusion XL via diffusers + Gradio).

Run:
    .venv/Scripts/python.exe app.py
Then open http://127.0.0.1:7860 in your browser.

Everything is controlled from the prompt. Examples:
    "a cozy cabin in a snowy forest, 16:9, 4k"
    "portrait of a cyberpunk samurai, 3 images, seed 42"
    "desktop wallpaper of a mountain lake, photorealistic"
See prompt_parser.py for the full list of understood words.

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
from diffusers import StableDiffusionXLPipeline

import db
import gpu_info
from prompt_parser import parse
from upscaler import Upscaler

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_NEGATIVE = "blurry, low quality, deformed, watermark, text"

HELP = """
**Write everything in the prompt.** The app reads these hints:

| Want | Say |
|---|---|
| Aspect ratio | `16:9`, `9:16`, `4:3`, `3:2`, `1:1`, `21:9` |
| Exact size | `1920x1080`, `1024x768` (scaled to the GPU-friendly equivalent) |
| Orientation | `portrait`, `landscape`, `square`, `wide`, `tall`, `wallpaper`, `phone wallpaper`, `poster`, `banner` |
| Real 4K output | `4k` (3840 px long side), `8k` (7680 px), `2k` (2560 px), or a big size like `3840x2160`. Generated at native size, then upscaled with Real-ESRGAN. |
| Quality | `highly detailed`, `masterpiece` (more steps) or `quick`, `draft` (fewer steps) |
| Fidelity | `strict` (follow prompt closely) or `creative` (looser) |
| Style | `photorealistic`, `anime`, `cinematic`, `watercolor`, `oil painting`, `3d render`, `pixel art`, `sketch`, `comic`, `cyberpunk`, `low poly` |
| Several images | `3 images`, `4 variations` |
| Reproducible | `seed 42` |
| Manual override | `30 steps`, `cfg 8` |
"""

HISTORY_COLUMNS = ["id", "when", "prompt", "size", "steps", "seconds", "it/s", "peak VRAM GB", "temp C", "seed"]


def load_pipeline() -> StableDiffusionXLPipeline:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available. Install the CUDA build of PyTorch.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading {MODEL_ID} ...")
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    print(f"Model ready in {time.time() - t0:.1f}s")
    return pipe


PIPE = load_pipeline()
UPSCALER = Upscaler()
print(f"Upscaler ready: {UPSCALER.name}")
STATIC_GPU = gpu_info.static_info()


# ----------------------------------------------------------------------------- generate tab

def preview(prompt: str) -> str:
    """Live 'interpreted as' line shown under the prompt box."""
    if not prompt.strip():
        return ""
    s = parse(prompt)
    line = f"**Will generate:** {s.summary()}"
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
        f"**Negative:** {g('negative_prompt')}",
    ]
    if g("parser_notes"):
        lines.append(f"**Parser notes:** {g('parser_notes')}")
    return "\n".join(lines)


def generate(prompt: str, negative: str, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    s = parse(prompt)
    negative = negative.strip() or DEFAULT_NEGATIVE
    base_seed = s.seed if s.seed is not None else random.randint(0, 2**32 - 1)
    total_steps = s.steps * s.count
    images, reports = [], []

    for i in range(s.count):
        seed = base_seed + i
        gen = torch.Generator(device="cuda").manual_seed(seed)

        def on_step(pipe, step, timestep, kwargs, i=i):
            progress((i * s.steps + step + 1) / total_steps, desc=f"Image {i + 1}/{s.count}")
            return kwargs

        torch.cuda.reset_peak_memory_stats()
        before = gpu_info.snapshot()
        t_start = time.perf_counter()

        img = PIPE(
            prompt=s.prompt,
            negative_prompt=negative,
            num_inference_steps=s.steps,
            guidance_scale=s.guidance,
            width=s.width,
            height=s.height,
            generator=gen,
            callback_on_step_end=on_step,
        ).images[0]
        torch.cuda.synchronize()
        t_infer = time.perf_counter()
        after = gpu_info.snapshot()
        mem = gpu_info.torch_memory()

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = OUTPUT_DIR / f"{stamp}_seed{seed}.png"
        img.save(path)
        t_end = time.perf_counter()

        upscale_rec: dict = {}
        final_img = img
        if s.upscale_to:
            progress(1.0, desc=f"Upscaling image {i + 1}/{s.count} to {s.upscale_to} px")
            up_img, up_info = UPSCALER.upscale_to(img, s.upscale_to)
            if s.exact_size:
                tw, th = s.exact_size
                left, top = max(0, (up_img.width - tw) // 2), max(0, (up_img.height - th) // 2)
                up_img = up_img.crop((left, top, left + min(tw, up_img.width), top + min(th, up_img.height)))
            up_path = OUTPUT_DIR / f"{stamp}_seed{seed}_{s.upscale_to}px.png"
            up_img.save(up_path)
            final_img = up_img
            upscale_rec = {
                "upscale_target": s.upscale_to,
                "upscaled_width": up_img.width,
                "upscaled_height": up_img.height,
                "upscaled_path": str(up_path),
                "upscaled_size_kb": round(up_path.stat().st_size / 1024, 1),
                "upscale_seconds": up_info["seconds"],
                "upscale_passes": up_info["passes"],
                "upscaler": up_info["upscaler"],
            }
            mem = gpu_info.torch_memory()  # peak now includes the upscaler

        rec = {
            **upscale_rec,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image_path": str(path),
            "file_size_kb": round(path.stat().st_size / 1024, 1),
            "user_prompt": prompt.strip(),
            "final_prompt": s.prompt,
            "negative_prompt": negative,
            "parser_notes": " · ".join(s.notes),
            "model_id": MODEL_ID,
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
        rec["id"] = db.insert(rec)
        print(f"#{rec['id']} saved {path.name}  {s.width}x{s.height}  {rec['inference_seconds']}s  "
              f"peak {rec['vram_peak_gb']} GB  {rec['temp_after_c']} C"
              + (f"  -> {final_img.width}x{final_img.height} in {upscale_rec['upscale_seconds']}s" if upscale_rec else ""))
        images.append(final_img)
        reports.append(details_markdown(rec))

    return images, "\n\n---\n\n".join(reports)


# ----------------------------------------------------------------------------- history tab

def history_table(search: str = "", limit: int = 50) -> pd.DataFrame:
    rows = db.fetch(limit=int(limit), search=search)
    data = [
        [r["id"], r["created_at"][:16].replace("T", " "), r["user_prompt"][:70],
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

with gr.Blocks(title="AI Image Generator") as demo:
    gr.Markdown(
        f"# AI Image Generator\n{STATIC_GPU['gpu_name']} · {STATIC_GPU['vram_total_gb']} GB VRAM · "
        f"CUDA {STATIC_GPU['cuda_version']} · torch {STATIC_GPU['torch_version']} · upscaler {UPSCALER.name}"
    )

    with gr.Tabs():
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt = gr.Textbox(
                        label="Prompt", lines=4,
                        placeholder="a cozy cabin in a snowy forest at dusk, cinematic, 16:9, 4k",
                    )
                    interpreted = gr.Markdown("")
                    negative = gr.Textbox(label="Negative prompt (what to avoid)", lines=2, value=DEFAULT_NEGATIVE)
                    run = gr.Button("Generate", variant="primary")
                    with gr.Accordion("What can I write in the prompt?", open=False):
                        gr.Markdown(HELP)

                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="Results", columns=2, height=560, object_fit="contain")
                    with gr.Accordion("Generation details (GPU, timing, settings)", open=True):
                        details = gr.Markdown("")

            prompt.change(preview, prompt, interpreted, show_progress="hidden")
            run.click(generate, [prompt, negative], [gallery, details])
            prompt.submit(generate, [prompt, negative], [gallery, details])

            gr.Examples(
                examples=[
                    ["a cozy cabin in a snowy forest at dusk, cinematic, 16:9, 4k"],
                    ["portrait of a cyberpunk samurai in neon rain, 2 images"],
                    ["desktop wallpaper of a misty mountain lake at sunrise, photorealistic"],
                    ["a corgi astronaut floating in space, 3d render, square, seed 7"],
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
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
