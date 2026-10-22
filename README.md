# Frazile AI - AI Image Generation (local GPU)

Runs Stable Diffusion XL on your NVIDIA GPU using Hugging Face `diffusers`.

**Project owner:** Frazile AI  
**Author:** Frazile AI

## Setup (once)

```powershell
cd C:\Users\frazi\Documents\apps\ai-image-gen
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Generate

```powershell
.\.venv\Scripts\Activate.ps1
python generate.py "a cozy cabin in a snowy forest at dusk, cinematic lighting"
python generate.py "a red sports car" --steps 40 --seed 42 --count 4
python generate.py --help
```

Images are saved to `outputs\`. The first run downloads the model (~7 GB) to
`%USERPROFILE%\.cache\huggingface`.

## Other models

Pass `--model` with any SDXL-compatible Hugging Face id, e.g.
`--model stabilityai/sdxl-turbo --steps 4 --guidance 0`.

## Web UI (recommended)

```powershell
.\.venv\Scripts\python.exe app.py
```

Opens http://127.0.0.1:7860. There are no sliders or dropdowns: write everything in the prompt.

| Want | Say in the prompt |
|---|---|
| Aspect ratio | `16:9`, `9:16`, `4:3`, `3:2`, `1:1`, `21:9` |
| Exact size | `1920x1080`, `1024x768` (scaled to the nearest GPU-friendly size) |
| Orientation | `portrait`, `landscape`, `square`, `wide`, `tall`, `wallpaper`, `phone wallpaper`, `poster`, `banner` |
| Real 4K output | `4k` (3840 px long side), `8k` (7680 px), `2k` (2560 px), or an exact size like `3840x2160`. The image is generated at native size, then upscaled with Real-ESRGAN. |
| Quality | `highly detailed`, `masterpiece` for more steps; `quick`, `draft` for fewer |
| Fidelity | `strict` follows the prompt closely; `creative` is looser |
| Style | `photorealistic`, `anime`, `cinematic`, `watercolor`, `oil painting`, `3d render`, `pixel art`, `sketch`, `comic`, `cyberpunk`, `low poly` |
| Several images | `3 images`, `4 variations` |
| Reproducible | `seed 42` |
| Manual override | `30 steps`, `cfg 8` |

Example: `a cozy cabin in a snowy forest at dusk, cinematic, 16:9, 4k, seed 42`

A line under the prompt box shows how the prompt was interpreted before you generate.
Every result is also saved to `outputs\`.

## Generation details and history

After every run the Generate tab shows a report under the gallery: inference time and
steps per second, image settings, peak VRAM, GPU temperature and power before and after,
clock speeds, fan, driver and library versions, and the exact prompt sent to the model.

Every image is also recorded in `generations.db` (SQLite). Browse it two ways:

- **History tab** in the app: search prompts, see totals and averages, click a row to view
  the image with its full details.
- **Command line:**

```powershell
.\.venv\Scripts\python.exe db.py          # last 20 generations
.\.venv\Scripts\python.exe db.py 12       # every field for generation #12
.\.venv\Scripts\python.exe db.py stats    # totals and averages
```

The database is a plain SQLite file, so DB Browser for SQLite or any SQL client can open it.

## Why 4K is a two-stage process

Stable Diffusion XL was trained at about 1 megapixel. Asking it for 3840x2160 directly produces
duplicated subjects and needs roughly 16x the memory. So the app generates at the model's native
size and then runs Real-ESRGAN (x4 per pass, tiled so it fits in VRAM) up to the requested size.
A 16:9 image goes 1344x768 -> 3840x2194 in about 5 to 10 seconds on an RTX 3090.

Weights live in `models/RealESRGAN_x4plus.pth`. If the file is missing, download it from
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

Both the native and the upscaled image are saved to `outputs\` and recorded in the database
(columns `upscaled_width`, `upscaled_height`, `upscale_seconds`, `upscaled_path`).

Upscale any existing image from the command line:

```powershell
.\.venv\Scripts\python.exe upscaler.py outputs\some_image.png 3840
```

## Documentation

Full technical documentation is in `docs\` as PDFs:

- `00-AI-Image-Generator-Complete-Technical-Documentation.pdf` is the complete document (29 pages).
- `01-Project-Overview.pdf` to `11-Appendix.pdf` are the same sections as standalone PDFs
  (requirements, architecture, workflows, module reference, prompt language, database schema,
  performance, operations, change log, appendix).
