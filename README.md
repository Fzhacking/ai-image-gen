# Frazile AI - AI Image Generation (local GPU)

Runs Z-Image Turbo, Stable Diffusion XL, its best community fine-tunes and FLUX.1 on your NVIDIA GPU using
Hugging Face `diffusers`, with prompt weighting, LoRAs and Real-ESRGAN upscaling.

**Project owner:** Frazile AI  
**Author:** Frazile AI

## Setup (once)

```powershell
cd C:\Users\frazi\Documents\apps\ai-image-gen
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Generate

```powershell
.\.venv\Scripts\Activate.ps1
python generate.py "a cozy cabin in a snowy forest at dusk, cinematic lighting"
python generate.py "a red sports car" --steps 40 --seed 42 --count 4
python generate.py "portrait of a fisherman, (weathered skin:1.2)" --model juggernaut
python generate.py --list-models
python generate.py --help
```

Images are saved to `outputs\`. The first use of a model downloads it (~7 GB each) to
`%USERPROFILE%\.cache\huggingface`.

## Models

| Say in the prompt | Model | Steps | CFG | Best for |
|---|---|---|---|---|
| `model zimage` | Z-Image Turbo (default) | 8 | 0 | photoreal, legible text, strong prompt following; Apache 2.0 |
| `model sdxl` | SDXL base 1.0 | 30 | 7 | general purpose baseline |
| `model juggernaut` | Juggernaut XL v9 | 30 | 5 | photoreal people, products and scenes; best all-round upgrade |
| `model realvis` | RealVisXL V5.0 | 30 | 4 | photoreal with strong skin, fabric and lighting detail |
| `model dreamshaper` | DreamShaper XL v2 Turbo | 6 | 2 | fast illustration, fantasy and stylised art |
| `model playground` | Playground v2.5 | 40 | 3 | highly aesthetic, painterly-real composition and colour |
| `model animagine` | Animagine XL 3.1 | 28 | 6 | anime and manga |
| `model qwen` | Qwen-Image + Lightning | 8 | 1 | 20B model in 4-bit: best text rendering and layout, posters, signage |
| `model flux` | FLUX.1 [schnell] | 4 | 0 | best prompt understanding: text, counting, complex scenes |

Steps and CFG are the model's defaults; `quick`, `highly detailed`, `strict`, `creative`,
`30 steps` and `cfg 8` still adjust them. Every SDXL-family model runs with the DPM++ 2M Karras
sampler (DPM++ SDE Karras for the turbo and RealVis models, Euler a for Animagine), which gives
cleaner results than the stock sampler at the same step count.

**Auto** (the default in the Model box) keeps Z-Image Turbo or FLUX while one of them is loaded.
When an SDXL-family model is loaded it picks a specialist from the style words once it is on disk:
`photorealistic`/`cinematic` -> Juggernaut, `anime` -> Animagine, `oil painting` / `watercolor` ->
Playground, `digital art`/`comic`/`cyberpunk` -> DreamShaper. Otherwise it keeps the model that is
already loaded. Switching between SDXL models takes about 26 s; Z-Image takes longer.

**Your own checkpoints:** drop any SDXL `.safetensors` file (Civitai downloads, merges, your own
fine-tunes) into `models\checkpoints\`. It appears in the model list under its file name after a
restart. Files with `turbo`, `lightning` or `hyper` in the name get 6-step defaults.

**LoRAs:** two starter SDXL LoRAs are set up in `models\loras\` (fetch them on a fresh clone with
`.\.venv\Scripts\python.exe models.py fetch-loras`): **Pixel Art XL** (`<lora:pixel-art-xl>`,
say "pixel art" in the prompt) and **Papercut** (`<lora:papercut>`, say "papercut"). Tick them in the
LoRAs box (applied at 0.8) or write the tag in the prompt with any weight, for example
`<lora:pixel-art-xl:0.9>`. Several can be stacked. Add more by dropping `.safetensors` files into the
folder and restarting. They apply to SDXL-family models; Z-Image and FLUX need LoRAs trained for them.

**Prompt weighting (SDXL family):** `(glowing sword:1.3)` or `(glowing sword)++` emphasises,
`(word:0.7)` or `(word)--` de-emphasises. Prompts longer than 77 tokens are no longer cut off.

**Z-Image Turbo** is the model loaded at start-up. It is a 6B flow model from Tongyi-MAI that renders
photoreal scenes and readable text in 8 steps with no guidance; it has no negative prompt and ignores LoRAs and
weighting. On a 24 GB card it runs with CPU offload (~20 GB download).

**Qwen-Image** is Alibaba's 20B image model (Apache 2.0), the strongest open model for rendering
text and complex layouts. To fit a 24 GB card it runs with the transformer quantised to 4-bit and the
Lightning 8-step LoRA attached, so no negative prompt or guidance. The first `model qwen` downloads
about 48 GB and then spends several minutes quantising; the 4-bit copy is cached under `models\cache\`
so later loads take about 90 s. On the RTX 3090 an image takes 13 to 14 s at 18 GB peak VRAM. Stop other
GPU programs first: it needs most of the card.

**FLUX.1 [schnell]** is gated on Hugging Face. Accept the terms at
https://huggingface.co/black-forest-labs/FLUX.1-schnell, run `.venv\Scripts\huggingface-cli login`,
then say `model flux`. The download is about 33 GB. On a 24 GB card the transformer is loaded in
4-bit and the T5 encoder in 8-bit with CPU offload, which needs `bitsandbytes` (in
requirements.txt). FLUX has no negative prompt and ignores LoRAs here.

## Web UI (recommended)

```powershell
.\.venv\Scripts\python.exe app.py
```

Opens http://127.0.0.1:7860 in a full-width, dark Frazile AI studio layout. Write everything in the
prompt; the only control is the Model box, and even that can be overridden from the prompt.
After a generation an **Upscale** bar appears: pick 2K / 4K / 8K and upscale the selected image
or the whole batch with Real-ESRGAN.

| Want | Say in the prompt |
|---|---|
| A specific model | `model juggernaut`, `model: flux`, `using realvis` |
| Emphasis | `(glowing sword:1.3)`, `(word)++`, `(word)--` |
| LoRA | `<lora:name:0.8>` |
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

## Admin panel

The **Admin** tab is a live system panel. Tiles at the top show GPU load, VRAM, GPU temperature
(with an OK / Warm / Hot / Critical state), GPU power against its limit, CPU usage with a
per-core bar strip, memory, system-disk space, network throughput and app uptime. Below them:

- **GPU:** card, driver / CUDA / cuDNN, current and maximum clocks, utilisation, VRAM from
  nvidia-smi and from torch, PCIe link, throttle reasons (decoded), VBIOS, processes on the GPU,
  loaded model and sampler.
- **CPU:** model, cores, frequency, usage per core, load averages, process and thread counts, boot time.
- **Memory and storage:** RAM, page file, disk I/O since boot, every partition, and the size of the
  outputs folder, models folder, Hugging Face cache and database.
- **Network:** hostname and primary IP, live up/down throughput, totals since boot, errors and drops,
  connections and listening ports, every interface with IPv4, MAC and link speed, and an optional
  reachability check of Hugging Face, GitHub and PyPI.
- **Power:** GPU draw and limits, performance state and fan, the active Windows power plan, battery
  if present, and an estimate of the GPU energy used by every recorded generation.
- **Software and app:** OS and library versions, whether it runs in Docker, the app process, and
  generation totals from the database.

Refresh by hand or tick **Auto-refresh every 5 s**. The panel is also available in the container.

## Docker

The whole app can run as a GPU container.

```powershell
docker compose up --build
```

Then open http://localhost:7860. Generated images land in `outputs\`, your checkpoints and
LoRAs are read from `models\`, the history database is `generations.db`, and downloaded models
go to a named volume (`hf-cache`) so rebuilds do not re-download them. Set `HF_TOKEN` in the
environment (or a `.env` file) if you want FLUX to download inside the container.

Requirements on the host:

- **Windows 11:** WSL 2 (`wsl --install`), Docker Desktop with the WSL 2 backend, and the normal
  NVIDIA Windows driver. GPU access is built into Docker Desktop; no extra toolkit is needed.
- **Linux:** Docker, the NVIDIA driver and `nvidia-container-toolkit`.

The image is based on `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` with PyTorch for CUDA 12.6.
The first build downloads several GB of Python packages. Inside the container the Admin panel shows
the container's own view of CPU, memory and network; GPU readings come from the host driver.

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
