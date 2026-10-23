# Frazile AI - AI Image Generation (local GPU)

Runs Z-Image Turbo, Stable Diffusion XL and its best community fine-tunes, Qwen-Image and FLUX.1 on your
NVIDIA GPU using Hugging Face `diffusers`, with prompt weighting, LoRAs and Real-ESRGAN upscaling. Since
version 2.2 it also makes **video** (LTX-Video, Wan 2.1, with frame-by-frame upscaling to 4K) and **music**
(MusicGen) from the same app. A web Studio drives everything from the prompt; an Admin panel manages
models, LoRAs, downloads and shows live system statistics.

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

Video and music models live in the same registry but are used from their own tabs (see
[Video and music](#video-and-music)):

| Model | Family | Defaults | What it does |
|---|---|---|---|
| LTX-Video 2B (`ltx`) | video | 40 steps, guidance 3, 24 fps | text-to-video and image-to-video; 768x512 clips of 2-10 s in about a minute; ~28 GB |
| Wan 2.1 T2V 1.3B (`wan`) | video | 30 steps, guidance 5, 16 fps | text-to-video at 832x480 with smoother motion; several minutes per clip; ~29 GB |
| MusicGen stereo medium (`musicgen`) | music | guidance 3 | instrumental music, 32 kHz stereo, up to 30 s; ~4 GB |
| MusicGen stereo small (`musicgen-small`) | music | guidance 3 | quick sketches; ~1.3 GB |
| Stable Audio Open 1.0 (`stable-audio`) | music | 100 steps, guidance 7 | 44.1 kHz music loops and sound effects, up to 47 s; gated on Hugging Face, untested here |

Steps and CFG are the model's defaults; `quick`, `highly detailed`, `strict`, `creative`,
`30 steps` and `cfg 8` still adjust them. Every SDXL-family model runs with the DPM++ 2M Karras
sampler (DPM++ SDE Karras for the turbo and RealVis models, Euler a for Animagine), which gives
cleaner results than the stock sampler at the same step count.

**Auto** (the default in the Model box) keeps Z-Image Turbo or FLUX while one of them is loaded.
When an SDXL-family model is loaded it picks a specialist from the style words once it is on disk:
`photorealistic`/`cinematic` -> Juggernaut, `anime` -> Animagine, `oil painting` / `watercolor` ->
Playground, `digital art`/`comic`/`cyberpunk` -> DreamShaper. Otherwise it keeps the model that is
already loaded. Switching between SDXL models takes about 26 s; Z-Image takes longer.

**Adding models:** open **Admin › Add**, paste a Hugging Face repo id (`RunDiffusion/Juggernaut-XL-v9`),
a file in a repo, a Civitai link, a direct `.safetensors` URL or a local path, press **Check** and then
**Add**. Diffusers repos become models in their own right (SDXL, Z-Image, Qwen-Image or FLUX family is
detected from the repo); single files are downloaded into `models\checkpoints\` and loaded with the
SDXL loader. Downloads run in the background (Admin › Downloads) and the model appears in the Model box
as soon as it finishes, no restart needed. You can still drop SDXL `.safetensors` files straight into
`models\checkpoints\` by hand; files with `turbo`, `lightning` or `hyper` in the name get 6-step defaults.

Every model, built-in or added, can be edited in **Admin › Models**: default steps and guidance, sampler,
description, aliases, the style words that auto-pick it, enabled or hidden, and which model loads at start.
Edits live in `models\registry.json` and apply to the next load.

**LoRAs:** two starter SDXL LoRAs are set up in `models\loras\` (fetch them on a fresh clone with
`.\.venv\Scripts\python.exe models.py fetch-loras`): **Pixel Art XL** (`<lora:pixel-art-xl>`,
say "pixel art" in the prompt) and **Papercut** (`<lora:papercut>`, say "papercut"). Tick them in the
LoRAs box or write the tag in the prompt with any weight, for example `<lora:pixel-art-xl:0.9>`. Several
can be stacked. Add more in **Admin › Add** (Civitai links bring their trigger words along) or by dropping
`.safetensors` files into the folder. **Admin › LoRAs** records, per file, the model family it was trained
for, its trigger words and its default weight: Auto then switches to a model of that family, the Studio
reminds you of the trigger words, and a LoRA is never applied to a model of the wrong family.

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
prompt; the quick-hint chips above it add or remove the common hints (aspect ratio, 2K/4K/8K, image
count, quick / highly detailed, styles, a random seed) and the Model and LoRAs boxes can be overridden
from the prompt. The header shows the loaded model, VRAM, GPU temperature and running downloads live.
After a generation an **Upscale** bar appears: pick 2K / 4K / 8K and upscale the selected image or the
whole batch with Real-ESRGAN; **Reuse this seed** puts the selected image's seed into the prompt and
**New variation** runs the prompt again with a fresh seed.

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

## Video and music

The **Video** and **Music** tabs use the same GPU, the same History and the same Admin panel as the
images. Both work entirely offline once the weights are downloaded (first use, or **Admin › Models ›
Download in background**).

**Video tab.** Describe the scene, the motion and the camera in a sentence or two, pick a video
model, a size (768x512, 512x768, 832x480, 480x832, 1024x576, 576x1024), a length (2-10 s), steps,
guidance and a seed, then press **Generate video**. Chips above the prompt add camera and motion
hints. Options:

- **Start image** (LTX-Video only): drop in any picture, or a Studio result, and the model animates it
  (image-to-video). The image is centre-cropped to the chosen size.
- **Upscale video**: every frame is run through Real-ESRGAN's compact video network
  (`realesr-general-x4v3`, fetched on first use into `models\`), resampled to the exact target and
  re-encoded to H.264. A 4 s LTX clip reaches 3840x2560 in about a minute on an RTX 3090.
- **Add soundtrack**: pick any track made in the Music tab; it is cut to the clip length with a short
  fade-out and muxed into a new MP4.

Video models need the whole 24 GB card, so the image model is unloaded while a video model is in use
and reloads automatically on the next image. Clips are H.264 MP4 files in `outputs\`
(`<timestamp>_seed<seed>.mp4`, `_up3840.mp4` for upscales, `_music.mp4` with a soundtrack).

**Music tab.** Describe genre, mood, instruments and tempo ("upbeat synthwave with punchy drums and
a warm bass line, 120 bpm"), choose a length (3-30 s with MusicGen), guidance, temperature and a
seed, then press **Generate music**. MusicGen writes instrumental stereo music at 32 kHz; a 15 s
track takes roughly 30-60 s. Tracks are 16-bit WAV files in `outputs\`. A small MusicGen model
(about 4 GB of VRAM) can stay loaded next to an SDXL model; the big image models get the card to
themselves.

Measured on the RTX 3090 (see `docs\10-Video-and-Music.pdf` for the full table):

| Operation | Settings | Time | VRAM |
|---|---|---|---|
| LTX-Video text-to-video | 768x512, 89 frames (3.7 s), 30 steps | 30 s | 15.5 GB peak |
| LTX-Video image-to-video | 768x512, 65 frames, 20 steps, from a Studio PNG | 16 s | 14.9 GB peak |
| LTX-Video load | first load 186 s (writes a bf16 text-encoder copy), then 94 s | | 14.2 GB resident |
| Wan 2.1 text-to-video | 832x480, 49 frames (3 s), 30 steps | 205 s | 12.9 GB peak |
| Wan 2.1 load | first load 266 s, then 164 s (8-bit text encoder) | | 11.1 GB resident |
| Video upscale to 4K | 65 frames to 3840x2560 | 49 s | under 1 GB extra |
| MusicGen stereo medium | 10 s track, guidance 3 | 36 s (+24 s load) | 4.2 GB peak |

Everything lands in **History** with a `kind` column (image, video, music), a first-frame or waveform
thumbnail, an inline player and **Reuse prompt**, which sends the prompt back to the tab that made it.

## Admin panel

The **Admin** tab manages the studio and watches the machine. Its sub-tabs:

- **Overview:** the system tiles (below), activity tiles (generations today and this week, images /
  videos / tracks with total seconds of video and music, GPU minutes in the last 24 h, loaded and
  start-up model, models on disk by kind, LoRAs, downloads, outputs size, Hugging Face login) and a
  per-model table of uses, average / fastest / slowest seconds, steps per second, VRAM and temperature.
- **Models:** every model (image, video and music) with family, type (built-in, local file, added), defaults,
  sampler, on-disk size, usage and enabled / start-up flags. Click one to edit it, **Load now**, **Download in background**,
  **Set as start-up model**, unload the current model to free VRAM, remove it from the list (or reset a
  built-in to its defaults) or delete its files from disk.
- **LoRAs:** every LoRA file with family, trigger words, default weight, size and recorded uses; edit the
  metadata, **Try it in the Studio** (fills a test prompt and switches tabs) or delete the file.
- **Add:** add a model or LoRA from a Hugging Face repo or file, a Civitai link, a direct URL, a local
  path or a browser upload. **Check** shows what the source is (pipeline class, family, size, gated,
  trigger words) and pre-fills the form.
- **Downloads:** progress, speed and ETA of background downloads, with cancel.
- **System:** the full telemetry described below.
- **Settings:** start-up model, default LoRA weight, default negative prompt, Hugging Face token (for
  FLUX and gated repos), Civitai token, clean-up of old generations, and the admin lock.
- **Logs:** the app's own log (model loads, generations, downloads, errors) with filter and auto-refresh.

Deleting files and cleaning outputs need their confirmation box ticked. When the app is reachable from
other machines (Docker or `GRADIO_SERVER_NAME=0.0.0.0`) set `FRAZILE_ADMIN_PASSWORD`; destructive
actions then also need the panel unlocked in Settings.

The **System** sub-tab is the live system panel. Tiles at the top show GPU load, VRAM, GPU temperature
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
Stats refresh while an image is generating: telemetry and GPU work run on separate queues.

## Docker

The whole app can run as a GPU container.

```powershell
docker compose up --build
```

Then open http://localhost:7860. Generated images land in `outputs\`, your checkpoints, LoRAs and
the Admin registry are read from `models\`, the history database is `generations.db`, and downloaded
models go to a named volume (`hf-cache`) so rebuilds do not re-download them. Set `HF_TOKEN` in the
environment (or a `.env` file) if you want FLUX to download inside the container, and
`FRAZILE_ADMIN_PASSWORD` to lock destructive Admin actions, since the container listens on all interfaces.

Requirements on the host:

- **Windows 11:** WSL 2 (`wsl --install`), Docker Desktop with the WSL 2 backend, and the normal
  NVIDIA Windows driver. GPU access is built into Docker Desktop; no extra toolkit is needed.
- **Linux:** Docker, the NVIDIA driver and `nvidia-container-toolkit`.

The image is based on `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` with PyTorch for CUDA 12.6.
The first build downloads several GB of Python packages. Inside the container the Admin panel shows
the container's own view of CPU, memory and network; GPU readings come from the host driver.

## Categories and batch generation

Give an image a **category** in the Studio (the Category box under the negative prompt; type a new
one or pick an existing one) and it is saved to `outputs\<Category>\` instead of `outputs\`, recorded
with that category in `generations.db`, and its prompt is stored under the category in the
`prompt_library` table. The History tab filters by category.

`batch.py` generates a whole category through the running app (start `app.py` first):

```powershell
.\.venv\Scripts\python.exe batch.py --category Nature --count 50              # built-in Nature prompt set
.\.venv\Scripts\python.exe batch.py --category Nature --count 20 --model juggernaut --seed-start 100
.\.venv\Scripts\python.exe batch.py --category Cities --prompts my_prompts.txt # one prompt per line
.\.venv\Scripts\python.exe batch.py --list                                     # built-in prompt sets
.\.venv\Scripts\python.exe db.py categories                                    # counts per category
.\.venv\Scripts\python.exe db.py prompts Nature                                # the stored prompts
```

Prompts are cycled with fresh seeds when `--count` exceeds the set, so every image differs. The
built-in Nature set has 50 prompts (mountains, forests, oceans, deserts, wildlife, macro, weather,
seasons); 50 Z-Image Turbo images take about 9 minutes on an RTX 3090.

`upscale_batch.py` upscales a whole category afterwards with Real-ESRGAN, exactly like the Studio's
Upscale bar, and records the results (it runs in its own process, so the app can stay up):

```powershell
.\.venv\Scripts\python.exe upscale_batch.py --category Nature --targets 4k 8k   # <stem>_3840px.png and _7680px.png
.\.venv\Scripts\python.exe upscale_batch.py --ids 31 32 --targets 2k
```

The database row keeps the largest target in its upscale columns and lists every target under
`extra_json["upscales"]`; existing files are skipped, so the command can be re-run.

## Generation details and history

After every run the Generate tab shows a report under the gallery: inference time and
steps per second, image settings, peak VRAM, GPU temperature and power before and after,
clock speeds, fan, driver and library versions, and the exact prompt sent to the model.

Every image, video clip and music track is also recorded in `generations.db` (SQLite). Browse it two ways:

- **History tab** in the app: search prompts, filter by kind (images, videos, music), model or
  upscaled-only, browse a table or a thumbnail grid, click an entry to view or play it with its full
  details, **Reuse prompt** (with or without the seed) to send it back to the tab that made it, or
  delete the record and its files.
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

- `00-AI-Image-Generator-Complete-Technical-Documentation.pdf` is the complete document.
- `01-Project-Overview.pdf` to `15-Appendix.pdf` are the same sections as standalone PDFs
  (requirements, architecture, workflows, module reference, models, prompt language, database schema,
  admin panel, video and music, performance, deployment, operations, change log, appendix).

Rebuild them after a change with `.\.venv\Scripts\python.exe docs-src\build_pdf.py`.
