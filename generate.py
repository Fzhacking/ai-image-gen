"""
AI image generation on a local NVIDIA GPU (diffusers), command-line version.

Usage:
    python generate.py "a cozy cabin in a snowy forest at dusk, cinematic lighting"
    python generate.py "a red sports car" --steps 40 --width 1024 --height 768 --seed 42
    python generate.py "an astronaut riding a horse" --negative "blurry, low quality" --count 4
    python generate.py "portrait of a fisherman, (weathered skin:1.2)" --model juggernaut
    python generate.py "cat astronaut" --model dreamshaper --lora pixel-style:0.7
    python generate.py --list-models

Models are named by key (see --list-models), Hugging Face repo id, or a local .safetensors path.
The first use of a model downloads it into the Hugging Face cache.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch is imported

import models

OUTPUT_DIR = Path(__file__).parent / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate images on your GPU.")
    p.add_argument("prompt", nargs="?", help="Text description of the image to generate")
    p.add_argument("--negative", default="blurry, low quality, deformed, watermark, text",
                   help="Things to avoid in the image (ignored by FLUX)")
    p.add_argument("--steps", type=int, default=None, help="Denoising steps (default: the model's own)")
    p.add_argument("--guidance", type=float, default=None, help="Prompt strength (default: the model's own)")
    p.add_argument("--width", type=int, default=1024, help="Image width (multiple of 8)")
    p.add_argument("--height", type=int, default=1024, help="Image height (multiple of 8)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible results")
    p.add_argument("--count", type=int, default=1, help="Number of images to generate")
    p.add_argument("--model", default="sdxl", help="Model key, Hugging Face repo id, or local .safetensors path")
    p.add_argument("--lora", action="append", default=[], metavar="NAME[:WEIGHT]",
                   help="LoRA from models/loras/ to apply (repeatable)")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR, help="Output directory")
    p.add_argument("--list-models", action="store_true", help="List available models and exit")
    return p.parse_args()


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not available. Install the CUDA build of PyTorch (see requirements.txt).")
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Using GPU: {name} ({vram_gb:.1f} GB VRAM)")
    return torch.device("cuda")


def list_models() -> None:
    for m in models.all_models():
        where = "downloaded" if models.is_downloaded(m) else f"not downloaded (~{m.download_gb:g} GB)"
        print(f"{m.key:12} {m.name:26} {m.steps:>3} steps  cfg {m.guidance:<4g} {where}")
        print(f"{'':12} {m.description}")
    print("LoRAs:", ", ".join(models.available_loras()) or "none in models/loras/")


def main() -> None:
    args = parse_args()
    if args.list_models:
        list_models()
        return
    if not args.prompt:
        sys.exit("Give a prompt, or --list-models.")

    import torch

    device = check_gpu()
    try:
        spec = models.resolve(args.model)
    except KeyError:
        sys.exit(f"Unknown model '{args.model}'. Known: {', '.join(m.key for m in models.all_models())}")

    manager = models.ModelManager(device=str(device))
    pipe = manager.load(spec)

    loras = []
    for item in args.lora:
        name, _, weight = item.partition(":")
        loras.append((name, float(weight) if weight else 1.0))
    for note in manager.apply_loras(loras):
        print(note)

    steps = args.steps if args.steps is not None else spec.steps
    guidance = args.guidance if args.guidance is not None else spec.guidance
    negative = args.negative if spec.negative else ""
    encoded = manager.encode(args.prompt, negative)

    args.out.mkdir(parents=True, exist_ok=True)
    base_seed = args.seed if args.seed is not None else torch.seed() % (2**32)

    for i in range(args.count):
        seed = base_seed + i
        generator = torch.Generator(device=device).manual_seed(seed)

        print(f"\n[{i + 1}/{args.count}] {spec.name}  seed={seed} steps={steps} cfg={guidance:g} "
              f"size={args.width}x{args.height}  scheduler={manager.scheduler_name}")
        t0 = time.time()
        image = pipe(
            **encoded,
            **manager.generation_kwargs(steps, guidance, args.width, args.height, generator),
        ).images[0]
        elapsed = time.time() - t0

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = args.out / f"{stamp}_seed{seed}.png"
        image.save(path)
        print(f"Saved {path}  ({elapsed:.1f}s)")
        manager.release_cache()

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nDone. Peak VRAM used: {peak:.1f} GB")


if __name__ == "__main__":
    main()
