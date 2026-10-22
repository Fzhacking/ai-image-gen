"""
AI image generation on a local NVIDIA GPU using Stable Diffusion XL (diffusers).

Usage:
    python generate.py "a cozy cabin in a snowy forest at dusk, cinematic lighting"
    python generate.py "a red sports car" --steps 40 --width 1024 --height 768 --seed 42
    python generate.py "an astronaut riding a horse" --negative "blurry, low quality" --count 4

The first run downloads the model (~7 GB) into the Hugging Face cache.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate images with Stable Diffusion XL on your GPU.")
    p.add_argument("prompt", help="Text description of the image to generate")
    p.add_argument("--negative", default="blurry, low quality, deformed, watermark, text",
                   help="Things to avoid in the image")
    p.add_argument("--steps", type=int, default=30, help="Denoising steps (20-50; more = slower, often better)")
    p.add_argument("--guidance", type=float, default=7.0, help="How strongly to follow the prompt (5-9 typical)")
    p.add_argument("--width", type=int, default=1024, help="Image width (multiple of 8)")
    p.add_argument("--height", type=int, default=1024, help="Image height (multiple of 8)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible results")
    p.add_argument("--count", type=int, default=1, help="Number of images to generate")
    p.add_argument("--model", default=MODEL_ID, help="Hugging Face model id or local path")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR, help="Output directory")
    return p.parse_args()


def check_gpu() -> torch.device:
    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not available. Install the CUDA build of PyTorch (see requirements.txt).")
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Using GPU: {name} ({vram_gb:.1f} GB VRAM)")
    return torch.device("cuda")


def load_pipeline(model_id: str, device: torch.device) -> StableDiffusionXLPipeline:
    print(f"Loading model: {model_id} (first run downloads ~7 GB)...")
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to(device)

    # Memory-friendly VAE decoding; harmless on a 24 GB card, essential on smaller ones.
    pipe.vae.enable_slicing()

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram_gb < 12:
        # Offload idle sub-models to CPU so the pipeline fits on 8-10 GB cards.
        pipe.enable_model_cpu_offload()

    print(f"Model ready in {time.time() - t0:.1f}s")
    return pipe


def main() -> None:
    args = parse_args()
    device = check_gpu()
    pipe = load_pipeline(args.model, device)
    args.out.mkdir(parents=True, exist_ok=True)

    base_seed = args.seed if args.seed is not None else torch.seed() % (2**32)

    for i in range(args.count):
        seed = base_seed + i
        generator = torch.Generator(device=device).manual_seed(seed)

        print(f"\n[{i + 1}/{args.count}] seed={seed} steps={args.steps} size={args.width}x{args.height}")
        t0 = time.time()
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            width=args.width,
            height=args.height,
            generator=generator,
        ).images[0]
        elapsed = time.time() - t0

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = args.out / f"{stamp}_seed{seed}.png"
        image.save(path)
        print(f"Saved {path}  ({elapsed:.1f}s)")

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nDone. Peak VRAM used: {peak:.1f} GB")


if __name__ == "__main__":
    main()
