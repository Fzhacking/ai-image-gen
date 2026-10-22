"""
Real-ESRGAN upscaling (x4 per pass, tiled so any image size fits in VRAM).

Weights: models/RealESRGAN_x4plus.pth
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

    python upscaler.py input.png 3840        # upscale so the long side is 3840 px
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_PATH = Path(__file__).parent / "models" / "RealESRGAN_x4plus.pth"

# Named targets: long side in pixels.
TARGETS = {"2k": 2560, "4k": 3840, "uhd": 3840, "8k": 7680}

TILE = 512       # input tile size in px
OVERLAP = 32     # overlap between tiles to hide seams


class Upscaler:
    def __init__(self, device: str = "cuda", model_path: Path = MODEL_PATH):
        import spandrel
        if not model_path.exists():
            raise FileNotFoundError(f"Upscaler weights missing: {model_path}")
        self.device = device
        self.model = spandrel.ModelLoader(device=device).load_from_file(str(model_path))
        self.model.eval()
        self.scale = self.model.scale
        self.name = f"{self.model.architecture.name} x{self.scale} ({model_path.name})"
        self._half = device == "cuda"
        if self._half:
            self.model.model.half()

    @torch.inference_mode()
    def _run_tile(self, tile: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t = t.half() if self._half else t.float()
        t = t / 255.0
        out = self.model(t).clamp_(0, 1)
        out = (out[0].permute(1, 2, 0).float().cpu().numpy() * 255.0).round().astype(np.uint8)
        return out

    def upscale_once(self, img: Image.Image) -> Image.Image:
        """One x4 pass, tiled with overlap."""
        src = np.array(img.convert("RGB"), copy=True)
        h, w, _ = src.shape
        s = self.scale
        out = np.zeros((h * s, w * s, 3), dtype=np.uint8)

        for y in range(0, h, TILE):
            for x in range(0, w, TILE):
                # tile with context padding
                y0, y1 = max(0, y - OVERLAP), min(h, y + TILE + OVERLAP)
                x0, x1 = max(0, x - OVERLAP), min(w, x + TILE + OVERLAP)
                up = self._run_tile(src[y0:y1, x0:x1])
                # crop the padding back off
                cy0, cx0 = (y - y0) * s, (x - x0) * s
                cy1 = cy0 + (min(h, y + TILE) - y) * s
                cx1 = cx0 + (min(w, x + TILE) - x) * s
                out[y * s: y * s + (cy1 - cy0), x * s: x * s + (cx1 - cx0)] = up[cy0:cy1, cx0:cx1]
        return Image.fromarray(out)

    def upscale_to(self, img: Image.Image, long_side: int) -> tuple[Image.Image, dict]:
        """
        Upscale until the long side is >= long_side, then resample down to exactly long_side.
        Returns (image, info). Runs x4 passes; before a second pass the intermediate is
        resized so the pass lands near the target instead of far beyond it.
        """
        t0 = time.perf_counter()
        cur = img
        passes = 0
        while max(cur.size) < long_side:
            if passes > 0 and max(cur.size) * self.scale > long_side:
                ratio = (long_side / self.scale) / max(cur.size)
                cur = cur.resize((max(1, round(cur.width * ratio)), max(1, round(cur.height * ratio))),
                                 Image.LANCZOS)
            cur = self.upscale_once(cur)
            passes += 1
        if max(cur.size) > long_side:
            ratio = long_side / max(cur.size)
            cur = cur.resize((round(cur.width * ratio), round(cur.height * ratio)), Image.LANCZOS)
        info = {
            "upscaler": self.name,
            "passes": passes,
            "from": f"{img.width}x{img.height}",
            "to": f"{cur.width}x{cur.height}",
            "seconds": round(time.perf_counter() - t0, 2),
        }
        return cur, info


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 3840
    up = Upscaler()
    out, info = up.upscale_to(Image.open(src), target)
    dst = src.with_name(f"{src.stem}_up{target}.png")
    out.save(dst)
    print(info, "->", dst)
