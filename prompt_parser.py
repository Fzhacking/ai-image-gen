"""
Turn a free-text prompt into generation settings.

Understood in the prompt (case-insensitive):
  Resolution      "1024x768", "1216 x 832", "1920x1080"  (snapped to multiples of 64, scaled to ~1 MP)
  Aspect ratio    "16:9", "9:16", "4:3", "3:2", "1:1", "21:9", "2:3", "3:4"
  Orientation     portrait, landscape, square, wide/widescreen/ultrawide, tall/vertical,
                  phone wallpaper, desktop wallpaper, banner, poster
  Output size     2k -> 2560 px long side, 4k / uhd -> 3840 px, 8k -> 7680 px
                  (generated at native size, then upscaled with Real-ESRGAN; also raises steps)
  Quality         ultra detailed, masterpiece, hd -> more steps
                  quick, draft, fast, rough, preview -> fewer steps
  Fidelity        strict/exact/precise -> higher guidance; creative/loose/dreamy -> lower
  Style           anime, photorealistic/photo, cinematic, watercolor, oil painting, 3d render,
                  pixel art, digital art/concept art, sketch, comic, cyberpunk, low poly
  Count           "3 images", "4 variations", "x4", "4 versions"
  Seed            "seed 42", "seed: 42", "seed=42"
  Steps           "30 steps", "steps 30", "steps=30"
  Guidance        "cfg 8", "guidance 8", "cfg=8"
  Model           "model juggernaut", "model: flux", "using realvis"  (see models.py for the list)
  LoRA            "<lora:name:0.8>", "<lora:name>"   (files in models/loras/)
  Weighting       "(red hair:1.3)", "(word)++"   kept in the prompt, applied by compel

Numeric directives (resolution, ratio, seed, count, steps, cfg), model and LoRA tags are removed
from the prompt sent to the model. Descriptive words (portrait, 4k, anime ...) stay in because
they help the model too.

Steps and guidance are only final after apply_defaults(): the parser records whether the user
set them explicitly, and the chosen model supplies the defaults (a turbo model wants 6 steps,
FLUX wants 4, SDXL wants 30).
"""

import re
from dataclasses import dataclass, field

from models import model_words

MAX_SIDE = 1536
MIN_SIDE = 512
TARGET_PIXELS = 1024 * 1024  # SDXL native area

# output size words -> long side in pixels after upscaling
UPSCALE_TARGETS = {"2k": 2560, "4k": 3840, "uhd": 3840, "8k": 7680}

# aspect ratio -> SDXL-friendly bucket (width, height)
RATIO_BUCKETS = {
    "1:1": (1024, 1024),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "21:9": (1536, 640),
    "9:21": (640, 1536),
    "5:4": (1088, 896),
    "4:5": (896, 1088),
}

ORIENTATION_WORDS = [
    (r"\b(ultra ?wide|cinemascope)\b", "21:9"),
    (r"\b(phone wallpaper|mobile wallpaper|instagram story|tiktok)\b", "9:16"),
    (r"\b(desktop wallpaper|wallpaper|widescreen|wide ?screen|banner|youtube thumbnail)\b", "16:9"),
    (r"\b(wide|horizontal)\b", "3:2"),
    (r"\b(tall|vertical|poster|book cover)\b", "2:3"),
    (r"\b(portrait)\b", "2:3"),
    (r"\b(landscape)\b", "3:2"),
    (r"\b(square|icon|avatar|profile picture)\b", "1:1"),
]

STYLE_SUFFIXES = [
    (r"\b(photo ?realistic|realistic photo|photograph|dslr|photo)\b",
     "photorealistic, 8k, DSLR photo, sharp focus, natural lighting"),
    (r"\b(anime|manga)\b", "anime style, vibrant colors, clean lines, studio quality"),
    (r"\b(cinematic|film still|movie scene)\b",
     "cinematic lighting, dramatic, film still, depth of field, color graded"),
    (r"\b(watercolou?r)\b", "watercolor painting, soft edges, paper texture, pastel tones"),
    (r"\b(oil painting|oil on canvas)\b", "oil painting, visible brush strokes, canvas texture, classical"),
    (r"\b(3d render|3d|octane|blender|unreal)\b", "3d render, octane render, volumetric lighting, highly detailed"),
    (r"\b(pixel art|8-bit|16-bit)\b", "pixel art, retro game sprite, crisp pixels"),
    (r"\b(digital art|concept art|digital painting)\b",
     "digital painting, concept art, trending on artstation, highly detailed"),
    (r"\b(pencil sketch|sketch|line art|ink drawing)\b", "pencil sketch, line art, monochrome, hand drawn"),
    (r"\b(comic|cartoon)\b", "comic book style, bold outlines, flat colors, halftone"),
    (r"\b(cyberpunk)\b", "cyberpunk, neon lights, rain, futuristic city, high contrast"),
    (r"\b(low ?poly)\b", "low poly, geometric, isometric, minimal shading"),
]

UPSCALE_WORDS = r"\b(2k|4k|uhd|8k)\b"
HIGH_QUALITY = r"\b(hd|ultra ?detailed|highly detailed|masterpiece|best quality|high quality)\b"
LOW_QUALITY = r"\b(quick|draft|fast|rough|preview)\b"
STRICT = r"\b(strict|exact|exactly|precise|literal)\b"
CREATIVE = r"\b(creative|loose|dreamy|abstract|surreal)\b"


@dataclass
class Settings:
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 7.0
    count: int = 1
    seed: int | None = None
    style: str | None = None
    upscale_to: int | None = None   # long side in px after upscaling, None = no upscale
    exact_size: tuple[int, int] | None = None  # crop the upscaled image to exactly this size
    notes: list[str] = field(default_factory=list)
    model: str | None = None                 # model key/alias named in the prompt, None = auto
    loras: list[tuple[str, float]] = field(default_factory=list)
    steps_explicit: bool = False
    guidance_explicit: bool = False
    quality: str | None = None               # "high" | "low" | None
    fidelity: str | None = None              # "strict" | "creative" | None

    def apply_defaults(self, base_steps: int, base_guidance: float) -> None:
        """Fill steps/guidance from the chosen model unless the prompt set them explicitly."""
        if not self.steps_explicit:
            steps = base_steps
            if self.quality == "high":
                steps = round(base_steps * 1.35)
            elif self.quality == "low":
                steps = max(1, round(base_steps * 0.5))
            self.steps = max(1, min(100, steps))
        if not self.guidance_explicit:
            g = base_guidance
            if g > 0:
                if self.fidelity == "strict":
                    g = base_guidance * 1.3
                elif self.fidelity == "creative":
                    g = base_guidance * 0.7
            elif self.fidelity:  # guidance-free models (FLUX schnell) ignore fidelity words
                self.notes.append("this model has no guidance scale, fidelity words ignored")
            self.guidance = round(max(0.0, min(20.0, g)), 1)

    def summary(self) -> str:
        parts = [f"{self.width}x{self.height}", f"{self.steps} steps", f"guidance {self.guidance:g}"]
        if self.upscale_to:
            uw, uh = self.upscaled_size()
            parts.insert(1, f"upscale to {uw}x{uh}")
        if self.style:
            parts.append(f"style: {self.style}")
        if self.count > 1:
            parts.append(f"{self.count} images")
        if self.seed is not None:
            parts.append(f"seed {self.seed}")
        if self.model:
            parts.append(f"model {self.model}")
        for name, w in self.loras:
            parts.append(f"lora {name} x{w:g}")
        return "  |  ".join(parts)

    def upscaled_size(self) -> tuple[int, int]:
        if self.exact_size:
            return self.exact_size
        if not self.upscale_to:
            return self.width, self.height
        ratio = self.upscale_to / max(self.width, self.height)
        return round(self.width * ratio), round(self.height * ratio)


def _snap(n: float) -> int:
    return max(MIN_SIDE, min(MAX_SIDE, int(round(n / 64) * 64)))


def _fit_resolution(w: int, h: int) -> tuple[int, int]:
    """Scale a requested resolution to roughly SDXL's native area, keeping the ratio."""
    scale = min(1.0, (TARGET_PIXELS / (w * h)) ** 0.5)
    return _snap(w * scale), _snap(h * scale)


def _closest_bucket(w: int, h: int) -> tuple[int, int]:
    ratio = w / h
    return min(RATIO_BUCKETS.values(), key=lambda wh: abs(wh[0] / wh[1] - ratio))


def _take(s: Settings, pattern: str) -> re.Match | None:
    """Find pattern in the current prompt (case-insensitive) and remove it. Returns the match."""
    m = re.search(pattern, s.prompt, re.IGNORECASE)
    if m:
        s.prompt = s.prompt[: m.start()] + s.prompt[m.end():]
    return m


def parse(text: str) -> Settings:
    s = Settings(prompt=text.strip())
    low = s.prompt.lower()  # for descriptive-word checks that do not alter the prompt
    ratio_set = False
    requested: tuple[int, int] | None = None

    # LoRA tags: <lora:name:0.8> or <lora:name>
    while (m := _take(s, r"<\s*lora\s*:\s*([^:<>]+?)\s*(?::\s*(-?\d*\.?\d+))?\s*>")):
        weight = float(m.group(2)) if m.group(2) else 1.0
        s.loras.append((m.group(1).strip(), max(-2.0, min(2.0, weight))))

    # model: "model juggernaut", "model: flux", "checkpoint=realvis", "using realvis", "with flux"
    m = _take(s, r"\b(?:model|checkpoint|ckpt)\s*[:=]?\s*([a-z0-9][a-z0-9_.\-]*)")
    if m:
        s.model = m.group(1).lower()
    else:
        words = "|".join(re.escape(w) for w in model_words())
        if words:
            m = _take(s, rf"\b(?:using|with|via|on)\s+(?:the\s+)?({words})(?:\s+model)?\b")
            if m:
                s.model = m.group(1).lower()

    # explicit resolution, e.g. 1920x1080
    m = _take(s, r"\b(\d{3,4})\s*[x×]\s*(\d{3,4})\b")
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        requested = (w, h)
        s.width, s.height = _fit_resolution(w, h)
        s.notes.append(f"resolution {w}x{h} -> generate at {s.width}x{s.height}")
        ratio_set = True

    # explicit aspect ratio, e.g. 16:9
    if not ratio_set:
        m = _take(s, r"\b(\d{1,2}):(\d{1,2})\b")
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            key = f"{a}:{b}"
            s.width, s.height = RATIO_BUCKETS.get(key) or _closest_bucket(a, b)
            s.notes.append(f"aspect {key}")
            ratio_set = True

    # orientation words (kept in the prompt)
    if not ratio_set:
        for pattern, key in ORIENTATION_WORDS:
            m = re.search(pattern, low)
            if m:
                s.width, s.height = RATIO_BUCKETS[key]
                s.notes.append(f"{key} from '{m.group(0)}'")
                break

    # output size: 2k / 4k / 8k words, or an explicit resolution larger than we can generate
    m = re.search(UPSCALE_WORDS, low)
    if m:
        s.upscale_to = UPSCALE_TARGETS[m.group(1)]
        s.notes.append(f"{m.group(1)} -> upscale to {s.upscale_to} px long side")
    elif requested and max(requested) > max(s.width, s.height):
        # upscale a little past the target so the exact size can be cropped out
        ratio = max(requested[0] / s.width, requested[1] / s.height)
        s.upscale_to = int(max(s.width, s.height) * ratio + 0.999)
        s.exact_size = requested
        s.notes.append(f"upscale and crop to exactly {requested[0]}x{requested[1]}")

    # steps
    m = _take(s, r"\b(?:steps?\s*[:=]?\s*(\d{1,3})|(\d{1,3})\s*steps?)\b")
    if m:
        s.steps = max(1, min(100, int(m.group(1) or m.group(2))))
        s.steps_explicit = True
    elif s.upscale_to or re.search(HIGH_QUALITY, low):
        s.quality = "high"
        s.notes.append("high quality -> more steps")
    elif re.search(LOW_QUALITY, low):
        s.quality = "low"
        s.notes.append("quick -> fewer steps")

    # guidance
    m = _take(s, r"\b(?:cfg|guidance)\s*[:=]?\s*(\d{1,2}(?:\.\d)?)\b")
    if m:
        s.guidance = max(0.0, min(20.0, float(m.group(1))))
        s.guidance_explicit = True
    elif re.search(STRICT, low):
        s.fidelity = "strict"
        s.notes.append("strict -> higher guidance")
    elif re.search(CREATIVE, low):
        s.fidelity = "creative"
        s.notes.append("creative -> lower guidance")

    # count
    m = _take(s, r"\b(?:(\d)\s*(?:images?|variations?|versions?|pictures?|pics?)|x\s?(\d))\b")
    if m:
        s.count = max(1, min(4, int(m.group(1) or m.group(2))))

    # seed
    m = _take(s, r"\bseed\s*[:=]?\s*(\d+)\b")
    if m:
        s.seed = int(m.group(1)) % (2**32)

    # style enrichment: add the preset words that are not already in the prompt
    for pattern, suffix in STYLE_SUFFIXES:
        if re.search(pattern, low):
            s.style = suffix.split(",")[0]
            extra = [t.strip() for t in suffix.split(",") if t.strip().lower() not in low]
            if extra:
                s.prompt = f"{s.prompt.strip().rstrip(',')}, {', '.join(extra)}"
            break

    # tidy leftovers from removed directives
    s.prompt = re.sub(r"\s*,(\s*,)+", ",", s.prompt)
    s.prompt = re.sub(r"\s+,", ",", s.prompt)
    s.prompt = re.sub(r"\s{2,}", " ", s.prompt).strip(" ,")
    s.apply_defaults(s.steps, s.guidance)  # SDXL baseline; the app re-applies for the chosen model
    return s


if __name__ == "__main__":
    tests = [
        "a cozy cabin in a snowy forest, 16:9, 4k",
        "portrait of a cyberpunk samurai, 3 images, seed 42",
        "desktop wallpaper of a mountain lake, photorealistic, 8k",
        "a cat, 1920x1080, quick",
        "a cat, 3840x2160",
        "anime girl with umbrella, tall, strict, 2 variations",
        "oil painting of a ship in a storm 21:9 steps 45 cfg 8",
        "square icon of a fox, 2k",
        "a bowl of ramen",
        "a knight in silver armour, model juggernaut, (glowing sword:1.3), highly detailed",
        "neon city street at night using flux, 16:9",
        "cat astronaut <lora:pixel-style:0.7> <lora:detailer> with dreamshaper, quick",
    ]
    for t in tests:
        r = parse(t)
        print(f"{t!r}\n   -> {r.summary()}\n   -> prompt: {r.prompt}\n   -> {r.notes}\n")
