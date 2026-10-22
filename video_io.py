"""
Video and audio file helpers (ffmpeg from the imageio-ffmpeg wheel, soundfile, PIL).

    write_mp4(frames, path, fps)            H.264 MP4 from an iterator of PIL frames (streamed, nothing is held in RAM)
    read_frames(path)                       (meta, iterator of PIL frames) from an MP4
    video_thumbnail(path, dest, width)      first frame as a small JPEG
    mux_audio(video, audio, dest)           copy the video stream and add an AAC sound track
    write_wav(samples, sr, path)            float samples (n, channels) -> 16-bit WAV
    waveform_png(wav, dest)                 small dark waveform picture for the History grid

No system ffmpeg is needed: imageio-ffmpeg ships a static binary for Windows, Linux and macOS.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from PIL import Image, ImageDraw


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError("ffmpeg failed: " + " | ".join(tail))


# ----------------------------------------------------------------------------- video

def write_mp4(frames: Iterable[Image.Image], path: str | Path, fps: float, crf: int = 17, audio: str | Path | None = None) -> dict:
    """Encode PIL frames (all the same size) into an H.264 MP4 that browsers play. Returns {width, height, frames}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    it = iter(frames)
    first = next(it)
    w, h = first.size
    w -= w % 2
    h -= h % 2   # yuv420p needs even dimensions
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
    if audio:
        cmd += ["-i", str(audio), "-shortest", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    n = 0
    try:
        for frame in _chain(first, it):
            if frame.size != (w, h):
                frame = frame.crop((0, 0, w, h)) if frame.size[0] >= w and frame.size[1] >= h else frame.resize((w, h), Image.LANCZOS)
            proc.stdin.write(np.asarray(frame.convert("RGB"), dtype=np.uint8).tobytes())
            n += 1
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        code = proc.wait()
    if code != 0:
        raise RuntimeError("ffmpeg failed: " + err.strip().splitlines()[-1] if err.strip() else f"ffmpeg exit {code}")
    return {"width": w, "height": h, "frames": n}


def _chain(first, rest) -> Iterator[Image.Image]:
    yield first
    yield from rest


def read_frames(path: str | Path) -> tuple[dict, Iterator[Image.Image]]:
    """Meta (width, height, fps, duration, nframes) and a lazy iterator of RGB PIL frames."""
    import imageio_ffmpeg
    gen = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(gen)
    w, h = meta["size"]
    info = {"width": w, "height": h, "fps": float(meta.get("fps") or 0), "duration": float(meta.get("duration") or 0)}

    def frames() -> Iterator[Image.Image]:
        for raw in gen:
            yield Image.frombuffer("RGB", (w, h), raw, "raw", "RGB", 0, 1)

    return info, frames()


def probe(path: str | Path) -> dict:
    info, frames = read_frames(path)
    frames.close()
    return info


def video_thumbnail(path: str | Path, dest: str | Path, width: int = 384) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path), "-frames:v", "1",
          "-vf", f"scale={width}:-2", "-q:v", "3", str(dest)])
    return dest


def mux_audio(video: str | Path, audio: str | Path, dest: str | Path, fade_out: float = 1.0) -> Path:
    """New MP4 with the video stream copied and the audio added (cut to the video length, short fade-out)."""
    dest = Path(dest)
    info = probe(video)
    dur = info.get("duration") or 0
    af = f"afade=t=out:st={max(0.0, dur - fade_out):.2f}:d={fade_out:.2f}" if dur > fade_out else "anull"
    _run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
          "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
          "-af", af, "-shortest", "-movflags", "+faststart", str(dest)])
    return dest


# ----------------------------------------------------------------------------- audio

def write_wav(samples: np.ndarray, sample_rate: int, path: str | Path) -> dict:
    """samples: float array (n,) or (n, channels) in -1..1 -> 16-bit PCM WAV. Returns {seconds, channels, sample_rate}."""
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    sf.write(str(path), arr, sample_rate, subtype="PCM_16")
    return {"seconds": round(arr.shape[0] / sample_rate, 2), "channels": arr.shape[1], "sample_rate": sample_rate}


def write_mp3(wav: str | Path, dest: str | Path, bitrate: str = "192k") -> Path:
    dest = Path(dest)
    _run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "libmp3lame", "-b:a", bitrate, str(dest)])
    return dest


def waveform_png(wav: str | Path, dest: str | Path, size: tuple[int, int] = (384, 216),
                 bg: str = "#0B0F1A", colors: tuple[str, str] = ("#EC4899", "#FACC15")) -> Path:
    """Draw the waveform as vertical bars with a pink-to-yellow gradient on the dark studio background."""
    import soundfile as sf
    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    mono = np.abs(data).mean(axis=1) if data.size else np.zeros(1, dtype=np.float32)
    w, h = size
    bars = w // 3
    chunk = max(1, len(mono) // bars)
    levels = np.array([mono[i * chunk:(i + 1) * chunk].max() if len(mono[i * chunk:(i + 1) * chunk]) else 0 for i in range(bars)])
    if levels.max() > 0:
        levels = levels / levels.max()
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    c0, c1 = Image.new("RGB", (1, 1), colors[0]).getpixel((0, 0)), Image.new("RGB", (1, 1), colors[1]).getpixel((0, 0))
    mid = h // 2
    for i, lv in enumerate(levels):
        t = i / max(1, bars - 1)
        col = tuple(int(c0[k] + (c1[k] - c0[k]) * t) for k in range(3))
        bar = max(2, int(lv * (h * 0.42)))
        x = i * 3
        d.rectangle([x, mid - bar, x + 1, mid + bar], fill=col)
    d.line([(0, mid), (w, mid)], fill=(40, 48, 70), width=1)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "PNG")
    return dest


def audio_seconds(path: str | Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return round(info.frames / info.samplerate, 2)


if __name__ == "__main__":
    import sys
    print("ffmpeg:", ffmpeg_exe())
    if len(sys.argv) > 1:
        print(probe(sys.argv[1]))
