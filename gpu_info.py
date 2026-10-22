"""
GPU details: static facts about the card plus a live snapshot (temperature, power, clocks, load).

Static facts come from torch; live readings come from nvidia-smi, which ships with the driver.
"""

import platform
import subprocess
from typing import Any

import torch

_SMI_FIELDS = [
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "clocks.sm",
    "clocks.mem",
    "fan.speed",
    "driver_version",
    "pstate",
]


def _num(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def snapshot() -> dict[str, Any]:
    """Live readings from nvidia-smi for GPU 0. Returns {} if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--id=0", f"--query-gpu={','.join(_SMI_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    vals = [v.strip() for v in out.split(",")]
    if len(vals) != len(_SMI_FIELDS):
        return {}
    raw = dict(zip(_SMI_FIELDS, vals))
    return {
        "temperature_c": _num(raw["temperature.gpu"]),
        "power_w": _num(raw["power.draw"]),
        "power_limit_w": _num(raw["power.limit"]),
        "gpu_util_pct": _num(raw["utilization.gpu"]),
        "mem_util_pct": _num(raw["utilization.memory"]),
        "mem_used_mb": _num(raw["memory.used"]),
        "mem_total_mb": _num(raw["memory.total"]),
        "sm_clock_mhz": _num(raw["clocks.sm"]),
        "mem_clock_mhz": _num(raw["clocks.mem"]),
        "fan_pct": _num(raw["fan.speed"]),
        "driver_version": raw["driver_version"],
        "pstate": raw["pstate"],
    }


def static_info() -> dict[str, Any]:
    """Facts that do not change between runs."""
    if not torch.cuda.is_available():
        return {"gpu_name": "none", "cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    import diffusers
    return {
        "cuda_available": True,
        "gpu_name": props.name,
        "vram_total_gb": round(props.total_memory / 1024**3, 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "multiprocessors": props.multi_processor_count,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
    }


def torch_memory() -> dict[str, float]:
    """Current torch allocator numbers in GB."""
    return {
        "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "vram_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"static": static_info(), "live": snapshot(), "torch": torch_memory()}, indent=2))
