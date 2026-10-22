"""
System telemetry for the Admin panel: CPU, memory, disks, network, GPU, power, software and app.

Everything is read on demand; nothing is cached except the previous network counters, which
are needed to turn byte totals into per-second rates.

    python sysinfo.py          # dump everything as JSON
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psutil

import gpu_info

ROOT = Path(__file__).parent
_PROC = psutil.Process(os.getpid())
_APP_STARTED = time.time()
_net_prev: tuple[float, int, int] | None = None  # (time, bytes_sent, bytes_recv)


def _run(cmd: list[str], timeout: float = 5) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def human_bytes(n: float | None, digits: int = 1) -> str:
    if n is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.{digits if unit != 'B' else 0}f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def human_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    days, rem = divmod(td.total_seconds(), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts = [f"{int(days)}d"] if days else []
    parts += [f"{int(hours)}h", f"{int(minutes)}m"]
    return " ".join(parts)


_size_cache: dict[str, tuple[float, int]] = {}


def _dir_size(path: Path, max_files: int = 20000, ttl: float = 60.0) -> int:
    """Total size of a folder, cached for ttl seconds (the HF cache is tens of GB; walking it costs ~2 s)."""
    key = str(path)
    hit = _size_cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    total, n = 0, 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
            n += 1
            if n >= max_files:
                break
    _size_cache[key] = (time.time(), total)
    return total


# ----------------------------------------------------------------------------- cpu

def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


_threads_cache: tuple[float, int] | None = None


def _thread_count(ttl: float = 30.0) -> int:
    """System-wide thread count; walking every process costs ~2 s on Windows, so cache it."""
    global _threads_cache
    if _threads_cache and time.time() - _threads_cache[0] < ttl:
        return _threads_cache[1]
    total = 0
    for proc in psutil.process_iter(["num_threads"]):
        total += proc.info["num_threads"] or 0
    _threads_cache = (time.time(), total)
    return total


def cpu() -> dict[str, Any]:
    freq = psutil.cpu_freq()
    per_core = psutil.cpu_percent(interval=0.25, percpu=True)
    try:
        load1, load5, load15 = psutil.getloadavg()
    except (OSError, AttributeError):
        load1 = load5 = load15 = None
    return {
        "name": _cpu_name(),
        "architecture": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "freq_current_mhz": round(freq.current) if freq else None,
        "freq_max_mhz": round(freq.max) if freq and freq.max else None,
        "usage_pct": round(sum(per_core) / len(per_core), 1) if per_core else None,
        "per_core_pct": per_core,
        "load_1m": round(load1, 2) if load1 is not None else None,
        "load_5m": round(load5, 2) if load5 is not None else None,
        "load_15m": round(load15, 2) if load15 is not None else None,
        "processes": len(psutil.pids()),
        "threads": _thread_count(),
        "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(timespec="seconds"),
        "uptime": human_duration(time.time() - psutil.boot_time()),
    }


# ----------------------------------------------------------------------------- memory & disks

def memory() -> dict[str, Any]:
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    return {
        "total": vm.total, "used": vm.used, "available": vm.available, "percent": vm.percent,
        "swap_total": sw.total, "swap_used": sw.used, "swap_percent": sw.percent,
    }


def disks() -> dict[str, Any]:
    parts = []
    for p in psutil.disk_partitions(all=False):
        try:
            u = psutil.disk_usage(p.mountpoint)
        except (PermissionError, OSError):
            continue
        parts.append({"mount": p.mountpoint, "fs": p.fstype, "total": u.total, "used": u.used,
                      "free": u.free, "percent": u.percent})
    io = psutil.disk_io_counters()
    hf_home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    project = {
        "outputs": _dir_size(ROOT / "outputs"),
        "outputs_files": sum(1 for p in (ROOT / "outputs").iterdir() if p.suffix.lower() in (".png", ".mp4", ".wav")) if (ROOT / "outputs").exists() else 0,
        "models_dir": _dir_size(ROOT / "models"),
        "hf_cache": _dir_size(hf_home / "hub"),
        "hf_cache_path": str(hf_home / "hub"),
        "database": (ROOT / "generations.db").stat().st_size if (ROOT / "generations.db").exists() else 0,
    }
    return {
        "partitions": parts,
        "read_total": io.read_bytes if io else None,
        "write_total": io.write_bytes if io else None,
        "project": project,
    }


# ----------------------------------------------------------------------------- network

def network() -> dict[str, Any]:
    global _net_prev
    now = time.time()
    io = psutil.net_io_counters()
    up_rate = down_rate = None
    if _net_prev:
        dt = max(now - _net_prev[0], 1e-6)
        up_rate = (io.bytes_sent - _net_prev[1]) / dt
        down_rate = (io.bytes_recv - _net_prev[2]) / dt
    _net_prev = (now, io.bytes_sent, io.bytes_recv)

    addrs, stats = psutil.net_if_addrs(), psutil.net_if_stats()
    per_nic = psutil.net_io_counters(pernic=True)
    interfaces = []
    for name, alist in addrs.items():
        st = stats.get(name)
        ipv4 = [a.address for a in alist if a.family == socket.AF_INET]
        ipv6 = [a.address for a in alist if a.family == socket.AF_INET6]
        mac = [a.address for a in alist if a.family == psutil.AF_LINK]
        nic = per_nic.get(name)
        interfaces.append({
            "name": name,
            "up": bool(st.isup) if st else None,
            "speed_mbps": st.speed if st and st.speed else None,
            "mtu": st.mtu if st else None,
            "ipv4": ", ".join(ipv4), "ipv6": ", ".join(ipv6)[:60], "mac": ", ".join(mac),
            "sent": nic.bytes_sent if nic else None, "recv": nic.bytes_recv if nic else None,
        })
    interfaces.sort(key=lambda i: (not i["up"], -(i["recv"] or 0)))

    try:
        conns = psutil.net_connections(kind="inet")
        established = sum(1 for c in conns if c.status == psutil.CONN_ESTABLISHED)
        listening = sorted({c.laddr.port for c in conns if c.status == psutil.CONN_LISTEN and c.laddr})
    except (psutil.AccessDenied, OSError):
        established, listening = None, []

    primary_ip = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        primary_ip = s.getsockname()[0]
        s.close()
    except OSError:
        pass

    return {
        "hostname": socket.gethostname(),
        "primary_ip": primary_ip,
        "bytes_sent": io.bytes_sent, "bytes_recv": io.bytes_recv,
        "packets_sent": io.packets_sent, "packets_recv": io.packets_recv,
        "errors_in": io.errin, "errors_out": io.errout, "drops_in": io.dropin, "drops_out": io.dropout,
        "up_rate": up_rate, "down_rate": down_rate,
        "established": established, "listening_ports": listening[:30],
        "interfaces": interfaces,
    }


def internet(timeout: float = 2.0) -> dict[str, Any]:
    """Reachability of the services the app depends on (DNS + TCP connect only, no payload)."""
    out = {}
    for host in ("huggingface.co", "github.com", "pypi.org"):
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, 443), timeout=timeout):
                out[host] = round((time.perf_counter() - t0) * 1000)
        except OSError:
            out[host] = None
    return out


# ----------------------------------------------------------------------------- gpu & power

def gpu() -> dict[str, Any]:
    static = gpu_info.static_info()
    live = gpu_info.snapshot()
    mem = gpu_info.torch_memory() if static.get("cuda_available") else {}
    procs = []
    for line in _run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader,nounits"]).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            procs.append({"pid": parts[0], "vram_mb": parts[1], "name": Path(parts[2]).name})
    extra = {}
    raw = _run(["nvidia-smi", "--id=0", "--query-gpu=pci.bus_id,vbios_version,clocks.max.sm,clocks.max.mem,"
                "temperature.memory,encoder.stats.sessionCount,pcie.link.gen.current,pcie.link.width.current,"
                "power.max_limit,power.default_limit,clocks_throttle_reasons.active", "--format=csv,noheader,nounits"])
    if raw:
        v = [x.strip() for x in raw.split(",")]
        keys = ["pci_bus", "vbios", "max_sm_clock_mhz", "max_mem_clock_mhz", "mem_temp_c", "encoder_sessions",
                "pcie_gen", "pcie_width", "power_max_limit_w", "power_default_limit_w", "throttle_reasons"]
        extra = dict(zip(keys, v))
    return {"static": static, "live": live, "torch": mem, "processes": procs, "extra": extra}


def power(gpu_live: dict[str, Any] | None = None) -> dict[str, Any]:
    live = gpu_live if gpu_live is not None else gpu_info.snapshot()
    plan = None
    if sys.platform == "win32":
        raw = _run(["powercfg", "/getactivescheme"])
        if "(" in raw:
            plan = raw[raw.find("(") + 1: raw.rfind(")")]
    bat = None
    try:
        b = psutil.sensors_battery()
        if b:
            bat = {"percent": b.percent, "plugged": b.power_plugged,
                   "minutes_left": None if b.secsleft in (psutil.POWER_TIME_UNLIMITED, psutil.POWER_TIME_UNKNOWN) else b.secsleft // 60}
    except (AttributeError, OSError):
        pass
    draw, limit = live.get("power_w"), live.get("power_limit_w")
    return {
        "gpu_draw_w": draw,
        "gpu_limit_w": limit,
        "gpu_pct_of_limit": round(draw / limit * 100, 1) if draw and limit else None,
        "gpu_pstate": live.get("pstate"),
        "gpu_fan_pct": live.get("fan_pct"),
        "windows_power_plan": plan,
        "battery": bat,
    }


# ----------------------------------------------------------------------------- software & app

def software() -> dict[str, Any]:
    import diffusers
    import gradio
    import torch
    import transformers
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "gradio": gradio.__version__,
        "psutil": psutil.__version__,
        "driver": gpu_info.snapshot().get("driver_version"),
        "in_container": Path("/.dockerenv").exists() or os.environ.get("FRAZILE_IN_DOCKER") == "1",
    }


def app(loaded_model: str | None = None) -> dict[str, Any]:
    with _PROC.oneshot():
        rss = _PROC.memory_info().rss
        threads = _PROC.num_threads()
        cpu_pct = _PROC.cpu_percent(interval=None)
        created = _PROC.create_time()
    return {
        "pid": _PROC.pid,
        "started": datetime.fromtimestamp(created).isoformat(timespec="seconds"),
        "uptime": human_duration(time.time() - created),
        "rss": rss,
        "threads": threads,
        "cpu_pct": cpu_pct,
        "loaded_model": loaded_model,
        "working_dir": str(ROOT),
    }


def everything(loaded_model: str | None = None, check_internet: bool = False) -> dict[str, Any]:
    g = gpu()
    data = {
        "cpu": cpu(), "memory": memory(), "disks": disks(), "network": network(),
        "gpu": g, "power": power(g["live"]), "software": software(), "app": app(loaded_model),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }
    if check_internet:
        data["internet"] = internet()
    return data


if __name__ == "__main__":
    print(json.dumps(everything(check_internet=True), indent=2, default=str))
