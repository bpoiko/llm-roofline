"""Host environment capture for env.json.

Only reads an allowlist of fields. Never records serial numbers, hardware UUIDs,
provisioning IDs, hostnames, or usernames."""

import json
import platform
import re
import subprocess
from pathlib import Path


def _run(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _sysctl(name: str) -> str:
    return _run("sysctl", "-n", name)


def _int(s: str):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def power_source() -> str | None:
    """'AC Power' or 'Battery Power' on a Mac; None where there's no pmset.
    A first sweep ran on battery until macOS forced Low Power Sleep at 1%, which
    looked like a 9-hour hang and throttled the runs before it."""
    m = re.search(r"drawing from '([^']+)'", _run("pmset", "-g", "batt"))
    return m.group(1) if m else None


def on_battery() -> bool:
    return power_source() == "Battery Power"


def battery_pct() -> int | None:
    m = re.search(r"(\d+)%;", _run("pmset", "-g", "batt"))
    return int(m.group(1)) if m else None


def charger_watts() -> int | None:
    """Connected charger's rating. "AC Power" alone isn't enough: on a 30 W charger an
    M4 Pro under GPU load drained 60% -> 9% while plugged in and throttled hard."""
    m = re.search(r"Wattage \(W\): (\d+)", _run("system_profiler", "SPPowerDataType"))
    return int(m.group(1)) if m else None


ENERGY_MODES = {0: "automatic", 1: "low power", 2: "high power"}


def energy_mode() -> str | None:
    """macOS Energy Mode for the current power source (System Settings > Battery).
    Automatic may lower clocks under sustained load; set High Power for sweeps with
    `sudo pmset -c powermode 2`. It doesn't add watts: an undersized charger still drains."""
    m = re.search(r"^\s*powermode\s+(\d+)", _run("pmset", "-g"), re.M)
    return ENERGY_MODES.get(int(m.group(1)), m.group(1)) if m else None


def power_problem(start_pct: int | None, max_drop: int = 10) -> str | None:
    """Why measurements would be power-limited right now, or None. A battery falling while
    plugged in means the charger can't keep up with the load, whatever its rating."""
    if on_battery():
        return "running on battery"
    pct = battery_pct()
    if start_pct is not None and pct is not None and start_pct - pct >= max_drop:
        return f"battery fell {start_pct}% -> {pct}% while plugged in: the charger can't keep up with the load"
    return None


def _darwin() -> dict:
    gpu = {}
    try:
        displays = json.loads(_run("system_profiler", "SPDisplaysDataType", "-json"))["SPDisplaysDataType"]
        gpu = displays[0] if displays else {}
    except (ValueError, KeyError):
        pass
    mem = _int(_sysctl("hw.memsize"))
    return {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "cpu_cores_total": _int(_sysctl("hw.physicalcpu")),
        "cpu_cores_performance": _int(_sysctl("hw.perflevel0.physicalcpu")),
        "cpu_cores_efficiency": _int(_sysctl("hw.perflevel1.physicalcpu")),
        "gpu_name": gpu.get("sppci_model"),
        "gpu_cores": _int(gpu.get("sppci_cores")),
        "ram_gb": round(mem / 2**30) if mem else None,
        # 0 = macOS default GPU wired-memory limit; raised values change what fits
        "gpu_wired_limit_mb": _int(_sysctl("iogpu.wired_limit_mb")),
        "power_source": power_source(),
        "charger_w": charger_watts(),
        "battery_pct": battery_pct(),
        "energy_mode": energy_mode(),
        "os": "macOS",
        "os_version": _run("sw_vers", "-productVersion"),
        "os_build": _run("sw_vers", "-buildVersion"),
    }


def _linux() -> dict:
    info = {"os": "Linux", "os_version": platform.release()}
    try:
        with open("/proc/meminfo") as f:
            kb = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1))
        info["ram_gb"] = round(kb / 2**20)
    except (OSError, AttributeError):
        pass
    try:
        with open("/proc/cpuinfo") as f:
            m = re.search(r"model name\s*:\s*(.+)", f.read())
        info["chip"] = m.group(1).strip() if m else None
    except OSError:
        pass
    # name/memory/driver only; deliberately not uuid or serial
    smi = _run("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits")
    if smi:
        name, mem_mib, driver = [s.strip() for s in smi.splitlines()[0].split(",")]
        info.update(gpu_name=name, gpu_mem_gb=round(int(mem_mib) / 1024), nvidia_driver=driver)
    return info


def tilde(s: str) -> str:
    """Home-relative path: portable across machines, and no username in published results."""
    home = str(Path.home())
    return "~" + s[len(home):] if s == home or s.startswith(home + "/") else s


def capture(runtime: dict) -> dict:
    """runtime: runtime-specific fields, e.g. {"name": "ollama", "version": ..., "commit": ...}."""
    system = platform.system()
    host = _darwin() if system == "Darwin" else _linux() if system == "Linux" else {"os": system}
    host["arch"] = platform.machine()
    host["python"] = platform.python_version()
    return {"host": host, "runtime": runtime}


def default_machine_slug(host: dict) -> str:
    """apple-silicon: chip + RAM (m4pro-24gb); discrete GPU: GPU + VRAM (rtx5070ti-16gb)."""
    if host.get("gpu_mem_gb"):
        name, mem = host.get("gpu_name") or "gpu", host["gpu_mem_gb"]
    else:
        name, mem = host.get("chip") or "unknown", host.get("ram_gb")
    name = re.sub(r"^apple\s+|\(.*?\)|nvidia\s+|geforce\s+", "", name.lower())
    slug = re.sub(r"[^a-z0-9]+", "", name)
    return f"{slug}-{mem}gb" if mem else slug
