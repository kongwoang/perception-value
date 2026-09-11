"""Every run directory records enough to reproduce it: config, git commit, environment."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

from .paths import RAW, REPO


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def environment() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    try:
        import torch
        env.update({"torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception:
        pass
    for path, key in [("/etc/nv_tegra_release", "tegra"),
                      ("/sys/devices/gpu.0/devfreq/17000000.gv11b/cur_freq", "gpu_freq_hz")]:
        try:
            env[key] = Path(path).read_text().strip()
        except Exception:
            pass
    try:
        env["nvpmodel"] = subprocess.check_output(["nvpmodel", "-q"], text=True,
                                                  stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    return env


def new_run(tag: str, config: dict) -> Path:
    run = RAW / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (run / "environment.json").write_text(json.dumps(environment(), indent=2))
    return run


def latest(tag: str) -> Path:
    runs = sorted(RAW.glob(f"*_{tag}"))
    if not runs:
        raise FileNotFoundError(f"no run with tag {tag!r} under {RAW}")
    return runs[-1]
