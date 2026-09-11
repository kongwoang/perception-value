"""Jetson AGX Xavier power rails via sysfs INA3221.

Returns milliwatts per rail. On R35 the rails live under
/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/ as in{n}_input (mV) / curr{n}_input (mA),
with older trees exposing in_power{n}_input (mW) directly.
"""
from __future__ import annotations

import glob
from pathlib import Path


def _rail_files() -> dict[str, tuple[Path, Path | None]]:
    rails: dict[str, tuple[Path, Path | None]] = {}
    for hw in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"):
        hwp = Path(hw)
        for lbl in sorted(hwp.glob("in*_label")):
            idx = lbl.name.split("_")[0].replace("in", "")
            name = lbl.read_text().strip()
            direct = hwp / f"in_power{idx}_input"
            if direct.exists():
                rails[name] = (direct, None)
                continue
            v, c = hwp / f"in{idx}_input", hwp / f"curr{idx}_input"
            if v.exists() and c.exists():
                rails[name] = (v, c)
    return rails


RAILS = _rail_files()
AVAILABLE = bool(RAILS)


def read_power_mw() -> dict[str, float]:
    out = {}
    for name, (a, b) in RAILS.items():
        try:
            if b is None:
                out[name] = float(a.read_text().strip())
            else:
                out[name] = float(a.read_text().strip()) * float(b.read_text().strip()) / 1000.0
        except Exception:
            pass
    return out
