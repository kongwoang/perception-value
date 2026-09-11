#!/usr/bin/env python
"""Profile each perception mode on this Jetson AGX Xavier.

Measures median/p95 latency, peak GPU memory and rail power under a sustained
load, so the CHEAP/FULL compute gap is a measured property of this board rather
than a FLOP estimate.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import power, runmeta                      # noqa: E402
from rap.detect import MODES, TwoFidelityDetector   # noqa: E402


class RailSampler(threading.Thread):
    def __init__(self, period=0.05):
        super().__init__(daemon=True)
        self.period, self.samples, self._halt = period, [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            self.samples.append((time.time(), power.read_power_mw()))
            time.sleep(self.period)

    def stop(self):
        self._halt.set()
        self.join(timeout=2)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        keys = self.samples[0][1].keys()
        arr = {k: np.array([s[1].get(k, np.nan) for s in self.samples]) for k in keys}
        out = {f"{k}_mw_mean": float(np.nanmean(v)) for k, v in arr.items()}
        out.update({f"{k}_mw_p95": float(np.nanpercentile(v, 95)) for k, v in arr.items()})
        out["n_samples"] = len(self.samples)
        return out


def profile_mode(det, mode, images, warmup, reps, idle_baseline):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup):
        det.detect(images[i % len(images)], mode)
    torch.cuda.synchronize()

    e2e, fwd = [], []
    sampler = RailSampler()
    sampler.start()
    t_start = time.time()
    for i in range(reps):
        img = images[i % len(images)]
        t0 = time.perf_counter()
        x, r, pad, _ = det.preprocess(img, mode)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        raw = det.forward(x, mode)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        det.postprocess(raw, mode, r, pad, img.shape)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        e2e.append((t3 - t0) * 1e3)
        fwd.append((t2 - t1) * 1e3)
    wall = time.time() - t_start
    sampler.stop()

    rails = sampler.summary()
    res = {
        "mode": mode.name, "net_h": mode.net_h, "net_w": mode.net_w,
        "pixels": mode.pixels, "reps": reps,
        "lat_e2e_ms_median": float(np.median(e2e)),
        "lat_e2e_ms_mean": float(np.mean(e2e)),
        "lat_e2e_ms_p95": float(np.percentile(e2e, 95)),
        "lat_e2e_ms_p99": float(np.percentile(e2e, 99)),
        "lat_fwd_ms_median": float(np.median(fwd)),
        "lat_fwd_ms_p95": float(np.percentile(fwd, 95)),
        "peak_mem_alloc_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_mem_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "throughput_fps": reps / wall,
        **rails,
    }
    for rail in ("GPU", "SOC", "CPU", "SYS5V"):
        k = f"{rail}_mw_mean"
        if k in rails and k in idle_baseline:
            res[f"{rail}_mw_over_idle"] = rails[k] - idle_baseline[k]
    if "GPU_mw_over_idle" in res:
        res["energy_gpu_mj_per_frame"] = res["GPU_mw_over_idle"] * res["lat_e2e_ms_median"] / 1e3
    total_over = sum(res.get(f"{r}_mw_over_idle", 0.0) for r in ("GPU", "SOC", "CPU"))
    res["module_mw_over_idle"] = total_over
    res["energy_module_mj_per_frame"] = total_over * res["lat_e2e_ms_median"] / 1e3
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/kongwoang/models/yolo/yolov8s.pt")
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--backend", default="torch", choices=["torch", "trt"])
    ap.add_argument("--images", default="", help="dir of real frames; synthetic if empty")
    ap.add_argument("--rounds", type=int, default=3, help="interleaved repeats of the mode sweep")
    args = ap.parse_args()

    run = runmeta.new_run("profile", vars(args))
    rng = np.random.default_rng(0)
    if args.images:
        import cv2
        paths = sorted(Path(args.images).rglob("*.png"))[:32]
        images = [cv2.imread(str(p)) for p in paths]
        assert images, f"no images under {args.images}"
    else:
        images = [(rng.random((375, 1242, 3)) * 255).astype(np.uint8) for _ in range(8)]

    det = TwoFidelityDetector(args.weights, half=bool(args.half), backend=args.backend)

    print("measuring idle baseline (10 s)...")
    idle = RailSampler()
    idle.start()
    time.sleep(10)
    idle.stop()
    idle_baseline = idle.summary()
    print({k: round(v, 1) for k, v in idle_baseline.items() if k.endswith("_mean")})

    rows = []
    for rnd in range(args.rounds):
        for name in args.modes:
            r = profile_mode(det, MODES[name], images, args.warmup, args.reps, idle_baseline)
            r["round"] = rnd
            rows.append(r)
            print(f"[r{rnd}] {name:10s} e2e={r['lat_e2e_ms_median']:6.2f}ms "
                  f"p95={r['lat_e2e_ms_p95']:6.2f} fwd={r['lat_fwd_ms_median']:6.2f} "
                  f"mem={r['peak_mem_alloc_mb']:6.1f}MB "
                  f"gpu+={r.get('GPU_mw_over_idle', float('nan')):7.0f}mW")

    (run / "profile_rows.json").write_text(json.dumps(rows, indent=2))
    (run / "idle_baseline.json").write_text(json.dumps(idle_baseline, indent=2))

    agg = {}
    for name in args.modes:
        sel = [r for r in rows if r["mode"] == name]
        agg[name] = {k: float(np.median([r[k] for r in sel]))
                     for k in sel[0] if isinstance(sel[0][k], (int, float))}
    (run / "profile_summary.json").write_text(json.dumps(agg, indent=2))
    print("\nwrote", run)
    return agg


if __name__ == "__main__":
    main()
