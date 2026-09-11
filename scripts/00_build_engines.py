#!/usr/bin/env python
"""Export one ONNX graph per perception mode and build an FP16 TensorRT engine.

PyTorch eager on Xavier has a ~19 ms per-layer launch-overhead floor that swamps
the resolution effect entirely, so a latency study in eager mode would be measuring
Python, not perception compute. TensorRT is the deployment path on this board and
restores the compute/latency relationship the experiment depends on.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap.detect import MODES  # noqa: E402

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/kongwoang/models/yolo/yolov8s.pt")
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--outdir", default="/home/kongwoang/models/yolo/engines")
    ap.add_argument("--workspace_mb", type=int, default=4096)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.weights).stem

    from ultralytics import YOLO

    for name in args.modes:
        m = MODES[name]
        onnx_path = out / f"{stem}_{name}.onnx"
        engine_path = out / f"{stem}_{name}.engine"
        if not onnx_path.exists():
            print(f"[onnx] {name} -> {onnx_path}")
            y = YOLO(args.weights)
            produced = y.export(format="onnx", imgsz=(m.net_h, m.net_w), opset=12,
                                simplify=False, dynamic=False, half=False, batch=1)
            Path(produced).replace(onnx_path)
        if engine_path.exists():
            print(f"[trt ] {name} already built")
            continue
        print(f"[trt ] {name} -> {engine_path}")
        cmd = [TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--fp16",
               f"--workspace={args.workspace_mb}", "--noDataTransfers", "--useSpinWait"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not engine_path.exists():
            print(r.stdout[-3000:])
            print(r.stderr[-3000:])
            raise SystemExit(f"trtexec failed for {name}")
        for line in r.stdout.splitlines():
            if "mean:" in line.lower() and "gpu compute" in line.lower().replace("  ", " "):
                print("   ", line.strip())
    print("done")


if __name__ == "__main__":
    main()
