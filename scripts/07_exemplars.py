#!/usr/bin/env python
"""Render the qualitative figure: frame pairs that look equally uncertain to the cheap
model but differ sharply in how much FULL is worth.

Each panel shows the CHEAP detections, the FULL detections, and the GT objects that
CHEAP missed, shaded by how critical they are.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import geometry as G, kitti, runmeta      # noqa: E402
from rap.cache import DetCache                     # noqa: E402
from rap.paths import CACHE, FIGURES               # noqa: E402
from rap.risk import RiskConfig, match             # noqa: E402
from rap.viz import SERIES                         # noqa: E402


def _bgr(hexcolor):
    h = hexcolor.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


CHEAP_C, FULL_C, MISS_C = _bgr(SERIES[0]), _bgr(SERIES[2]), _bgr("#e34948")


def render(seq, frame, cheap_dir, full_dir, model, cfg, op_conf):
    img = cv2.imread(str(kitti.image_path(seq, frame)))
    cheap, full = DetCache(cheap_dir / f"{seq}.npz"), DetCache(full_dir / f"{seq}.npz")
    dc, dfl = cheap.det(cheap.index[frame]), full.det(full.index[frame])

    geom = G.sequence_geometry(seq)
    sel = geom["frame"] == frame
    g = geom[sel]
    g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
    crit = G.criticality_for(g, model)
    gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(np.float64)

    for det, color in ((dc, CHEAP_C), (dfl, FULL_C)):
        k = det["conf"] >= op_conf
        for b in det["xyxy"][k]:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)

    kc = dc["conf"] >= op_conf
    best, _ = match(gt, np.array(["v"] * len(gt)), dc["xyxy"][kc].astype(np.float64),
                    dc["conf"][kc], dc["coarse"][kc], cfg)
    missed = best < cfg.iou_thr
    for b, c, m in zip(gt, crit, missed):
        if not m:
            continue
        thick = 2 + int(round(4 * c))
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), MISS_C, thick)
        cv2.putText(img, f"crit {c:.2f}", (int(b[0]), max(int(b[1]) - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, MISS_C, 1, cv2.LINE_AA)
    return img


def banner(img, lines, height=58):
    bar = np.full((height, img.shape[1], 3), 252, np.uint8)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(bar, txt, (10, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    col, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="")
    ap.add_argument("--cheap", default="cheap_320")
    ap.add_argument("--full", default="full_640")
    ap.add_argument("--det", default=str(CACHE / "det"))
    ap.add_argument("--op_conf", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default=str(FIGURES / "fig7_exemplar_pairs.png"))
    args = ap.parse_args()

    run = Path(args.analysis) if args.analysis else runmeta.latest("analysis")
    ex = pd.read_csv(run / "pair_exemplars.csv").head(args.n)
    cfg, model = RiskConfig(op_conf=args.op_conf), G.PRIMARY
    cheap_dir, full_dir = Path(args.det) / args.cheap, Path(args.det) / args.full

    rows = []
    for _, r in ex.iterrows():
        panels = []
        for side in ("hi", "lo"):
            img = render(r[f"{side}_seq"], int(r[f"{side}_frame"]), cheap_dir, full_dir,
                         model, cfg, args.op_conf)
            tag = "FULL IS WORTH IT" if side == "hi" else "FULL IS WASTED"
            panels.append(banner(img, [
                (f"{tag}   seq {r[f'{side}_seq']} frame {int(r[f'{side}_frame'])}",
                 _bgr(SERIES[1]) if side == "hi" else (90, 90, 90)),
                (f"Value_task={r[f'{side}_value']:.2f}   predicted criticality={r[f'{side}_crit']:.2f}"
                 f"   mean conf={r[f'{side}_conf_mean']:.2f}   entropy sum={r[f'{side}_ent_sum']:.2f}",
                 (60, 60, 60))]))
        h = max(p.shape[0] for p in panels)
        w = max(p.shape[1] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, w - p.shape[1],
                                     cv2.BORDER_CONSTANT, value=(252, 252, 252)) for p in panels]
        rows.append(np.vstack(panels))

    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 14, 0, w - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(252, 252, 252)) for r in rows]
    canvas = np.vstack(rows)
    legend = np.full((34, canvas.shape[1], 3), 252, np.uint8)
    for i, (txt, col) in enumerate([("cheap detections", CHEAP_C), ("full detections", FULL_C),
                                    ("GT missed by cheap (thickness = criticality)", MISS_C)]):
        cv2.putText(legend, txt, (10 + i * 430, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2,
                    cv2.LINE_AA)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, np.vstack([legend, canvas]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
