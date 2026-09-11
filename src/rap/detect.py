"""Two-fidelity detection with full class-score retention.

CHEAP and FULL are the *same weights and the same architecture* run at different
input resolutions, so any accuracy difference is attributable to perception compute
and not to a change of model.

Ultralytics' own `predict` discards the per-box class distribution, which we need
for entropy/margin uncertainty features, so inference is driven manually through
letterbox -> forward -> NMS while keeping the full score vector of every surviving
box.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torchvision

# COCO ids that are traffic participants. Everything else is dropped before NMS so
# the detector is scored only on what the driving task cares about.
COCO_TRAFFIC = {0: "person", 1: "cyclist", 2: "vehicle", 3: "cyclist",
                5: "vehicle", 7: "vehicle"}
COCO_KEEP = sorted(COCO_TRAFFIC)


@dataclass(frozen=True)
class Mode:
    name: str
    net_h: int
    net_w: int
    conf: float = 0.10          # low operating threshold; analysis re-thresholds as needed
    iou: float = 0.65
    max_det: int = 100
    arch: str = "yolo"          # "yolo" (dense head + NMS) or "detr" (set prediction)

    @property
    def pixels(self) -> int:
        return self.net_h * self.net_w


# KITTI frames are ~1242x375 (3.3:1). These keep that aspect and land on /32.
MODES = {
    "cheap_320": Mode("cheap_320", 96, 320),
    "cheap_384": Mode("cheap_384", 128, 384),
    "cheap_512": Mode("cheap_512", 160, 512),
    "full_640": Mode("full_640", 192, 640),
    "full_960": Mode("full_960", 288, 960),
    # nuScenes CAM_FRONT is 1600x900 (16:9). These keep that aspect and preserve the
    # KITTI pair's 4x pixel ratio, so the fidelity gap is comparable across datasets.
    "ns_cheap_320": Mode("ns_cheap_320", 192, 320),
    "ns_full_640": Mode("ns_full_640", 384, 640),
    # Detector-B. RT-DETR is a set-prediction transformer: no NMS, and ultralytics only
    # supports square input for it, so these are square and the aspect padding differs
    # from the YOLO modes. The 4x pixel ratio between the pair is preserved.
    "rt_cheap_320": Mode("rt_cheap_320", 320, 320, arch="detr", max_det=300),
    "rt_full_640": Mode("rt_full_640", 640, 640, arch="detr", max_det=300),
    "rt_mid_480": Mode("rt_mid_480", 480, 480, arch="detr", max_det=300),
}


def letterbox(img: np.ndarray, net_h: int, net_w: int):
    """Resize keeping aspect, pad to (net_h, net_w). Returns image, scale, (padx, pady)."""
    h, w = img.shape[:2]
    r = min(net_h / h, net_w / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padx, pady = (net_w - new_w) / 2, (net_h - new_h) / 2
    top, bottom = int(round(pady - 0.1)), int(round(pady + 0.1))
    left, right = int(round(padx - 0.1)), int(round(padx + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, (left, top)


class TwoFidelityDetector:
    """Runs both fidelities from one set of weights.

    backend="trt" executes the same graph through the per-mode TensorRT engine that
    the profiler measures; backend="torch" is the eager reference used to validate
    the engines.
    """

    def __init__(self, weights: str, device: str = "cuda:0", half: bool = True,
                 backend: str = "torch", engine_dir: str | None = None):
        self.device = torch.device(device)
        self.backend = backend
        self.half = half and self.device.type == "cuda" and backend == "torch"
        self._keep = torch.tensor(COCO_KEEP, device=self.device)
        self.weights = weights
        self._engines: dict[str, object] = {}
        self.engine_dir = engine_dir
        if backend == "torch":
            from ultralytics import RTDETR, YOLO
            self.yolo = (RTDETR if "rtdetr" in str(weights).lower() else YOLO)(weights)
            self.net = self.yolo.model.to(self.device).eval()
            if self.half:
                self.net = self.net.half()
            self.names = self.yolo.names
        else:
            from pathlib import Path as _P
            self.names = None
            self.engine_dir = engine_dir or str(_P(weights).parent / "engines")

    def engine_for(self, mode: "Mode"):
        from pathlib import Path as _P
        from .trt import TRTModule
        if mode.name not in self._engines:
            stem = _P(self.weights).stem
            path = _P(self.engine_dir) / f"{stem}_{mode.name}.engine"
            self._engines[mode.name] = TRTModule(path, device=str(self.device))
        return self._engines[mode.name]

    # -- preprocessing -------------------------------------------------------
    def preprocess(self, img_bgr: np.ndarray, mode: Mode):
        lb, r, pad = letterbox(img_bgr, mode.net_h, mode.net_w)
        x = torch.from_numpy(np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)))
        x = x.to(self.device, non_blocking=True)
        x = (x.half() if self.half else x.float()) / 255.0
        return x.unsqueeze(0), r, pad, lb

    @torch.no_grad()
    def forward(self, x: torch.Tensor, mode: "Mode" = None) -> torch.Tensor:
        if self.backend == "trt":
            return self.engine_for(mode)(x)
        out = self.net(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    # -- postprocessing ------------------------------------------------------
    def postprocess(self, raw: torch.Tensor, mode: Mode, r: float, pad, img_shape):
        """raw: (1, 4+nc, N). Returns dict of numpy arrays in ORIGINAL image pixels."""
        p = raw[0].float()                      # (4+nc, N)
        boxes_xywh, cls_scores = p[:4], p[4:]   # (4,N), (nc,N)
        traffic = cls_scores[self._keep]        # (k, N)
        best, best_i = traffic.max(0)
        n_cand_raw = int((cls_scores.max(0).values > mode.conf).sum())
        keep = best > mode.conf
        n_cand = int(keep.sum())
        if n_cand == 0:
            return _empty_dets(n_cand_raw, 0)

        boxes_xywh = boxes_xywh[:, keep].T      # (n,4) cx,cy,w,h in net pixels
        traffic = traffic[:, keep].T            # (n,k)
        full_scores = cls_scores[:, keep].T     # (n,nc)
        best, best_i = best[keep], best_i[keep]

        xy = boxes_xywh[:, :2]
        wh = boxes_xywh[:, 2:]
        xyxy = torch.cat([xy - wh / 2, xy + wh / 2], dim=1)

        coarse_id = self._keep[best_i]
        sel = torchvision.ops.batched_nms(xyxy, best, coarse_id, mode.iou)[: mode.max_det]
        n_post = int(sel.numel())

        xyxy = xyxy[sel]
        xyxy[:, [0, 2]] -= pad[0]
        xyxy[:, [1, 3]] -= pad[1]
        xyxy /= r
        h0, w0 = img_shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, w0 - 1)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, h0 - 1)

        tr = traffic[sel]
        top2 = tr.topk(min(2, tr.shape[1]), dim=1).values
        margin = (top2[:, 0] - top2[:, 1]) if top2.shape[1] > 1 else top2[:, 0]

        fs = full_scores[sel].clamp_min(1e-9)
        pnorm = fs / fs.sum(1, keepdim=True)
        entropy = -(pnorm * pnorm.log()).sum(1)

        conf = best[sel].clamp(1e-6, 1 - 1e-6)
        binent = -(conf * conf.log() + (1 - conf) * (1 - conf).log())

        coarse = np.array([COCO_TRAFFIC[int(c)] for c in self._keep[best_i[sel]].cpu().numpy()])
        return {
            "xyxy": xyxy.cpu().numpy().astype(np.float32),
            "conf": best[sel].cpu().numpy().astype(np.float32),
            "coco_id": self._keep[best_i[sel]].cpu().numpy().astype(np.int16),
            "coarse": coarse,
            "entropy": entropy.cpu().numpy().astype(np.float32),
            "margin": margin.cpu().numpy().astype(np.float32),
            "binent": binent.cpu().numpy().astype(np.float32),
            "n_cand_raw": n_cand_raw,
            "n_cand": n_cand,
            "n_post": n_post,
        }

    # -- set-prediction postprocessing (RT-DETR) -----------------------------
    def postprocess_detr(self, raw: torch.Tensor, mode: Mode, r: float, pad, img_shape):
        """raw: (1, num_queries, 4+nc), boxes normalised cxcywh in network-input space.

        No NMS: the decoder already emits a set. Scores are per-class sigmoids, so a box
        may be kept under several classes; we take its best traffic class, as for YOLO.
        """
        p = raw[0].float()
        boxes, cls_scores = p[:, :4], p[:, 4:]
        traffic = cls_scores[:, self._keep]
        best, best_i = traffic.max(1)
        n_cand_raw = int((cls_scores.max(1).values > mode.conf).sum())
        keep = best > mode.conf
        n_cand = int(keep.sum())
        if n_cand == 0:
            return _empty_dets(n_cand_raw, 0)

        b = boxes[keep]
        xy = b[:, :2] * torch.tensor([mode.net_w, mode.net_h], device=b.device)
        wh = b[:, 2:] * torch.tensor([mode.net_w, mode.net_h], device=b.device)
        xyxy = torch.cat([xy - wh / 2, xy + wh / 2], dim=1)

        sel = torch.argsort(best[keep], descending=True)[: mode.max_det]
        xyxy = xyxy[sel]
        xyxy[:, [0, 2]] -= pad[0]
        xyxy[:, [1, 3]] -= pad[1]
        xyxy /= r
        h0, w0 = img_shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, w0 - 1)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, h0 - 1)

        tr = traffic[keep][sel]
        top2 = tr.topk(min(2, tr.shape[1]), dim=1).values
        margin = (top2[:, 0] - top2[:, 1]) if top2.shape[1] > 1 else top2[:, 0]
        fs = cls_scores[keep][sel].clamp_min(1e-9)
        pnorm = fs / fs.sum(1, keepdim=True)
        entropy = -(pnorm * pnorm.log()).sum(1)
        conf = best[keep][sel].clamp(1e-6, 1 - 1e-6)
        binent = -(conf * conf.log() + (1 - conf) * (1 - conf).log())
        coarse = np.array([COCO_TRAFFIC[int(c)]
                           for c in self._keep[best_i[keep][sel]].cpu().numpy()])
        return {
            "xyxy": xyxy.cpu().numpy().astype(np.float32),
            "conf": conf.cpu().numpy().astype(np.float32),
            "coco_id": self._keep[best_i[keep][sel]].cpu().numpy().astype(np.int16),
            "coarse": coarse,
            "entropy": entropy.cpu().numpy().astype(np.float32),
            "margin": margin.cpu().numpy().astype(np.float32),
            "binent": binent.cpu().numpy().astype(np.float32),
            "n_cand_raw": n_cand_raw, "n_cand": n_cand, "n_post": int(len(xyxy)),
        }

    @torch.no_grad()
    def detect(self, img_bgr: np.ndarray, mode: Mode):
        x, r, pad, lb = self.preprocess(img_bgr, mode)
        raw = self.forward(x, mode)
        post = self.postprocess_detr if mode.arch == "detr" else self.postprocess
        det = post(raw, mode, r, pad, img_bgr.shape)
        return det, lb


def _empty_dets(n_cand_raw: int, n_cand: int):
    return {
        "xyxy": np.zeros((0, 4), np.float32), "conf": np.zeros(0, np.float32),
        "coco_id": np.zeros(0, np.int16), "coarse": np.array([], dtype="U8"),
        "entropy": np.zeros(0, np.float32), "margin": np.zeros(0, np.float32),
        "binent": np.zeros(0, np.float32),
        "n_cand_raw": n_cand_raw, "n_cand": n_cand, "n_post": 0,
    }
