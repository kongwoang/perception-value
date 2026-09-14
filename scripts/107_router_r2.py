#!/usr/bin/env python
"""Task 2, R2: a raw-pixel weak-skipping router -- MobileNetV2-style CNN at 128 x 128.

Pre-registered in RESEARCH_LOG.md (2026-09-14, "Task 2 pre-registration").

  input    the CHEAP camera image (nuScenes CAM_FRONT, KITTI image_02), resized to 128 x 128,
           ImageNet-normalised, no augmentation
  network  torchvision MobileNetV2, width multiplier in {1.0, ..., 1.4} closest to 0.15 GFLOPs at
           128 x 128 (counted before training); one network per dataset, one sigmoid head per cell
  loss     masked BCE on 1[V>0], pos_weight = negatives / positives per head on train + val
  training from scratch (no cached weights exist), AdamW lr 1e-3, wd 1e-4, batch 128, 30 epochs,
           cosine schedule, seed 0, train + val frames, no early stopping or selection
  export   ONNX -> TensorRT FP16 (trtexec); test scores come from the engine, and the Spearman
           correlation with the PyTorch scores is reported per head
  scoring  eta at 10/20/30/50 on test with the benchmark's evaluate(), paired against random

Stages run in separate processes so the nuScenes tables and the GPU job never share memory:
  --stage cache   decode and resize every image once (nuScenes paths need the nuScenes tables)
  --stage train   labels, FLOP count, training, export, engine build, scoring
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta                                                         # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402

SIDE = 128
CACHE_DIR = Path(CACHE) / "router_r2"
KITTI_IMG = Path("/home/kongwoang/datasets/kitti_tracking/training/image_02")
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
WIDTHS = (1.0, 1.1, 1.2, 1.3, 1.4)
TARGET_FLOPS = 0.15e9
EPOCHS, BATCH, LR, WD = 30, 128, 1e-3, 1e-4
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def stage_cache():
    import cv2
    from rap.cache import DetCache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, det in (("KITTI", Path(CACHE) / "det" / "cheap_320"), ("nuScenes", Path(CACHE) / "nusc_det_tv" / "ns_cheap_320")):
        out = CACHE_DIR / f"{dataset}.npz"
        if out.exists():
            print("  exists", out); continue
        if dataset == "nuScenes":
            from rap.nusc import NuScenesDB
            db = NuScenesDB("/home/kongwoang/datasets/nuscenes/trainval", "v1.0-trainval")
            name_to_scene = {db.scene_name(s): s for s in db.scenes}
        seqs, frames, imgs = [], [], []
        t0 = time.time()
        for f in sorted(det.glob("*.npz")):
            c = DetCache(f)
            toks = db.samples(name_to_scene[f.stem]) if dataset == "nuScenes" else None
            for fr in c.frames:
                fr = int(fr)
                path = (db.image_path(toks[fr]) if dataset == "nuScenes" else KITTI_IMG / f.stem / f"{fr:06d}.png")
                im = cv2.imread(str(path), cv2.IMREAD_COLOR)
                assert im is not None, path
                imgs.append(cv2.resize(im, (SIDE, SIDE), interpolation=cv2.INTER_AREA)[:, :, ::-1])
                seqs.append(f.stem); frames.append(fr)
        np.savez_compressed(out, seq=np.array(seqs, "U32"), frame=np.array(frames), img=np.stack(imgs))
        print(f"  {dataset}: {len(imgs)} images cached in {time.time() - t0:.0f}s -> {out}")


def count_flops(model, side=SIDE):
    import torch
    flops = []

    def conv_hook(m, i, o):
        k = m.kernel_size[0] * m.kernel_size[1]
        flops.append(o.shape[2] * o.shape[3] * (m.in_channels // m.groups) * k * m.out_channels)

    def lin_hook(m, i, o):
        flops.append(m.in_features * m.out_features)
    hs = [mod.register_forward_hook(conv_hook) for mod in model.modules() if isinstance(mod, torch.nn.Conv2d)]
    hs += [mod.register_forward_hook(lin_hook) for mod in model.modules() if isinstance(mod, torch.nn.Linear)]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, side, side))
    for h in hs:
        h.remove()
    return float(sum(flops))


def labels(dataset, splits):
    """Frame table with V and CHEAP cost per cell; cells are built once (their features are costly)."""
    _s = importlib.util.spec_from_file_location("t92", ROOT / "scripts" / "92_benchmark_table.py")
    t92 = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(t92)
    cells = list((t92.nuscenes_cells if dataset == "nuScenes" else t92.kitti_cells)(splits))
    # the braking cell covers every frame (plan cells only frames with trajectory truth)
    full = max(cells, key=lambda c: len(c["d"]))
    base = full["d"][["seq", "frame", "unit", "split"]].astype({"seq": str, "frame": int, "unit": str}).copy()
    heads = []
    for c in cells:
        name = f"{c['geometry']}__{c['system']}__{c['target']}"
        d = c["d"][["seq", "frame"]].astype({"seq": str, "frame": int}).copy()
        d[f"V__{name}"] = (c["d"][c["cheap"]] - c["d"][c["full"]]).to_numpy(float)
        d[f"Jc__{name}"] = c["d"][c["cheap"]].to_numpy(float)
        base = base.merge(d, on=["seq", "frame"], how="left", validate="one_to_one")
        heads.append(name)
    return base, heads, t92


def stage_train(args):
    import torch, torchvision
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    run = runmeta.new_run(args.tag, vars(args))
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    dev = torch.device("cuda:0")
    rows_out = []
    rng = np.random.default_rng(0)
    for dataset in ("nuScenes", "KITTI"):
        z = np.load(CACHE_DIR / f"{dataset}.npz", allow_pickle=False)
        imgs = pd.DataFrame({"seq": z["seq"].astype(str), "frame": z["frame"].astype(int), "_i": np.arange(len(z["seq"]))})
        lab, heads, t92 = labels(dataset, splits)
        lab = lab.merge(imgs, on=["seq", "frame"], how="left", validate="one_to_one")
        assert lab._i.notna().all(), "an image is missing from the cache"
        X = z["img"][lab._i.to_numpy(int)]
        Y = np.stack([np.where(np.isnan(lab[f"V__{h}"]), np.nan, (lab[f"V__{h}"] > t92.EPS).astype(float)) for h in heads], 1)
        fit = lab.split.isin(["train", "val"]).to_numpy()
        test = (lab.split == "test").to_numpy()

        flops = {w: count_flops(torchvision.models.mobilenet_v2(width_mult=w, num_classes=len(heads))) for w in WIDTHS}
        width = min(WIDTHS, key=lambda w: abs(flops[w] - TARGET_FLOPS))
        model = torchvision.models.mobilenet_v2(width_mult=width, num_classes=len(heads)).to(dev)
        print(f"  {dataset}: {len(heads)} heads, width {width} -> {flops[width] / 1e9:.3f} GFLOPs "
              f"({', '.join(f'{w}:{f / 1e9:.3f}' for w, f in flops.items())})", flush=True)

        yf = Y[fit]
        pos = np.nansum(yf, 0); neg = np.sum(~np.isnan(yf), 0) - pos
        pos_weight = torch.tensor(neg / np.maximum(pos, 1), dtype=torch.float32, device=dev)
        mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
        Xf = torch.from_numpy(X[fit]).permute(0, 3, 1, 2).contiguous()
        Yf = torch.from_numpy(np.nan_to_num(yf, nan=0.0)).float()
        Mf = torch.from_numpy(~np.isnan(yf)).float()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        steps = EPOCHS * int(np.ceil(len(Xf) / BATCH))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        g = torch.Generator().manual_seed(0)
        t0 = time.time()
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(Xf), generator=g)
            tot = 0.0
            for b in range(0, len(Xf), BATCH):
                idx = perm[b:b + BATCH]
                x = (Xf[idx].to(dev, non_blocking=True).float() / 255.0 - mean) / std
                y, m = Yf[idx].to(dev), Mf[idx].to(dev)
                l = (lossf(model(x), y) * m).sum() / m.sum().clamp(min=1.0)
                opt.zero_grad(set_to_none=True); l.backward(); opt.step(); sched.step()
                tot += float(l) * len(idx)
            if ep in (0, EPOCHS // 2, EPOCHS - 1):
                print(f"    epoch {ep + 1}/{EPOCHS} loss {tot / len(Xf):.4f}  {time.time() - t0:.0f}s", flush=True)
        model.eval()
        torch.save(model.state_dict(), run / f"r2_{dataset}.pt")

        with torch.no_grad():
            xa = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
            torch_scores = torch.cat([torch.sigmoid(model((xa[b:b + 256].to(dev).float() / 255.0 - mean) / std)).cpu()
                                      for b in range(0, len(xa), 256)]).numpy()

        # ONNX -> TensorRT FP16, batch 1, preprocessing (normalisation) inside the graph
        class Wrapped(torch.nn.Module):
            def __init__(self, net):
                super().__init__(); self.net = net
                self.register_buffer("mean", mean.cpu()); self.register_buffer("std", std.cpu())

            def forward(self, x):
                return torch.sigmoid(self.net((x / 255.0 - self.mean) / self.std))
        wrapped = Wrapped(model.cpu()).eval()
        onnx_path, engine_path = run / f"r2_{dataset}.onnx", run / f"r2_{dataset}_fp16.engine"
        torch.onnx.export(wrapped, torch.zeros(1, 3, SIDE, SIDE), str(onnx_path), opset_version=13,
                          input_names=["image"], output_names=["p"])
        subprocess.run([TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--fp16"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from rap.trt import TRTModule
        eng = TRTModule(engine_path)
        trt_scores = np.zeros_like(torch_scores)
        xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
        for i in np.flatnonzero(test):
            trt_scores[i] = eng(xt[i:i + 1].to(dev).float()).float().cpu().numpy()[0]
        del eng
        from scipy import stats
        for hi, h in enumerate(heads):
            ok = test & ~np.isnan(Y[:, hi])
            rho = float(stats.spearmanr(trt_scores[ok, hi], torch_scores[ok, hi]).correlation)
            track, geometry = dataset, h.split("__")[0]
            system, target = h.split("__")[1], h.split("__")[2]
            v = lab[f"V__{h}"].to_numpy(float)
            m = ok
            ks, prize0, point, draws, dropped = t92.evaluate(v[m], lab.unit.to_numpy()[m],
                                                             {"random": None, "R2_cnn_clf": trt_scores[m, hi]}, 1000, rng)
            np.savez_compressed(run / f"scores__{track}__{geometry}__{system}__{target}.npz",
                                seq=lab.seq.to_numpy().astype("U32")[m], frame=lab.frame.to_numpy()[m],
                                unit=lab.unit.to_numpy().astype("U40")[m], split=lab.split.to_numpy().astype("U8")[m],
                                V=v[m], J_cheap=lab[f"Jc__{h}"].to_numpy(float)[m], R2_cnn_clf=trt_scores[m, hi],
                                R2_cnn_clf_torch=torch_scores[m, hi])
            for qi, q in enumerate(t92.QUOTAS):
                dr = draws["R2_cnn_clf"][:, qi] - draws["random"][:, qi]
                lo, hi_ = t92.ci(draws["R2_cnn_clf"][:, qi]); dlo, dhi = t92.ci(dr)
                rows_out.append({"track": track, "geometry": geometry, "system": system, "target": target, "split": "test",
                                 "signal": "R2_cnn_clf", "deployable": True, "input_dim": 3 * SIDE * SIDE,
                                 "width_mult": width, "gflops": flops[width] / 1e9, "trt_vs_torch_spearman": rho,
                                 "quota": q, "k": int(ks[qi]), "eta": float(point["R2_cnn_clf"]["eta"][qi]),
                                 "eta_lo": lo, "eta_hi": hi_, "minus_random": float(np.nanmean(dr)),
                                 "minus_random_lo": dlo, "minus_random_hi": dhi,
                                 "p_le_random": float(np.mean(dr[np.isfinite(dr)] <= 0)),
                                 "tie_frac": float(point["R2_cnn_clf"]["tie"][qi]),
                                 "responsive_frac": float(point["R2_cnn_clf"]["resp"][qi]),
                                 "boot_dropped": int(dropped[qi]), "n_units": int(len(np.unique(lab.unit[m]))),
                                 "n_frames": int(m.sum())})
            r = [x for x in rows_out if x["quota"] == 0.2][-1]
            print(f"  {dataset:8s} {h:30s} @20 {r['eta']:+.3f} (vs random {r['minus_random']:+.3f} "
                  f"[{r['minus_random_lo']:+.3f}, {r['minus_random_hi']:+.3f}])  TRT~torch rho {rho:.4f}", flush=True)
        (run / f"r2_{dataset}_meta.json").write_text(json.dumps(
            {"width_mult": width, "gflops": flops[width] / 1e9, "flops_by_width": flops, "heads": heads,
             "pos_weight": pos_weight.cpu().tolist(), "train_frames": int(fit.sum()), "epochs": EPOCHS}, indent=1))
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows_out)
    out = Path(RESULTS) / "final" / "benchmark_table_routers.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old.signal.str.startswith("R2_")], df], ignore_index=True)
    df.to_csv(out, index=False)
    df.to_csv(run / out.name, index=False)
    print("  wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cache", "train"])
    ap.add_argument("--tag", default="router_r2")
    args = ap.parse_args()
    stage_cache() if args.stage == "cache" else stage_train(args)


if __name__ == "__main__":
    main()
