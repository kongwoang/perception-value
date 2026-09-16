#!/usr/bin/env python
"""Task 2, R2: a raw-pixel router under the cascade -- MobileNetV2-style CNN at 128 x 128.

CHEAP runs on every input, and R2 only decides whether FULL is added; the skipping design, where an
escalated input runs FULL instead of CHEAP, is evaluated separately in scripts/128_skip_accounting.py.

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

Every stage is its own process, one dataset at a time.  The first version trained, scored, exported
and built the engine in one process; the board rebooted right after the nuScenes weights were saved
(RESEARCH_LOG 12:55), most likely because trtexec's default workspace -- the whole device memory --
met a process still holding its CUDA context on unified memory.  So:
  --stage cache    decode and resize every image once (nuScenes paths need the nuScenes tables)
  --stage train    --dataset D: labels, FLOP count, training (or --reuse_pt), PyTorch scores; exits
  --stage export   --dataset D --run R: ONNX on CPU, trtexec with the workspace capped, TensorRT
                   scores on test frames, evaluation; writes the dataset's rows
"""
from __future__ import annotations

import argparse, importlib.util, json, subprocess, sys, time
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
TRT_WORKSPACE_MIB = 512
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
    full = max(cells, key=lambda c: len(c["d"]))          # the braking cell covers every frame
    base = full["d"][["seq", "frame", "unit", "split"]].astype({"seq": str, "frame": int, "unit": str}).copy()
    heads = []
    for c in cells:
        name = f"{c['geometry']}__{c['system']}__{c['target']}"
        d = c["d"][["seq", "frame"]].astype({"seq": str, "frame": int}).copy()
        d[f"V__{name}"] = (c["d"][c["cheap"]] - c["d"][c["full"]]).to_numpy(float)
        d[f"Jc__{name}"] = c["d"][c["cheap"]].to_numpy(float)
        base = base.merge(d, on=["seq", "frame"], how="left", validate="one_to_one")
        heads.append(name)
    return base, heads


def _images(dataset, lab):
    z = np.load(CACHE_DIR / f"{dataset}.npz", allow_pickle=False)
    imgs = pd.DataFrame({"seq": z["seq"].astype(str), "frame": z["frame"].astype(int), "_i": np.arange(len(z["seq"]))})
    j = lab[["seq", "frame"]].merge(imgs, on=["seq", "frame"], how="left", validate="one_to_one")
    assert j._i.notna().all(), "an image is missing from the cache"
    return z["img"][j._i.to_numpy(int)]


def stage_train(args):
    import torch, torchvision
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    run = Path(args.run) if args.run else runmeta.new_run(args.tag, vars(args))
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    dev = torch.device("cuda:0")
    dataset = args.dataset
    lab, heads = labels(dataset, splits)
    lab.to_pickle(run / f"r2_{dataset}_labels.pkl")
    X = _images(dataset, lab)
    Y = np.stack([np.where(np.isnan(lab[f"V__{h}"]), np.nan, (lab[f"V__{h}"] > 1e-9).astype(float)) for h in heads], 1)
    fit = lab.split.isin(["train", "val"]).to_numpy()
    flops = {w: count_flops(torchvision.models.mobilenet_v2(width_mult=w, num_classes=len(heads))) for w in WIDTHS}
    width = min(WIDTHS, key=lambda w: abs(flops[w] - TARGET_FLOPS))
    model = torchvision.models.mobilenet_v2(width_mult=width, num_classes=len(heads)).to(dev)
    print(f"  {dataset}: {len(heads)} heads, width {width} -> {flops[width] / 1e9:.3f} GFLOPs "
          f"({', '.join(f'{w}:{f / 1e9:.3f}' for w, f in flops.items())})", flush=True)
    yf = Y[fit]
    pos = np.nansum(yf, 0); neg = np.sum(~np.isnan(yf), 0) - pos
    pos_weight = torch.tensor(neg / np.maximum(pos, 1), dtype=torch.float32, device=dev)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    reused = None
    if args.reuse_pt and Path(args.reuse_pt).exists():
        model.load_state_dict(torch.load(args.reuse_pt, map_location="cpu"))
        reused = str(args.reuse_pt)
        print(f"  weights reused from {reused} (same procedure, 30 epochs completed before the reboot)", flush=True)
    else:
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
                loss = (lossf(model(x), y) * m).sum() / m.sum().clamp(min=1.0)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                tot += float(loss) * len(idx)
            if ep in (0, EPOCHS // 2, EPOCHS - 1):
                print(f"    epoch {ep + 1}/{EPOCHS} loss {tot / len(Xf):.4f}  {time.time() - t0:.0f}s", flush=True)
        del opt, Xf, Yf, Mf
    model.eval()
    torch.save(model.state_dict(), run / f"r2_{dataset}.pt")
    with torch.no_grad():
        xa = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
        scores = torch.cat([torch.sigmoid(model((xa[b:b + 64].to(dev).float() / 255.0 - mean) / std)).cpu()
                            for b in range(0, len(xa), 64)]).numpy()
    np.savez_compressed(run / f"r2_{dataset}_torch_scores.npz", scores=scores)
    (run / f"r2_{dataset}_meta.json").write_text(json.dumps(
        {"width_mult": width, "gflops": flops[width] / 1e9, "flops_by_width": flops, "heads": heads,
         "pos_weight": pos_weight.cpu().tolist(), "train_frames": int(fit.sum()), "epochs": EPOCHS,
         "weights_reused_from": reused}, indent=1))
    print(f"  wrote {run}", flush=True)


def stage_export(args):
    import torch, torchvision
    from scipy import stats
    run = Path(args.run)
    dataset = args.dataset
    meta = json.loads((run / f"r2_{dataset}_meta.json").read_text())
    heads = meta["heads"]
    lab = pd.read_pickle(run / f"r2_{dataset}_labels.pkl")
    torch_scores = np.load(run / f"r2_{dataset}_torch_scores.npz")["scores"]

    # ONNX on the CPU: no CUDA context exists while the engine is being built
    net = torchvision.models.mobilenet_v2(width_mult=meta["width_mult"], num_classes=len(heads))
    net.load_state_dict(torch.load(run / f"r2_{dataset}.pt", map_location="cpu"))
    net.eval()

    class Wrapped(torch.nn.Module):
        def __init__(self, n):
            super().__init__(); self.net = n
            self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

        def forward(self, x):
            return torch.sigmoid(self.net((x / 255.0 - self.mean) / self.std))
    onnx_path, engine_path = run / f"r2_{dataset}.onnx", run / f"r2_{dataset}_fp16.engine"
    torch.onnx.export(Wrapped(net).eval(), torch.zeros(1, 3, SIDE, SIDE), str(onnx_path), opset_version=13,
                      input_names=["image"], output_names=["p"])
    t0 = time.time()
    subprocess.run([TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--fp16",
                    f"--memPoolSize=workspace:{TRT_WORKSPACE_MIB}"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  {dataset}: engine built in {time.time() - t0:.0f}s (workspace {TRT_WORKSPACE_MIB} MiB)", flush=True)

    from rap.trt import TRTModule
    _s = importlib.util.spec_from_file_location("t92", ROOT / "scripts" / "92_benchmark_table.py")
    t92 = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(t92)
    X = _images(dataset, lab)
    test = (lab.split == "test").to_numpy()
    dev = torch.device("cuda:0")
    eng = TRTModule(engine_path)
    trt_scores = np.full_like(torch_scores, np.nan)
    xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    for i in np.flatnonzero(test):
        trt_scores[i] = eng(xt[i:i + 1].to(dev).float()).float().cpu().numpy()[0]
    del eng

    rng = np.random.default_rng(0)
    rows = []
    for hi, h in enumerate(heads):
        geometry, system, target = h.split("__")
        v = lab[f"V__{h}"].to_numpy(float)
        m = test & ~np.isnan(v)
        rho = float(stats.spearmanr(trt_scores[m, hi], torch_scores[m, hi]).correlation)
        ks, prize0, point, draws, dropped = t92.evaluate(v[m], lab.unit.to_numpy()[m],
                                                         {"random": None, "R2_cnn_clf": trt_scores[m, hi]}, 1000, rng)
        np.savez_compressed(run / f"scores__{dataset}__{geometry}__{system}__{target}.npz",
                            seq=lab.seq.to_numpy().astype("U32")[m], frame=lab.frame.to_numpy()[m],
                            unit=lab.unit.to_numpy().astype("U40")[m], split=lab.split.to_numpy().astype("U8")[m],
                            V=v[m], J_cheap=lab[f"Jc__{h}"].to_numpy(float)[m], R2_cnn_clf=trt_scores[m, hi],
                            R2_cnn_clf_torch=torch_scores[m, hi])
        for qi, q in enumerate(t92.QUOTAS):
            dr = draws["R2_cnn_clf"][:, qi] - draws["random"][:, qi]
            lo, hi_ = t92.ci(draws["R2_cnn_clf"][:, qi]); dlo, dhi = t92.ci(dr)
            rows.append({"track": dataset, "geometry": geometry, "system": system, "target": target, "split": "test",
                         "signal": "R2_cnn_clf", "deployable": True, "input_dim": 3 * SIDE * SIDE,
                         "width_mult": meta["width_mult"], "gflops": meta["gflops"], "trt_vs_torch_spearman": rho,
                         "quota": q, "k": int(ks[qi]), "eta": float(point["R2_cnn_clf"]["eta"][qi]),
                         "eta_lo": lo, "eta_hi": hi_, "minus_random": float(np.nanmean(dr)),
                         "minus_random_lo": dlo, "minus_random_hi": dhi,
                         "p_le_random": float(np.mean(dr[np.isfinite(dr)] <= 0)),
                         "tie_frac": float(point["R2_cnn_clf"]["tie"][qi]),
                         "responsive_frac": float(point["R2_cnn_clf"]["resp"][qi]),
                         "boot_dropped": int(dropped[qi]), "n_units": int(len(np.unique(lab.unit[m]))),
                         "n_frames": int(m.sum())})
        r = [x for x in rows if x["quota"] == 0.2][-1]
        print(f"  {dataset:8s} {h:30s} @20 {r['eta']:+.3f} (vs random {r['minus_random']:+.3f} "
              f"[{r['minus_random_lo']:+.3f}, {r['minus_random_hi']:+.3f}])  TRT~torch rho {rho:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(run / f"r2_{dataset}_rows.csv", index=False)
    out = Path(RESULTS) / "final" / "benchmark_table_routers.csv"
    if out.exists():
        old = pd.read_csv(out)
        keep = ~(old.signal.str.startswith("R2_") & (old.track == dataset))
        df = pd.concat([old[keep], df], ignore_index=True)
    df.to_csv(out, index=False)
    print("  wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cache", "train", "export"])
    ap.add_argument("--dataset", choices=["nuScenes", "KITTI"])
    ap.add_argument("--run", default=None, help="router_r2 run dir shared by the train and export stages")
    ap.add_argument("--reuse_pt", default=None, help="stage train: load finished weights instead of training")
    ap.add_argument("--tag", default="router_r2")
    args = ap.parse_args()
    if args.stage == "cache":
        stage_cache()
    elif args.stage == "train":
        stage_train(args)
    else:
        stage_export(args)


if __name__ == "__main__":
    main()
