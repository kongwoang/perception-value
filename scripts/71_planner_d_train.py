#!/usr/bin/env python
"""Phase 0G: train Planner D on the frozen split, and gate it on validation trajectory error.

Architecture, target, loss, both variants and their corruption values were frozen in the
Phase 0G pre-registration before this script existed.  The 85 Phase-0F test scenes are not
read here at all.

D-GT trains on ground-truth rasters.  D-Aug redraws the object channel each time a sample is
drawn, from boxes corrupted by the pre-registered dropout/jitter process, using PKL's own
drawing code -- so an augmented raster is drawn exactly the way a clean one is.

The viability gate is validation ADE at least 10% below a constant-velocity predictor, checked
before any cheap/full evaluation.  Because the test scenes are only boston-seaport and
singapore-onenorth while training is mostly boston-seaport, ADE is also reported restricted to
validation scenes in those two locations.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rap import runmeta                                                        # noqa: E402
from rap.planner_d import (Corruption, HORIZONS, OBJ_CHANNEL, PlannerD,        # noqa: E402
                           ade_fde, constant_velocity, object_channel)
from rap.paths import CACHE, RESULTS                                           # noqa: E402

NX = NY = 256
CELLS = 5 * NX * NY


class RasterSet(torch.utils.data.Dataset):
    """Bit-packed rasters with optional object-channel corruption."""

    def __init__(self, files, corrupt: Corruption | None = None, seed: int = 0):
        packs, tg, vel, sc, oc, ox, ol = [], [], [], [], [], [], []
        for f in files:
            z = np.load(f, allow_pickle=False)
            if len(z["sample_token"]) == 0:
                continue
            packs.append(z["packed_gt"]); tg.append(z["target"]); vel.append(z["ego_v"])
            sc.append(z["scene"]); oc.append(z["obj_count"])
            ox.append(z["obj_xy_cs"]); ol.append(z["obj_lw"])
        self.packed = np.concatenate(packs)
        self.target = np.concatenate(tg).astype(np.float32)
        self.ego_v = np.concatenate(vel).astype(np.float32)
        self.scene = np.concatenate(sc)
        counts = np.concatenate(oc)
        self.obj_xy = np.concatenate(ox); self.obj_lw = np.concatenate(ol)
        self.obj_off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.corrupt = corrupt
        self.seed = seed

    def __len__(self):
        return len(self.target)

    def raster(self, i: int, rng=None) -> np.ndarray:
        x = np.unpackbits(self.packed[i])[:CELLS].reshape(5, NX, NY).astype(np.float32)
        if self.corrupt is not None:
            a, b = self.obj_off[i], self.obj_off[i + 1]
            o, w = self.corrupt.apply(self.obj_xy[a:b], self.obj_lw[a:b], rng)
            x[OBJ_CHANNEL] = object_channel(o, w)
        return x

    def __getitem__(self, i):
        rng = (np.random.default_rng((self.seed, i, int(time.time() * 1e3) % 100000))
               if self.corrupt is not None else None)
        return torch.from_numpy(self.raster(i, rng)), torch.from_numpy(self.target[i])


@torch.no_grad()
def predict(model, ds, device, bsz=32) -> np.ndarray:
    model.eval()
    out = []
    for a in range(0, len(ds), bsz):
        x = np.stack([ds.raster(i) for i in range(a, min(a + bsz, len(ds)))])
        out.append(model(torch.from_numpy(x).to(device)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="gt", choices=["gt", "aug"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=str(CACHE / "planner_d"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bsz", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--nworkers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"planner_d_{args.variant}_s{args.seed}"
    run = runmeta.new_run(tag, vars(args))

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    # Claim the CUDA context and a reusable allocator pool before the raster cache and the
    # dataloader workers fill host memory.  GPU memory here is the same physical RAM, and three
    # metric chunks died earlier asking for their first 20-128 MiB after other allocations had
    # fragmented the largest free block.
    if torch.cuda.is_available():
        torch.cuda.init()
        pool = torch.empty(int(800e6 // 4), dtype=torch.float32, device="cuda:0")
        del pool
        print(f"  reserved CUDA pool: {torch.cuda.memory_reserved() / 1e6:.0f} MB")
    data = Path(args.data)
    tr_files = sorted((data / "train").glob("chunk_*.npz"))
    va_files = sorted((data / "val").glob("chunk_*.npz"))
    if not tr_files or not va_files:
        raise SystemExit("no cached Planner D data; run 70_planner_d_data.py first")

    corrupt = Corruption() if args.variant == "aug" else None
    tr = RasterSet(tr_files, corrupt, args.seed)
    va = RasterSet(va_files, None)          # validation always on clean rasters
    print(f"  train {len(tr)} samples, val {len(va)} samples, variant D-{args.variant.upper()}")

    # scene -> map location, read from the two small nuScenes tables rather than the big ones
    T = Path("/home/kongwoang/datasets/nuscenes/trainval/v1.0-trainval")
    _logs = {r["token"]: r["location"] for r in json.loads((T / "log.json").read_text())}
    loc_of = {sc["name"]: _logs[sc["log_token"]]
              for sc in json.loads((T / "scene.json").read_text())}
    TEST_LOCS = {"boston-seaport", "singapore-onenorth"}
    in_test_locs = np.array([loc_of[s] in TEST_LOCS for s in va.scene])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PlannerD().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.MSELoss()
    dl = torch.utils.data.DataLoader(tr, batch_size=args.bsz, shuffle=True,
                                     num_workers=args.nworkers, drop_last=True)

    cv = constant_velocity(va.ego_v)
    cv_ade, cv_fde = ade_fde(cv, va.target)
    print(f"  constant-velocity baseline: val ADE {cv_ade.mean():.3f} m  "
          f"FDE {cv_fde.mean():.3f} m")

    hist, best, bad = [], np.inf, 0
    for ep in range(args.epochs):
        model.train(); tot = n = 0
        for x, y in dl:
            opt.zero_grad()
            loss = lossf(model(x.to(device)), y.to(device))
            loss.backward(); opt.step()
            tot += float(loss) * len(x); n += len(x)
        sched.step()
        pred = predict(model, va, device)
        ade, fde = ade_fde(pred, va.target)
        hist.append({"epoch": ep, "train_mse": tot / max(n, 1),
                     "val_ade": float(ade.mean()), "val_fde": float(fde.mean())})
        print(f"  ep {ep:3d}  train MSE {tot/max(n,1):8.4f}  val ADE {ade.mean():6.3f} m"
              f"  FDE {fde.mean():6.3f} m", flush=True)
        if ade.mean() < best - 1e-4:
            best, bad = float(ade.mean()), 0
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_ade": best, "args": vars(args)}, run / "best.pt")
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_ade": best, "args": vars(args)},
                       Path(args.data) / f"{tag}_best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at epoch {ep}"); break

    ck = torch.load(run / "best.pt", map_location="cpu")
    model.load_state_dict(ck["model"]); model.to(device)
    pred = predict(model, va, device)
    ade, fde = ade_fde(pred, va.target)

    # viability restricted to the two locations the test split actually covers, because
    # training is 58% boston-seaport while the test scenes are 69% singapore-onenorth
    ade_tl, fde_tl = ade[in_test_locs], fde[in_test_locs]
    cv_ade_tl = cv_ade[in_test_locs]
    summary = {
        "variant": f"D-{args.variant.upper()}", "seed": args.seed,
        "best_epoch": int(ck["epoch"]), "n_train": len(tr), "n_val": len(va),
        "val_ade": float(ade.mean()), "val_fde": float(fde.mean()),
        "cv_val_ade": float(cv_ade.mean()), "cv_val_fde": float(cv_fde.mean()),
        "ade_improvement_vs_cv": 1.0 - float(ade.mean()) / float(cv_ade.mean()),
        "viability_pass": bool(float(ade.mean()) <= 0.90 * float(cv_ade.mean())),
        "n_val_test_locations": int(in_test_locs.sum()),
        "val_ade_test_locations": float(ade_tl.mean()),
        "val_fde_test_locations": float(fde_tl.mean()),
        "cv_val_ade_test_locations": float(cv_ade_tl.mean()),
        "ade_improvement_vs_cv_test_locations":
            1.0 - float(ade_tl.mean()) / float(cv_ade_tl.mean()),
        "viability_pass_test_locations":
            bool(float(ade_tl.mean()) <= 0.90 * float(cv_ade_tl.mean())),
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    import pandas as pd
    pd.DataFrame(hist).to_csv(run / "history.csv", index=False)
    out = Path(RESULTS) / "final"; out.mkdir(parents=True, exist_ok=True)
    f = out / "phase0g_planner_d_seeds.csv"
    row = pd.DataFrame([summary])
    row.to_csv(f, mode="a", header=not f.exists(), index=False)
    print(f"\n  D-{args.variant.upper()} seed {args.seed}: val ADE {ade.mean():.3f} m vs "
          f"constant-velocity {cv_ade.mean():.3f} m  "
          f"({100*summary['ade_improvement_vs_cv']:+.1f}%)  "
          f"viability {'PASS' if summary['viability_pass'] else 'FAIL'}")
    print(f"  on the {int(in_test_locs.sum())} val samples in the test set's two locations: "
          f"ADE {ade_tl.mean():.3f} m vs {cv_ade_tl.mean():.3f} m "
          f"({100*summary['ade_improvement_vs_cv_test_locations']:+.1f}%)  "
          f"{'PASS' if summary['viability_pass_test_locations'] else 'FAIL'}")
    print("  wrote", run)


if __name__ == "__main__":
    main()
