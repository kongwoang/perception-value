"""Diagnostic on VALIDATION rasters only: which pixel->metre convention makes PKL's planner agree with the real future?"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/kongwoang/research/risk-aware-perception")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "third_party/pkl"))
s = importlib.util.spec_from_file_location("m74", ROOT / "scripts/74_plannerC_vs_truth.py"); m74 = importlib.util.module_from_spec(s); s.loader.exec_module(m74)
from planning_centric_metrics.models import compile_model
from rap.planner_d import ade_fde, constant_velocity
files = sorted((ROOT / "data/cache/planner_d/val").glob("chunk_*.npz"))
z = np.load(files[0], allow_pickle=False); print("val chunk", files[0].name, "keys", z.files)
n = min(400, len(z["sample_token"]))
packed, true, vel = z["packed_gt"][:n] if "packed_gt" in z.files else z["packed"][:n], z["target"][:n], z["ego_v"][:n]
dev = torch.device("cuda:0")
model = compile_model(cin=5, cout=16, with_skip=True, dropout_p=0.0).to(dev)
model.load_state_dict(torch.load(ROOT / "third_party/tip/planner.pt", map_location="cpu")); model.eval()
masks = (torch.Tensor(json.loads((ROOT / "third_party/tip/masks_trainval.json").read_text())) == 1).to(dev)
p = m74.argmax_paths(model, masks, packed, dev, 16)              # metres, as the pipeline decodes them
pix = (p - m74.LO) / m74.RES                                      # back to pixel indices (i = first axis, j = second)
alts = {"as used (i->x, j->y)": p,
        "flip y": np.stack([p[..., 0], -p[..., 1]], -1),
        "swap axes": np.stack([pix[..., 1] * m74.RES[1] + m74.LO[1], pix[..., 0] * m74.RES[0] + m74.LO[0]], -1),
        "flip x": np.stack([-p[..., 0], p[..., 1]], -1)}
cv, _ = ade_fde(constant_velocity(vel), true)
print(f"frames {n}; truth mean |x| {np.abs(true[..., 0]).mean():.2f} m, |y| {np.abs(true[..., 1]).mean():.2f} m; constant-velocity ADE {cv.mean():.3f}")
for k, v in alts.items():
    ade, _ = ade_fde(v, true)
    ex, ey = (v - true)[..., 0], (v - true)[..., 1]
    print(f"  {k:24s} ADE {ade.mean():.3f}  mean err x {ex.mean():+.2f} y {ey.mean():+.2f}  corr(x) {np.corrcoef(v[...,0].ravel(), true[...,0].ravel())[0,1]:+.3f} corr(y) {np.corrcoef(v[...,1].ravel(), true[...,1].ravel())[0,1]:+.3f}")
t = 3   # 1.0 s
print("  at 1.0 s: truth x median %.2f, planner x median %.2f; truth y median %.2f, planner y median %.2f" % (
    np.median(true[:, t, 0]), np.median(p[:, t, 0]), np.median(true[:, t, 1]), np.median(p[:, t, 1])))
print("  planner x at 4.0 s median %.2f vs truth %.2f; stationary-truth frames (|x4|<1 m): %d, planner moves >5 m on %d of them" % (
    np.median(p[:, -1, 0]), np.median(true[:, -1, 0]), int((np.abs(true[:, -1, 0]) < 1).sum()), int(((np.abs(true[:, -1, 0]) < 1) & (np.abs(p[:, -1, 0]) > 5)).sum())))
