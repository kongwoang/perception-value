#!/usr/bin/env python3
"""Task 17 item 2: apply the registered three-decimal rule to every C14 run available. Not shipped."""
import sys, numpy as np, pandas as pd

SIG = ["all_learned", "gate_ridge", "gate_gbm", "R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf"]
STAT = ["ndg", "ndg_lo", "ndg_hi"]
STABLE = "stable"                       # reported as unchanged at three decimals, exact value unknown

def pooled(path):
    d = pd.read_csv(path, float_precision="round_trip")
    p = d[(d.track == "pooled") & (d.target == "B_minus_official")].drop_duplicates("signal").set_index("signal")
    return {(s, t): float(p.loc[s, t]) for s in SIG for t in STAT}

ship = pooled(sys.argv[1])
runs = {"Jetson regen (Task 13)": dict(ship), "Jetson regen (Task 14)": dict(ship)}
runs["Jetson regen (Task 13)"].update({
    ("all_learned", "ndg"): -0.0891322470276259, ("all_learned", "ndg_lo"): -0.142657334240754,
    ("all_learned", "ndg_hi"): -0.0240217506289352,
    ("R1_mlp_clf", "ndg"): -0.0956753499904087, ("R1_mlp_clf", "ndg_lo"): -0.1764920692388275,
    ("R1_mlp_clf", "ndg_hi"): -0.0167396745285437})
runs["Jetson regen (Task 14)"].update({
    ("all_learned", "ndg"): -0.089056, ("all_learned", "ndg_lo"): -0.142610, ("all_learned", "ndg_hi"): -0.023838,
    ("R1_mlp_clf", "ndg"): -0.095219, ("R1_mlp_clf", "ndg_lo"): -0.176406, ("R1_mlp_clf", "ndg_hi"): -0.016029})
mac = {k: None for k in ship}                     # None: not reported
mac.update({("all_learned", "ndg"): STABLE, ("all_learned", "ndg_lo"): -0.142434, ("all_learned", "ndg_hi"): STABLE,
            ("R1_gbm_clf", "ndg"): -0.125136, ("R1_mlp_clf", "ndg"): STABLE,
            ("gate_ridge", "ndg"): STABLE, ("gate_gbm", "ndg"): STABLE, ("R1_mlp_reg", "ndg"): STABLE,
            ("R1_gbm_reg", "ndg"): STABLE})
runs["macOS (reported)"] = mac
for label, path in zip(("Jetson, 1 thread", "Jetson, 4 threads"), sys.argv[2:4]):
    runs[label] = pooled(path)

def r3(x):
    return float(np.floor(x * 1000 + 0.5) / 1000)

def margin(x):
    k = np.floor(x * 1000 + 0.5)
    return min(abs(x * 1000 - (k - 0.5)), abs(x * 1000 - (k + 0.5))) / 1000

rows = []
for s in SIG:
    for t in STAT:
        x0 = ship[(s, t)]
        vals = {k: v[(s, t)] for k, v in runs.items()}
        exact = [v for v in vals.values() if isinstance(v, float)]
        agree = all(r3(v) == r3(x0) for v in exact)          # "stable" reports agree by definition
        drift = max([abs(v - x0) for v in exact] + [0.0])
        m = margin(x0)
        verdict = "not safe" if not agree else ("fragile" if drift >= m else "safe")
        rows.append(dict(signal=s, stat=t, shipped=x0, shipped_3dp=r3(x0), margin=m, max_drift=drift,
                         values_3dp=sorted({r3(v) for v in exact}), n_exact_runs=len(exact),
                         macos=vals["macOS (reported)"], verdict=verdict))
df = pd.DataFrame(rows)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 40)
print(df.to_string(index=False, formatters={"shipped": "{:+.6f}".format, "margin": "{:.1e}".format,
                                            "max_drift": "{:.1e}".format}))
print()
ok = []
for k, v in runs.items():
    lo, hi = v[("all_learned", "ndg_lo")], v[("all_learned", "ndg_hi")]
    lo = lo if isinstance(lo, float) else ship[("all_learned", "ndg_lo")]
    hi = hi if isinstance(hi, float) else ship[("all_learned", "ndg_hi")]
    ok.append(lo < -0.05 < hi)
print("pooled reading 'inconclusive' (interval straddles -0.05) in every run:", all(ok))
df.to_csv(sys.argv[4], index=False)
