#!/usr/bin/env python3
"""Task 17 item 2: finest decimals at which each pooled C14 figure passes the registered rule. Not shipped."""
import sys, numpy as np, pandas as pd
exec(open(sys.argv[5]).read().split("def r3(x):")[0])      # the run data exactly as derive.py builds it

CLASS = {"all_learned": "average", "gate_ridge": "ridge", "gate_gbm": "gbm", "R1_gbm_reg": "gbm",
         "R1_gbm_clf": "gbm", "R1_mlp_reg": "mlp", "R1_mlp_clf": "mlp"}

def rnd(x, d):
    return float(np.floor(x * 10 ** d + 0.5) / 10 ** d)

def margin(x, d):
    k = np.floor(x * 10 ** d + 0.5)
    return min(abs(x * 10 ** d - (k - 0.5)), abs(x * 10 ** d - (k + 0.5))) / 10 ** d

drift = {}
for s in SIG:
    for t in STAT:
        exact = [v[(s, t)] for v in runs.values() if isinstance(v[(s, t)], float)]
        drift[(s, t)] = max([abs(v - ship[(s, t)]) for v in exact] + [0.0])
class_drift = {}
for (s, t), v in drift.items():
    class_drift[CLASS[s]] = max(class_drift.get(CLASS[s], 0.0), v)

rows = []
for s in SIG:
    for t in STAT:
        x0 = ship[(s, t)]
        exact = [v[(s, t)] for v in runs.values() if isinstance(v[(s, t)], float)]
        out = {"signal": s, "stat": t, "shipped": x0}
        best = None
        for d in (3, 2, 1):
            agree = all(rnd(v, d) == rnd(x0, d) for v in exact)
            m = margin(x0, d)
            rule = "not safe" if not agree else ("fragile" if drift[(s, t)] >= m else "safe")
            risk = m <= class_drift[CLASS[s]]
            out[f"{d}dp"] = f"{rnd(x0, d):+.{d}f} {rule}{' (class drift ' + format(class_drift[CLASS[s]], '.1e') + ' >= margin ' + format(m, '.1e') + ')' if rule == 'safe' and risk else ''}"
            if best is None and rule == "safe" and not risk:
                best = f"{rnd(x0, d):+.{d}f}"
        out["quote_as"] = best or "range across runs"
        rows.append(out)
df = pd.DataFrame(rows)
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 70)
print(df.to_string(index=False, formatters={"shipped": "{:+.6f}".format}))
print("\nclass drift:", {k: f"{v:.1e}" for k, v in class_drift.items()})
df.to_csv(sys.argv[6], index=False)
