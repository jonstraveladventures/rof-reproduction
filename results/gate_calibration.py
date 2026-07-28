"""Calibration study for the collapse-diagnostic gate (exploratory, in-sample).

Candidate per-run statistics, all on seed-averaged pool ROF after warmup
(epoch > 50), MA-7 smoothed:
  A. good-pool net drop: trailing-max minus FINAL value  (Nikolai's step 5)
  B. good-pool max drawdown: max over t of (trailing-max - value)
  C. bad-pool max drawdown
  D. max of B and C (either pool)
  E. B normalised by the run's good-pool range (drawdown / (max-min))

For each statistic: per-run values grouped by collapse label, plus the
separation margin (min over collapsing minus max over healthy; positive
margin = a perfect in-sample threshold exists) and the best threshold's
confusion counts. 18 runs: 8 LL + 777 + paper + 8 Reacher (6 collapse).
"""
import glob
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "LunarLander_RSSM"
spec = importlib.util.spec_from_file_location("an", REPO / "analyze_llc.py")
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)

WARMUP = 50


def pool(metric_glob, key, epochs):
    runs = [an.parse_metrics(f) for f in sorted(glob.glob(str(metric_glob)))]
    return [sum(m[e][key] for m in runs) / len(runs) for e in epochs]


def stats_for(mpc_path, metric_glob):
    mruns = [an.parse_metrics(f) for f in sorted(glob.glob(str(metric_glob)))]
    m_epochs = set.intersection(*[set(m) for m in mruns])
    mpc = an.parse_mpc(str(mpc_path))
    epochs = sorted(set(mpc) & m_epochs)
    out = {}
    for label, key in (("good", "jac_rof"), ("bad", "jac_rof_bad")):
        raw = pool(metric_glob, key, epochs)
        w = [v for e, v in zip(epochs, raw) if e > WARMUP]
        sm = an.ma(w)
        rmax, dd = sm[0], 0.0
        for v in sm:
            rmax = max(rmax, v)
            dd = max(dd, rmax - v)
        out[f"{label}_net"] = max(sm) - sm[-1]
        out[f"{label}_dd"] = dd
        out[f"{label}_range"] = max(sm) - min(sm)
    out["either_dd"] = max(out["good_dd"], out["bad_dd"])
    out["good_dd_norm"] = out["good_dd"] / out["good_range"] if out["good_range"] > 0 else 0.0
    return out


RUNS = []
LL = [(101, False), (202, True), (303, True), (404, False),
      (505, True), (606, True), (707, True), (808, False)]
for s, c in LL:
    mpc = HERE / ("mpc_sp707_merged.txt" if s == 707 else f"mpc_sp{s}.txt")
    RUNS.append((f"LL-{s}", c, mpc, HERE / f"metrics_sp{s}_seed*.txt"))
RUNS.append(("LL-777", False, HERE / "mpc_dq_f1rs.txt", HERE / "metrics_dq_f1rs_seed*.txt"))
RUNS.append(("paper", True, REPO / "logs" / "mpc_eval_logs.txt",
             REPO / "logs" / "metrics_eval_logs.txt"))
for s in (101, 202, 303, 404, 505, 606, 707, 808):
    RUNS.append((f"R-{s}", False, HERE / f"mpc_r{s}.txt", HERE / f"metrics_r{s}_seed*.txt"))

vals = {name: stats_for(m, g) for name, c, m, g in RUNS}
labels = {name: c for name, c, m, g in RUNS}

STATS = ["good_net", "good_dd", "bad_dd", "either_dd", "good_dd_norm"]
print(f"{'run':10} {'collapse':>8}", *[f"{s:>13}" for s in STATS])
for name, c, _, _ in RUNS:
    v = vals[name]
    print(f"{name:10} {str(c):>8}", *[f"{v[s]:>13.4f}" for s in STATS])

print()
for s in STATS:
    coll = sorted(vals[n][s] for n in vals if labels[n])
    heal = sorted(vals[n][s] for n in vals if not labels[n])
    margin = coll[0] - heal[-1]
    # best threshold: maximise (TP + TN)
    cands = sorted(set(coll + heal))
    best = None
    for t in cands:
        tp = sum(1 for v in coll if v >= t)
        tn = sum(1 for v in heal if v < t)
        if best is None or tp + tn > best[0]:
            best = (tp + tn, t, tp, tn)
    _, t, tp, tn = best
    print(f"{s:14}: collapse range [{coll[0]:+.4f}, {coll[-1]:+.4f}]  "
          f"healthy range [{heal[0]:+.4f}, {heal[-1]:+.4f}]  "
          f"margin {margin:+.4f}  best-thr {t:+.4f} -> {tp}/6 caught, {tn}/12 silent")
