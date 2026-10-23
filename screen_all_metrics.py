"""Screen every offline metric against closed-loop outcome across all runs.

For each of the 27 metrics common to all runs, four summary statistics are
formed over epoch > 50 (seed-averaged, MA-7 smoothed) and rank-correlated
against two closed-loop targets.

With 27 x 4 = 108 candidates and n = 17 runs, the largest |rho| is expected to
be sizeable by chance alone, so significance is judged against a permutation
null on the MAXIMUM |rho| across all candidates rather than a per-test p-value.

Usage, from the root of the reproduction repository (or pass its path):
    python screen_all_metrics.py [REPO]
"""

import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np

CROF = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
RES = CROF / "results"
if not (CROF / "analyze_llc.py").is_file():
    sys.exit(f"analyze_llc.py not found in {CROF}; run from the repo root or pass its path")

spec = importlib.util.spec_from_file_location("an", CROF / "analyze_llc.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)

WARMUP = 50
N_PERM = 4000
RNG = np.random.default_rng(0)

RUNS = [
    ("z-only 3080", "zonly", RES / "metrics_zonly3080_seed*.txt", RES / "mpc_zonly_12345_3080.txt"),
    ("z-only 5090", "zonly", RES / "metrics_zonly_seed*.txt", RES / "mpc_zonly_12345.txt"),
    ("V7 3080", "human", RES / "metrics_v7_seed*.txt", RES / "mpc_v7_12345_continuous.txt"),
    ("paper", "human", CROF / "logs" / "metrics_eval_logs.txt", CROF / "logs" / "mpc_eval_logs.txt"),
    ("777", "human", RES / "metrics_dq_f1rs_seed*.txt", RES / "mpc_dq_f1rs.txt"),
]
for s in (101, 202, 303, 404, 505, 606, 707, 808):
    RUNS.append((f"sp{s}", "human", RES / f"metrics_sp{s}_seed*.txt",
                 RES / ("mpc_sp707_merged.txt" if s == 707 else f"mpc_sp{s}.txt")))
for t in ("dq_f25", "dq_f75", "dq_h25"):
    RUNS.append((t, "mixture", RES / f"metrics_{t}_seed*.txt", RES / f"mpc_{t}.txt"))
RUNS.append(("disc_scripted", "scripted", RES / "metrics_disc_scripted_seed*.txt",
             RES / "mpc_eval_disc_scripted.txt"))

KEYS = ["emp_C", "emp_L", "emp_O", "jac_ctrl_cond", "jac_ctrl_rank", "jac_obs_cond",
        "jac_obs_rank", "jac_ocf", "jac_rcf", "jac_rof", "jac_rof_bad",
        "jac_spec_radius", "jac_spec_radius_max", "ol_cumrew_err", "ol_obs_avg",
        "ol_obs_end", "ol_obs_max", "ol_obs_start", "ol_rew_avg", "ol_rew_end",
        "ol_rew_max", "ol_rew_start", "post_obs_rmse", "post_rew_rmse",
        "val_kl", "val_loss", "val_recon"]


def summarise(metric_glob, mpc_path):
    files = sorted(glob.glob(str(metric_glob)))
    mruns = [an.parse_metrics(f) for f in files]
    m_epochs = set.intersection(*[set(m) for m in mruns])
    mpc = an.parse_mpc(str(mpc_path))
    epochs = sorted(e for e in (set(mpc) & m_epochs) if e > WARMUP)

    means = [sum(mpc[e]) / len(mpc[e]) for e in epochs]
    msm = an.ma(means)
    targets = {"mpc_drawdown": max(msm) - msm[-1], "mpc_final10": sum(means[-10:]) / 10}

    feats = {}
    for k in KEYS:
        vals = []
        for e in epochs:
            per = [m[e][k] for m in mruns if k in m.get(e, {})]
            vals.append(sum(per) / len(per) if per else np.nan)
        if any(not np.isfinite(v) for v in vals):
            for suf in ("mean", "final", "drawdown", "trend"):
                feats[f"{k}.{suf}"] = np.nan
            continue
        sm = an.ma(vals)
        rmax, dd = sm[0], 0.0
        for v in sm:
            rmax = max(rmax, v)
            dd = max(dd, rmax - v)
        feats[f"{k}.mean"] = float(np.mean(sm))
        feats[f"{k}.final"] = sm[-1]
        feats[f"{k}.drawdown"] = dd
        feats[f"{k}.trend"] = sm[-1] - sm[0]
    return targets, feats


names, provs, T, F = [], [], [], []
for name, prov, mg, mp in RUNS:
    t, f = summarise(mg, mp)
    names.append(name)
    provs.append(prov)
    T.append(t)
    F.append(f)

feat_names = [k for k in F[0] if all(np.isfinite(f[k]) for f in F)]
X = np.array([[f[k] for k in feat_names] for f in F])
provs = np.array(provs)
print(f"{len(names)} runs, {len(feat_names)} candidate statistics "
      f"({len(feat_names)//4} metrics x 4 summaries)\n")


def rank(a):
    order = a.argsort()
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def screen(mask, target_key, label):
    Xs = X[mask]
    y = np.array([t[target_key] for t, m in zip(T, mask) if m])
    n = len(y)
    Rx = np.array([rank(Xs[:, j]) for j in range(Xs.shape[1])])
    Rx = (Rx - Rx.mean(1, keepdims=True))
    Rx /= np.linalg.norm(Rx, axis=1, keepdims=True) + 1e-12
    ry = rank(y)
    ry = ry - ry.mean()
    ry /= np.linalg.norm(ry) + 1e-12
    rho = Rx @ ry

    null_max = np.empty(N_PERM)
    for i in range(N_PERM):
        p = RNG.permutation(n)
        null_max[i] = np.abs(Rx @ ry[p]).max()
    thr95 = np.percentile(null_max, 95)

    print("=" * 96)
    print(f"{label}   target = {target_key}   n = {n}")
    print("=" * 96)
    print(f"  permutation null: max|rho| over {len(feat_names)} candidates exceeds "
          f"{thr95:.3f} in 5% of shuffles")
    order = np.argsort(-np.abs(rho))
    print(f"\n  {'statistic':<30}{'rho_s':>9}   {'beats chance?':<14}")
    print("  " + "-" * 60)
    for j in order[:12]:
        verdict = "YES" if abs(rho[j]) > thr95 else "no"
        print(f"  {feat_names[j]:<30}{rho[j]:>+9.3f}   {verdict:<14}")
    hits = [feat_names[j] for j in order if abs(rho[j]) > thr95]
    print(f"\n  -> {len(hits)} of {len(feat_names)} statistics beat the permutation null"
          f"{': ' + ', '.join(hits) if hits else ''}")
    for probe in ("jac_rof.drawdown", "jac_rof_bad.drawdown", "val_loss.final"):
        if probe in feat_names:
            j = feat_names.index(probe)
            print(f"     [{probe:<20}] rho = {rho[j]:+.3f}")
    print()


allm = np.ones(len(names), bool)
human = provs == "human"
for tgt in ("mpc_drawdown", "mpc_final10"):
    screen(allm, tgt, "ALL LUNARLANDER RUNS")
for tgt in ("mpc_drawdown", "mpc_final10"):
    screen(human, tgt, "HUMAN-DATA RUNS ONLY")
