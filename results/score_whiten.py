"""
Score the 2026-09-24 coordinate-invariance test (W1-W5).

Reads the whitened metric sweeps (metrics_w_<run>_seed{12345,1,2}.txt, written by
eval_metrics.py --whiten) next to the stored sweeps and MPC logs. Every
input goes through analyze_llc.load_run, so an incomplete sweep stops the script.
Arms whose sweeps are absent are reported as pending and not scored.

The five checks and their thresholds were fixed before the sweeps ran. "raw" is
the native-coordinate jac_rof; levels are seed-averaged over epochs >= 50. Each
check holds if:
  W1  the Spearman correlation over the twelve panel runs between raw and wfull
      bad-pool levels is >= 0.6
  W2  the top three of the eight screening runs by wfull bad-pool level are all
      majority-scripted (f25, scr)
  W3  the h128 offset ratio under wfull, |mean(h128 pair) - mean(sp pair)| over
      the spread of the eight screening runs, is >= 0.47 (half the raw ratio)
  W4  z-only 3080's wfull good-pool level over epochs >= 400 exceeds V7's
  W5  the median over the twelve panel runs of the within-run Spearman, across
      checkpoints at epochs >= 50, between raw and wfull good-pool series is >= 0.5
wdiag is reported beside each as a secondary variant. Write-up:
docs/crof-coordinate-invariance.md.

    python results/score_whiten.py
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p / s for p in HERE.parents for s in ("", "LunarLander_RSSM")
            if (p / s / "analyze_llc.py").exists())
spec = importlib.util.spec_from_file_location("an", REPO / "analyze_llc.py")
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)

SCREEN = ["sp909", "sp1010", "f25_909", "f25_1010", "f75_909", "f75_1010",
          "scr_909", "scr_1010"]
PANEL = SCREEN + ["h128_909", "h128_1010", "z8_909", "z8_1010"]
SCRIPTED = {"f25_909", "f25_1010", "scr_909", "scr_1010"}
LOCAL = {"v7": ("mpc_v7_12345_continuous.txt", {}),
         "zonly3080": ("mpc_zonly_12345_3080.txt", {"expected_epochs": range(55, 501, 5)})}
W_FIELDS = ("jac_rof", "jac_rof_bad", "jac_rof_wfull", "jac_rof_wfull_bad",
            "jac_rof_wdiag", "jac_rof_wdiag_bad")


def mpc_name(run):
    return LOCAL[run][0] if run in LOCAL else f"mpc_{run}.txt"


def load(run):
    """Seed-averaged series for every W field, or None if the sweep is absent."""
    if not list(HERE.glob(f"metrics_w_{run}_seed*.txt")):
        return None
    kw = LOCAL.get(run, (None, {}))[1]
    epochs, _, runs = an.load_run(HERE / mpc_name(run), HERE / f"metrics_w_{run}_seed*.txt",
                                  fields=W_FIELDS, **kw)
    return epochs, {k: [sum(m[e][k] for m in runs) / len(runs) for e in epochs]
                    for k in W_FIELDS}, runs


def level(ep, series, lo=50):
    v = [x for e, x in zip(ep, series) if e >= lo]
    return sum(v) / len(v)


def manipulation(run, runs):
    """jac_rof and jac_rof_bad in the new sweep against the stored sweep, per seed file."""
    kw = LOCAL.get(run, (None, {}))[1]
    _, _, old = an.load_run(HERE / mpc_name(run), HERE / f"metrics_{run}_seed*.txt", **kw)
    old = {an.metric_seed(str(f))[0]: m for f, m in
           zip(sorted(HERE.glob(f"metrics_{run}_seed*.txt")), old)}
    new = {an.metric_seed(str(f))[0]: m for f, m in
           zip(sorted(HERE.glob(f"metrics_w_{run}_seed*.txt")), runs)}
    worst = 0.0
    for s in new:
        for e in new[s]:
            for k in ("jac_rof", "jac_rof_bad"):
                worst = max(worst, abs(new[s][e][k] - old[s][e][k]))
    return worst


def main():
    data = {r: load(r) for r in PANEL + list(LOCAL)}
    print("manipulation check: max |new - stored| over jac_rof, jac_rof_bad")
    for r, d in data.items():
        if d is None:
            print(f"  {r:10} pending")
        else:
            print(f"  {r:10} {manipulation(r, d[2]):.4f}")

    print(f"\n{'run':10} {'bad raw':>8} {'bad wfull':>9} {'bad wdiag':>9} "
          f"{'good raw':>8} {'good wfull':>10} {'good wdiag':>10}")
    lv = {}
    for r, d in data.items():
        if d is None:
            continue
        ep, s, _ = d
        lv[r] = {k: level(ep, s[k]) for k in W_FIELDS}
        print(f"{r:10} {lv[r]['jac_rof_bad']:8.4f} {lv[r]['jac_rof_wfull_bad']:9.4f} "
              f"{lv[r]['jac_rof_wdiag_bad']:9.4f} {lv[r]['jac_rof']:8.4f} "
              f"{lv[r]['jac_rof_wfull']:10.4f} {lv[r]['jac_rof_wdiag']:10.4f}")

    print()
    if all(data[r] is not None for r in PANEL):
        for v in ("wfull", "wdiag"):
            tag = "" if v == "wfull" else "  [wdiag, secondary]"
            key = f"jac_rof_{v}_bad"
            rho = an.spearman([lv[r]["jac_rof_bad"] for r in PANEL], [lv[r][key] for r in PANEL])
            print(f"W1 Spearman(raw, {v}) bad level, 12 runs = {rho:+.3f} "
                  f"(>= 0.6: {'HELD' if rho >= 0.6 else 'FAILED'}){tag}")
            top = sorted(SCREEN, key=lambda r: -lv[r][key])[:3]
            print(f"W2 top three by {v}: {top} "
                  f"({'HELD' if set(top) <= SCRIPTED else 'FAILED'}){tag}")
            b = [lv[r][key] for r in SCREEN]
            sp = (lv["sp909"][key] + lv["sp1010"][key]) / 2
            for arch in ("h128", "z8"):
                x = (lv[f"{arch}_909"][key] + lv[f"{arch}_1010"][key]) / 2
                ratio = abs(x - sp) / (max(b) - min(b))
                verdict = f"(>= 0.47: {'HELD' if ratio >= 0.47 else 'FAILED'})" \
                    if arch == "h128" else "(reported)"
                print(f"W3 {arch} offset {x - sp:+.4f}, ratio {ratio:.2f} {verdict}{tag}")
            rhos = []
            for r in PANEL:
                ep, s, _ = data[r]
                idx = [i for i, e in enumerate(ep) if e >= 50]
                rhos.append(an.spearman([s["jac_rof"][i] for i in idx],
                                        [s[f"jac_rof_{v}"][i] for i in idx]))
            med = sorted(rhos)[len(rhos) // 2 - 1: len(rhos) // 2 + 1]
            med = sum(med) / 2
            print(f"W5 median within-run Spearman(raw, {v}) good = {med:+.3f} "
                  f"(>= 0.5: {'HELD' if med >= 0.5 else 'FAILED'}); "
                  f"range {min(rhos):+.3f} .. {max(rhos):+.3f}{tag}")
    else:
        print("W1, W2, W3, W5: pending (panel sweeps not all present)")

    if data["v7"] is not None and data["zonly3080"] is not None:
        for v in ("", "_wfull", "_wdiag"):
            z = level(data["zonly3080"][0], data["zonly3080"][1][f"jac_rof{v}"], 400)
            c = level(data["v7"][0], data["v7"][1][f"jac_rof{v}"], 400)
            name = v[1:] or "raw"
            verdict = f" ({'HELD' if z > c else 'FAILED'})" if v == "_wfull" else ""
            print(f"W4 good ep>=400, {name:6}: z-only {z:.4f} vs V7 {c:.4f}{verdict}")
    else:
        print("W4: pending")


if __name__ == "__main__":
    main()
