"""
Score the 2026-09-26 domain-contrast registration (D1-D3).

Per run: Spearman between the seed-averaged pool mean 0.5*good + 0.5*bad
(unsmoothed) and MA-7 MPC mean return over all checkpoints, as in
make_panel_figures.run_stats, for the sealed ROF, wfull and wdiag. Inputs go
through analyze_llc.load_run, so an incomplete sweep stops the script.

    python results/score_contrast.py
"""

import importlib.util
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
REPO = next(p / s for p in HERE.parents for s in ("", "LunarLander_RSSM")
            if (p / s / "analyze_llc.py").exists())
spec = importlib.util.spec_from_file_location("an", REPO / "analyze_llc.py")
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)

SEEDS = (101, 202, 303, 404, 505, 606, 707, 808)
LL_COLLAPSE = {202, 303, 505, 606, 707}          # frozen collapse boolean, seed panel
FIELDS = ("jac_rof", "jac_rof_bad", "jac_rof_wfull", "jac_rof_wfull_bad",
          "jac_rof_wdiag", "jac_rof_wdiag_bad")
VARIANTS = (("sealed", "jac_rof", "jac_rof_bad"),
            ("wfull", "jac_rof_wfull", "jac_rof_wfull_bad"),
            ("wdiag", "jac_rof_wdiag", "jac_rof_wdiag_bad"))


def rhos(task, s):
    if task == "LL":
        mpc = HERE / ("mpc_sp707_merged.txt" if s == 707 else f"mpc_sp{s}.txt")
        tag, kw = f"sp{s}", {}
    else:
        mpc = HERE / f"mpc_r{s}.txt"
        tag, kw = f"r{s}", ({"known_missing": (10,)} if s == 606 else {})
    ep, m, runs = an.load_run(mpc, HERE / f"metrics_w_{tag}_seed*.txt", fields=FIELDS, **kw)
    _, _, old = an.load_run(mpc, HERE / f"metrics_{tag}_seed*.txt", **kw)
    worst = max(abs(a[e][k] - b[e][k]) for a, b in zip(runs, old) for e in ep
                for k in ("jac_rof", "jac_rof_bad"))
    sm = an.ma([sum(m[e]) / len(m[e]) for e in ep])
    out = {}
    for name, g, b in VARIANTS:
        comb = [sum(0.5 * r[e][g] + 0.5 * r[e][b] for r in runs) / len(runs) for e in ep]
        out[name] = an.spearman(comb, sm)
    return out, worst


def main():
    res = {}
    print(f"{'run':8} {'sealed':>7} {'wfull':>7} {'wdiag':>7}  max|new-stored|")
    for task in ("LL", "R"):
        for s in SEEDS:
            r, worst = rhos(task, s)
            res[(task, s)] = r
            flag = " (collapse)" if task == "LL" and s in LL_COLLAPSE else ""
            print(f"{task}-{s:<5} {r['sealed']:+7.2f} {r['wfull']:+7.2f} {r['wdiag']:+7.2f}  "
                  f"{worst:.4f}{flag}")
    print()
    for name, _, _ in VARIANTS:
        ll = [res[("LL", s)][name] for s in SEEDS]
        rr = [res[("R", s)][name] for s in SEEDS]
        d1 = all(x > 0 for x in rr)
        d2 = min(ll) <= -0.1 and max(ll) >= 0.1
        tag = "" if name == "wfull" else "  [not a registered test]"
        print(f"{name:6}: Reacher {min(rr):+.2f}..{max(rr):+.2f} (median {median(rr):+.2f}); "
              f"LunarLander {min(ll):+.2f}..{max(ll):+.2f} (median {median(ll):+.2f}){tag}")
        if name == "wfull":
            print(f"  D1 all Reacher > 0: {'HELD' if d1 else 'FAILED'}")
            print(f"  D2 LunarLander spans zero (<= -0.1 and >= +0.1): {'HELD' if d2 else 'FAILED'}")
            print(f"  D3 contrast survives: {'HELD' if d1 and d2 else 'FAILED'}")
        coll = [res[("LL", s)][name] for s in SEEDS if s in LL_COLLAPSE]
        heal = [res[("LL", s)][name] for s in SEEDS if s not in LL_COLLAPSE]
        print(f"  LunarLander collapsing median {median(coll):+.2f}, healthy median {median(heal):+.2f}")


if __name__ == "__main__":
    main()
