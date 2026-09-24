"""Figures for the z-only seed-12345 3080 arm (wm_checkpoints/zonly_12345_3080).

Produces:
  fig_zonly3080_mpc.png  -- MA-7 MPC return for the z-only 3080 run against its
                            [h, z] control on the same seed and GPU, plus the
                            5090 z-only arm and the healthy 777 reference.
  fig_zonly3080_rof.png  -- ROF over training for the same arms, showing that
                            the ablation raised ROF while the collapse persisted.

Run from the repository root:
    python results/figures/make_zonly3080_figures.py
"""

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "figures"

spec = importlib.util.spec_from_file_location("an", ROOT / "analyze_llc.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)

WARMUP = 50
# Every curve must cover these checkpoints in full (the z-only metric sweeps
# start at 55, so the window after WARMUP is the common expectation).
EXPECTED = [e for e in an.EXPECTED_EPOCHS if e > WARMUP]

# (label, mpc log, metrics glob, colour, linewidth, z-order)
ARMS = [
    ("z-only 12345, 3080 (this run)",
     RES / "mpc_zonly_12345_3080.txt", "metrics_zonly3080_seed*.txt",
     "#d62728", 2.6, 5),
    ("[h, z] V7 12345, 3080 (control)",
     RES / "mpc_v7_12345_continuous.txt", "metrics_v7_seed*.txt",
     "#1f77b4", 2.0, 4),
    ("z-only 12345, 5090",
     RES / "mpc_zonly_12345.txt", "metrics_zonly_seed*.txt",
     "#ff7f0e", 1.6, 3),
    ("[h, z] 777, 3080 (healthy ref)",
     RES / "mpc_dq_f1rs.txt", "metrics_dq_f1rs_seed*.txt",
     "#2ca02c", 1.6, 2),
    ("[h, z] 12345, paper, resume @300",
     ROOT / "logs" / "mpc_eval_logs.txt", "__none__",
     "#7f7f7f", 1.3, 1),
]


def mpc_curve(path):
    """MA-7 smoothed MPC mean over epochs > WARMUP.

    The warmup filter matters: before epoch 50 the sweeps carry large early
    transients that would otherwise dominate the drawdown statistic and have
    nothing to do with the late collapse this figure is about.
    """
    mpc = an.parse_mpc(str(path))
    short = [e for e in EXPECTED if len(mpc.get(e, [])) != an.EXPECTED_EPISODES]
    if short:
        raise SystemExit(f"ERROR: {path.name} lacks full {an.EXPECTED_EPISODES}-episode "
                         f"blocks at epochs {short}")
    eps = EXPECTED
    means = [float(np.mean(mpc[e])) for e in eps]
    return eps, means, an.ma(means)


def metric_curve(glob_pat, key):
    files = sorted(RES.glob(glob_pat))
    if not files:
        return [], []
    seeds = sorted(s for f in files for s in an.metric_seed(str(f)))
    if seeds != sorted(an.EXPECTED_SEEDS):
        raise SystemExit(f"ERROR: {glob_pat} covers metric seeds {seeds}, "
                         f"expected {sorted(an.EXPECTED_SEEDS)}")
    runs = [an.parse_metrics(str(f)) for f in files]
    for f, r in zip(files, runs):
        short = [e for e in EXPECTED if key not in r.get(e, {})]
        if short:
            raise SystemExit(f"ERROR: {f.name} lacks {key} at epochs {short}")
    eps = EXPECTED
    vals = [float(np.mean([r[e][key] for r in runs])) for e in eps]
    return eps, an.ma(vals)


def summarise(eps, raw, sm):
    """Peak and drawdown on the smoothed curve; final-10 on raw checkpoint means.

    final-10 is left unsmoothed so it matches analyze_llc.py's reported value.
    """
    peak_i = int(np.argmax(sm))
    tail = [e for e in eps if e > 300]
    if len(tail) > 2:
        i0 = eps.index(tail[0])
        r_ep = np.argsort(np.argsort(np.array(eps[i0:], float))).astype(float)
        r_sm = np.argsort(np.argsort(np.array(sm[i0:], float))).astype(float)
        r_ep -= r_ep.mean()
        r_sm -= r_sm.mean()
        rho = float(r_ep @ r_sm / (np.linalg.norm(r_ep) * np.linalg.norm(r_sm)))
    else:
        rho = float("nan")
    return {
        "peak": sm[peak_i], "peak_ep": eps[peak_i],
        "final10": float(np.mean(raw[-10:])),
        "drawdown": max(max(sm[:i + 1]) - v for i, v in enumerate(sm)),
        "rho_late": rho,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: closed-loop MPC ----
    # Legend sits outside the axes: placed inside it covers the control's
    # collapsed tail, which is the whole point of the figure.
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    rows = []
    for label, mpc_path, _, colour, lw, z in ARMS:
        if not mpc_path.exists():
            print(f"  [skip] {mpc_path.name} not found")
            continue
        eps, raw, sm = mpc_curve(mpc_path)
        ax.plot(eps, sm, label=label, color=colour, lw=lw, zorder=z)
        s = summarise(eps, raw, sm)
        rows.append((label, s))
        ax.plot([s["peak_ep"]], [s["peak"]], "o", color=colour, ms=6, zorder=z + 1)

    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("MPC mean return (MA-7, 20 episodes/checkpoint)")
    ax.set_title("z-only reward head does not prevent collapse on the 3080\n"
                 "seed 12345, eval seed 12345", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9,
              framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_zonly3080_mpc.png", dpi=160)
    plt.close(fig)

    print(f"\n{'run':<36}{'peak':>16}{'final-10':>10}{'drawdown':>11}"
          f"{'rho ep>300':>12}")
    print("-" * 85)
    for label, s in rows:
        print(f"{label:<36}{s['peak']:>+9.1f} @{s['peak_ep']:<5}"
              f"{s['final10']:>+10.1f}{s['drawdown']:>11.1f}"
              f"{s['rho_late']:>+12.2f}")

    # ---- Figure 2: ROF ----
    fig, ax = plt.subplots(figsize=(12.4, 4.8))
    for label, _, glob_pat, colour, lw, z in ARMS:
        eps, sm = metric_curve(glob_pat, "jac_rof")
        if not eps:
            print(f"  [skip] {glob_pat} not found")
            continue
        ax.plot(eps, sm, label=label, color=colour, lw=lw, zorder=z)
        print(f"{label:<36} ROF mean (ep>{WARMUP}) = {np.mean(sm):.4f}")

    ax.set_xlabel("training epoch")
    ax.set_ylabel("ROF (jac_rof, MA-7, mean over 3 eval seeds)")
    ax.set_title("The ablation raised ROF as predicted; the collapse was "
                 "unaffected", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9,
              framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_zonly3080_rof.png", dpi=160)
    plt.close(fig)

    print(f"\nwrote {OUT / 'fig_zonly3080_mpc.png'}")
    print(f"wrote {OUT / 'fig_zonly3080_rof.png'}")


if __name__ == "__main__":
    main()
