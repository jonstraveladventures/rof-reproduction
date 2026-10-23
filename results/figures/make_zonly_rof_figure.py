"""Good-pool ROF over training for the reward-head ablation, both seeds.

Produces:
  fig_zonly_rof.png  -- (a) seed 12345: the z-only runs on the 3080 and 5090,
                        the [h, z] control on the 3080 and the healthy
                        777-L40S-C reference (the arms of the paper's earlier
                        single-panel figure); (b) seed 777 on the 3080: the
                        z-only run against its [h, z] control.

Each curve is jac_rof (raw coordinates, good validation pool), averaged over
metric seeds {12345, 1, 2} at each checkpoint after epoch 50, then centred MA-7.
Every curve must cover every checkpoint from 55 to 500.

Run from the repository root:
    python results/figures/make_zonly_rof_figure.py
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
EXPECTED = [e for e in an.EXPECTED_EPOCHS if e > WARMUP]
SEEDS = (12345, 1, 2)

# (label, metrics file stem, colour, linewidth, z-order)
PANELS = [
    ("(a) seed 12345", [
        ("12345-3080-Z (z-only head)", "metrics_zonly3080", "#d62728", 2.6, 5),
        ("12345-3080-C ([h, z] head, control)", "metrics_v7", "#1f77b4", 2.0, 4),
        ("12345-5090-Z (z-only head)", "metrics_zonly", "#ff7f0e", 1.6, 3),
        ("777-L40S-C ([h, z] head, healthy reference)", "metrics_dq_f1rs", "#2ca02c", 1.6, 2),
    ]),
    ("(b) seed 777, RTX 3080", [
        ("777-3080-Z (z-only head)", "metrics_w_777-3080-Z", "#d62728", 2.6, 5),
        ("777-3080-C ([h, z] head, control)", "metrics_w_777-3080-C", "#1f77b4", 2.0, 4),
    ]),
]


def rof_curve(stem):
    runs = [an.parse_metrics(str(RES / f"{stem}_seed{s}.txt")) for s in SEEDS]
    for s, r in zip(SEEDS, runs):
        missing = [e for e in EXPECTED if "jac_rof" not in r.get(e, {})]
        if missing:
            raise SystemExit(f"{stem}_seed{s}.txt lacks jac_rof at epochs {missing}")
    vals = [float(np.mean([r[e]["jac_rof"] for r in runs])) for e in EXPECTED]
    return EXPECTED, an.ma(vals)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True, layout="constrained")
    for ax, (title, arms) in zip(axes, PANELS):
        for label, stem, colour, lw, z in arms:
            eps, ys = rof_curve(stem)
            ax.plot(eps, ys, label=label, color=colour, lw=lw, zorder=z)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("world-model training epoch", fontsize=9)
        ax.set_xlim(WARMUP, 505)
        ax.tick_params(labelsize=8.5)
        ax.grid(alpha=0.25)
        # Below the axes, so no curve is covered.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, frameon=False)
    axes[0].set_ylabel("good-pool ROF\n(MA-7, mean of 3 metric seeds)", fontsize=9)
    out = OUT / "fig_zonly_rof.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
