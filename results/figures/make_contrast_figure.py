"""Two-environment contrast: per-run rho_s in LunarLander vs Reacher panels.

Statistics computed at runtime from raw logs via analyze_llc conventions
(reuses run_stats from make_panel_figures).
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent

spec = importlib.util.spec_from_file_location("mpf", HERE / "make_panel_figures.py")
mpf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mpf)  # computes mpf.stats for LL panel + refs

C_HEALTHY, C_COLLAPSE, C_777, C_PAPER = mpf.C_HEALTHY, mpf.C_COLLAPSE, mpf.C_777, mpf.C_PAPER
PANEL = mpf.PANEL

# r606 epoch 10 was overwritten (recorded 2026-07-21).
r_stats = {s: mpf.run_stats(RES / f"mpc_r{s}.txt", RES / f"metrics_r{s}_seed*.txt",
                            known_missing=(10,) if s == 606 else ())
           for s in PANEL}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
panels = [
    ("LunarLander (non-Markovian reward): 5/8 collapse",
     [(f"seed {s}", mpf.stats[s]) for s in PANEL]
     + [("777 (ref)", mpf.stats["777"]), ("paper (ref)", mpf.stats["paper"])],
     True),
    ("Reacher (Markovian reward): 0/8 collapse",
     [(f"seed {s}", r_stats[s]) for s in PANEL], False),
]
for ax, (title, entries, has_refs) in zip(axes, panels):
    entries = sorted(entries, key=lambda t: t[1]["rho"])
    ax.axvline(0, lw=0.8, color="#bbbbbb", zorder=0)
    for y, (name, st) in enumerate(entries):
        ref = "(ref)" in name
        col = (C_777 if name.startswith("777") else C_PAPER) if ref else \
              (C_COLLAPSE if st["collapse"] else C_HEALTHY)
        ax.plot(st["rho"], y, "D" if ref else "o", ms=8 if ref else 9,
                color=col, zorder=3)
        right = 0 <= st["rho"] < 0.8
        ax.text(st["rho"] + (0.05 if right else -0.05), y,
                mpf.neg(f"{st['rho']:+.2f}"), va="center",
                ha="left" if right else "right",
                fontsize=8.5, color="#333333")
    ax.set_yticks(range(len(entries)), [e[0] for e in entries], fontsize=9)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("ρₛ(jac_rof_combined, smoothed MPC)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="x", lw=0.3, color="#eeeeee")
h = [plt.Line2D([], [], marker="o", ls="", color=C_COLLAPSE, label="collapse"),
     plt.Line2D([], [], marker="o", ls="", color=C_HEALTHY, label="healthy"),
     plt.Line2D([], [], marker="D", ls="", color="#888888", label="reference runs")]
axes[0].legend(handles=h, loc="lower right", fontsize=8.5, frameon=False)
fig.suptitle("Per-run ROF correlation across the two seed panels (same eight seeds)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(HERE / "fig_two_env_rho.png", dpi=170)
print("contrast figure written")
