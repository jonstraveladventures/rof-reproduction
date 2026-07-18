"""Seed-panel figures: heterogeneity grid, ROF-lead dumbbells, rho_s dot plot.

Conventions inherited from analyze_llc.py (centred MA-7, alpha=0.5 combined).
Colours: Okabe-Ito subset, validated (healthy #0072B2, collapse #D55E00,
777 #009E73, paper reference #555555).
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent
REPO = RES.parent / "LunarLander_RSSM"

spec = importlib.util.spec_from_file_location("an", REPO / "analyze_llc.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)

C_HEALTHY, C_COLLAPSE, C_777, C_PAPER = "#0072B2", "#D55E00", "#009E73", "#555555"

PANEL = [101, 202, 303, 404, 505, 606, 707, 808]
COLLAPSE = {202, 303, 505, 606, 707}
RHO = {101: +0.320, 202: -0.548, 303: -0.307, 404: +0.190,
       505: -0.224, 606: +0.082, 707: -0.015, 808: -0.019}


def mpc_curve(path):
    r = an.parse_mpc(str(path))
    eps = sorted(r)
    means = [sum(r[e]) / len(r[e]) for e in eps]
    return eps, means, an.ma(means)


def rof_curve(paths):
    per = [an.parse_metrics(str(p)) for p in paths]
    eps = sorted(set.intersection(*[set(p) for p in per]))
    comb = []
    for e in eps:
        vals = [0.5 * p[e]["jac_rof"] + 0.5 * p[e]["jac_rof_bad"] for p in per]
        comb.append(sum(vals) / len(vals))
    return eps, comb


runs = {}
for s in PANEL:
    runs[s] = mpc_curve(RES / f"mpc_sp{s}.txt" if s != 707 else RES / "mpc_sp707_merged.txt")
runs["777"] = mpc_curve(RES / "mpc_dq_f1rs.txt")
runs["paper"] = mpc_curve(REPO / "logs" / "mpc_eval_logs.txt")

# ---------- Figure 1: small-multiples heterogeneity grid ----------
fig, axes = plt.subplots(2, 5, figsize=(16, 6.2), sharex=True, sharey=True)
order = PANEL + ["777", "paper"]
for ax, key in zip(axes.flat, order):
    eps, raw, sm = runs[key]
    if key == "777":
        col, label = C_777, "seed 777 (reference) · healthy · ρₛ +0.76"
    elif key == "paper":
        col, label = C_PAPER, "paper 12345 (resumed) · collapse · ρₛ −0.71"
    else:
        healthy = key not in COLLAPSE
        col = C_HEALTHY if healthy else C_COLLAPSE
        label = (f"seed {key} · {'healthy' if healthy else 'collapse'}"
                 f" · ρₛ {RHO[key]:+.2f}".replace("-", "−"))
    ax.plot(eps, raw, lw=0.7, color=col, alpha=0.30)
    ax.plot(eps, sm, lw=2.0, color=col)
    pk = max(range(len(sm)), key=lambda i: sm[i])
    ax.plot(eps[pk], sm[pk], "o", ms=5, color=col)
    ax.set_title(label, fontsize=9.5)
    ax.axhline(0, lw=0.6, color="#cccccc", zorder=0)
    ax.grid(True, lw=0.3, color="#eeeeee")
for ax in axes[1]:
    ax.set_xlabel("epoch")
for ax in axes[:, 0]:
    ax.set_ylabel("MPC return (20 eps)")
axes.flat[0].set_ylim(-115, 235)
fig.suptitle("MPC return across training for ten runs on the same dataset "
             "(thin lines: raw per-checkpoint means, thick: centred MA-7, dots: smoothed peaks)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(HERE / "fig_panel_curves.png", dpi=170)
plt.close(fig)

# ---------- Figure 2: ROF argmin vs MPC peak dumbbells ----------
rows = [("paper 12345", 250, 310, C_PAPER),
        ("seed 707", 60, 135, C_COLLAPSE),
        ("seed 606", 175, 135, C_COLLAPSE),
        ("seed 505", 5, 265, C_COLLAPSE),
        ("seed 303", 5, 110, C_COLLAPSE),
        ("seed 202", 30, 160, C_COLLAPSE)]
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.axvspan(0, 35, color="#f0f0f0", zorder=0)
ax.text(17, len(rows) - 0.45, "training\nedge", ha="center", va="top",
        fontsize=8, color="#888888")
for y, (name, rmin, peak, col) in enumerate(rows):
    ax.annotate("", xy=(peak, y), xytext=(rmin, y),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8,
                                shrinkA=6, shrinkB=6))
    ax.plot(rmin, y, "o", ms=9, mfc="white", mec=col, mew=2, zorder=3)
    ax.plot(peak, y, "o", ms=9, color=col, zorder=3)
    lead = peak - rmin
    ax.text(max(rmin, peak) + 18, y, f"lead {lead:+d}", va="center",
            fontsize=9.5, color="#333333")
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.set_xlim(-10, 560)
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xlabel("epoch")
ax.set_title("ROF minimum (open circles) and MPC peak (filled) in each collapsing run",
             fontsize=12)
ax.grid(True, axis="x", lw=0.3, color="#eeeeee")
fig.tight_layout()
fig.savefig(HERE / "fig_rof_lead.png", dpi=170)
plt.close(fig)

# ---------- Figure 3: rho_s dot plot ----------
entries = [(f"seed {s}", RHO[s],
            C_HEALTHY if s not in COLLAPSE else C_COLLAPSE, "o") for s in PANEL]
entries += [("seed 777 (ref)", +0.763, C_777, "D"),
            ("paper 12345 (ref)", -0.707, C_PAPER, "D")]
entries.sort(key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.axvline(0, lw=0.8, color="#bbbbbb", zorder=0)
for y, (name, rho, col, mk) in enumerate(entries):
    ax.plot(rho, y, mk, ms=9 if mk == "o" else 8, color=col, zorder=3)
    ax.text(rho + (0.045 if rho >= 0 else -0.045), y,
            f"{rho:+.2f}".replace("-", "−"),
            va="center", ha="left" if rho >= 0 else "right",
            fontsize=9, color="#333333")
ax.set_yticks(range(len(entries)), [e[0] for e in entries])
ax.set_xlim(-1.0, 1.0)
ax.set_xlabel("ρₛ(jac_rof_combined, smoothed MPC)")
ax.set_title("Correlation between jac_rof_combined and smoothed MPC return, per run",
             fontsize=12)
ax.grid(True, axis="x", lw=0.3, color="#eeeeee")
h = [plt.Line2D([], [], marker="o", ls="", color=C_COLLAPSE, label="collapse"),
     plt.Line2D([], [], marker="o", ls="", color=C_HEALTHY, label="healthy"),
     plt.Line2D([], [], marker="D", ls="", color="#888888", label="reference runs")]
ax.legend(handles=h, loc="lower right", fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(HERE / "fig_rho_dist.png", dpi=170)
plt.close(fig)
print("figures written")
