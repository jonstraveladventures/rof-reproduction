"""
Stage-B analysis for the CROF cross-environment validation (pre-registered).

Inputs: the LLC MPC sweep log, three metric sweep logs (seeds averaged), and
optionally the exploratory belief-drift log. Produces:
  1. Regime characterisation of the MPC-over-training curve (pre-analysis
     note of 2026-07-06: collapse-present vs Reacher-like monotone).
  2. Correlation table (Pearson, Spearman, quadratic R^2) of metrics vs
     MA-7-smoothed MPC mean return. Headline (pre-committed): jac_rof_combined
     and CROF-A/B. Everything else labelled exploratory/baseline.
  3. CROF checkpoint picks vs the smoothed-MPC oracle.
  4. The pre-registered decision-rule verdict.

Conventions inherited from the paper: centred MA-7 smoothing, alpha=0.5 in
rof_combined, min-max normalised CROF composites, N=100 checkpoints.
"""

import argparse
import glob
import math
import re


def parse_metrics(path):
    out, cur = {}, None
    for line in open(path):
        m = re.match(r"=== Epoch (\d+) ===", line.strip())
        if m:
            cur = int(m.group(1)); out[cur] = {}
            continue
        if cur is not None:
            for k, v in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([-+0-9.eE]+)", line):
                try:
                    out[cur][k] = float(v)
                except ValueError:
                    pass
    return out


def parse_mpc(path):
    returns, cur = {}, None
    for line in open(path):
        m = re.search(r"\[\d+/\d+\] Epoch (\d+) --", line)
        if m:
            cur = int(m.group(1)); returns[cur] = []
            continue
        m = re.search(r"\[Episode \d+\] return=([-+0-9.]+),", line)
        if m and cur is not None:
            returns[cur].append(float(m.group(1)))
    return returns


def ma(xs, w=7):
    out = []
    for i in range(len(xs)):
        lo, hi = max(0, i - w // 2), min(len(xs), i + w // 2 + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def pearson(x, y):
    n = len(x); mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((a - my) ** 2 for a in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def rank(x):
    idx = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            r[idx[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    return pearson(rank(x), rank(y))


def r2_quad(x, y):
    n = len(x)
    X = [[1.0, xi, xi * xi] for xi in x]
    # normal equations, 3x3
    A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(3)] for a in range(3)]
    B = [sum(X[i][a] * y[i] for i in range(n)) for a in range(3)]
    # gaussian elimination
    for c in range(3):
        p = max(range(c, 3), key=lambda r_: abs(A[r_][c]))
        A[c], A[p] = A[p], A[c]; B[c], B[p] = B[p], B[c]
        if abs(A[c][c]) < 1e-12:
            return float("nan")
        for r_ in range(c + 1, 3):
            f = A[r_][c] / A[c][c]
            for cc in range(c, 3):
                A[r_][cc] -= f * A[c][cc]
            B[r_] -= f * B[c]
    coef = [0.0] * 3
    for c in (2, 1, 0):
        coef[c] = (B[c] - sum(A[c][cc] * coef[cc] for cc in range(c + 1, 3))) / A[c][c]
    yhat = [coef[0] + coef[1] * xi + coef[2] * xi * xi for xi in x]
    my = sum(y) / n
    ss_res = sum((yi - yh) ** 2 for yi, yh in zip(y, yhat))
    ss_tot = sum((yi - my) ** 2 for yi in y)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def norm(xs):
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in xs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mpc_log", required=True)
    p.add_argument("--metrics_logs", required=True, help="glob, e.g. 'metrics_llc_seed*.txt'")
    p.add_argument("--drift_log", default=None)
    args = p.parse_args()

    mpc = parse_mpc(args.mpc_log)
    metric_runs = [parse_metrics(f) for f in sorted(glob.glob(args.metrics_logs))]
    assert metric_runs, "no metrics logs matched"
    drift = parse_metrics(args.drift_log) if args.drift_log else {}

    epochs = sorted(set(mpc) & set.intersection(*[set(m) for m in metric_runs]))
    print(f"matched epochs: {len(epochs)} ({epochs[0]}-{epochs[-1]}), "
          f"metric seeds: {len(metric_runs)}, episodes/ckpt: {len(mpc[epochs[0]])}")

    mean_ret = [sum(mpc[e]) / len(mpc[e]) for e in epochs]
    se_ret = [(sum((r - sum(mpc[e]) / len(mpc[e])) ** 2 for r in mpc[e]) / (len(mpc[e]) - 1)) ** 0.5
              / len(mpc[e]) ** 0.5 for e in epochs]
    sm = ma(mean_ret)

    # ---- 1. Regime characterisation (interpretation gate, pre-analysis note) ----
    imax = sm.index(max(sm))
    tail = sum(sm[-10:]) / 10
    span = max(sm) - min(sm)
    collapse = (max(sm) - tail) > 0.5 * span
    rho_epoch = spearman(list(range(len(sm))), sm)
    print("\n--- REGIME CHECK (read before the correlations) ---")
    print(f"smoothed return: start {sm[0]:+.1f}, peak {max(sm):+.1f} @ epoch {epochs[imax]}, "
          f"final-10 mean {tail:+.1f}, span {span:.1f}")
    print(f"monotonicity rho(epoch, smoothed return) = {rho_epoch:+.2f}")
    print(f"collapse-present (peak - tail > 0.5*span): {collapse}")
    print(f"median per-checkpoint SE of mean return: {sorted(se_ret)[len(se_ret)//2]:.1f}")
    print("interpretation: collapse-present -> pre-registered thresholds carry their meaning;"
          " near-monotone -> a weak rho is a 'no selection problem in this regime' null.")

    # ---- 2. Averaged metrics + composites ----
    def col(name):
        vals = []
        for e in epochs:
            per_seed = [m[e][name] for m in metric_runs if name in m[e]]
            vals.append(sum(per_seed) / len(per_seed))
        return vals

    rof_comb = [0.5 * a + 0.5 * b for a, b in zip(col("jac_rof"), col("jac_rof_bad"))]
    n_rof, n_kc, n_ko, n_eo = (norm(rof_comb), norm(col("jac_ctrl_rank")),
                               norm(col("jac_obs_rank")), norm(col("ol_obs_avg")))
    crof_a = [r + (1 - kc) + (1 - ko) + e for r, kc, ko, e in zip(n_rof, n_kc, n_ko, n_eo)]
    crof_b = [r + 0.5 * ((1 - kc) + (1 - ko) + e) for r, kc, ko, e in zip(n_rof, n_kc, n_ko, n_eo)]

    table = [("jac_rof_combined [HEADLINE]", rof_comb), ("CROF-A [HEADLINE]", crof_a),
             ("CROF-B [HEADLINE]", crof_b), ("jac_rof", col("jac_rof")),
             ("jac_rof_bad", col("jac_rof_bad")), ("val_loss [baseline]", col("val_loss")),
             ("val_kl [baseline]", col("val_kl")), ("post_obs_rmse [baseline]", col("post_obs_rmse")),
             ("post_rew_rmse [baseline]", col("post_rew_rmse")), ("ol_obs_avg [baseline]", col("ol_obs_avg")),
             ("ol_rew_end [baseline]", col("ol_rew_end")), ("ol_cumrew_err [baseline]", col("ol_cumrew_err")),
             ("jac_ctrl_rank", col("jac_ctrl_rank")), ("jac_obs_rank", col("jac_obs_rank")),
             ("emp_C", col("emp_C")), ("emp_O", col("emp_O")), ("emp_L", col("emp_L"))]
    if drift:
        d_epochs = set(drift)
        if all(e in d_epochs for e in epochs):
            for name in ("drift_rew_avg", "drift_rew_end", "drift_kl_avg", "drift_kl_end", "drift_h_avg"):
                table.append((f"{name} [exploratory]", [drift[e][name] for e in epochs]))

    print("\n--- CORRELATIONS vs MA-7 smoothed MPC mean ---")
    print(f"{'metric':34} {'pearson':>9} {'spearman':>9} {'R2quad':>8}")
    rows = []
    for name, x in table:
        rows.append((name, pearson(x, sm), spearman(x, sm), r2_quad(x, sm)))
    for name, pe, sp, r2 in sorted(rows, key=lambda t: -abs(t[2])):
        print(f"{name:34} {pe:>+9.3f} {sp:>+9.3f} {r2:>8.3f}")

    # ---- 3. Checkpoint selection ----
    oracle_e, oracle_v = epochs[imax], max(sm)
    print("\n--- CHECKPOINT SELECTION ---")
    print(f"oracle (smoothed MPC): epoch {oracle_e} at {oracle_v:+.1f}")
    for name, score in (("CROF-A", crof_a), ("CROF-B", crof_b), ("rof_combined", rof_comb)):
        pick = epochs[score.index(min(score))]
        pick_v = sm[epochs.index(pick)]
        gap = oracle_v - pick_v
        rel = gap / abs(oracle_v) if oracle_v != 0 else float("nan")
        print(f"min {name} (raw): epoch {pick}, smoothed return {pick_v:+.1f} "
              f"(gap {gap:+.1f}, {100*rel:.0f}% of oracle)")

    # ---- 4. Decision rule ----
    rho = spearman(rof_comb, sm)
    print("\n--- PRE-REGISTERED DECISION RULE ---")
    print(f"rho_s(jac_rof_combined, smoothed MPC) = {rho:+.3f}")
    if rho <= -0.45:
        print("VERDICT: rho <= -0.45 -> ROF generalises. Proceed: second environment + write-up.")
    elif abs(rho) < 0.30:
        print("VERDICT: |rho| < 0.30 -> run dataset-provenance control; scope-limit write-up; stop.")
    else:
        print("VERDICT: ambiguous band (-0.45, -0.30] -> two more metric seeds, judge pooled estimate.")
    print("(Prediction on record: rho in [-0.65, -0.45], P(generalise)=0.65. "
          "Interpret via the regime check above.)")


if __name__ == "__main__":
    main()
