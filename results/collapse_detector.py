"""Nikolai's proposed collapse-detection procedure, run on both seed panels.

His procedure (email 2026-07-23), verbatim:
  1. Warmup: ignore epochs <= 50.
  2. Smooth: MA-7 on each pool (good=jac_rof, bad=jac_rof_bad) separately.
  3. Per pool, trailing running maximum since warmup.
  4. Fire if EITHER pool's smoothed ROF drops >= ~2x the pool's
     checkpoint-to-checkpoint noise std below its trailing max, for
     >= 4 consecutive checkpoints (~20 epochs).
  5. Optional: require good-pool net drop |delta| > ~0.02.

Ground-truth collapse labels come from analyze_llc's boolean on the MPC curve.
Reports a confusion matrix and, for each fire, its epoch vs the MPC peak
(lead > 0 = fires before the peak = genuine early warning).
"""
import importlib.util
from pathlib import Path
from statistics import pstdev

HERE = Path(__file__).resolve().parent
# The analysis repo: the parent of results/, or a LunarLander_RSSM folder
# beside one of its parents.
REPO = next(p / s for p in HERE.parents for s in ("", "LunarLander_RSSM")
            if (p / s / "analyze_llc.py").exists())
spec = importlib.util.spec_from_file_location("an", REPO / "analyze_llc.py")
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)

WARMUP = 50
NOISE_MULT = 2.0
CONSEC = 4
GOOD_DROP = 0.02


def pool_series(runs, key, epochs):
    return [sum(m[e][key] for m in runs) / len(runs) for e in epochs]


def detect_pool(raw, epochs):
    """Return (fires, first_fire_epoch) for one pool's raw ROF series."""
    idx = [i for i, e in enumerate(epochs) if e > WARMUP]
    if len(idx) < CONSEC + 1:
        return False, None
    ep = [epochs[i] for i in idx]
    raw_w = [raw[i] for i in idx]
    sm = an.ma(raw_w)                                   # step 2
    diffs = [raw_w[i] - raw_w[i - 1] for i in range(1, len(raw_w))]
    noise = pstdev(diffs) if len(diffs) > 1 else 0.0    # checkpoint-to-checkpoint noise
    thr = NOISE_MULT * noise
    run_max, flags = sm[0], []
    for v in sm:                                        # steps 3-4
        run_max = max(run_max, v)
        flags.append((run_max - v) >= thr and thr > 0)
    c = 0
    for i, f in enumerate(flags):
        c = c + 1 if f else 0
        if c >= CONSEC:
            return True, ep[i - CONSEC + 1]
    return False, None


def run(name, mpc_path, metric_glob, collapse, **expect):
    epochs, mpc, mruns = an.load_run(mpc_path, metric_glob, **expect)
    means = [sum(mpc[e]) / len(mpc[e]) for e in epochs]
    sm = an.ma(means)
    mpc_peak_ep = epochs[sm.index(max(sm))]

    good = pool_series(mruns, "jac_rof", epochs)
    bad = pool_series(mruns, "jac_rof_bad", epochs)
    fg, eg = detect_pool(good, epochs)
    fb, eb = detect_pool(bad, epochs)
    fires_core = fg or fb
    fire_ep = min([e for e in (eg, eb) if e is not None], default=None)

    # optional good-pool net-drop gate
    gi = [i for i, e in enumerate(epochs) if e > WARMUP]
    gsm = an.ma([good[i] for i in gi])
    good_net = max(gsm) - gsm[-1]
    fires_gated = fires_core and good_net > GOOD_DROP

    lead = (mpc_peak_ep - fire_ep) if fire_ep is not None else None
    return dict(name=name, collapse=collapse, fires=fires_core,
                fires_gated=fires_gated, fire_ep=fire_ep,
                mpc_peak=mpc_peak_ep, lead=lead, good_net=good_net)


LL = [("LL-101", False), ("LL-202", True), ("LL-303", True), ("LL-404", False),
      ("LL-505", True), ("LL-606", True), ("LL-707", True), ("LL-808", False)]
REACHER = [f"R-{s}" for s in (101, 202, 303, 404, 505, 606, 707, 808)]

rows = []
for tag, coll in LL:
    s = tag.split("-")[1]
    mpc_f = HERE / ("mpc_sp707_merged.txt" if s == "707" else f"mpc_sp{s}.txt")
    rows.append(run(tag, mpc_f, HERE / f"metrics_sp{s}_seed*.txt", coll))
rows.append(run("LL-777 (ref)", HERE / "mpc_dq_f1rs.txt",
                HERE / "metrics_dq_f1rs_seed*.txt", False))
rows.append(run("LL-paper (ref)", REPO / "logs" / "mpc_eval_logs.txt",
                REPO / "logs" / "metrics_eval_logs.txt", True, expected_seeds=(12345,)))
for tag in REACHER:
    s = tag.split("-")[1]
    # r606 epoch 10 was overwritten (recorded 2026-07-21).
    rows.append(run(tag, HERE / f"mpc_r{s}.txt", HERE / f"metrics_r{s}_seed*.txt", False,
                    known_missing=(10,) if s == "606" else ()))

print(f"{'run':16} {'collapse':>8} {'fires':>6} {'gated':>6} {'fire@':>6} "
      f"{'peak@':>6} {'lead':>6} {'good_net':>9}")
for r in rows:
    print(f"{r['name']:16} {str(r['collapse']):>8} {str(r['fires']):>6} "
          f"{str(r['fires_gated']):>6} {str(r['fire_ep'] or '-'):>6} "
          f"{r['mpc_peak']:>6} {str(r['lead'] if r['lead'] is not None else '-'):>6} "
          f"{r['good_net']:>+9.3f}")

for label, key in (("CORE (steps 1-4)", "fires"), ("GATED (+ step 5)", "fires_gated")):
    tp = sum(1 for r in rows if r["collapse"] and r[key])
    fn = sum(1 for r in rows if r["collapse"] and not r[key])
    fp = sum(1 for r in rows if not r["collapse"] and r[key])
    tn = sum(1 for r in rows if not r["collapse"] and not r[key])
    leads = [r["lead"] for r in rows if r["collapse"] and r[key] and r["lead"] is not None]
    print(f"\n{label}: sensitivity {tp}/{tp+fn} caught, "
          f"specificity {tn}/{tn+fp} silent, false alarms {fp}")
    if leads:
        print(f"  leads on caught collapses (epochs, + = before MPC peak): "
              f"{sorted(leads, reverse=True)}")
