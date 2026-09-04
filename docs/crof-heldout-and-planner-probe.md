# Two follow-ups on the twelve August runs (September 2026)

Both analyses had their predictions written down before any number was computed. Runs,
conventions and raw files are as in `crof-screening-offset-panel.md`; all files are in the
shared repo under `results/`.

## 1. Held-out test of `ol_cumrew_err.trend`

Your best-of-108 candidate, tested on the twelve August runs (not among your seventeen).
Statistic: seed-averaged `ol_cumrew_err`, epochs > 50, centred MA-7, OLS slope against
epoch. Targets as in your screen: MPC drawdown (MA-7 peak minus mean of the last 10 raw
checkpoint means) and final-10.

| run | ol trend (/epoch) | ρ(epoch, ol) | ol mean | MPC drawdown | final-10 |
|---|---|---|---|---|---|
| sp909 | −0.043 | −0.95 | 27.9 | 42.4 | +117.7 |
| sp1010 | −0.039 | −0.93 | 32.4 | 82.3 | +55.2 |
| f25_909 | −0.100 | −0.94 | 57.9 | 142.3 | +17.2 |
| f25_1010 | −0.062 | −0.91 | 52.4 | 79.1 | +2.6 |
| f75_909 | −0.061 | −0.86 | 41.3 | 145.2 | +6.6 |
| f75_1010 | −0.032 | −0.79 | 39.2 | 68.9 | +67.1 |
| scr_909 | −0.005 | −0.12 | 46.4 | 38.2 | −108.5 |
| scr_1010 | +0.028 | +0.52 | 48.2 | 114.3 | −214.5 |
| h128_909 | −0.023 | −0.69 | 28.5 | 81.9 | +85.5 |
| h128_1010 | −0.046 | −0.98 | 35.1 | 32.1 | +117.1 |
| z8_909 | −0.044 | −0.95 | 29.5 | 76.9 | +37.1 |
| z8_1010 | −0.043 | −0.90 | 31.2 | 31.3 | +120.2 |

Predicted: ρ(trend, drawdown) ≥ +0.5 and ρ(trend, final-10) ≤ −0.5. Observed: −0.17
(p = 0.60) and −0.09 (p = 0.78). Not supported.

The table's other message: the open-loop cumulative reward error falls through training in
10 of 12 runs, including every collapsing human-data run (f25_909: −0.10 per epoch while MPC
sheds 142 points). The exceptions are the two scripted (never-good) runs. For comparison,
`jac_rof.drawdown` on the same twelve gives −0.01 against drawdown and −0.65 against
final-10, the latter carried entirely by the two scripted runs (data-family effect).

## 2. Planner-side (edge-of-reach) probe

Script: `eval_edge.py`. Runs: three same-stack pairs from the August panel (sp909/sp1010,
h128_1010/h128_909, z8_1010/z8_909; healthy listed first), plus V7 12345 (3080),
777 (mine), and z-only 12345 (3080). Checkpoints every 25 epochs (20 per run).

Start states: 59 states from the first 40 val episodes at steps t ∈ {20, 100, 200},
kept only when t + 25 precedes both episode end and first leg contact, since env replay
(reset with the episode seed, replay recorded actions) is exact to ~1e-6 before contact
and diverges at it. The same 59 states are used for every checkpoint of every run.

Per state and checkpoint: 5-step posterior warmup on real observations (posterior mean);
CEM plan at the MPC-eval defaults (horizon 25, population 384, elites 48, 4 iterations,
alpha 0.7, gamma 0.97, reward weight 0.2, done penalty 2.5); imagined rollout of the
chosen sequence (prior mean) vs its real execution in the reconstructed env; the recorded
human sequence from the same state as baseline.

- gap_plan: imagined minus real cumulative reward over the horizon, planner sequence
- gap_human: the same for the recorded human sequence
- supp_plan: mean 5-NN Euclidean distance of imagined observations (standardised by
  training-set channel std) to the full training-set observations
- crash: fraction of real planner rollouts terminating within the horizon

Window means (early = epochs 100–200, late = 400–500):

| run | gap_plan early | late | gap_human early | late | supp_plan early | late | crash late |
|---|---|---|---|---|---|---|---|
| sp909 (healthy) | 5.11 | 5.87 | −0.53 | −0.33 | 0.373 | 0.372 | 0.02 |
| sp1010 (collapse) | 4.47 | 5.00 | −0.74 | −1.07 | 0.374 | 0.374 | 0.02 |
| h128_1010 (healthy) | 5.04 | 4.70 | −0.11 | −0.57 | 0.380 | 0.372 | 0.02 |
| h128_909 (collapse) | 0.53 | 3.97 | −2.14 | −0.50 | 0.369 | 0.366 | 0.02 |
| z8_1010 (healthy) | 6.21 | 6.13 | 0.89 | −1.63 | 0.391 | 0.376 | 0.02 |
| z8_909 (collapse) | 3.68 | 6.80 | −3.28 | −0.34 | 0.378 | 0.370 | 0.02 |
| V7 12345 3080 (collapse) | 6.80 | 9.09 | 1.05 | 0.87 | 0.374 | 0.377 | 0.02 |
| 777 (healthy) | 3.41 | 2.37 | −0.61 | −1.47 | 0.361 | 0.377 | 0.02 |
| z-only 12345 3080 (collapse) | 3.36 | 2.96 | −1.61 | 0.40 | 0.383 | 0.370 | 0.02 |

Predictions and outcomes:

- Planner optimism (late − early gap_plan) rises more in the collapsing arm than in its
  healthy partner: 2 of 3 pairs (h128: +3.45 vs −0.34; z8: +3.12 vs −0.08; sp: +0.53 vs
  +0.77).
- The rise is planner-specific (exceeds the rise in gap_human) in the collapsing runs:
  4 of 5 (z-only 3080 the exception: −0.40 vs +2.01).
- Support drift: Δsupp_plan lies between −0.015 and +0.001 on a level of 0.37 in every
  run. Imagined trajectories do not move away from the training support anywhere; the
  pairwise comparison is noise-scale and should not be read as support.
- Within-run tracking, Spearman(gap_plan, MA-7 MPC) over the 20 checkpoints: sp1010 +0.21,
  h128_909 +0.06, z8_909 −0.28, V7 −0.49, z-only −0.49; healthy sp909 gives −0.43. No
  discrimination.

Reading: the strong edge-of-reach form (planner leaving the data support) is not
supported. A weak form is present (a few reward points more optimism per 25-step plan in
three of five collapsing runs) but is not a diagnostic. The probe is blind to the landing
phase by construction.
