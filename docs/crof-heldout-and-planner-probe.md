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

## 3. Closed-loop probe (instrumented MPC episodes)

Script: `eval_closedloop.py`. The MPC-eval loop reproduced exactly (env reset seed 12345 +
episode, `set_seed` per episode, `rssm.step` belief update with the previous action, CEM at
the eval defaults, action 0 once landed, 600-step cap), with logging added at every planner
step. Same nine arms; checkpoints every 50 epochs (ten per arm); ten episodes per
checkpoint. Predictions written down before running.

Per step: R̂ (the planner's predicted 25-step return of its chosen sequence, prior-mean
rollout), R (the realised return over the next 25 real steps), gap = R̂ − R, recon (standardised
error between the decoded belief and the current observation), pred1 (standardised error
between the imagined next observation under the executed action and the real one).
Phase: near-ground if y < 0.35, else flight. Windows: early = epochs 100–200, late = 400–500.

Manipulation check: per-checkpoint mean return of the instrumented episodes against the MPC
log's mean over episodes 1–10 at the same epochs: mean differences −15 to +10 return points,
correlations 0.46–0.92, no systematic offset. The residual scatter is CPU-vs-GPU divergence
on ten-episode means and is large: late-window crash fractions run 0.2–0.8, so window means
move by tens of return points (z8_1010, healthy by its 20-episode sweep, gives 98 early and
63 late here).

| run | return E | L | gap near E | L | gap flight E | L | R̂ near E | L | recon near E | L | pred1 near E | L | crash L |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sp909 (healthy) | 126 | 89 | 17.3 | 25.3 | 16.7 | 17.5 | 17.3 | 23.3 | 0.184 | 0.218 | 0.220 | 0.261 | 0.33 |
| sp1010 (collapse) | 65 | 51 | 34.7 | 29.6 | 21.5 | 14.6 | 26.9 | 18.3 | 0.282 | 0.395 | 0.337 | 0.427 | 0.63 |
| h128_1010 (healthy) | 82 | 137 | 25.4 | 26.2 | 20.3 | 18.2 | 20.5 | 28.1 | 0.261 | 0.339 | 0.314 | 0.416 | 0.23 |
| h128_909 (collapse) | 146 | 100 | 25.2 | 20.1 | 19.1 | 13.7 | 24.0 | 16.9 | 0.276 | 0.260 | 0.352 | 0.309 | 0.27 |
| z8_1010 (healthy) | 98 | 63 | 23.4 | 40.7 | 21.4 | 20.9 | 19.1 | 27.9 | 0.229 | 0.315 | 0.268 | 0.346 | 0.43 |
| z8_909 (collapse) | 33 | 58 | 28.9 | 35.4 | 16.6 | 18.9 | 13.6 | 25.3 | 0.396 | 0.477 | 0.465 | 0.542 | 0.47 |
| V7 12345 3080 (collapse) | 57 | −17 | 32.2 | 83.9 | 21.1 | 24.0 | 25.6 | 48.7 | 0.189 | 0.292 | 0.253 | 0.367 | 0.80 |
| 777 (healthy) | 121 | 168 | 22.7 | 20.6 | 16.1 | 12.6 | 21.2 | 25.1 | 0.239 | 0.334 | 0.288 | 0.368 | 0.20 |
| z-only 12345 3080 (collapse) | −21 | 25 | 37.8 | 46.2 | 17.2 | 10.7 | 16.1 | 20.5 | 0.230 | 0.331 | 0.269 | 0.387 | 0.73 |

Predictions and outcomes:

- Near-ground optimism (late − early gap) rises more in the collapsing arm than in its
  healthy partner: 0 of 3 pairs (z8_1010, healthy, rose +17.3; sp1010 and h128_909 fell).
- In collapsing runs the near-ground rise exceeds the flight rise: 5 of 5 formally, but in
  sp1010 and h128_909 both changes are negative (−5.2 vs −6.9, −5.1 vs −5.5); the real
  cases are V7 (+51.7 vs +2.9), z-only (+8.4 vs −6.5), z8_909 (+6.6 vs +2.3).
- Near-ground belief error (recon) rises more in the collapsing arm: 1 of 3. It rises in
  every run, healthy included (+0.03 to +0.11), so the rise is a universal late-training
  effect, not the collapse.
- The planner's near-ground R̂ holds up (late ≥ 0.8 × early) in collapsing runs: 3 of 5. It
  falls in sp1010 and h128_909; it doubles in V7 (25.6 → 48.7) while V7's return goes to −17.
- Within-run, near-ground recon tracks episode return (Spearman ≤ −0.5): 1 of 5 (V7, −0.68;
  its pred1 −0.75). gap tracks return in 4 of 5 collapsing runs (−0.61 to −0.83), but that is
  the mechanical link through the realised return; healthy arms give −0.16 to −0.49.

Reading: no belief-drift or model-accuracy mechanism. One arm shows the confidently-wrong
landing signature in full: V7, the severe collapse, whose predicted near-ground return doubles
while its realised return collapses, with near-ground belief error tracking the fall. The
moderate collapses do not show it, and at ten episodes per checkpoint the pairwise
comparisons have little power. Five mechanism accounts have now been tested this way and
none survives as a general one.
